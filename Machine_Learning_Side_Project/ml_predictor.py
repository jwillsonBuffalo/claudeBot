"""
ml_predictor.py — XGBoost-based ML trading engine for Polymarket BTC Up/Down markets (5m and 15m).

Classes
-------
BinanceDataFetcher  — fetches OHLCV 1m candles, funding rates from Binance REST
FeatureEngine       — computes 27 features from raw OHLCV + Polymarket context
XGBoostModel        — wraps XGBClassifier with Platt calibration; save/load
BacktestEngine      — walk-forward validation on Binance historical data
MLTrader            — thread-based trader; mirrors AdvancedTrader lifecycle

Supports 5-minute and 15-minute Polymarket markets (BTC Up/Down).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Persistence paths (relative to cwd — app.py sets cwd to project root)
# ---------------------------------------------------------------------------
ML_MODEL_FILE   = "ml_model.pkl"
ML_STATE_FILE   = "ml_state.json"
ML_OUTCOMES_FILE = "ml_outcomes.json"

# ---------------------------------------------------------------------------
# 1. BinanceDataFetcher
# ---------------------------------------------------------------------------

class BinanceDataFetcher:
    """
    Fetches asset/USDT 1m candles and funding rates from Binance REST APIs.
    Supports any asset (BTC, BNB, etc.) via the symbol parameter.
    Uses a simple TTL cache to avoid hammering the API.
    """

    _CANDLE_URL  = "https://api.binance.com/api/v3/klines"
    _CANDLE_FALLBACK = "https://api.binance.us/api/v3/klines"
    _FUNDING_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
    _CACHE_TTL   = 10  # seconds

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self._symbol = symbol.upper()
        self._candle_cache: Optional[List[dict]] = None
        self._candle_ts: float = 0.0
        self._candle_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-ml-bot/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candles(self, limit: int = 60) -> List[dict]:
        """
        Returns last `limit` completed 1m candles.
        Each candle: {ts_ms, open, high, low, close, volume}
        Cached for _CACHE_TTL seconds.
        """
        with self._candle_lock:
            now = time.monotonic()
            if self._candle_cache and (now - self._candle_ts) < self._CACHE_TTL:
                cached = self._candle_cache
                return cached[-limit:] if len(cached) >= limit else cached

            candles = self._fetch_candles(max(limit, 100))
            if candles:
                self._candle_cache = candles
                self._candle_ts = now
            return (candles[-limit:] if candles and len(candles) >= limit
                    else (candles or []))

    def get_candles_historical(
        self,
        days: int = 90,
        progress_callback=None,
    ) -> List[dict]:
        """
        Batch-fetches `days` worth of 1m candles (1440 per day).
        Paginates backwards from now. Used only by BacktestEngine.
        Returns candles sorted oldest → newest.

        progress_callback(fetched: int, total: int) is called after each batch.
        """
        total_needed = days * 24 * 60
        all_candles: List[dict] = []
        end_ms = int(time.time() * 1000)
        batch = 1000  # Binance klines max per request

        while len(all_candles) < total_needed:
            needed = total_needed - len(all_candles)
            fetch_n = min(needed, batch)
            start_ms = end_ms - fetch_n * 60_000

            raw = self._fetch_raw(
                self._CANDLE_URL,
                params={
                    "symbol": self._symbol,
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms - 1,
                    "limit": fetch_n,
                },
                fallback_url=self._CANDLE_FALLBACK,
            )
            if not raw:
                break
            batch_candles = [self._parse_kline(k) for k in raw]
            all_candles = batch_candles + all_candles
            end_ms = start_ms
            if progress_callback:
                try:
                    progress_callback(len(all_candles), total_needed)
                except Exception:
                    pass
            if len(raw) < fetch_n:
                break  # no more data

        return all_candles

    def get_funding_rate(self) -> float:
        """Returns current Binance USDT-M perpetual funding rate, or 0.0 on error."""
        try:
            resp = self._session.get(
                self._FUNDING_URL, params={"symbol": self._symbol}, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("lastFundingRate", 0) or 0)
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_candles(self, limit: int) -> List[dict]:
        raw = self._fetch_raw(
            self._CANDLE_URL,
            params={"symbol": self._symbol, "interval": "1m", "limit": limit},
            fallback_url=self._CANDLE_FALLBACK,
        )
        if not raw:
            return []
        # Drop the in-progress (last) candle — only return completed candles
        completed = raw[:-1] if len(raw) > 1 else raw
        return [self._parse_kline(k) for k in completed]

    def _fetch_raw(self, url: str, params: dict,
                   fallback_url: Optional[str] = None) -> Optional[list]:
        for attempt_url in ([url, fallback_url] if fallback_url else [url]):
            if not attempt_url:
                continue
            try:
                resp = self._session.get(attempt_url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return None

    @staticmethod
    def _parse_kline(k: list) -> dict:
        return {
            "ts_ms":  int(k[0]),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        }


# ---------------------------------------------------------------------------
# 2. FeatureEngine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Pure-function class (no mutable state) that computes 27 features from
    OHLCV candles + Polymarket context data.

    Feature order is fixed — changing it invalidates saved models.
    """

    FEATURE_NAMES: List[str] = [
        "ret_1m",        # 0
        "ret_5m",        # 1
        "ret_15m",       # 2
        "ret_30m",       # 3
        "rsi_14",        # 4  normalized [0,1]
        "rsi_delta",     # 5  RSI momentum
        "macd_hist",     # 6  MACD histogram / close
        "bb_pct",        # 7  Bollinger %B
        "atr_pct",       # 8  ATR(14) / close
        "ema_diff",      # 9  (EMA5-EMA20)/close
        "ema_cross",     # 10 cross signal ∈ {-2,-1,0,1,2}
        "vol_ratio_5",   # 11 volume vs 5-bar MA
        "vol_ratio_20",  # 12 volume vs 20-bar MA
        "body_size",     # 13 |open-close| / (high-low)
        "upper_shadow",  # 14
        "lower_shadow",  # 15
        "momentum_3",    # 16 3-bar momentum ∈ {-3..3}
        "hour_sin",      # 17
        "hour_cos",      # 18
        "dow_sin",       # 19
        "dow_cos",       # 20
        "funding_rate",  # 21 clipped ±0.01
        "up_ask",        # 22 Polymarket Up best ask
        "down_ask",      # 23 Polymarket Down best ask
        "efficiency_gap",# 24 up_ask + down_ask - 1.0
        "price_vs_ptb",  # 25 (btc - price_to_beat) / price_to_beat
        "secs_into_win", # 26 secs elapsed / window_sec
        # ── Analysis engine signals (indices 27-37) ──────────────────────────
        # Set to 0 during backtest (not available historically).
        # Live prediction populates these from /api/analysis.
        "an_direction",  # 27 composite direction: -1=DOWN, 0=NEUTRAL, +1=UP
        "an_confidence", # 28 composite confidence 0.0–1.0
        "an_mom",        # 29 momentum signal
        "an_rsi",        # 30 RSI signal
        "an_ema",        # 31 EMA cross signal
        "an_macd",       # 32 MACD signal
        "an_bb",         # 33 Bollinger Bands signal
        "an_supertrend", # 34 Supertrend signal
        "an_stoch_rsi",  # 35 Stochastic RSI signal
        "an_vol",        # 36 volume signal (HIGH=+1, LOW=-1, NORMAL=0)
        "an_adx",        # 37 ADX strength 0.0–1.0
    ]

    MIN_CANDLES = 60  # need at least 60 candles for stable indicator values

    @staticmethod
    def _sig_to_int(sig: Optional[str]) -> float:
        """Convert a signal string to -1 / 0 / +1."""
        if not sig:
            return 0.0
        s = sig.upper()
        if s in ("UP", "HIGH", "BULLISH"):
            return 1.0
        if s in ("DOWN", "LOW", "BEARISH"):
            return -1.0
        return 0.0

    @classmethod
    def compute(
        cls,
        candles: List[dict],
        funding_rate: float = 0.0,
        up_ask: float = 0.0,
        down_ask: float = 0.0,
        price_to_beat: float = 0.0,
        window_start_ts: float = 0.0,
        analysis_signals: Optional[dict] = None,
        window_sec: int = 300,
    ) -> Optional[Dict[str, float]]:
        """
        Returns a {feature_name: float} dict, or None if insufficient data.

        Args:
            candles:         list of completed 1m candles (oldest first)
            funding_rate:    Binance USDT-M perpetual funding rate
            up_ask:          Polymarket Up outcome best ask (0.0 if unknown)
            down_ask:        Polymarket Down outcome best ask (0.0 if unknown)
            price_to_beat:   Polymarket oracle BTC price at window open
            window_start_ts: Unix timestamp of window open
        """
        if len(candles) < cls.MIN_CANDLES:
            return None

        closes  = [c["close"]  for c in candles]
        opens   = [c["open"]   for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]

        last_close = closes[-1]
        if last_close <= 0:
            return None

        # --- Momentum returns ---
        ret_1m  = closes[-1] / closes[-2] - 1 if closes[-2] > 0 else 0.0
        ret_5m  = closes[-1] / closes[-6] - 1 if len(closes) >= 6  and closes[-6] > 0 else 0.0
        ret_15m = closes[-1] / closes[-16] - 1 if len(closes) >= 16 and closes[-16] > 0 else 0.0
        ret_30m = closes[-1] / closes[-31] - 1 if len(closes) >= 31 and closes[-31] > 0 else 0.0

        # --- RSI ---
        rsi_raw   = cls._calc_rsi(closes[-15:], 14)       # 0..100
        rsi_14    = rsi_raw / 100.0                        # normalized
        rsi_raw_prev = cls._calc_rsi(closes[-16:-1], 14)
        rsi_delta = (rsi_raw - rsi_raw_prev) / 100.0

        # --- MACD ---
        macd_line, signal_line = cls._calc_macd(closes)
        macd_hist = (macd_line - signal_line) / last_close

        # --- Bollinger Bands ---
        bb_upper, bb_mid, bb_lower = cls._calc_bbands(closes)
        bb_range = bb_upper - bb_lower
        bb_pct = (last_close - bb_lower) / bb_range if bb_range > 0 else 0.5

        # --- ATR ---
        atr_pct = cls._calc_atr(candles[-15:], 14) / last_close

        # --- EMA cross ---
        ema5_series  = cls._calc_ema(closes, 5)
        ema20_series = cls._calc_ema(closes, 20)
        ema_diff = (ema5_series[-1] - ema20_series[-1]) / last_close
        sign_now  = 1 if ema5_series[-1] > ema20_series[-1] else -1
        sign_prev = 1 if ema5_series[-2] > ema20_series[-2] else -1
        ema_cross = float(sign_now - sign_prev)  # ∈ {-2, 0, 2}

        # --- Volume ratios ---
        vol_last = volumes[-1]
        vol_ratio_5  = vol_last / (sum(volumes[-6:-1]) / 5) if sum(volumes[-6:-1]) > 0 else 1.0
        vol_ratio_20 = vol_last / (sum(volumes[-21:-1]) / 20) if sum(volumes[-21:-1]) > 0 else 1.0

        # --- Candle shape (last completed candle) ---
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        hl = h - l + 1e-8
        body_size    = abs(o - c) / hl
        upper_shadow = (h - max(o, c)) / hl
        lower_shadow = (min(o, c) - l) / hl

        # --- 3-bar momentum ---
        def _sign(x: float) -> int:
            return 1 if x > 0 else (-1 if x < 0 else 0)
        momentum_3 = float(sum(
            _sign(closes[-i] - closes[-i - 1]) for i in range(1, 4)
        ))

        # --- Time encoding ---
        ts = window_start_ts if window_start_ts > 0 else time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_sin = math.sin(2 * math.pi * dt.hour / 24)
        hour_cos = math.cos(2 * math.pi * dt.hour / 24)
        dow_sin  = math.sin(2 * math.pi * dt.weekday() / 7)
        dow_cos  = math.cos(2 * math.pi * dt.weekday() / 7)

        # --- Funding rate (clamp) ---
        funding_clamped = max(-0.01, min(0.01, float(funding_rate)))

        # --- Polymarket context ---
        efficiency_gap = (up_ask + down_ask - 1.0) if (up_ask > 0 and down_ask > 0) else 0.0
        price_vs_ptb   = ((last_close - price_to_beat) / price_to_beat
                          if price_to_beat > 0 else 0.0)

        # --- Window timing ---
        secs_into_win = 0.0
        if window_start_ts > 0:
            secs_into_win = min(1.0, max(0.0, (time.time() - window_start_ts) / window_sec))

        # --- Analysis engine signals (0 when not available, e.g. during backtest) ---
        an = analysis_signals or {}
        sigs = an.get("signals", {})
        an_direction  = cls._sig_to_int(an.get("prediction"))
        an_confidence = (an.get("confidence") or 0) / 100.0
        an_mom        = cls._sig_to_int(sigs.get("momentum", {}).get("signal"))
        an_rsi        = cls._sig_to_int(sigs.get("rsi", {}).get("signal"))
        an_ema        = cls._sig_to_int(sigs.get("ema_cross", {}).get("signal"))
        an_macd       = cls._sig_to_int(sigs.get("macd", {}).get("signal"))
        an_bb         = cls._sig_to_int(sigs.get("bbands", {}).get("signal"))
        an_supertrend = cls._sig_to_int(sigs.get("supertrend", {}).get("signal"))
        an_stoch_rsi  = cls._sig_to_int(sigs.get("stoch_rsi", {}).get("signal"))
        an_vol        = cls._sig_to_int(sigs.get("volume", {}).get("signal"))
        an_adx        = min((sigs.get("adx", {}).get("adx") or 0) / 100.0, 1.0)

        return {
            "ret_1m":         ret_1m,
            "ret_5m":         ret_5m,
            "ret_15m":        ret_15m,
            "ret_30m":        ret_30m,
            "rsi_14":         rsi_14,
            "rsi_delta":      rsi_delta,
            "macd_hist":      macd_hist,
            "bb_pct":         bb_pct,
            "atr_pct":        atr_pct,
            "ema_diff":       ema_diff,
            "ema_cross":      ema_cross,
            "vol_ratio_5":    vol_ratio_5,
            "vol_ratio_20":   vol_ratio_20,
            "body_size":      body_size,
            "upper_shadow":   upper_shadow,
            "lower_shadow":   lower_shadow,
            "momentum_3":     momentum_3,
            "hour_sin":       hour_sin,
            "hour_cos":       hour_cos,
            "dow_sin":        dow_sin,
            "dow_cos":        dow_cos,
            "funding_rate":   funding_clamped,
            "up_ask":         float(up_ask),
            "down_ask":       float(down_ask),
            "efficiency_gap": efficiency_gap,
            "price_vs_ptb":   price_vs_ptb,
            "secs_into_win":  secs_into_win,
            "an_direction":   an_direction,
            "an_confidence":  an_confidence,
            "an_mom":         an_mom,
            "an_rsi":         an_rsi,
            "an_ema":         an_ema,
            "an_macd":        an_macd,
            "an_bb":          an_bb,
            "an_supertrend":  an_supertrend,
            "an_stoch_rsi":   an_stoch_rsi,
            "an_vol":         an_vol,
            "an_adx":         an_adx,
        }

    @classmethod
    def to_array(cls, features: Dict[str, float]) -> List[float]:
        """Converts a features dict to an ordered array matching FEATURE_NAMES."""
        return [features.get(name, 0.0) for name in cls.FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Private indicator helpers (no external deps)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ema(closes: List[float], period: int) -> List[float]:
        """Returns EMA series same length as closes. First value = SMA of first `period` bars."""
        if len(closes) < period:
            return [closes[-1]] * len(closes)
        alpha = 2.0 / (period + 1)
        ema = [sum(closes[:period]) / period]
        for c in closes[period:]:
            ema.append(ema[-1] + alpha * (c - ema[-1]))
        # Prepend fill to match length
        pad = [ema[0]] * (len(closes) - len(ema))
        return pad + ema

    @staticmethod
    def _calc_rsi(closes: List[float], period: int = 14) -> float:
        """Wilder RSI. Returns 0-100. Needs period+1 values."""
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calc_macd(closes: List[float],
                   fast: int = 6, slow: int = 13, signal: int = 5
                   ) -> Tuple[float, float]:
        """Returns (macd_line, signal_line) for the final bar."""
        def ema_last(data: List[float], p: int) -> List[float]:
            if len(data) < p:
                v = data[-1] if data else 0.0
                return [v] * len(data)
            alpha = 2.0 / (p + 1)
            e = sum(data[:p]) / p
            out = [e]
            for x in data[p:]:
                e = e + alpha * (x - e)
                out.append(e)
            return [out[0]] * (len(data) - len(out)) + out

        fast_ema = ema_last(closes, fast)
        slow_ema = ema_last(closes, slow)
        macd = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_ema = ema_last(macd, signal)
        return macd[-1], signal_ema[-1]

    @staticmethod
    def _calc_bbands(closes: List[float],
                     period: int = 20, num_std: float = 2.0
                     ) -> Tuple[float, float, float]:
        """Returns (upper, middle, lower) for the final bar."""
        if len(closes) < period:
            last = closes[-1]
            return last, last, last
        window = closes[-period:]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)
        return mid + num_std * std, mid, mid - num_std * std

    @staticmethod
    def _calc_atr(candles: List[dict], period: int = 14) -> float:
        """Average True Range over last `period` bars."""
        if len(candles) < 2:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]
            l = candles[i]["low"]
            prev_c = candles[i - 1]["close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        if not trs:
            return 0.0
        use = trs[-period:]
        return sum(use) / len(use)


# ---------------------------------------------------------------------------
# 3. XGBoostModel
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    Wraps XGBClassifier + Platt scaling (LogisticRegression) for calibrated P(UP).

    Persistence:
      ml_model.pkl   — pickled (xgb_model, platt_scaler) tuple
      ml_state.json  — metadata dict (accuracy, n_samples, trained_at, etc.)
    """

    DEFAULT_XGB_PARAMS = {
        "n_estimators":      200,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "eval_metric":       "logloss",
        "random_state":      42,
        "use_label_encoder": False,
    }

    def __init__(
        self,
        timeframe: str = "5m",
        model_file: str = None,
        state_file: str = None,
        asset: str = "btc",
    ) -> None:
        tf = timeframe if timeframe in ("5m", "15m") else "5m"
        ast = asset.lower() if asset else "btc"
        self.model_file  = model_file or f"ml_model_{tf}_{ast}_xgb.pkl"
        self.state_file  = state_file or f"ml_state_{tf}_{ast}_xgb.json"
        self._xgb        = None
        self._platt      = None
        self._is_trained = False
        self._meta: dict = {}

    # ------------------------------------------------------------------

    def train(self, X: List[List[float]], y: List[int]) -> dict:
        """
        Trains XGBClassifier, then Platt-scales on OOF predictions.
        Returns metadata dict. Auto-saves to disk.
        """
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        import numpy as np
        if len(X) < 50:
            raise ValueError(f"Need at least 50 samples, got {len(X)}")
        if len(np.unique(y)) < 2:
            raise ValueError("Training data has only one class — need both UP and DOWN samples")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        xgb = XGBClassifier(**{k: v for k, v in self.DEFAULT_XGB_PARAMS.items()
                               if k != "use_label_encoder"})
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Platt scaling on val set OOF probs
        val_probs = xgb.predict_proba(X_val)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0)
        platt.fit(val_probs, y_val)

        # Accuracy after calibration
        cal_probs = platt.predict_proba(val_probs)[:, 1]
        preds = (cal_probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)

        # Feature importances (gain-based, normalized)
        raw_imp = xgb.get_booster().get_score(importance_type="gain")
        total_imp = sum(raw_imp.values()) or 1.0
        feat_imp = {}
        for i, name in enumerate(FeatureEngine.FEATURE_NAMES):
            key = f"f{i}"
            feat_imp[name] = round(raw_imp.get(key, 0.0) / total_imp, 4)
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        self._xgb = xgb
        self._platt = platt
        self._is_trained = True

        self._meta = {
            "accuracy":           round(acc, 4),
            "n_samples":          len(X),
            "n_features":         len(FeatureEngine.FEATURE_NAMES),
            "trained_at":         datetime.now(timezone.utc).isoformat(),
            "feature_importances": feat_imp,
        }
        self.save()
        return dict(self._meta)

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Returns calibrated P(UP) ∈ [0, 1]. Raises RuntimeError if not trained."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Run backtest or train first.")
        arr = FeatureEngine.to_array(features)
        import numpy as np
        raw_prob = self._xgb.predict_proba([arr])[0][1]
        cal_prob = self._platt.predict_proba([[raw_prob]])[0][1]
        return float(cal_prob)

    def save(self) -> None:
        with open(self.model_file, "wb") as f:
            pickle.dump((self._xgb, self._platt), f)
        with open(self.state_file, "w") as f:
            json.dump(self._meta, f, indent=2)

    def load(self) -> bool:
        """Returns True on success. Falls back through naming convention migrations."""
        try:
            target = self.model_file
            old_state = None
            if not os.path.exists(target):
                # Migration v2: ml_model_{tf}_{asset}_xgb.pkl → ml_model_{tf}_xgb.pkl
                v2 = target.replace(f"_btc_xgb.pkl", "_xgb.pkl")
                v2_state = self.state_file.replace(f"_btc_xgb.json", "_xgb.json")
                if os.path.exists(v2):
                    target, old_state = v2, v2_state
                else:
                    # Migration v1: ml_model_{tf}_xgb.pkl → ml_model_{tf}.pkl
                    v1 = target.replace("_btc_xgb.pkl", ".pkl").replace("_xgb.pkl", ".pkl")
                    v1_state = self.state_file.replace("_btc_xgb.json", ".json").replace("_xgb.json", ".json")
                    if os.path.exists(v1):
                        target, old_state = v1, v1_state
                    else:
                        return False
            with open(target, "rb") as f:
                self._xgb, self._platt = pickle.load(f)
            state_path = old_state if (old_state and os.path.exists(old_state)) else self.state_file
            if os.path.exists(state_path):
                with open(state_path) as f:
                    self._meta = json.load(f)
            self._is_trained = True
            if old_state:  # re-save under new name
                self.save()
            return True
        except Exception:
            return False

    def is_trained(self) -> bool:
        return self._is_trained

    def get_meta(self) -> dict:
        return dict(self._meta)


# ---------------------------------------------------------------------------
# 3b. RandomForestModel
# ---------------------------------------------------------------------------

class RandomForestModel:
    """
    sklearn RandomForestClassifier + Platt calibration.
    Identical interface to XGBoostModel.
    """

    DEFAULT_RF_PARAMS: dict = {
        "n_estimators":   300,
        "max_depth":      None,
        "min_samples_leaf": 5,
        "max_features":   "sqrt",
        "class_weight":   "balanced",
        "random_state":   42,
        "n_jobs":         -1,
    }

    def __init__(self, timeframe: str = "5m", model_file: str = None, state_file: str = None, asset: str = "btc") -> None:
        tf = timeframe if timeframe in ("5m", "15m") else "5m"
        ast = asset.lower() if asset else "btc"
        self.model_file  = model_file or f"ml_model_{tf}_{ast}_rf.pkl"
        self.state_file  = state_file or f"ml_state_{tf}_{ast}_rf.json"
        self._rf         = None
        self._platt      = None
        self._is_trained = False
        self._meta: dict = {}

    def train(self, X: List[List[float]], y: List[int]) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        import numpy as np
        if len(X) < 50:
            raise ValueError(f"Need at least 50 samples, got {len(X)}")
        if len(np.unique(y)) < 2:
            raise ValueError("Training data has only one class — need both UP and DOWN samples")

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        rf = RandomForestClassifier(**self.DEFAULT_RF_PARAMS)
        rf.fit(X_tr, y_tr)

        val_probs = rf.predict_proba(X_val)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0)
        platt.fit(val_probs, y_val)

        cal_probs = platt.predict_proba(val_probs)[:, 1]
        acc = accuracy_score(y_val, (cal_probs >= 0.5).astype(int))

        raw_imp = rf.feature_importances_
        total = raw_imp.sum() or 1.0
        feat_imp = {name: round(float(raw_imp[i] / total), 4) for i, name in enumerate(FeatureEngine.FEATURE_NAMES)}
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        self._rf = rf
        self._platt = platt
        self._is_trained = True
        self._meta = {
            "accuracy": round(acc, 4),
            "n_samples": len(X),
            "n_features": len(FeatureEngine.FEATURE_NAMES),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_importances": feat_imp,
            "model_type": "random_forest",
        }
        self.save()
        return dict(self._meta)

    def predict_proba(self, features: Dict[str, float]) -> float:
        if not self._is_trained:
            raise RuntimeError("RandomForestModel not trained.")
        arr = FeatureEngine.to_array(features)
        raw_prob = self._rf.predict_proba([arr])[0][1]
        return float(self._platt.predict_proba([[raw_prob]])[0][1])

    def save(self) -> None:
        with open(self.model_file, "wb") as f:
            pickle.dump((self._rf, self._platt), f)
        with open(self.state_file, "w") as f:
            json.dump(self._meta, f, indent=2)

    def load(self) -> bool:
        try:
            if not os.path.exists(self.model_file):
                return False
            with open(self.model_file, "rb") as f:
                self._rf, self._platt = pickle.load(f)
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    self._meta = json.load(f)
            self._is_trained = True
            return True
        except Exception:
            return False

    def is_trained(self) -> bool:
        return self._is_trained

    def get_meta(self) -> dict:
        return dict(self._meta)


# ---------------------------------------------------------------------------
# 3c. ExtraTreesModel
# ---------------------------------------------------------------------------

class ExtraTreesModel:
    """
    sklearn ExtraTreesClassifier + Platt calibration.
    Identical interface to XGBoostModel.
    """

    DEFAULT_ET_PARAMS: dict = {
        "n_estimators":   300,
        "min_samples_leaf": 3,
        "max_features":   "sqrt",
        "class_weight":   "balanced",
        "random_state":   42,
        "n_jobs":         -1,
    }

    def __init__(self, timeframe: str = "5m", model_file: str = None, state_file: str = None, asset: str = "btc") -> None:
        tf = timeframe if timeframe in ("5m", "15m") else "5m"
        ast = asset.lower() if asset else "btc"
        self.model_file  = model_file or f"ml_model_{tf}_{ast}_et.pkl"
        self.state_file  = state_file or f"ml_state_{tf}_{ast}_et.json"
        self._et         = None
        self._platt      = None
        self._is_trained = False
        self._meta: dict = {}

    def train(self, X: List[List[float]], y: List[int]) -> dict:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        import numpy as np
        if len(X) < 50:
            raise ValueError(f"Need at least 50 samples, got {len(X)}")
        if len(np.unique(y)) < 2:
            raise ValueError("Training data has only one class — need both UP and DOWN samples")

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        et = ExtraTreesClassifier(**self.DEFAULT_ET_PARAMS)
        et.fit(X_tr, y_tr)

        val_probs = et.predict_proba(X_val)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0)
        platt.fit(val_probs, y_val)

        cal_probs = platt.predict_proba(val_probs)[:, 1]
        acc = accuracy_score(y_val, (cal_probs >= 0.5).astype(int))

        raw_imp = et.feature_importances_
        total = raw_imp.sum() or 1.0
        feat_imp = {name: round(float(raw_imp[i] / total), 4) for i, name in enumerate(FeatureEngine.FEATURE_NAMES)}
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        self._et = et
        self._platt = platt
        self._is_trained = True
        self._meta = {
            "accuracy": round(acc, 4),
            "n_samples": len(X),
            "n_features": len(FeatureEngine.FEATURE_NAMES),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_importances": feat_imp,
            "model_type": "extra_trees",
        }
        self.save()
        return dict(self._meta)

    def predict_proba(self, features: Dict[str, float]) -> float:
        if not self._is_trained:
            raise RuntimeError("ExtraTreesModel not trained.")
        arr = FeatureEngine.to_array(features)
        raw_prob = self._et.predict_proba([arr])[0][1]
        return float(self._platt.predict_proba([[raw_prob]])[0][1])

    def save(self) -> None:
        with open(self.model_file, "wb") as f:
            pickle.dump((self._et, self._platt), f)
        with open(self.state_file, "w") as f:
            json.dump(self._meta, f, indent=2)

    def load(self) -> bool:
        try:
            if not os.path.exists(self.model_file):
                return False
            with open(self.model_file, "rb") as f:
                self._et, self._platt = pickle.load(f)
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    self._meta = json.load(f)
            self._is_trained = True
            return True
        except Exception:
            return False

    def is_trained(self) -> bool:
        return self._is_trained

    def get_meta(self) -> dict:
        return dict(self._meta)


# ---------------------------------------------------------------------------
# 3d. EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Weighted average of XGBoost + RandomForest + ExtraTrees.
    Each sub-model is weighted by its validation accuracy.
    """

    def __init__(self, timeframe: str = "5m", asset: str = "btc") -> None:
        self._timeframe = timeframe
        self._models = [
            XGBoostModel(timeframe, asset=asset),
            RandomForestModel(timeframe, asset=asset),
            ExtraTreesModel(timeframe, asset=asset),
        ]
        self._is_trained = False
        self._meta: dict = {}

    def load_all(self) -> int:
        """Load all sub-models from disk. Returns count of successfully loaded models."""
        loaded = sum(1 for m in self._models if m.load())
        self._is_trained = loaded > 0
        self._refresh_meta()
        return loaded

    def _refresh_meta(self) -> None:
        trained = [m for m in self._models if m.is_trained()]
        if not trained:
            return
        best = max(trained, key=lambda m: m.get_meta().get("accuracy", 0))
        self._meta = {
            **best.get_meta(),
            "model_type": "ensemble",
            "sub_models": [
                {"model_type": type(m).__name__, **m.get_meta()}
                for m in trained
            ],
        }

    def is_trained(self) -> bool:
        return self._is_trained

    def predict_proba(self, features: Dict[str, float]) -> float:
        probs, weights = [], []
        for m in self._models:
            if m.is_trained():
                acc = m.get_meta().get("accuracy", 0.5)
                probs.append(m.predict_proba(features) * acc)
                weights.append(acc)
        return sum(probs) / sum(weights) if weights else 0.5

    def get_meta(self) -> dict:
        return dict(self._meta)

    # EnsembleModel has no save/load of its own — sub-models persist individually
    def load(self) -> bool:
        return self.load_all() > 0

    def save(self) -> None:
        pass  # sub-models save themselves


# ---------------------------------------------------------------------------
# 3e. Factory
# ---------------------------------------------------------------------------

MODEL_TYPES = ("xgboost", "random_forest", "extra_trees", "ensemble")


def make_model(model_type: str, timeframe: str = "5m", asset: str = "btc"):
    """Return a model instance for the given type and asset. Defaults to XGBoost/BTC."""
    if model_type == "random_forest":
        return RandomForestModel(timeframe, asset=asset)
    if model_type == "extra_trees":
        return ExtraTreesModel(timeframe, asset=asset)
    if model_type == "ensemble":
        return EnsembleModel(timeframe, asset=asset)
    return XGBoostModel(timeframe, asset=asset)


# ---------------------------------------------------------------------------
# 4. BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward validation on Binance historical 1m data.

    Algorithm:
    1. Fetch months_history months of 1m candles
    2. Build windows: group 5 (5m) or 15 (15m) consecutive 1m candles
       label = 1 if window_close > window_open else 0
    3. Walk-forward: train on first train_frac, then expand
    4. Retrain every retrain_every new windows
    5. Report metrics
    """

    def __init__(
        self,
        fetcher: BinanceDataFetcher,
        months_history: int   = 6,
        train_frac: float     = 0.70,
        retrain_every: int    = 100,
        timeframe: str        = "5m",
        asset: str            = "btc",
    ) -> None:
        self.fetcher         = fetcher
        self.months_history  = months_history
        self.train_frac      = train_frac
        self.retrain_every   = retrain_every
        self.timeframe       = timeframe if timeframe in ("5m", "15m") else "5m"
        self.asset           = asset.lower() if asset else "btc"
        self._window_steps   = 15 if self.timeframe == "15m" else 5  # 1m candles per window

    def run(self, progress_callback=None) -> dict:
        """
        Runs full walk-forward backtest. Returns results dict.
        progress_callback(pct: float, msg: str) is called periodically.
        """
        def _prog(pct: float, msg: str):
            if progress_callback:
                try:
                    progress_callback(pct, msg)
                except Exception:
                    pass

        total_batches = max(1, (self.months_history * 43200) // 1000)
        _prog(0.0, f"Fetching {self.months_history}mo of candles from Binance (~{total_batches} requests)…")
        def _fetch_progress(fetched: int, total: int) -> None:
            pct = min(4.5, fetched / total * 4.5)
            _prog(pct, f"Fetching candles: {fetched:,} / {total:,}  ({fetched * 100 // total}%)")
        candles = self.fetcher.get_candles_historical(
            days=self.months_history * 30,
            progress_callback=_fetch_progress,
        )
        if len(candles) < 200:
            return {"status": "error", "error": f"Insufficient data: {len(candles)} candles"}

        _prog(5.0, f"Building {self.timeframe} windows from {len(candles)} candles...")
        windows = self._build_windows(candles)
        if len(windows) < 100:
            return {"status": "error", "error": f"Insufficient windows: {len(windows)}"}

        _prog(10.0, f"Computing features for {len(windows)} windows...")
        X_all, y_all, meta_all = self._compute_features(windows)
        if len(X_all) < 80:
            return {"status": "error", "error": "Insufficient feature rows"}

        n_total  = len(X_all)
        n_train  = int(n_total * self.train_frac)
        n_test   = n_total - n_train

        _prog(15.0, f"Walk-forward: {n_train} train, {n_test} test windows...")

        # Train all 3 algorithms on the same walk-forward data
        algo_defs = [
            ("xgboost",       XGBoostModel(timeframe=self.timeframe, asset=self.asset)),
            ("random_forest", RandomForestModel(timeframe=self.timeframe, asset=self.asset)),
            ("extra_trees",   ExtraTreesModel(timeframe=self.timeframe, asset=self.asset)),
        ]

        model_comparison = {}
        primary_model = None
        pct_per_algo  = 26.0  # 15→97 split 3 ways

        for algo_idx, (algo_name, model) in enumerate(algo_defs):
            pct_start = 15.0 + algo_idx * pct_per_algo
            _prog(pct_start, f"Training {algo_name} ({algo_idx + 1}/3)...")
            results = self._walk_forward_eval(
                model, X_all, y_all, meta_all,
                n_train, n_total, _prog, pct_start, pct_per_algo - 2.0,
            )
            row_metrics = self._compute_metrics(results, n_train, n_test)
            model_comparison[algo_name] = {
                "accuracy":    row_metrics["accuracy"],
                "brier_score": row_metrics["brier_score"],
                "sim_pnl_10":  row_metrics["sim_pnl_10"],
                "trained_model": model,
            }
            if primary_model is None:
                primary_model = model   # XGBoost is the primary (first)

        _prog(97.0, "Computing metrics...")
        xgb_metrics = self._compute_metrics(
            self._walk_forward_results_cache if hasattr(self, "_walk_forward_results_cache") else [],
            n_train, n_test,
        )
        # Use XGBoost's metrics as the top-level summary (backward compat)
        xgb_row = model_comparison["xgboost"]
        metrics = {
            "accuracy":          xgb_row["accuracy"],
            "accuracy_all":      xgb_row["accuracy"],
            "brier_score":       xgb_row["brier_score"],
            "sim_pnl_10":        xgb_row["sim_pnl_10"],
            "n_windows":         n_total,
            "n_train":           n_train,
            "n_test":            n_test,
            "n_predicted":       n_test,
            "trained_model":     primary_model,
            "all_trained_models": {k: v["trained_model"] for k, v in model_comparison.items()},
            "model_comparison":  {
                k: {kk: vv for kk, vv in v.items() if kk != "trained_model"}
                for k, v in model_comparison.items()
            },
            "status":            "complete",
            "run_at":            datetime.now(timezone.utc).isoformat(),
            "months_history":    self.months_history,
            "timeframe":         self.timeframe,
        }

        _prog(100.0, "Backtest complete.")
        return metrics

    # ------------------------------------------------------------------

    def _walk_forward_eval(
        self,
        model,
        X_all: List[List[float]],
        y_all: List[int],
        meta_all: List[dict],
        n_train: int,
        n_total: int,
        _prog,
        pct_start: float,
        pct_range: float,
    ) -> list:
        """Train model on first n_train samples, then walk-forward predict remaining."""
        model.train(X_all[:n_train], y_all[:n_train])
        results = []
        retrain_counter = 0
        n_test = n_total - n_train
        for i in range(n_train, n_total):
            prob = model.predict_proba(dict(zip(FeatureEngine.FEATURE_NAMES, X_all[i])))
            results.append({
                "label":     y_all[i],
                "prob":      prob,
                "window_ts": meta_all[i]["window_ts"],
                "hour_utc":  meta_all[i]["hour_utc"],
            })
            retrain_counter += 1
            if retrain_counter >= self.retrain_every:
                model.train(X_all[:i + 1], y_all[:i + 1])
                retrain_counter = 0
            if i % 200 == 0:
                pct = pct_start + pct_range * (i - n_train) / max(n_test, 1)
                _prog(pct, f"Testing {type(model).__name__} {i - n_train}/{n_test}...")
        return results

    def _build_windows(self, candles: List[dict]) -> List[dict]:
        """Groups sorted 1m candles into window records (5m or 15m)."""
        steps = self._window_steps
        windows = []
        i = 0
        while i + steps - 1 < len(candles):
            w_open  = candles[i]
            w_close = candles[i + steps - 1]
            # label: 1=UP (close > open), 0=DOWN
            label = 1 if w_close["close"] > w_open["open"] else 0
            # feature candles: the 100 candles ending just before window open
            feat_start = max(0, i - 100)
            windows.append({
                "window_ts":       w_open["ts_ms"] / 1000,
                "open_price":      w_open["open"],
                "close_price":     w_close["close"],
                "label":           label,
                "feature_candles": candles[feat_start:i],
            })
            i += steps
        return windows

    def _compute_features(
        self, windows: List[dict]
    ) -> Tuple[List[List[float]], List[int], List[dict]]:
        """Computes FeatureEngine.compute() for each window. Skips None returns."""
        X, y, meta = [], [], []
        for w in windows:
            if len(w["feature_candles"]) < FeatureEngine.MIN_CANDLES:
                continue
            feat = FeatureEngine.compute(
                candles=w["feature_candles"],
                funding_rate=0.0,      # not available historically
                up_ask=0.0,
                down_ask=0.0,
                price_to_beat=w["open_price"],
                window_start_ts=w["window_ts"],
                analysis_signals=None, # not available historically — features default to 0
            )
            if feat is None:
                continue
            X.append(FeatureEngine.to_array(feat))
            y.append(w["label"])
            dt = datetime.fromtimestamp(w["window_ts"], tz=timezone.utc)
            meta.append({"window_ts": w["window_ts"], "hour_utc": dt.hour})
        return X, y, meta

    def _compute_metrics(self, results: list, n_train: int, n_test: int) -> dict:
        """Computes all backtest metrics from walk-forward results."""
        import math as _math

        if not results:
            return {"accuracy": 0.0, "n_windows": 0, "n_test": 0, "n_predicted": 0}

        n_predicted = len(results)

        def _accuracy(rows):
            if not rows:
                return 0.0
            correct = sum(1 for r in rows if (r["prob"] >= 0.5) == bool(r["label"]))
            return round(correct / len(rows), 4)

        acc_filtered = _accuracy(results)
        acc_all      = acc_filtered

        # Log-loss & Brier score
        eps = 1e-7
        log_loss = -sum(
            r["label"] * _math.log(max(r["prob"], eps)) +
            (1 - r["label"]) * _math.log(max(1 - r["prob"], eps))
            for r in results
        ) / len(results)
        brier = sum((r["prob"] - r["label"]) ** 2 for r in results) / len(results)

        # Confusion matrix
        tp = sum(1 for r in results if r["prob"] >= 0.5 and r["label"] == 1)
        fp = sum(1 for r in results if r["prob"] >= 0.5 and r["label"] == 0)
        tn = sum(1 for r in results if r["prob"] <  0.5 and r["label"] == 0)
        fn = sum(1 for r in results if r["prob"] <  0.5 and r["label"] == 1)

        # 10-bin calibration
        bins = [{"prob_range": f"{i/10:.1f}-{(i+1)/10:.1f}", "actual_rate": 0.0, "count": 0}
                for i in range(10)]
        for r in results:
            b = min(int(r["prob"] * 10), 9)
            bins[b]["count"] += 1
            bins[b]["actual_rate"] += r["label"]
        for b in bins:
            if b["count"] > 0:
                b["actual_rate"] = round(b["actual_rate"] / b["count"], 3)

        # Per-hour accuracy
        per_hour: Dict[int, List] = {h: [] for h in range(24)}
        for r in results:
            # hour is stored in meta; re-derive from results if available
            pass  # populated below
        per_hour_results: Dict[int, list] = {h: [] for h in range(24)}
        for r in results:
            # window_ts present on every result row
            h = int(datetime.fromtimestamp(r.get("window_ts", 0),
                                           tz=timezone.utc).hour)
            per_hour_results[h].append(r)
        per_hour_acc = [
            {
                "hour_utc": h,
                "accuracy": _accuracy(per_hour_results[h]),
                "count":    len(per_hour_results[h]),
            }
            for h in range(24)
        ]

        # Simulated P&L (buy at 0.50, win 0.50, lose 0.50)
        sim_pnl = sum(
            0.50 if (r["prob"] >= 0.5) == bool(r["label"]) else -0.50
            for r in results
        )

        return {
            "n_windows":         len(results) + n_train,
            "n_train":           n_train,
            "n_test":            n_test,
            "n_predicted":       n_predicted,
            "accuracy":          acc_filtered,
            "accuracy_all":      acc_all,
            "log_loss":          round(log_loss, 4),
            "brier_score":       round(brier, 4),
            "calibration_bins":  bins,
            "confusion_matrix":  {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "per_hour_accuracy": per_hour_acc,
            "sim_pnl_10":        round(sim_pnl, 2),
        }


# ---------------------------------------------------------------------------
# 5. MLTrader
# ---------------------------------------------------------------------------

class MLTrader:
    """
    XGBoost-based autonomous trader for Polymarket BTC Up/Down markets.

    Supports both 5-minute (300s) and 15-minute (900s) timeframes.

    Lifecycle mirrors AdvancedTrader exactly:
    - run() called in a daemon thread by app.py
    - Polls every POLL_INTERVAL_SEC; monitors positions every MONITOR_INTERVAL_SEC
    - Launches _auto_redeem_loop thread when config.auto_redeem=True

    Trading flow (per tick):
      candles → features → XGBoost P(UP) → confidence gate →
      window-open gate → dedup gate → lookup token → place BUY
    """

    POLL_INTERVAL_SEC    = 30
    MONITOR_INTERVAL_SEC = 10

    def __init__(self, config, log_callback=None) -> None:
        self.config        = config
        self._log_cb       = log_callback
        self.running       = False
        self._stop_event   = threading.Event()

        # Timeframe (5m or 15m)
        self._timeframe  = getattr(config, "timeframe", "5m")
        if self._timeframe not in ("5m", "15m"):
            self._timeframe = "5m"
        self._window_sec = 900 if self._timeframe == "15m" else 300

        # Asset (btc / bnb)
        self._asset  = getattr(config, "asset", "btc").lower()
        self._symbol = f"{self._asset.upper()}USDT"

        # Model type (xgboost / random_forest / extra_trees / ensemble)
        self._model_type = getattr(config, "model_type", "xgboost")
        if self._model_type not in MODEL_TYPES:
            self._model_type = "xgboost"

        # Lazy import to avoid circular at module load
        from copy_trader import TradeExecutor
        self.executor = TradeExecutor(config)

        self.fetcher = BinanceDataFetcher(symbol=self._symbol)
        self.model   = make_model(self._model_type, self._timeframe, self._asset)
        self._model_loaded = (
            self.model.load_all() > 0 if self._model_type == "ensemble"
            else self.model.load()
        )

        self.trade_journal: list = []
        self._journal_lock = threading.Lock()
        self._last_monitor_ts: float = 0.0
        self._last_traded_window: Optional[int] = None

        # Online retraining
        self._outcome_buffer: List[dict] = []
        self._outcome_lock   = threading.Lock()
        self._load_outcome_buffer()

        # Latest features (for /api/ml-features)
        self._last_features: Optional[Dict[str, float]] = None
        self._last_prediction: Optional[dict] = None
        self._prediction_log: list = []

        self.stats = {
            "predictions_made": 0,
            "trades_entered":   0,
            "exits_taken":      0,
            "errors":           0,
            "retrains":         0,
            "outcome_buffer":   0,
            "model_accuracy":   self.model.get_meta().get("accuracy"),
            "model_samples":    self.model.get_meta().get("n_samples"),
        }

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def run(self) -> None:
        self.running = True
        self._stop_event.clear()
        mode = "DRY RUN" if self.config.dry_run else "LIVE"
        self._log("info", f"[ML] Started | mode={mode} | timeframe={self._timeframe} | asset={self._asset} | model={self._model_type}")

        # Auto-redeem thread (reuses module-level function from copy_trader)
        if getattr(self.config, "auto_redeem", False):
            from copy_trader import _auto_redeem_loop as _arl
            threading.Thread(
                target=_arl,
                args=(self.config, self._log, self._stop_event),
                name="MLAutoRedeem",
                daemon=True,
            ).start()
            self._log("info", "[ML] Auto-redeem thread started")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self.stats["errors"] += 1
                self._log("error", f"[ML] Tick error: {exc}")

            now = time.monotonic()
            if now - self._last_monitor_ts > self.MONITOR_INTERVAL_SEC:
                try:
                    self._monitor_positions()
                except Exception as exc:
                    self._log("error", f"[ML] Monitor error: {exc}")
                self._last_monitor_ts = now

            self._stop_event.wait(self.POLL_INTERVAL_SEC)

        self.running = False
        self._log("info", f"[ML] Stopped. Stats: {self.stats}")

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False

    def get_trade_journal(self) -> list:
        with self._journal_lock:
            return list(self.trade_journal)

    def get_last_features(self) -> Optional[Dict[str, float]]:
        return self._last_features

    def get_last_prediction(self) -> Optional[dict]:
        return self._last_prediction

    def get_prediction_log(self) -> list:
        return list(self._prediction_log)

    def set_timeframe(self, tf: str) -> None:
        """Switch to a different market timeframe without restarting the trader."""
        if tf not in ("5m", "15m"):
            return
        self._timeframe  = tf
        self._window_sec = 900 if tf == "15m" else 300
        self._last_traded_window = None   # reset per-window dedup
        self.model = make_model(self._model_type, tf, self._asset)
        self._model_loaded = (
            self.model.load_all() > 0 if self._model_type == "ensemble"
            else self.model.load()
        )
        self._prediction_log.clear()
        self._log("info", f"[ML] Switched to {tf} | asset={self._asset} | model={self._model_type} | {'loaded' if self._model_loaded else 'not trained — run backtest first'}")

    def execute_if_agree(self) -> dict:
        """
        One-shot trade: executes immediately if the last ML prediction and the
        current Analysis engine both point the same direction.
        Called from the /api/ml-agree-trade endpoint (runs on the HTTP thread).
        """
        if not self.model.is_trained():
            return {"status": "no_model", "message": "Model not trained — run backtest first"}

        pred = self._last_prediction
        if not pred:
            return {"status": "no_prediction", "message": "No ML prediction yet — wait for first tick"}

        analysis = self._get_analysis_signals()
        if not analysis:
            return {"status": "no_analysis", "message": "Analysis engine unavailable or wrong timeframe"}

        ml_dir = pred.get("direction")
        an_raw = (analysis.get("prediction") or "").upper()
        if an_raw in ("UP", "BULLISH", "HIGH"):
            an_dir = "UP"
        elif an_raw in ("DOWN", "BEARISH", "LOW"):
            an_dir = "DOWN"
        else:
            return {"status": "analysis_neutral", "message": f"Analysis is neutral ({an_raw}) — no trade"}

        if ml_dir != an_dir:
            return {
                "status": "disagree",
                "message": f"ML says {ml_dir}, Analysis says {an_dir} — no trade",
                "ml_direction": ml_dir,
                "an_direction": an_dir,
            }

        # Both agree — derive context and execute
        window_id = pred.get("window_start") or (
            int(time.time() // self._window_sec) * self._window_sec
        )
        up_ask, down_ask, _ptb, _ws = self._get_polymarket_context()
        token_id, slug, outcome_label = self._lookup_token(window_id, ml_dir)
        if not token_id:
            return {"status": "no_token", "message": f"Token lookup failed for {ml_dir} window {window_id}"}

        ask_price = (up_ask if ml_dir == "UP" else down_ask) or 0.5
        if ask_price <= 0 or ask_price >= 1.0:
            ask_price = 0.5
        trade_usd = getattr(self.config, "trade_amount_usd", 10.0)
        shares    = round(trade_usd / ask_price, 2)
        cost_usd  = round(shares * ask_price, 4)

        self._execute_trade(
            token_id=token_id,
            slug=slug,
            direction=ml_dir,
            outcome_label=outcome_label,
            shares=shares,
            ask_price=ask_price,
            cost_usd=cost_usd,
            prob=pred.get("prob", 0.5),
            confidence=pred.get("confidence", 0),
            window_start=window_id,
            features=self._last_features,
        )
        return {
            "status": "simulated" if self.config.dry_run else "traded",
            "direction": ml_dir,
            "ml_direction": ml_dir,
            "an_direction": an_dir,
            "confidence": pred.get("confidence"),
            "prob": pred.get("prob"),
            "an_confidence": analysis.get("confidence"),
            "dry_run": self.config.dry_run,
            "shares": shares,
            "ask_price": ask_price,
            "cost_usd": cost_usd,
        }

    def _get_analysis_signals(self) -> Optional[dict]:
        """Fetch current Analysis engine signals from the local API."""
        try:
            r = requests.get("http://localhost:5002/api/analysis", timeout=3)
            if r.status_code == 200:
                d = r.json()
                got_tf = d.get("timeframe")
                if got_tf == self._timeframe:
                    return d
                # Timeframe in response doesn't match — log once and return anyway
                # (downstream handles NEUTRAL; at least we surface the mismatch)
                self._log("warn",
                    f"[ML] Analysis timeframe mismatch: got={got_tf!r} expected={self._timeframe!r} — using anyway")
                return d
            else:
                self._log("warn", f"[ML] Analysis API returned HTTP {r.status_code}")
        except Exception as exc:
            self._log("warn", f"[ML] Analysis fetch error: {exc}")
        return None

    # -----------------------------------------------------------------------
    # Core tick
    # -----------------------------------------------------------------------

    def _tick(self) -> None:
        if not self.model.is_trained():
            self._log("warn", "[ML] Model not trained — run backtest first to train model")
            return

        # --- Fetch data ---
        candles = self.fetcher.get_candles(limit=100)
        if len(candles) < FeatureEngine.MIN_CANDLES:
            self._log("warn", f"[ML] Insufficient candles: {len(candles)}")
            return

        funding_rate = self.fetcher.get_funding_rate()

        # --- Polymarket context ---
        up_ask, down_ask, price_to_beat, window_start = self._get_polymarket_context()

        # --- Analysis engine signals (best-effort; None if unavailable) ---
        analysis_signals = self._get_analysis_signals()

        # --- Compute features ---
        features = FeatureEngine.compute(
            candles=candles,
            funding_rate=funding_rate,
            up_ask=up_ask,
            down_ask=down_ask,
            price_to_beat=price_to_beat,
            window_start_ts=window_start,
            analysis_signals=analysis_signals,
            window_sec=self._window_sec,
        )
        if features is None:
            return
        self._last_features = features

        # --- Outcome recording / retrain (every tick, not just after trades) ---
        self._maybe_record_outcome()
        self._maybe_retrain()

        # --- Predict ---
        prob = self.model.predict_proba(features)
        confidence = round(abs(prob - 0.5) * 200)
        direction = "UP" if prob >= 0.5 else "DOWN"

        self._last_prediction = {
            "direction": direction,
            "confidence": confidence,
            "prob": round(prob, 4),
            "timestamp": time.time(),
            "timeframe": self._timeframe,
            "analysis_used": analysis_signals is not None,
        }
        self.stats["predictions_made"] += 1

        # --- Window ID and elapsed time ---
        window_id  = int(window_start) if window_start > 0 else int(time.time() // self._window_sec) * self._window_sec
        secs_into  = max(0.0, time.time() - window_id)

        # Record every prediction for comparison history (before any early-return gates)
        _prec = {"window_start": window_id, "direction": direction, "prob": round(prob, 4), "confidence": confidence, "window_sec": self._window_sec}
        if not self._prediction_log or self._prediction_log[-1].get("window_start") != window_id:
            self._prediction_log.append(_prec)
            if len(self._prediction_log) > 500:
                self._prediction_log = self._prediction_log[-500:]

        # Tracker mode — prediction recorded, nothing else to do
        if getattr(self.config, "predict_only", False):
            return

        # Tick summary — always visible so the user can see the bot is alive
        require_agree = getattr(self.config, "require_analysis_agreement", False)
        an_dir_tick = None
        if analysis_signals:
            _ar = (analysis_signals.get("prediction") or "").upper()
            an_dir_tick = "UP" if _ar in ("UP","BULLISH","HIGH") else "DOWN" if _ar in ("DOWN","BEARISH","LOW") else "NEUTRAL"
        an_label  = an_dir_tick or "N/A"
        agree_str = "✓AGREE" if an_dir_tick == direction else "✗DISAGREE"
        self._log("info", f"[ML] Tick | Analysis={an_label} | ML={direction} ({confidence}% conf) | {agree_str} | {secs_into:.0f}s in, {self._window_sec - secs_into:.0f}s remain")
        secs_remaining = max(0.0, self._window_sec - secs_into)
        if getattr(self.config, "trade_at_window_open_only", True):
            require_agree = getattr(self.config, "require_analysis_agreement", False)
            if require_agree:
                # Agreement mode: trade any time both agree, but stop when window almost expired
                min_remaining = getattr(self.config, "min_window_secs_remaining", 60)
                if secs_remaining < min_remaining:
                    self._log("warn", f"[ML] Skipping — only {secs_remaining:.0f}s left in window (min={min_remaining}s)")
                    return
            else:
                window_open_sec = getattr(self.config, "window_open_sec", 60)
                if secs_into > window_open_sec:
                    self._log("warn", f"[ML] Skipping — {secs_into:.0f}s into window, gate={window_open_sec}s. Next window in ~{secs_remaining:.0f}s")
                    return

        # --- Per-window dedup ---
        # NOTE: dedup lock is set AFTER token lookup succeeds so a transient
        # API failure doesn't permanently lock out the window.
        if self._last_traded_window == window_id:
            self._log("info", f"[ML] Already traded window {window_id} — waiting for next")
            return

        # --- Analysis agreement gate ---
        if getattr(self.config, "require_analysis_agreement", False):
            if not analysis_signals:
                self._log("warn", "[ML] BLOCKED — Analysis API unreachable — no trade")
                return
            an_raw = (analysis_signals.get("prediction") or "").upper()
            if an_raw in ("UP", "BULLISH", "HIGH"):
                an_dir = "UP"
            elif an_raw in ("DOWN", "BEARISH", "LOW"):
                an_dir = "DOWN"
            else:
                an_dir = None
            an_conf = int(analysis_signals.get("confidence") or 0)
            an_err  = analysis_signals.get("error")
            if an_dir != direction:
                reason = f"transient fetch error: {an_err}" if an_err and an_dir is None else f"Analysis={an_dir or 'NEUTRAL'} ({an_conf}%)"
                self._log("warn", f"[ML] BLOCKED — ML={direction} ({confidence}% conf), {reason} — no trade")
                return
            self._log("copy" if not self.config.dry_run else "dry_run",
                      f"[ML] AGREE — ML={direction} ({confidence}% conf), Analysis={an_dir} ({an_conf}%) — proceeding")

        # --- Lookup token ---
        self._log("info", f"[ML] All gates passed — looking up {direction} token for window {window_id}")
        token_id, slug, outcome_label = self._lookup_token(window_id, direction)
        if not token_id:
            self._log("warn", f"[ML] Token lookup failed for {direction} (slug={self._asset}-updown-{self._timeframe}-{window_id}) — will retry next poll")
            return

        # --- Lock window (only after a successful token lookup) ---
        self._last_traded_window = window_id

        # --- Determine ask price ---
        ask_price = (up_ask if direction == "UP" else down_ask) or 0.5
        if ask_price <= 0 or ask_price >= 1.0:
            ask_price = 0.5

        trade_usd = getattr(self.config, "trade_amount_usd", 10.0)
        shares = round(trade_usd / ask_price, 2)
        cost_usd = round(shares * ask_price, 4)

        self._execute_trade(
            token_id=token_id,
            slug=slug,
            direction=direction,
            outcome_label=outcome_label,
            shares=shares,
            ask_price=ask_price,
            cost_usd=cost_usd,
            prob=prob,
            confidence=confidence,
            window_start=window_id,
            features=features,
        )

        # (outcome recording / retrain already called at top of _tick)

    # -----------------------------------------------------------------------
    # Polymarket context
    # -----------------------------------------------------------------------

    def _get_polymarket_context(self) -> Tuple[float, float, float, float]:
        """Returns (up_ask, down_ask, price_to_beat, window_start_ts).

        window_start_ts is derived from the timeframe epoch boundary so it's
        consistent with _lookup_token's slug format ({asset}-updown-{tf}-{ts}).
        """
        import json as _json
        clob  = getattr(self.config, "clob_host", "https://clob.polymarket.com")
        gamma = getattr(self.config, "gamma_api_base", "https://gamma-api.polymarket.com")

        # Use clock-derived window start aligned to our timeframe
        window_start = float(int(time.time() // self._window_sec) * self._window_sec)
        slug = f"{self._asset}-updown-{self._timeframe}-{int(window_start)}"

        up_ask, down_ask = 0.0, 0.0
        try:
            resp = requests.get(
                f"{gamma}/markets",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code == 200:
                data   = resp.json()
                market = (data[0] if isinstance(data, list) else data) if data else {}

                raw_outcomes = market.get("outcomes", "[]")
                raw_ids      = market.get("clobTokenIds", "[]")
                if isinstance(raw_outcomes, str):
                    raw_outcomes = _json.loads(raw_outcomes)
                if isinstance(raw_ids, str):
                    raw_ids = _json.loads(raw_ids)

                # Map Up/Down token_id → fetch midpoint
                for i, outcome in enumerate(raw_outcomes):
                    if i >= len(raw_ids):
                        break
                    tok = raw_ids[i]
                    if not tok:
                        continue
                    try:
                        r = requests.get(f"{clob}/midpoint", params={"token_id": tok}, timeout=3)
                        if r.status_code == 200:
                            val = float(r.json().get("mid", 0) or 0)
                            if outcome.strip().lower() == "up":
                                up_ask = val
                            elif outcome.strip().lower() == "down":
                                down_ask = val
                    except Exception:
                        pass
        except Exception:
            pass

        # Price to beat from local /api/price
        price_to_beat = 0.0
        try:
            r = requests.get("http://localhost:5000/api/price", params={"asset": self._asset}, timeout=2)
            if r.status_code == 200:
                price_to_beat = float(r.json().get("price_to_beat", 0) or 0)
        except Exception:
            pass

        return up_ask, down_ask, price_to_beat, window_start

    # -----------------------------------------------------------------------
    # Token lookup
    # -----------------------------------------------------------------------

    def _lookup_token(
        self, window_start: int, direction: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Returns (token_id, slug, outcome_label) for the active market.

        Gamma-api markets encode outcomes/clobTokenIds as JSON-encoded string arrays
        e.g. outcomes='["Up","Down"]', clobTokenIds='["123...","456..."]'.
        The per-window slug is  {asset}-updown-{tf}-{unix_timestamp}.
        """
        import json as _json
        gamma   = getattr(self.config, "gamma_api_base", "https://gamma-api.polymarket.com")
        slug    = f"{self._asset}-updown-{self._timeframe}-{window_start}"
        target  = "Up" if direction == "UP" else "Down"
        try:
            resp = requests.get(
                f"{gamma}/markets",
                params={"slug": slug},
                timeout=8,
            )
            resp.raise_for_status()
            data   = resp.json()
            if not data:
                self._log("warn", f"[ML] No market found for slug {slug}")
                return None, None, None
            market = data[0] if isinstance(data, list) else data

            raw_outcomes = market.get("outcomes", "[]")
            raw_ids      = market.get("clobTokenIds", "[]")
            if isinstance(raw_outcomes, str):
                raw_outcomes = _json.loads(raw_outcomes)
            if isinstance(raw_ids, str):
                raw_ids = _json.loads(raw_ids)

            for i, outcome in enumerate(raw_outcomes):
                if outcome.strip().lower() == target.lower() and i < len(raw_ids):
                    return raw_ids[i], slug, target

            self._log("warn", f"[ML] '{target}' not in outcomes={raw_outcomes} for {slug}")
            return None, None, None
        except Exception as exc:
            self._log("warn", f"[ML] Token lookup error for {slug}: {exc}")
            return None, None, None

    # -----------------------------------------------------------------------
    # Trade execution
    # -----------------------------------------------------------------------

    def _execute_trade(
        self,
        token_id: str, slug: str, direction: str, outcome_label: str,
        shares: float, ask_price: float, cost_usd: float,
        prob: float, confidence: int, window_start: int,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        if self.config.dry_run:
            self.stats["trades_entered"] += 1
            self._append_journal(
                slug=slug, token_id=token_id, outcome=outcome_label,
                side="BUY", shares=shares, entry_price=ask_price,
                cost_usd=cost_usd, dry_run=True,
                prob=prob, confidence=confidence, window_start=window_start,
                features=features, status="SIMULATED",
            )
            self._log(
                "dry_run",
                f"[ML DRY] Trade logged — {direction} {outcome_label} | "
                f"conf={confidence}% prob={prob:.3f} | "
                f"{shares:.2f} shares @ {ask_price:.3f} = ${cost_usd:.2f}",
            )
        else:
            self._log(
                "copy",
                f"[ML LIVE] Placing BUY {direction} | conf={confidence}% "
                f"prob={prob:.3f} | {shares:.2f} shares @ {ask_price:.3f} "
                f"= ${cost_usd:.2f}",
            )
            order_status = "OPEN"
            try:
                self.executor.place_order(token_id, "BUY", shares, ask_price, use_gtc=True)
                self.stats["trades_entered"] += 1
            except Exception as exc:
                exc_l = str(exc).lower()
                if (
                    "empty order book" in exc_l
                    or "insufficient liquidity" in exc_l
                    or "invalid amount for a marketable" in exc_l
                    or "lower than the minimum" in exc_l
                    or "size" in exc_l and "lower than" in exc_l
                ):
                    order_status = "SKIPPED"
                    self._log("warn", f"[ML] Trade skipped — {exc}")
                else:
                    self.stats["errors"] += 1
                    order_status = "FAILED"
                    self._log("error", f"[ML] Order failed: {exc}")
            # Always record — so the attempt is visible in the trade tab
            self._append_journal(
                slug=slug, token_id=token_id, outcome=outcome_label,
                side="BUY", shares=shares, entry_price=ask_price,
                cost_usd=cost_usd, dry_run=False,
                prob=prob, confidence=confidence, window_start=window_start,
                features=features, status=order_status,
            )

    def _append_journal(
        self,
        slug: str, token_id: str, outcome: str, side: str,
        shares: float, entry_price: float, cost_usd: float,
        dry_run: bool, prob: float, confidence: int, window_start: int,
        features: Optional[Dict[str, float]], status: str,
    ) -> None:
        entry = {
            "id":          f"ml-{int(time.time() * 1000)}",
            "entry_time":  datetime.now(timezone.utc).isoformat(),
            "market":      slug,
            "outcome":     outcome,
            "side":        side,
            "our_size":    shares,
            "entry_price": entry_price,
            "cost_usd":    cost_usd,
            "token_id":    token_id,
            "status":      status,
            "cur_price":   entry_price,
            "pnl_usd":     0.0,
            "dry_run":     dry_run,
            "source":      "ml",
            "prob":        round(prob, 4),
            "confidence":  confidence,
            "window_start": window_start,
            "window_sec":  self._window_sec,
            "features":    features,
        }
        with self._journal_lock:
            self.trade_journal.append(entry)

    # -----------------------------------------------------------------------
    # Position monitoring
    # -----------------------------------------------------------------------

    def _monitor_positions(self) -> None:
        with self._journal_lock:
            active = [e for e in self.trade_journal
                      if e["status"] in ("OPEN", "SIMULATED")]
        if not active:
            return

        clob = getattr(self.config, "clob_host", "https://clob.polymarket.com")
        for entry in active:
            try:
                resp = requests.get(
                    f"{clob}/midpoint",
                    params={"token_id": entry["token_id"]},
                    timeout=5,
                )
                if resp.status_code != 200:
                    continue
                mid = float(resp.json().get("mid", 0) or 0)
                if mid <= 0:
                    self._try_settle_from_gamma(entry)
                    continue

                roi = (mid - entry["entry_price"]) / entry["entry_price"] if entry["entry_price"] > 0 else 0
                pnl = (mid - entry["entry_price"]) * entry["our_size"]

                with self._journal_lock:
                    entry["cur_price"] = round(mid, 4)
                    entry["pnl_usd"]   = round(pnl, 4)

                # Take-profit exit
                tp = getattr(self.config, "take_profit_pct", 0.0)
                if tp > 0 and roi * 100 >= tp:
                    self._exit_position(entry, mid, "take_profit")

            except Exception:
                pass

    def _exit_position(self, entry: dict, cur_price: float, reason: str) -> None:
        if entry["dry_run"]:
            with self._journal_lock:
                entry["status"]    = "SIM_PROFIT_TAKEN"
                entry["cur_price"] = round(cur_price, 4)
            self._log("dry_run", f"[ML DRY] Exit {reason} | {entry['outcome']} @ {cur_price:.3f}")
            self.stats["exits_taken"] += 1
            return

        try:
            self.executor.place_order(
                entry["token_id"], "SELL", entry["our_size"], cur_price
            )
            with self._journal_lock:
                entry["status"]    = "PROFIT_TAKEN"
                entry["cur_price"] = round(cur_price, 4)
            self._log("copy", f"[ML LIVE] Exit {reason} | {entry['outcome']} @ {cur_price:.3f}")
            self.stats["exits_taken"] += 1
        except Exception as exc:
            self._log("error", f"[ML] Exit order failed: {exc}")

    def _try_settle_from_gamma(self, entry: dict) -> None:
        try:
            gamma = getattr(self.config, "gamma_api_base", "https://gamma-api.polymarket.com")
            resp = requests.get(
                f"{gamma}/markets",
                params={"clob_token_id": entry["token_id"]},
                timeout=5,
            )
            if resp.status_code != 200:
                return
            data = resp.json()
            markets = data if isinstance(data, list) else [data]
            for m in markets:
                prices = m.get("outcomePrices", [])
                if not prices:
                    continue
                # outcomePrices: ["1", "0"] or ["0", "1"] depending on which settled
                for i, tok in enumerate(m.get("tokens") or []):
                    tid = tok if isinstance(tok, str) else tok.get("token_id", "")
                    if tid == entry["token_id"] and i < len(prices):
                        final = float(prices[i])
                        pnl = (final - entry["entry_price"]) * entry["our_size"]
                        new_status = "WON" if final >= 0.99 else "LOST"
                        if entry["dry_run"]:
                            new_status = "WON" if final >= 0.99 else "LOST"
                        with self._journal_lock:
                            entry["status"]    = new_status
                            entry["cur_price"] = final
                            entry["pnl_usd"]   = round(pnl, 4)
                        return
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Online retraining
    # -----------------------------------------------------------------------

    def _maybe_record_outcome(self) -> None:
        """For any journal entry where window has ended, record features+label."""
        now = time.time()
        with self._journal_lock:
            pending = [e for e in self.trade_journal
                       if e.get("features") and
                          not e.get("_outcome_recorded") and
                          e["status"] in ("OPEN", "SIMULATED") and
                          now > e["window_start"] + self._window_sec + 30]

        for entry in pending:
            try:
                start_ms = entry["window_start"] * 1000
                end_ms   = start_ms + self._window_sec * 1000

                # Fetch BTC open at window start
                raw_open = self.fetcher._fetch_raw(
                    BinanceDataFetcher._CANDLE_URL,
                    params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": int(start_ms),
                        "endTime":   int(start_ms) + 60000,
                        "limit": 1,
                    },
                )
                # Fetch BTC close at window end
                raw_close = self.fetcher._fetch_raw(
                    BinanceDataFetcher._CANDLE_URL,
                    params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": int(end_ms) - 60000,
                        "endTime":   int(end_ms),
                        "limit": 1,
                    },
                )
                if not raw_open or not raw_close:
                    continue

                btc_open  = float(raw_open[0][1])   # open of first candle at window start
                btc_close = float(raw_close[-1][4])  # close of last candle at window end
                label = 1 if btc_close > btc_open else 0

                feat_array = FeatureEngine.to_array(entry["features"])
                with self._outcome_lock:
                    self._outcome_buffer.append({"X": feat_array, "y": label})
                    buf_len = len(self._outcome_buffer)
                self._save_outcome_buffer()
                self.stats["outcome_buffer"] = buf_len

                # Mark entry as resolved so we don't process again
                with self._journal_lock:
                    entry["_outcome_recorded"] = True

                threshold = getattr(self.config, "retrain_threshold", 10)
                self._log("info",
                    f"[ML] Outcome recorded | label={'UP' if label == 1 else 'DOWN'} | "
                    f"buffer={buf_len}/{threshold}")

            except Exception:
                pass

    def _maybe_retrain(self) -> None:
        threshold = getattr(self.config, "retrain_threshold", 10)
        with self._outcome_lock:
            n = len(self._outcome_buffer)
        self.stats["outcome_buffer"] = n
        if n < threshold:
            return

        window_size = getattr(self.config, "retrain_window_size", 500)
        with self._outcome_lock:
            buf = list(self._outcome_buffer[-window_size:])

        X = [row["X"] for row in buf]
        y = [row["y"] for row in buf]

        import numpy as np
        from collections import Counter as _Counter
        class_counts = _Counter(y)
        if len(class_counts) < 2 or min(class_counts.values()) < 2:
            self._log("warn",
                f"[ML] Retrain skipped — need ≥2 samples per class, "
                f"got {dict(class_counts)} in {len(X)} samples")
            return

        try:
            meta = self.model.train(X, y)
            self.stats["retrains"] += 1
            self.stats["model_accuracy"] = meta.get("accuracy")
            self.stats["model_samples"]  = meta.get("n_samples")
            self._log("info",
                f"[ML] Retrained on {len(X)} samples | "
                f"accuracy={meta.get('accuracy', 0):.1%}")
            with self._outcome_lock:
                self._outcome_buffer.clear()
            self._save_outcome_buffer()
        except Exception as exc:
            self._log("error", f"[ML] Retrain failed: {exc}")

    def _load_outcome_buffer(self) -> None:
        try:
            if os.path.exists(ML_OUTCOMES_FILE):
                with open(ML_OUTCOMES_FILE) as f:
                    self._outcome_buffer = json.load(f)
        except Exception:
            self._outcome_buffer = []

    def _save_outcome_buffer(self) -> None:
        try:
            with self._outcome_lock:
                data = list(self._outcome_buffer)
            with open(ML_OUTCOMES_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log(self, log_type: str, message: str, data: Optional[dict] = None) -> None:
        entry: dict = {
            "type":      log_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message":   message,
        }
        if data:
            entry.update(data)
        if self._log_cb:
            try:
                self._log_cb(entry)
            except Exception:
                pass
        else:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] [{log_type.upper()}] {message}")
