"""
market_analysis.py — BTC 5-minute directional predictor.

Primary source: Chainlink BTC/USD on-chain price feed (Polygon mainnet).
  Contract : 0xc907E116054Ad103354f2D350FD2514433D57F6f
  Access   : free eth_call via public Polygon JSON-RPC (no API key needed)
  Method   : batch latestRoundData + getRoundData → 5-min OHLCV buckets

Fallback sources (if Chainlink RPC calls fail):
  1. Bybit  — spot kline REST API
  2. Kraken — OHLC REST API
  3. Binance.US — klines REST API

Signals
-------
1. Momentum  — 3-candle rolling close direction (majority vote)
2. RSI(14)   — Wilder RSI; overbought > 70 → DOWN, oversold < 30 → UP
3. EMA cross — EMA(5) vs EMA(20) relative position
4. Volume    — last completed candle volume vs 5-candle rolling average
             (skipped / set NORMAL when using Chainlink data — no volume)

No new dependencies — only `requests` (already in requirements.txt) + stdlib.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Persisted across restarts — adaptive weights + resolved prediction history
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_state.json")

CACHE_TTL_SECONDS   = 30
FETCH_TIMEOUT_SECONDS = 10

# ── Chainlink on-chain constants ───────────────────────────────────────────────
_CL_BTC_USD    = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
_CL_BNB_USD    = "0x82a6c4AF830caa6c97bb504425f6A9921C08aAB2"  # Chainlink BNB/USD on Polygon
CHAINLINK_FEEDS = {"btc": _CL_BTC_USD, "bnb": _CL_BNB_USD}
_CL_DECIMALS   = 8           # answer / 1e8 = USD price
_LATEST_SEL    = "0xfeaf968c"  # latestRoundData()
_ROUND_SEL     = "0x9a6fc8f5"  # getRoundData(uint80)
# Polygon public RPC endpoints (same ones used by ChainMonitor in copy_trader.py)
_POLYGON_RPCS  = [
    "https://polygon.drpc.org",
    "https://polygon.publicnode.com",
    "https://polygon-public.nodies.app",
    "https://polygon-bor-rpc.publicnode.com",
]
# How many historical Chainlink rounds to fetch per refresh (one batch request)
# Reduced to 40 so public RPCs don't throttle/reject the batch
_CL_BATCH_ROUNDS = 40
# Target window size in seconds for OHLCV bucketing
_CANDLE_SEC = 300  # 5 minutes (default; overridden per-timeframe)

# ── Per-timeframe analysis configuration ──────────────────────────────────────
# Encodes the best-practice indicator settings for each Polymarket window size.
#
# 5m  (scalping)  — fast EMAs, MACD(6,13,5), momentum dominant.
#                   Candles: polymarket 5m / bybit 5 / kraken 5 / binance 5m
# 15m (intraday)  — standard EMAs(9,21), MACD(12,26,9), volume and Bollinger matter more.
#                   Candles: polymarket 15m / bybit 15 / kraken 15 / binance 15m
# 1h  (swing)     — wide EMAs(21,50), RSI(21), ADX(21), trend signals dominate.
#                   Candles: polymarket 1h  / bybit 60  / kraken 60 / binance 1h
TIMEFRAME_CONFIG: Dict[str, Any] = {
    "5m": {
        "window_sec":            300,
        "candle_interval":       "5m",   # Polymarket / Binance
        "candle_interval_bybit": "5",
        "candle_interval_kraken":"5",
        "ema_fast": 5, "ema_slow": 20,
        "rsi_period": 14,
        "macd_fast": 6, "macd_slow": 13, "macd_signal": 5,
        "bbands_period": 20,
        "supertrend_period": 10, "supertrend_mult": 2.5,
        "adx_period": 14,
        # Faster signals (momentum, MACD, supertrend) dominate in 5m scalping;
        # liquidity grabs very common in 5m — high weight
        "default_weights": {
            "ema": 3.0, "momentum": 2.0, "rsi": 2.0, "volume": 1.0,
            "macd": 2.5, "bbands": 2.0, "stoch_rsi": 2.0, "supertrend": 2.5,
            "fvg": 2.0, "choch": 2.0, "liquidity_grab": 2.5,
        },
    },
    "15m": {
        "window_sec":            900,
        "candle_interval":       "15m",
        "candle_interval_bybit": "15",
        "candle_interval_kraken":"15",
        "ema_fast": 9, "ema_slow": 21,
        "rsi_period": 14,
        # Standard MACD(12,26,9) is the gold standard for 15m intraday crypto
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bbands_period": 20,
        "supertrend_period": 10, "supertrend_mult": 3.0,
        "adx_period": 14,
        # Bollinger, MACD and supertrend get more weight; pure momentum less;
        # FVG and CHoCH useful for 15m intraday reversals
        "default_weights": {
            "ema": 2.5, "momentum": 1.5, "rsi": 2.0, "volume": 1.5,
            "macd": 3.0, "bbands": 2.5, "stoch_rsi": 2.0, "supertrend": 2.5,
            "fvg": 2.5, "choch": 2.5, "liquidity_grab": 2.0,
        },
    },
    "1h": {
        "window_sec":            3600,
        "candle_interval":       "1h",
        "candle_interval_bybit": "60",
        "candle_interval_kraken":"60",
        "ema_fast": 21, "ema_slow": 50,  # 21/50 cross is classic swing filter
        "rsi_period": 21,                # Wilder RSI(21) — smoother on hourly
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bbands_period": 20,
        "supertrend_period": 14, "supertrend_mult": 3.5,
        "adx_period": 21,                # wider ADX smoothing for hourly
        # Trend signals (EMA, supertrend, MACD) dominate for 1h swing trades;
        # CHoCH most valuable for hourly reversals
        "default_weights": {
            "ema": 3.5, "momentum": 1.5, "rsi": 2.0, "volume": 2.0,
            "macd": 3.0, "bbands": 2.0, "stoch_rsi": 1.5, "supertrend": 3.5,
            "fvg": 1.5, "choch": 3.0, "liquidity_grab": 1.5,
        },
    },
}


# ── Prediction tracking + adaptive learning ────────────────────────────────────

class PredictionTracker:
    """
    Tracks one pending prediction per market window.
    When the window flips, resolves the previous prediction and returns the record.
    """
    MAX_RECORDS = 20
    WINDOW_SEC  = 300  # updated at runtime by MarketAnalyzer.analyze(timeframe=...)

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._records: List[dict] = []
        self._pending: Optional[dict] = None

    def record(
        self,
        btc_price:         float,
        prediction:        str,
        signal_directions: dict,
        unix_ts:           int,
        confidence:        int = 0,
    ) -> Optional[dict]:
        """
        Register analysis result for the current window.
        Returns a resolved record dict if the previous window just closed, else None.
        """
        window_start = (unix_ts // self.WINDOW_SEC) * self.WINDOW_SEC
        resolved = None
        with self._lock:
            if self._pending and self._pending["window_start"] < window_start:
                resolved = self._resolve(self._pending, btc_price)
                self._records.append(resolved)
                self._pending = None

            if self._pending is None or self._pending["window_start"] != window_start:
                self._pending = {
                    "window_start":      window_start,
                    "price_at_start":    btc_price,
                    "prediction":        prediction,
                    "signal_directions": signal_directions,
                    "confidence":        confidence,
                    "resolved":          False,
                    "correct":           None,
                    "price_at_close":    None,
                }
            else:
                # Same window — update prediction with latest analysis
                self._pending["prediction"]        = prediction
                self._pending["signal_directions"] = signal_directions
                self._pending["confidence"]        = confidence
        return resolved

    def get_stats(self) -> dict:
        with self._lock:
            resolved = [r for r in self._records if r["resolved"]]
        total   = len(resolved)
        correct = sum(1 for r in resolved if r["correct"])
        last    = resolved[-1] if resolved else None
        records_out = [
            {
                "window_start":   r["window_start"],
                "prediction":     r["prediction"],
                "confidence":     r.get("confidence", 0),
                "price_at_start": round(r["price_at_start"], 2),
                "price_at_close": round(r["price_at_close"], 2) if r["price_at_close"] is not None else None,
                "correct":        r["correct"],
                "signals":        r.get("signal_directions", {}),
            }
            for r in reversed(resolved)  # newest first
        ]
        return {
            "total":        total,
            "correct":      correct,
            "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
            "last_outcome":     last["correct"]    if last else None,
            "last_prediction":  last["prediction"] if last else None,
            "records":          records_out,
        }

    @staticmethod
    def _resolve(pending: dict, close_price: float) -> dict:
        p = pending["prediction"]
        s = pending["price_at_start"]
        if p == "UP":
            correct: Optional[bool] = close_price > s
        elif p == "DOWN":
            correct = close_price < s
        else:
            correct = None  # NEUTRAL — unscored
        return {**pending, "resolved": True, "correct": correct, "price_at_close": close_price}


class AdaptiveWeights:
    """
    Holds the four signal weights and updates them when a prediction resolves.
    Reward signals that agreed with a correct prediction; penalise those that agreed with a wrong one.
    """
    DEFAULT = {
        "ema": 3.0, "momentum": 2.0, "rsi": 2.0, "volume": 1.0,
        "macd": 2.5, "bbands": 2.0, "stoch_rsi": 2.0, "supertrend": 2.5,
        "fvg": 2.0, "choch": 2.0, "liquidity_grab": 2.5,
    }
    W_MIN, W_MAX, VOL_MAX     = 0.5, 5.0, 1.5
    REWARD_AGREE, REWARD_CONTRA = 0.15, 0.05
    PUNISH_AGREE, PUNISH_CONTRA = -0.15, -0.05
    SIG_TO_KEY = {
        "momentum":       "momentum",
        "rsi":            "rsi",
        "ema_cross":      "ema",
        "volume":         "volume",
        "macd":           "macd",
        "bbands":         "bbands",
        "stoch_rsi":      "stoch_rsi",
        "supertrend":     "supertrend",
        "fvg":            "fvg",
        "choch":          "choch",
        "liquidity_grab": "liquidity_grab",
    }

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._weights = dict(self.DEFAULT)

    def get(self) -> dict:
        with self._lock:
            return dict(self._weights)

    def update(self, resolved_record: dict) -> dict:
        """Adjust weights based on the resolved prediction. Returns new weights dict."""
        if resolved_record["correct"] is None:
            return self.get()
        correct = resolved_record["correct"]
        pred    = resolved_record["prediction"]
        sigs    = resolved_record["signal_directions"]
        with self._lock:
            for sig_name, sig_val in sigs.items():
                key = self.SIG_TO_KEY.get(sig_name)
                if not key:
                    continue
                agreed = self._agrees(sig_val, pred)
                if agreed is None:
                    continue
                if correct:
                    delta = self.REWARD_AGREE if agreed else self.REWARD_CONTRA
                else:
                    delta = self.PUNISH_AGREE if agreed else self.PUNISH_CONTRA
                cap = self.VOL_MAX if key == "volume" else self.W_MAX
                self._weights[key] = max(self.W_MIN, min(cap, self._weights[key] + delta))
            return dict(self._weights)

    @staticmethod
    def _agrees(sig_val: str, prediction: str) -> Optional[bool]:
        s, p = sig_val.upper(), prediction.upper()
        if p == "UP":
            return True  if s in ("UP",   "HIGH") else \
                   False if s in ("DOWN", "LOW")  else None
        if p == "DOWN":
            return True  if s in ("DOWN", "LOW")  else \
                   False if s in ("UP",   "HIGH") else None
        return None


class MarketAnalyzer:
    """
    Thread-safe, lazily-refreshed market predictor.
    Supports BTC and BNB (asset parameter).

    Usage
    -----
        analyzer = MarketAnalyzer()                    # BTC (default)
        analyzer = MarketAnalyzer(asset="bnb")         # BNB
        result = analyzer.get_analysis()   # always returns a dict; never raises
    """

    def __init__(self, asset: str = "btc", state_file: Optional[str] = None) -> None:
        self._asset      = asset.lower()          # "btc" or "bnb"
        self._symbol     = f"{self._asset.upper()}USDT"  # "BTCUSDT" or "BNBUSDT"
        self._cl_feed    = CHAINLINK_FEEDS.get(self._asset)  # Chainlink contract or None
        self._state_file = state_file or STATE_FILE
        self._lock = threading.Lock()
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_fetched: float = 0.0
        self._session    = self._make_session()
        self._tracker    = PredictionTracker()
        self._weights_aw = AdaptiveWeights()
        self._last_points: Optional[List[dict]] = None
        self._load_state()

    # ── State persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load adaptive weights and prediction history from disk (if present)."""
        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)
            weights = state.get("weights", {})
            if weights:
                with self._weights_aw._lock:
                    for k, v in weights.items():
                        if k in self._weights_aw._weights:
                            self._weights_aw._weights[k] = float(v)
            records = state.get("records", [])
            if records:
                with self._tracker._lock:
                    self._tracker._records = records
            logger.info(
                "MarketAnalyzer: restored state — %d prediction records, weights=%s",
                len(records), weights,
            )
        except FileNotFoundError:
            pass  # first run — start fresh
        except Exception as exc:
            logger.warning("MarketAnalyzer: could not load state file: %s", exc)

    def _save_state(self) -> None:
        """Persist adaptive weights and resolved prediction history to disk."""
        try:
            with self._tracker._lock:
                records = list(self._tracker._records)
            state = {
                "weights":   self._weights_aw.get(),
                "records":   records,
                "saved_at":  _utc_now(),
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.warning("MarketAnalyzer: could not save state file: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_analysis(self, timeframe: str = "5m") -> Dict[str, Any]:
        """
        Return cached analysis if < 30 s old; otherwise fetch + recompute.
        timeframe: "5m" | "15m" | "1h"
        Never raises — always returns a well-formed dict (error key set on failure).
        """
        base_cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["5m"])
        # Inject asset-specific slug_prefix
        tf_cfg = dict(base_cfg)
        tf_cfg["slug_prefix"] = f"{self._asset}-updown-{timeframe}"
        # Invalidate cache when timeframe changes
        with self._lock:
            if self._last_result is not None and self._last_result.get("timeframe") != timeframe:
                self._last_fetched = 0
            age = time.monotonic() - self._last_fetched
            if self._last_result is not None and age < CACHE_TTL_SECONDS:
                return self._last_result

        # Update tracker window to match timeframe
        self._tracker.WINDOW_SEC = tf_cfg["window_sec"]

        # Network call outside the lock so Flask SSE threads aren't serialised.
        result = self._safe_refresh(timeframe, tf_cfg)

        with self._lock:
            self._last_result = result
            self._last_fetched = time.monotonic()
            return self._last_result

    # ── Internal fetch + compute ───────────────────────────────────────────────

    def _safe_refresh(self, timeframe: str = "5m", tf_cfg: Optional[dict] = None) -> Dict[str, Any]:
        """Wrap _fetch_candles + _analyze; return error dict on any failure."""
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["5m"])
        try:
            candles, source = self._fetch_candles(tf_cfg=tf_cfg)
            if not candles or len(candles) < 20:
                raise ValueError(f"Insufficient candle data ({len(candles or [])} candles) from {source}")
            return self._analyze(candles, source, timeframe, tf_cfg)
        except Exception as exc:
            logger.warning("MarketAnalyzer refresh failed: %s", exc)
            return {
                "price":         None,
                "btc_price":     None,   # backward-compat alias
                "asset":         self._asset,
                "price_to_beat": None,
                "prediction":    "NEUTRAL",
                "confidence":    0,
                "signals":       {},
                "source":        "none",
                "timeframe":     timeframe,
                "weights":       self._weights_aw.get(),
                "accuracy":      self._tracker.get_stats(),
                "candles_chart": None,
                "interval_label": None,
                "updated_at":    _utc_now(),
                "error":         str(exc),
            }

    def _fetch_candles(self, limit: int = 40, tf_cfg: Optional[dict] = None) -> Tuple[List[dict], str]:
        """
        Try each source in order. Returns (candles, source_name).
        Polymarket's own Chainlink Data Streams feed is primary — exact settlement prices.
        """
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        self._last_points = None  # reset; set only by Chainlink direct path
        errors = []
        sources = [
            ("polymarket_chainlink", lambda n: self._fetch_polymarket_chainlink(n, tf_cfg)),
            ("chainlink",            lambda n: self._fetch_chainlink(n, tf_cfg)),
            ("bybit",                lambda n: self._fetch_bybit(n, tf_cfg)),
            ("kraken",               lambda n: self._fetch_kraken(n, tf_cfg)),
            ("binance_us",           lambda n: self._fetch_binance_us(n, tf_cfg)),
        ]
        for name, fetcher in sources:
            try:
                candles = fetcher(limit)
                if candles and len(candles) >= 15:
                    logger.debug("MarketAnalyzer: using %s (%d candles)", name, len(candles))
                    return candles, name
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.debug("MarketAnalyzer source %s failed: %s", name, exc)

        raise RuntimeError("All candle sources failed — " + "; ".join(errors))

    def _analyze(self, candles: List[dict], source: str,
                 timeframe: str = "5m", tf_cfg: Optional[dict] = None) -> Dict[str, Any]:
        """Compute all signals with timeframe-appropriate parameters, return full dict."""
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["5m"])
        closes    = [c["close"]  for c in candles]
        volumes   = [c["volume"] for c in candles]
        highs     = [c["high"]   for c in candles]
        lows      = [c["low"]    for c in candles]
        btc_price = closes[-1]
        unix_ts   = int(time.time())

        ema_fast  = tf_cfg.get("ema_fast", 5)
        ema_slow  = tf_cfg.get("ema_slow", 20)
        rsi_per   = tf_cfg.get("rsi_period", 14)
        adx_per   = tf_cfg.get("adx_period", 14)
        st_per    = tf_cfg.get("supertrend_period", 10)
        st_mult   = tf_cfg.get("supertrend_mult", 2.5)

        mom       = self._signal_momentum(closes)
        rsi_val   = self._calc_rsi(closes, period=rsi_per)
        rsi_sig   = self._signal_rsi(rsi_val)
        ema_f     = self._calc_ema(closes, ema_fast)
        ema_s     = self._calc_ema(closes, ema_slow)
        ema_cross = self._signal_ema_cross(ema_f, ema_s)
        # Volume signal is unreliable from Chainlink (no trade volume on-chain)
        vol_sig   = {"signal": "NORMAL", "ratio": 1.0} if source == "chainlink" \
                    else self._signal_volume(volumes)
        macd_sig  = self._calc_macd(closes, tf_cfg)
        bb_sig    = self._calc_bbands(closes, period=tf_cfg.get("bbands_period", 20))
        srsi_sig  = self._calc_stoch_rsi(closes, rsi_period=rsi_per)
        st_sig    = self._calc_supertrend(highs, lows, closes, period=st_per, mult=st_mult)
        adx_info  = self._calc_adx(highs, lows, closes, period=adx_per)
        # SMC signals
        fvg_sig      = self._signal_fvg(highs, lows, closes)
        choch_sig    = self._signal_choch(highs, lows, closes)
        liq_grab_sig = self._signal_liquidity_grab(highs, lows, closes)

        signal_directions = {
            "momentum":       mom["signal"],
            "rsi":            rsi_sig["signal"],
            "ema_cross":      ema_cross["signal"],
            "volume":         vol_sig["signal"],
            "macd":           macd_sig["signal"],
            "bbands":         bb_sig["signal"],
            "stoch_rsi":      srsi_sig["signal"],
            "supertrend":     st_sig["signal"],
            "fvg":            fvg_sig["signal"],
            "choch":          choch_sig["signal"],
            "liquidity_grab": liq_grab_sig["signal"],
        }

        # Merge adaptive weights with timeframe defaults (adaptive weights take precedence)
        tf_defaults = tf_cfg.get("default_weights", {})
        adaptive    = self._weights_aw.get()
        weights     = {k: adaptive.get(k, tf_defaults.get(k, 1.0)) for k in
                       set(tf_defaults) | set(adaptive)}
        prediction, confidence = self._composite(
            mom["signal"], rsi_sig["signal"], ema_cross["signal"], vol_sig["signal"],
            macd_sig["signal"], bb_sig["signal"], srsi_sig["signal"], st_sig["signal"],
            fvg_sig["signal"], choch_sig["signal"], liq_grab_sig["signal"],
            weights=weights,
        )

        # ADX confidence modifier — regime detection
        adx = adx_info["adx"]
        if adx < 20:
            confidence = max(0,  int(confidence * 0.80))  # choppy market — reduce trust
        elif adx > 25:
            confidence = min(95, int(confidence * 1.10))  # trending market — boost trust

        # Track prediction; update weights if previous window resolved
        resolved = self._tracker.record(btc_price, prediction, signal_directions, unix_ts, confidence)
        if resolved is not None:
            weights = self._weights_aw.update(resolved)
            logger.info(
                "Prediction resolved: %s → %s (correct=%s)  new weights: %s",
                resolved["prediction"], resolved["price_at_close"],
                resolved["correct"], weights,
            )
            self._save_state()

        # Price to beat = open of the current window.
        # ONLY computed from polymarket_chainlink — exchange fallbacks (bybit /
        # kraken / binance_us) use exchange-aggregated prices that can differ
        # hundreds of dollars from Polymarket's Chainlink Data Streams oracle.
        window_sec = tf_cfg.get("window_sec", 300)
        current_window_ms = ((unix_ts // window_sec) * window_sec) * 1000
        price_to_beat: Optional[float] = None
        if source == "polymarket_chainlink":
            for c in reversed(candles):
                if c["open_time"] == current_window_ms:
                    price_to_beat = round(c["open"], 2)
                    break
            # Only fall back to last candle when it IS the current window —
            # at window start the new candle may not exist yet; using the
            # previous window's open would show the wrong price-to-beat.
            if price_to_beat is None and candles and candles[-1]["open_time"] == current_window_ms:
                price_to_beat = round(candles[-1]["open"], 2)

        # 1-minute candles for chart display
        candles_1m: Optional[List[dict]] = None
        interval_label: Optional[str] = None
        if self._last_points:
            # Chainlink direct: bucket raw oracle rounds into 30s candles
            candles_chart  = _points_to_candles(self._last_points, 30)[-30:]
            interval_label = "30s"
        else:
            # Try Kraken 1m for chart display; fall back to the 5m signal candles
            try:
                candles_chart  = self._fetch_kraken_1m(20)
                interval_label = "1m"
            except Exception:
                # Always show something — use the 5m candles we already fetched
                candles_chart  = candles[-20:]
                interval_label = "5m"

        return {
            "price":         round(btc_price, 2),
            "btc_price":     round(btc_price, 2),   # backward-compat alias
            "asset":         self._asset,
            "price_to_beat": price_to_beat,
            "prediction":    prediction,
            "confidence":    confidence,
            "source":        source,
            "timeframe":     timeframe,
            "signals": {
                "momentum":       mom,
                "rsi":            {**rsi_sig, "value": round(rsi_val, 1)},
                "ema_cross":      ema_cross,
                "volume":         vol_sig,
                "macd":           macd_sig,
                "bbands":         bb_sig,
                "stoch_rsi":      srsi_sig,
                "supertrend":     st_sig,
                "adx":            adx_info,
                "fvg":            fvg_sig,
                "choch":          choch_sig,
                "liquidity_grab": liq_grab_sig,
            },
            "weights":        weights,
            "accuracy":       self._tracker.get_stats(),
            "candles_chart":  candles_chart,
            "interval_label": interval_label,
            "updated_at":     _utc_now(),
            "error":          None,
        }

    # ── Polymarket Chainlink Data Streams (primary) ────────────────────────────

    def _fetch_polymarket_chainlink(self, limit: int = 20, tf_cfg: Optional[dict] = None) -> List[dict]:
        """
        Polymarket's own Chainlink Data Streams feed — the exact prices used for
        settlement. Returns 5m OHLCV candles.
        Response: {"candles": [{time, open, high, low, close}, ...]}
        """
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        r = self._session.get(
            "https://polymarket.com/api/chainlink-candles",
            params={"symbol": self._asset.upper(), "interval": tf_cfg.get("candle_interval", "5m")},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=FETCH_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        raw = r.json().get("candles", [])
        if not raw:
            raise ValueError("Polymarket chainlink-candles: empty response")
        candles = [
            {
                "open_time": int(c["time"]) * 1000,
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c.get("volume", 1.0)),
            }
            for c in raw
        ]
        return candles[-limit:]

    # ── Kraken 1-minute klines (chart display fallback) ────────────────────────

    def _fetch_kraken_1m(self, limit: int = 20) -> List[dict]:
        """Kraken 1-minute OHLC — used for chart display when Chainlink direct
        is unavailable. Only available for BTC (XBTUSD)."""
        if self._asset != "btc":
            raise RuntimeError(f"Kraken 1m source not available for asset={self._asset!r}")
        url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
        resp = self._session.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise ValueError(f"Kraken 1m: {data['error']}")
        pair_key = next(k for k in data["result"] if k != "last")
        rows = data["result"][pair_key][-limit:]
        return [
            {
                "open_time": int(row[0]) * 1000,
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[6]),
            }
            for row in rows
        ]

    # ── Chainlink on-chain source ──────────────────────────────────────────────

    def _fetch_chainlink(self, limit: int = 20, tf_cfg: Optional[dict] = None) -> List[dict]:
        """
        Read Chainlink asset/USD feed on Polygon via batch JSON-RPC eth_call.

        1. Call latestRoundData() → get current price + round ID
        2. Batch getRoundData() for the previous _CL_BATCH_ROUNDS rounds
           (single HTTP POST, so total latency ≈ one round-trip)
        3. Decode all price points, bucket into 5-minute OHLCV candles
        """
        if not self._cl_feed:
            raise RuntimeError(f"No Chainlink feed configured for asset={self._asset!r}")
        errors = []
        for rpc_url in _POLYGON_RPCS:
            try:
                return self._chainlink_via_rpc(rpc_url, limit)
            except Exception as exc:
                errors.append(f"{rpc_url}: {exc}")
        raise RuntimeError("Chainlink RPC all failed: " + "; ".join(errors))

    def _chainlink_via_rpc(self, rpc_url: str, limit: int) -> List[dict]:
        # ── Step 1: latest round ──────────────────────────────────────────────
        raw = self._rpc_single(rpc_url, "eth_call", [
            {"to": self._cl_feed, "data": _LATEST_SEL}, "latest"
        ])
        round_id, price, updated_at = self._decode_round(raw)
        if price <= 0 or updated_at == 0:
            raise ValueError("Chainlink: invalid latest round data")

        points: List[dict] = [{"price": price, "ts": updated_at}]

        # ── Step 2: build batch for historical rounds ─────────────────────────
        phase_id = round_id >> 64
        aggr_id  = round_id & 0xFFFF_FFFF_FFFF_FFFF

        batch = []
        for i in range(1, _CL_BATCH_ROUNDS + 1):
            prev_aggr = aggr_id - i
            if prev_aggr <= 0:
                break
            prev_rid = (phase_id << 64) | prev_aggr
            # ABI-encode: selector + uint80 padded to 32 bytes
            data = _ROUND_SEL + format(prev_rid, "064x")
            batch.append({
                "jsonrpc": "2.0",
                "method":  "eth_call",
                "params":  [{"to": self._cl_feed, "data": data}, "latest"],
                "id":      i,
            })

        if not batch:
            raise ValueError("Chainlink: could not build batch (aggr_id too small)")

        # ── Step 3: send batch ────────────────────────────────────────────────
        resp = self._session.post(rpc_url, json=batch, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, list):
            raise ValueError(f"Chainlink batch: unexpected response type {type(results)}")

        for item in results:
            raw_result = item.get("result", "")
            if not raw_result or raw_result == "0x":
                continue
            try:
                _, r_price, r_ts = self._decode_round(raw_result)
                if r_price > 0 and r_ts > 0:
                    points.append({"price": r_price, "ts": r_ts})
            except Exception:
                continue

        if len(points) < 5:
            raise ValueError(f"Chainlink: only {len(points)} valid price points")

        # ── Step 4: store raw points for 30s chart, then bucket into timeframe candles ──
        self._last_points = list(points)
        candle_sec = tf_cfg.get("window_sec", _CANDLE_SEC) if tf_cfg else _CANDLE_SEC
        candles = _points_to_candles(points, candle_sec)
        return candles[-limit:]

    @staticmethod
    def _decode_round(hex_data: str) -> Tuple[int, float, int]:
        """
        ABI-decode a latestRoundData / getRoundData response.
        Returns (round_id, price_usd, updated_at_unix).
        Layout: 5 × 32-byte words = 160 bytes minimum.
          [0]  roundId        uint80  → uint256 ABI
          [1]  answer         int256  (price × 1e8)
          [2]  startedAt      uint256
          [3]  updatedAt      uint256
          [4]  answeredInRound uint80 → uint256 ABI
        """
        raw = hex_data[2:] if hex_data.startswith("0x") else hex_data
        data = bytes.fromhex(raw)
        if len(data) < 160:
            raise ValueError(f"Round data too short: {len(data)} bytes")

        def u256(offset: int) -> int:
            return int.from_bytes(data[offset: offset + 32], "big")

        round_id   = u256(0)
        answer_raw = int.from_bytes(data[32:64], "big", signed=True)  # int256
        updated_at = u256(96)
        price_usd  = answer_raw / (10 ** _CL_DECIMALS)
        return round_id, price_usd, updated_at

    def _rpc_single(self, rpc_url: str, method: str, params: list) -> str:
        """Single JSON-RPC call, returns the 'result' string."""
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        resp = self._session.post(rpc_url, json=payload, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error: {data['error']}")
        return data.get("result", "")

    # ── Exchange REST fallback sources ─────────────────────────────────────────

    def _fetch_bybit(self, limit: int = 20, tf_cfg: Optional[dict] = None) -> List[dict]:
        """Bybit V5 spot klines (newest-first, so reversed)."""
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        url = (
            "https://api.bybit.com/v5/market/kline"
            f"?category=spot&symbol={self._symbol}&interval={tf_cfg.get('candle_interval_bybit', '5')}&limit={limit}"
        )
        resp = self._session.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            raise ValueError(f"Bybit error: {data.get('retMsg')}")
        rows = data["result"]["list"]
        candles = [
            {
                "open_time": int(row[0]),
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
        ]
        candles.reverse()
        return candles

    def _fetch_kraken(self, limit: int = 20, tf_cfg: Optional[dict] = None) -> List[dict]:
        """Kraken OHLC — columns: [time, open, high, low, close, vwap, volume, count].
        Only available for BTC (XBTUSD). Skipped for other assets."""
        if self._asset != "btc":
            raise RuntimeError(f"Kraken source not available for asset={self._asset!r}")
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        url = (
            "https://api.kraken.com/0/public/OHLC"
            f"?pair=XBTUSD&interval={tf_cfg.get('candle_interval_kraken', '5')}"
        )
        resp = self._session.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise ValueError(f"Kraken error: {data['error']}")
        pair_key = next(k for k in data["result"] if k != "last")
        rows = data["result"][pair_key][-limit:]
        return [
            {
                "open_time": int(row[0]) * 1000,
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[6]),
            }
            for row in rows
        ]

    def _fetch_binance_us(self, limit: int = 20, tf_cfg: Optional[dict] = None) -> List[dict]:
        """Binance.US klines — same format as Binance global."""
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        url = (
            "https://api.binance.us/api/v3/klines"
            f"?symbol={self._symbol}&interval={tf_cfg.get('candle_interval', '5m')}&limit={limit}"
        )
        resp = self._session.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        raw: List[List] = resp.json()
        return [
            {
                "open_time": int(row[0]),
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            }
            for row in raw
        ]

    # ── Individual signals ─────────────────────────────────────────────────────

    def _signal_momentum(self, closes: List[float]) -> dict:
        """Count up vs down moves in the last 3 completed candles (index -4:-1)."""
        last3 = closes[-4:-1]
        if len(last3) < 3:
            last3 = closes[-3:]
        pairs    = len(last3) - 1
        up_count = sum(1 for i in range(1, len(last3)) if last3[i] > last3[i - 1])
        dn_count = pairs - up_count
        if up_count > dn_count:
            sig = "UP"
        elif dn_count > up_count:
            sig = "DOWN"
        else:
            sig = "NEUTRAL"
        return {"signal": sig, "detail": f"{up_count}/{pairs} recent candles bullish"}

    def _signal_rsi(self, rsi: float) -> dict:
        """Map RSI value to a directional signal."""
        if rsi > 70:
            sig = "DOWN"
        elif rsi < 30:
            sig = "UP"
        elif rsi >= 55:
            sig = "UP"
        elif rsi <= 45:
            sig = "DOWN"
        else:
            sig = "NEUTRAL"
        return {"signal": sig}

    def _signal_ema_cross(self, ema5: List[float], ema20: List[float]) -> dict:
        """UP if EMA5 > EMA20, DOWN if below, NEUTRAL if equal."""
        if not ema5 or not ema20:
            return {"signal": "NEUTRAL", "ema5": None, "ema20": None}
        e5, e20 = ema5[-1], ema20[-1]
        sig = "UP" if e5 > e20 else "DOWN" if e5 < e20 else "NEUTRAL"
        return {"signal": sig, "ema5": round(e5, 2), "ema20": round(e20, 2)}

    def _signal_volume(self, volumes: List[float]) -> dict:
        """Last completed candle volume vs 5-candle avg. Not used for Chainlink."""
        if len(volumes) < 7:
            return {"signal": "NORMAL", "ratio": 1.0}
        last_vol = volumes[-2]
        avg5     = sum(volumes[-7:-2]) / 5
        ratio    = last_vol / avg5 if avg5 > 0 else 1.0
        sig = "HIGH" if ratio >= 1.3 else "LOW" if ratio <= 0.7 else "NORMAL"
        return {"signal": sig, "ratio": round(ratio, 2)}

    @staticmethod
    def _signal_fvg(highs: List[float], lows: List[float], closes: List[float]) -> dict:
        """
        Fair Value Gap (FVG) — 3-candle price imbalance where candle N and candle N-2
        don't overlap, leaving a gap that price is 'drawn' to fill.

        Bullish FVG: candle[n].low > candle[n-2].high → strong upward impulse → UP
        Bearish FVG: candle[n].high < candle[n-2].low → strong downward impulse → DOWN

        Scans the last 5 candles newest-first, returns the first (freshest) FVG found.
        """
        n = len(closes)
        if n < 5:
            return {"signal": "NEUTRAL", "detail": "insufficient data"}
        for i in range(n - 2, max(n - 6, 1), -1):
            if i - 2 < 0:
                break
            if lows[i] > highs[i - 2]:
                gap = round(lows[i] - highs[i - 2], 0)
                return {"signal": "UP",   "detail": f"Bullish FVG ${gap:.0f}"}
            if highs[i] < lows[i - 2]:
                gap = round(lows[i - 2] - highs[i], 0)
                return {"signal": "DOWN", "detail": f"Bearish FVG ${gap:.0f}"}
        return {"signal": "NEUTRAL", "detail": "no FVG"}

    @staticmethod
    def _signal_choch(highs: List[float], lows: List[float], closes: List[float],
                      lookback: int = 8) -> dict:
        """
        Change of Character (CHoCH) — first sign of trend reversal.

        Bullish CHoCH: recent candles were mostly falling, current close breaks
                       above the highest high of the lookback window → UP.
        Bearish CHoCH: recent candles were mostly rising, current close breaks
                       below the lowest low of the lookback window → DOWN.
        """
        if len(closes) < lookback + 2:
            return {"signal": "NEUTRAL"}
        prior_c = closes[-(lookback + 1):-1]
        prior_h = highs[-(lookback + 1):-1]
        prior_l = lows[-(lookback + 1):-1]
        cur = closes[-1]
        ups = sum(1 for j in range(1, len(prior_c)) if prior_c[j] > prior_c[j - 1])
        dns = len(prior_c) - 1 - ups
        swing_high = max(prior_h)
        swing_low  = min(prior_l)
        if dns > ups and cur > swing_high:
            return {"signal": "UP",   "detail": f"CHoCH bull: broke ${swing_high:.0f}"}
        if ups > dns and cur < swing_low:
            return {"signal": "DOWN", "detail": f"CHoCH bear: broke ${swing_low:.0f}"}
        return {"signal": "NEUTRAL"}

    @staticmethod
    def _signal_liquidity_grab(highs: List[float], lows: List[float], closes: List[float],
                               lookback: int = 8) -> dict:
        """
        Liquidity Grab — price spikes past a recent swing high/low (sweeping stops),
        then closes back within range, signalling a sharp reversal.

        Bearish grab: current candle's high > lookback swing high AND close < prev close → DOWN
        Bullish grab: current candle's low < lookback swing low  AND close > prev close → UP
        """
        if len(closes) < lookback + 2:
            return {"signal": "NEUTRAL"}
        prior_h = highs[-(lookback + 1):-1]
        prior_l = lows[-(lookback + 1):-1]
        cur_high  = highs[-1]
        cur_low   = lows[-1]
        cur_close = closes[-1]
        prev_close = closes[-2]
        swing_high = max(prior_h)
        swing_low  = min(prior_l)
        if cur_high > swing_high and cur_close < prev_close:
            return {"signal": "DOWN", "detail": f"Bear liq grab: swept ${swing_high:.0f}"}
        if cur_low < swing_low and cur_close > prev_close:
            return {"signal": "UP",   "detail": f"Bull liq grab: swept ${swing_low:.0f}"}
        return {"signal": "NEUTRAL"}

    def _calc_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Wilder-smoothed ATR. Returns list aligned with closes (length == len(closes))."""
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i]  - closes[i - 1]))
            trs.append(tr)
        if not trs or len(trs) < period:
            return [highs[0] - lows[0]] * len(closes)
        atr_vals = [sum(trs[:period]) / period]
        for tr in trs[period:]:
            atr_vals.append((atr_vals[-1] * (period - 1) + tr) / period)
        pad = len(closes) - len(atr_vals)
        return [atr_vals[0]] * pad + atr_vals

    def _calc_supertrend(self, highs: List[float], lows: List[float], closes: List[float],
                         period: int = 10, mult: float = 2.5) -> dict:
        """Supertrend(10, 2.5). Price above lower band → UP; below upper band → DOWN."""
        if len(closes) < period + 2:
            return {"signal": "NEUTRAL", "value": None}
        atr = self._calc_atr(highs, lows, closes, period)
        upper = [(highs[i] + lows[i]) / 2 + mult * atr[i] for i in range(len(closes))]
        lower = [(highs[i] + lows[i]) / 2 - mult * atr[i] for i in range(len(closes))]
        final_upper = list(upper)
        final_lower = list(lower)
        trend = [1] * len(closes)  # 1 = UP, -1 = DOWN
        for i in range(1, len(closes)):
            final_upper[i] = upper[i] if (upper[i] < final_upper[i-1] or closes[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = lower[i] if (lower[i] > final_lower[i-1] or closes[i-1] < final_lower[i-1]) else final_lower[i-1]
            if trend[i-1] == 1:
                trend[i] = -1 if closes[i] < final_lower[i] else 1
            else:
                trend[i] =  1 if closes[i] > final_upper[i] else -1
        sig = "UP" if trend[-1] == 1 else "DOWN"
        st_val = final_lower[-1] if trend[-1] == 1 else final_upper[-1]
        return {"signal": sig, "value": round(st_val, 2)}

    def _calc_adx(self, highs: List[float], lows: List[float], closes: List[float],
                  period: int = 14) -> dict:
        """ADX(14). Returns adx score + strength label (STRONG/MODERATE/WEAK)."""
        if len(closes) < period * 2 + 1:
            return {"adx": 20.0, "strength": "MODERATE"}
        dm_pos, dm_neg, trs = [], [], []
        for i in range(1, len(closes)):
            up   = highs[i]  - highs[i - 1]
            down = lows[i-1] - lows[i]
            dm_pos.append(up   if up > down and up > 0   else 0.0)
            dm_neg.append(down if down > up and down > 0 else 0.0)
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i]  - closes[i - 1])))

        def _wilder(vals: list) -> list:
            s = [sum(vals[:period])]
            for v in vals[period:]:
                s.append(s[-1] - s[-1] / period + v)
            return s

        sm_tr  = _wilder(trs)
        sm_pos = _wilder(dm_pos)
        sm_neg = _wilder(dm_neg)
        dx_vals = []
        for i in range(len(sm_tr)):
            if sm_tr[i] == 0:
                dx_vals.append(0.0)
                continue
            di_pos = 100 * sm_pos[i] / sm_tr[i]
            di_neg = 100 * sm_neg[i] / sm_tr[i]
            denom  = di_pos + di_neg
            dx_vals.append(100 * abs(di_pos - di_neg) / denom if denom else 0.0)
        adx_vals = [sum(dx_vals[:period]) / period]
        for dx in dx_vals[period:]:
            adx_vals.append((adx_vals[-1] * (period - 1) + dx) / period)
        adx = adx_vals[-1]
        strength = "STRONG" if adx > 25 else "WEAK" if adx < 20 else "MODERATE"
        return {"adx": round(adx, 1), "strength": strength}

    def _calc_macd(self, closes: List[float], tf_cfg: Optional[dict] = None) -> dict:
        """MACD with timeframe-aware defaults.
        Histogram > 0 and rising → UP; < 0 and falling → DOWN."""
        if tf_cfg is None:
            tf_cfg = TIMEFRAME_CONFIG["5m"]
        if len(closes) < 20:
            return {"signal": "NEUTRAL", "histogram": 0.0}
        macd_fast   = int(tf_cfg.get("macd_fast", 6))
        macd_slow   = int(tf_cfg.get("macd_slow", 13))
        macd_signal = int(tf_cfg.get("macd_signal", 5))
        ema12       = self._calc_ema(closes, macd_fast)
        ema26       = self._calc_ema(closes, macd_slow)
        macd_line   = [ema12[i] - ema26[i] for i in range(len(closes))]
        signal_line = self._calc_ema(macd_line, macd_signal)
        histogram   = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]
        h, h_prev   = histogram[-1], histogram[-2]
        if h > 0 and h >= h_prev:   sig = "UP"
        elif h < 0 and h <= h_prev: sig = "DOWN"
        elif h > 0:                 sig = "UP"
        elif h < 0:                 sig = "DOWN"
        else:                       sig = "NEUTRAL"
        return {"signal": sig, "histogram": round(h, 2)}

    def _calc_bbands(self, closes: List[float], period: int = 20, mult: float = 2.0) -> dict:
        """Bollinger Bands(20,2σ). Above upper band → DOWN; below lower → UP; squeeze flag."""
        if len(closes) < period:
            return {"signal": "NEUTRAL", "squeeze": False, "pct_b": 0.5}
        window = closes[-period:]
        sma    = sum(window) / period
        std    = (sum((c - sma) ** 2 for c in window) / period) ** 0.5
        upper  = sma + mult * std
        lower  = sma - mult * std
        brange = upper - lower
        price  = closes[-1]
        pct_b  = (price - lower) / brange if brange > 0 else 0.5
        bw     = brange / sma if sma > 0 else 0.0
        # Squeeze: current bandwidth < 80% of avg of previous 5 bandwidths
        squeeze = False
        if len(closes) >= period + 5:
            prev_bws = []
            for i in range(1, 6):
                sl  = closes[-(period + i): len(closes) - i]
                s2  = sum(sl) / period
                v2  = (sum((c - s2) ** 2 for c in sl) / period) ** 0.5
                prev_bws.append((2 * mult * v2) / s2 if s2 > 0 else 0.0)
            squeeze = bw < (sum(prev_bws) / len(prev_bws)) * 0.8
        if pct_b > 1.0:    sig = "DOWN"
        elif pct_b < 0.0:  sig = "UP"
        elif pct_b > 0.55: sig = "UP"
        elif pct_b < 0.45: sig = "DOWN"
        else:              sig = "NEUTRAL"
        return {"signal": sig, "squeeze": squeeze,
                "pct_b": round(pct_b, 3), "bandwidth": round(bw * 100, 2)}

    def _calc_stoch_rsi(self, closes: List[float],
                        rsi_period: int = 14, stoch_period: int = 14,
                        smooth_k: int = 3,   smooth_d: int = 3) -> dict:
        """%K < 20 → UP (oversold); %K > 80 → DOWN (overbought); crossovers used in mid-range."""
        min_needed = rsi_period + stoch_period + smooth_k + smooth_d
        if len(closes) < min_needed:
            return {"signal": "NEUTRAL", "k": 50.0, "d": 50.0}
        # Build Wilder RSI series
        deltas   = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains    = [max(d, 0.0) for d in deltas]
        losses   = [max(-d, 0.0) for d in deltas]
        avg_gain = sum(gains[:rsi_period]) / rsi_period
        avg_loss = sum(losses[:rsi_period]) / rsi_period
        rsi_vals: List[float] = []
        for i in range(rsi_period, len(gains)):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
            rsi_vals.append(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss else 100.0)
        if len(rsi_vals) < stoch_period + smooth_k + smooth_d - 1:
            return {"signal": "NEUTRAL", "k": 50.0, "d": 50.0}
        # Stochastic of RSI
        raw_k: List[float] = []
        for i in range(len(rsi_vals) - stoch_period + 1):
            w   = rsi_vals[i: i + stoch_period]
            lo, hi = min(w), max(w)
            raw_k.append((rsi_vals[i + stoch_period - 1] - lo) / (hi - lo) * 100.0 if hi > lo else 50.0)
        k_vals = [sum(raw_k[i: i + smooth_k]) / smooth_k for i in range(len(raw_k) - smooth_k + 1)]
        d_vals = [sum(k_vals[i: i + smooth_d]) / smooth_d for i in range(len(k_vals) - smooth_d + 1)]
        if not d_vals:
            return {"signal": "NEUTRAL", "k": 50.0, "d": 50.0}
        k, d   = k_vals[-1], d_vals[-1]
        k_prev = k_vals[-2] if len(k_vals) >= 2 else k
        if k < 20:                  sig = "UP"
        elif k > 80:                sig = "DOWN"
        elif k > d and k_prev <= d: sig = "UP"
        elif k < d and k_prev >= d: sig = "DOWN"
        elif k > d:                 sig = "UP"
        elif k < d:                 sig = "DOWN"
        else:                       sig = "NEUTRAL"
        return {"signal": sig, "k": round(k, 1), "d": round(d, 1)}

    # ── Composite vote ─────────────────────────────────────────────────────────

    def _composite(
        self,
        mom_sig:  str,
        rsi_sig:  str,
        ema_sig:  str,
        vol_sig:  str,
        macd_sig:       str = "NEUTRAL",
        bb_sig:         str = "NEUTRAL",
        srsi_sig:       str = "NEUTRAL",
        st_sig:         str = "NEUTRAL",
        fvg_sig:        str = "NEUTRAL",
        choch_sig:      str = "NEUTRAL",
        liq_grab_sig:   str = "NEUTRAL",
        weights: Optional[dict] = None,
    ) -> tuple:
        """
        Weighted vote using adaptive weights.
        weights defaults to AdaptiveWeights.DEFAULT if not supplied.
        """
        if weights is None:
            weights = {
                "ema": 3.0, "momentum": 2.0, "rsi": 2.0, "volume": 1.0,
                "macd": 2.5, "bbands": 2.0, "stoch_rsi": 2.0, "supertrend": 2.5,
                "fvg": 2.0, "choch": 2.0, "liquidity_grab": 2.5,
            }

        def score(sig: str, w: float) -> float:
            return w if sig == "UP" else -w if sig == "DOWN" else 0.0

        vol_w  = weights.get("volume", 1.0)
        total  = (
            score(ema_sig,      weights.get("ema", 3.0))
            + score(mom_sig,    weights.get("momentum", 2.0))
            + score(rsi_sig,    weights.get("rsi", 2.0))
            + score(macd_sig,   weights.get("macd", 2.5))
            + score(bb_sig,     weights.get("bbands", 2.0))
            + score(srsi_sig,   weights.get("stoch_rsi", 2.0))
            + score(st_sig,     weights.get("supertrend", 2.5))
            + score(fvg_sig,    weights.get("fvg", 2.0))
            + score(choch_sig,  weights.get("choch", 2.0))
            + score(liq_grab_sig, weights.get("liquidity_grab", 2.5))
            + (vol_w if vol_sig == "HIGH" else -vol_w if vol_sig == "LOW" else 0.0)
        )
        max_possible = max(sum(weights.values()), 0.01)  # avoid div-by-zero

        if total > 0:
            return "UP",   min(int(40 + (total / max_possible) * 55), 95)
        if total < 0:
            return "DOWN", min(int(40 + (abs(total) / max_possible) * 55), 95)
        return "NEUTRAL", 0

    # ── Technical indicators ───────────────────────────────────────────────────

    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        """Wilder RSI with SMA seed. Returns 50 if insufficient data."""
        if len(closes) < period + 1:
            return 50.0
        deltas   = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains    = [max(d, 0.0) for d in deltas]
        losses   = [max(-d, 0.0) for d in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    def _calc_ema(self, closes: List[float], period: int) -> List[float]:
        """EMA with SMA seed. Leading positions filled with seed value."""
        if len(closes) < period:
            return closes[:]
        k    = 2.0 / (period + 1)
        seed = sum(closes[:period]) / period
        out  = [seed] * period
        prev = seed
        for price in closes[period:]:
            prev = price * k + prev * (1 - k)
            out.append(prev)
        return out

    # ── HTTP session ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers["User-Agent"] = "polymarket-copy-bot/1.0"
        return session


# ── Module helpers ─────────────────────────────────────────────────────────────

def _points_to_candles(points: List[dict], interval_sec: int) -> List[dict]:
    """
    Group {price, ts} points into OHLCV candles of `interval_sec` seconds.
    Volume = count of Chainlink price updates in the bucket (no trade volume).
    """
    buckets: Dict[int, dict] = {}
    for pt in points:
        bucket = (pt["ts"] // interval_sec) * interval_sec
        if bucket not in buckets:
            buckets[bucket] = {
                "open_time": bucket * 1000,
                "open":   pt["price"],
                "high":   pt["price"],
                "low":    pt["price"],
                "close":  pt["price"],
                "volume": 1.0,
            }
        else:
            c = buckets[bucket]
            c["high"]   = max(c["high"],  pt["price"])
            c["low"]    = min(c["low"],   pt["price"])
            c["close"]  = pt["price"]
            c["volume"] += 1.0
    return sorted(buckets.values(), key=lambda c: c["open_time"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
