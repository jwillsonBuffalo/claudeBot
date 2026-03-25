"""
app.py — Flask web server for the Polymarket Copy Trading Bot.

Features
--------
• Config form: private key (password field), target wallet, all risk controls
• DRY_RUN vs LIVE indicator: full-page colour theme + persistent banner
• Real-time log feed via Server-Sent Events (no WebSocket dep required)
• /api/start  POST — start the bot with supplied config
• /api/stop   POST — gracefully stop the bot
• /api/status GET  — running state + stats
• /api/stream GET  — SSE endpoint (EventSource in browser)

Security notes
--------------
• Private key is never logged, printed, or stored to disk
• config dict is sanitised before any str() output (see _safe_config_log)
• Run behind HTTPS in production (see README)
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time as _time
from datetime import datetime, timezone
from typing import Optional

import requests as _requests

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from config import AdvancedTraderConfig, BotConfig, PredictionTraderConfig
from copy_trader import AdvancedTrader, CopyTradingBot, PredictionTrader, TradeExecutor
from market_analysis import MarketAnalyzer, AdaptiveWeights, PredictionTracker, TIMEFRAME_CONFIG

import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))

_analyzers: dict = {
    tf: MarketAnalyzer(
        state_file=_os.path.join(_BASE_DIR, f"analysis_state_{tf.replace('h','h')}.json")
    )
    for tf in ("5m", "15m", "1h")
}
analyzer = _analyzers["5m"]  # backwards-compat alias

# ── Selected market (mutable dict — GIL is sufficient for single-key updates) ──
_selected_market: dict = {"timeframe": "5m"}


def _bg_analysis_poll() -> None:
    """Keep all three timeframe analyzers warm so adaptive weights build up continuously."""
    import time as _t
    while True:
        for tf, az in _analyzers.items():
            try:
                az.get_analysis(tf)
            except Exception:
                pass
        _t.sleep(25)


threading.Thread(target=_bg_analysis_poll, name="AnalysisBgPoll", daemon=True).start()

# ── Live price cache (5 s TTL) ────────────────────────────────────────────────
_price_cache: dict = {"price": None, "price_to_beat": None, "source": None, "ts": 0.0}
_price_lock  = threading.Lock()
_PRICE_TTL   = 5  # seconds


def _fetch_ptb_authoritative(now: float, timeframe: str = "5m") -> Optional[float]:
    """
    Get price-to-beat from Polymarket's own sources only — guaranteed to match
    what Polymarket displays in its UI.

    Step 1: Query Gamma API for the active market whose slug contains the
            authoritative window-start Unix timestamp.
    Step 2: Fetch polymarket.com/api/chainlink-candles and return the `open`
            price for that exact timestamp.

    Returns None if either call fails or no matching candle is found — callers
    should leave price_to_beat unchanged rather than showing a wrong value.
    """
    tf_cfg       = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["5m"])
    win_sec      = tf_cfg["window_sec"]
    slug_prefix  = tf_cfg["slug_prefix"]
    interval     = tf_cfg["candle_interval"]

    computed_win = (int(now) // win_sec) * win_sec
    window_ts    = computed_win  # default; may be overridden by Gamma API

    # Step 1 — confirm window timestamp from Polymarket's own market slug
    try:
        slug = f"{slug_prefix}-{computed_win}"
        g = _requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": slug, "active": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        g.raise_for_status()
        markets = g.json()
        if markets:
            parts = markets[0].get("slug", "").rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                window_ts = int(parts[1])
    except Exception:
        pass  # fall through with computed_win

    # Step 2 — fetch Chainlink candles and match to authoritative window timestamp
    try:
        r = _requests.get(
            "https://polymarket.com/api/chainlink-candles",
            params={"symbol": "BTC", "interval": interval},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        r.raise_for_status()
        candles = r.json().get("candles", [])
        for c in reversed(candles):
            if int(c["time"]) == window_ts:
                return round(float(c["open"]), 2)
    except Exception:
        pass

    return None
_spread_cache: dict = {"data": None, "ts": 0.0}
_spread_lock = threading.Lock()
_SPREAD_TTL  = 2  # seconds

# ── Wallet balance cache (15 s TTL) ──────────────────────────────────────────
_balance_cache: dict = {}   # keyed by wallet address (lowercase)
_balance_lock  = threading.Lock()
_BALANCE_TTL   = 15  # seconds
_POLYGON_RPC   = "https://polygon.drpc.org"
# Polymarket accepts both: native USDC (Circle) and USDC.e (bridged). Check both.
_USDC_NATIVE   = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"  # native USDC on Polygon
_USDC_E        = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (bridged)
_CLOB_HOST     = "https://clob.polymarket.com"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────────────────────
# Log broadcaster (SSE fan-out)
# ─────────────────────────────────────────────────────────────────────────────


class LogBroadcaster:
    """
    Thread-safe broadcast hub.  The bot calls `broadcast()` from its thread;
    each SSE response generator reads from its own per-client queue.
    """

    def __init__(self) -> None:
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._recent: list = []          # replay buffer for new SSE subscribers
        self._max_recent: int = 150      # keep last N entries

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            # Replay recent log entries so a page refresh shows existing activity
            for entry in self._recent:
                try:
                    q.put_nowait(entry)
                except queue.Full:
                    break
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    def broadcast(self, entry: dict) -> None:
        with self._lock:
            # Append to replay buffer
            self._recent.append(entry)
            if len(self._recent) > self._max_recent:
                self._recent.pop(0)
            dead: set = set()
            for q in self._clients:
                try:
                    q.put_nowait(entry)
                except queue.Full:
                    dead.add(q)
            self._clients -= dead

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


broadcaster = LogBroadcaster()

# ─────────────────────────────────────────────────────────────────────────────
# Global bot state
# ─────────────────────────────────────────────────────────────────────────────

_bot: Optional[CopyTradingBot] = None
_bot_thread: Optional[threading.Thread] = None
_bot_lock = threading.Lock()

_predictor: Optional[PredictionTrader] = None
_predictor_thread: Optional[threading.Thread] = None
_predictor_lock = threading.Lock()

_advanced: Optional[AdvancedTrader] = None
_advanced_thread: Optional[threading.Thread] = None
_advanced_lock = threading.Lock()

_verbose: bool = True    # verbose test button always enabled

# Wallet connection status (explicit connect button).
_wallet_status_lock = threading.Lock()
_wallet_status: dict = {
    "connected": False,
    "wallet": "",
    "funder": "",
    "signature_type": 0,
    "message": "not connected",
    "timestamp": None,
}

# Manual test trades (from verbose endpoint), shown in Trades tab with distinct styling.
_manual_trade_journal: list[dict] = []
_manual_trade_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_bot():
    global _bot, _bot_thread

    with _bot_lock:
        if _bot and _bot.running:
            return jsonify({"status": "error", "message": "Bot is already running"}), 409

        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

        # Build config from request data
        try:
            bot_tf = str(body.get("market_timeframe", _selected_market["timeframe"]))
            if bot_tf not in TIMEFRAME_CONFIG:
                bot_tf = "5m"
            bot_slug = TIMEFRAME_CONFIG[bot_tf]["slug_prefix"]
            config = BotConfig(
                target_wallet=str(body.get("target_wallet", "")).strip(),
                private_key=str(body.get("private_key", "")).strip(),
                my_wallet=str(body.get("my_wallet", "")).strip(),
                signature_type=int(body.get("signature_type", 0)),
                funder_address=str(body.get("funder_address", "")).strip(),
                copy_multiplier=float(body.get("copy_multiplier", 0.10)),
                max_usd_per_trade=float(body.get("max_usd_per_trade", 50.0)),
                min_usd_to_copy=float(body.get("min_usd_to_copy", 5.0)),
                max_total_exposure_usd=float(body.get("max_total_exposure_usd", 500.0)),
                dry_run=bool(body.get("dry_run", True)),
                take_profit_pct=float(body.get("take_profit_pct", 0.0)),
                notification_method=str(body.get("notification_method", "none")),
                telegram_bot_token=str(body.get("telegram_bot_token", "")),
                telegram_chat_id=str(body.get("telegram_chat_id", "")),
                slack_webhook_url=str(body.get("slack_webhook_url", "")),
                builder_api_key=str(body.get("builder_api_key", "")).strip(),
                builder_api_secret=str(body.get("builder_api_secret", "")).strip(),
                builder_api_passphrase=str(body.get("builder_api_passphrase", "")).strip(),
                auto_redeem=bool(body.get("auto_redeem", False)),
                market_timeframe=bot_tf,
                market_slug_patterns=[bot_slug],
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"status": "error", "message": f"Invalid config value: {exc}"}), 400

        err = config.validate()
        if err:
            return jsonify({"status": "error", "message": err}), 400

        _bot = CopyTradingBot(
            config,
            log_callback=broadcaster.broadcast,
        )
        _bot_thread = threading.Thread(
            target=_bot.run,
            name="CopyTradingBot",
            daemon=True,
        )
        _bot_thread.start()
        logger.info("Bot started | %s", config.safe_repr())

    return jsonify({"status": "started", "dry_run": config.dry_run})


@app.route("/api/stop", methods=["POST"])
def stop_bot():
    global _bot
    with _bot_lock:
        if _bot:
            _bot.stop()
            logger.info("Stop requested via API")
            return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})


@app.route("/api/status")
def status():
    with _bot_lock:
        if _bot and _bot.running:
            stats = dict(_bot.stats)
            stats["total_usd_copied"] = round(
                sum(e.get("cost_usd", 0) for e in _bot.get_trade_journal()), 2
            )
            return jsonify({
                "running": True,
                "dry_run": _bot.config.dry_run,
                "target_wallet": _bot.config.target_wallet,
                "my_wallet": _bot.config.my_wallet,
                "stats": stats,
                "sse_clients": broadcaster.client_count,
            })
    return jsonify({"running": False, "sse_clients": broadcaster.client_count})


@app.route("/api/wallet-status")
def wallet_status():
    with _wallet_status_lock:
        return jsonify(dict(_wallet_status))


@app.route("/api/connect-wallet", methods=["POST"])
def connect_wallet():
    """
    Validate wallet credentials and initialize CLOB client without starting a bot.
    """
    body = request.get_json(force=True, silent=True) or {}
    private_key = str(body.get("private_key", "")).strip()
    my_wallet = str(body.get("my_wallet", "")).strip()
    signature_type = int(body.get("signature_type", 0))
    funder_address = str(body.get("funder_address", "")).strip()

    if not private_key:
        with _wallet_status_lock:
            _wallet_status.update({
                "connected": False,
                "wallet": "",
                "funder": funder_address,
                "signature_type": signature_type,
                "message": "private key required",
                "timestamp": _utc_now(),
            })
        broadcaster.broadcast({
            "type": "error",
            "timestamp": _utc_now(),
            "message": "[WALLET] Connect failed — private key required",
            "data": {
                "trade_source": "wallet",
                "signature_type": signature_type,
                "funder": funder_address,
            },
        })
        return jsonify({"ok": False, "error": "private key required"}), 400

    try:
        cfg = BotConfig(
            private_key=private_key,
            my_wallet=my_wallet,
            signature_type=signature_type,
            funder_address=funder_address,
            dry_run=False,
        )
        executor = TradeExecutor(cfg)
        if not executor.client:
            err = executor.init_error or "CLOB client init failed"
            raise RuntimeError(err)

        derived = executor.derive_wallet_address() or my_wallet
        with _wallet_status_lock:
            _wallet_status.update({
                "connected": True,
                "wallet": derived or "",
                "funder": funder_address,
                "signature_type": signature_type,
                "message": "connected",
                "timestamp": _utc_now(),
            })
        broadcaster.broadcast({
            "type": "info",
            "timestamp": _utc_now(),
            "message": (
                "[WALLET] Connected | "
                f"wallet={derived or my_wallet or 'unknown'} | "
                f"sig_type={signature_type} | "
                f"funder={funder_address or '—'}"
            ),
            "data": {
                "trade_source": "wallet",
                "wallet": derived or my_wallet or "",
                "funder": funder_address,
                "signature_type": signature_type,
            },
        })
        return jsonify({
            "ok": True,
            "wallet": derived or "",
            "funder": funder_address,
            "signature_type": signature_type,
        })
    except Exception as exc:
        with _wallet_status_lock:
            _wallet_status.update({
                "connected": False,
                "wallet": "",
                "funder": funder_address,
                "signature_type": signature_type,
                "message": str(exc),
                "timestamp": _utc_now(),
            })
        broadcaster.broadcast({
            "type": "error",
            "timestamp": _utc_now(),
            "message": f"[WALLET] Connect failed — {exc}",
            "data": {
                "trade_source": "wallet",
                "signature_type": signature_type,
                "funder": funder_address,
            },
        })
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/trades")
def get_trades():
    """Return the in-memory trade journal with PnL data from all sources."""
    bot_trades: list[dict] = []
    with _bot_lock:
        if _bot:
            bot_trades = _bot.get_trade_journal()
    with _manual_trade_lock:
        manual_trades = list(_manual_trade_journal)
    pred_trades: list[dict] = []
    with _predictor_lock:
        if _predictor:
            pred_trades = _predictor.get_trade_journal()
    adv_trades: list[dict] = []
    with _advanced_lock:
        if _advanced:
            adv_trades = _advanced.get_trade_journal()

    merged = bot_trades + manual_trades + pred_trades + adv_trades
    merged.sort(key=lambda t: t.get("entry_time", ""), reverse=True)
    return jsonify({"trades": merged})


@app.route("/api/set-market", methods=["POST"])
def set_market():
    """Switch the active market timeframe. Returns new window_sec so the UI can update its clock."""
    body = request.get_json(force=True, silent=True) or {}
    tf   = str(body.get("timeframe", "5m"))
    if tf not in _analyzers:
        return jsonify({"status": "error", "message": f"Unknown timeframe '{tf}'. Use 5m, 15m, or 1h."}), 400
    _selected_market["timeframe"] = tf
    cfg = TIMEFRAME_CONFIG[tf]
    # Invalidate spread cache so next poll fetches the new timeframe's market
    with _spread_lock:
        _spread_cache["data"] = None
        _spread_cache["ts"]   = 0.0
    logger.info("Market timeframe switched to %s", tf)
    broadcaster.broadcast({
        "type":      "info",
        "timestamp": _utc_now(),
        "message":   f"[MARKET] Switched to BTC {tf} — window {cfg['window_sec']}s",
        "data":      {"trade_source": "system"},
    })
    return jsonify({
        "status":      "ok",
        "timeframe":   tf,
        "window_sec":  cfg["window_sec"],
        "slug_prefix": cfg["slug_prefix"],
    })


@app.route("/api/analysis")
def get_analysis():
    """Directional prediction for the selected market timeframe (30s cached)."""
    tf = _selected_market["timeframe"]
    return jsonify(_analyzers[tf].get_analysis(tf))


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Trader endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/start-predictor", methods=["POST"])
def start_predictor():
    global _predictor, _predictor_thread
    with _predictor_lock:
        if _predictor and _predictor.running:
            return jsonify({"status": "error", "message": "Predictor already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        try:
            tf = str(body.get("market_timeframe", _selected_market["timeframe"]))
            if tf not in _analyzers:
                tf = "5m"
            config = PredictionTraderConfig(
                private_key=str(body.get("private_key", "")).strip(),
                my_wallet=str(body.get("my_wallet", "")).strip(),
                signature_type=int(body.get("signature_type", 0)),
                funder_address=str(body.get("funder_address", "")).strip(),
                auto_trade_usd=float(body.get("auto_trade_usd", 10.0)),
                min_confidence_pct=int(body.get("min_confidence_pct", 65)),
                trade_at_window_open_only=bool(body.get("trade_at_window_open_only", True)),
                take_profit_pct=float(body.get("take_profit_pct", 0.0)),
                builder_api_key=str(body.get("builder_api_key", "")).strip(),
                builder_api_secret=str(body.get("builder_api_secret", "")).strip(),
                builder_api_passphrase=str(body.get("builder_api_passphrase", "")).strip(),
                auto_redeem=bool(body.get("auto_redeem", False)),
                dry_run=bool(body.get("dry_run", True)),
                market_timeframe=tf,
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"status": "error", "message": f"Invalid config: {exc}"}), 400

        err = config.validate()
        if err:
            return jsonify({"status": "error", "message": err}), 400

        _predictor = PredictionTrader(
            config=config,
            analyzer=_analyzers[tf],
            log_callback=broadcaster.broadcast,
        )
        _predictor_thread = threading.Thread(
            target=_predictor.run, name="PredictionTrader", daemon=True
        )
        _predictor_thread.start()
        logger.info("Predictor started | %s", config.safe_repr())

    return jsonify({"status": "started", "dry_run": config.dry_run})


@app.route("/api/stop-predictor", methods=["POST"])
def stop_predictor():
    global _predictor
    with _predictor_lock:
        if _predictor and _predictor.running:
            _predictor.stop()
            logger.info("Predictor stop requested via API")
            return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})


@app.route("/api/status-predictor")
def status_predictor():
    with _predictor_lock:
        if _predictor and _predictor.running:
            now      = int(_time.time())
            tf       = getattr(_predictor.config, "market_timeframe", _selected_market["timeframe"])
            win_sec  = TIMEFRAME_CONFIG.get(tf, TIMEFRAME_CONFIG["5m"])["window_sec"]
            window_start = (now // win_sec) * win_sec
            secs_left    = win_sec - (now - window_start)
            return jsonify({
                "running":                   True,
                "dry_run":                   _predictor.config.dry_run,
                "last_traded_window":        _predictor._last_traded_window,
                "secs_remaining_in_window":  secs_left,
                "window_sec":                win_sec,
            })
    return jsonify({"running": False})


@app.route("/api/reset-predictor", methods=["POST"])
def reset_predictor():
    """Reset adaptive weights, prediction history, and trade journal for the selected timeframe."""
    import os
    tf  = _selected_market["timeframe"]
    az  = _analyzers[tf]
    az._tracker    = PredictionTracker()
    az._weights_aw = AdaptiveWeights()
    az._tracker.WINDOW_SEC = TIMEFRAME_CONFIG[tf]["window_sec"]
    try:
        os.remove(az._state_file)
    except FileNotFoundError:
        pass
    # Also clear the running predictor's trade journal and window dedup
    with _predictor_lock:
        if _predictor:
            with _predictor._journal_lock:
                _predictor.trade_journal = []
            _predictor._last_traded_window = None
    logger.info("Prediction weights, history, and trade journal reset for timeframe=%s via API", tf)
    return jsonify({"status": "reset"})


@app.route("/api/start-advanced", methods=["POST"])
def start_advanced():
    global _advanced, _advanced_thread
    with _advanced_lock:
        if _advanced and _advanced.running:
            return jsonify({"status": "error", "message": "Advanced trader already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        try:
            adv_tf = str(body.get("market_timeframe", _selected_market["timeframe"]))
            if adv_tf not in _analyzers:
                adv_tf = "5m"
            config = AdvancedTraderConfig(
                private_key=str(body.get("private_key", "")).strip(),
                my_wallet=str(body.get("my_wallet", "")).strip(),
                signature_type=int(body.get("signature_type", 0)),
                funder_address=str(body.get("funder_address", "")).strip(),
                aggression=str(body.get("aggression", "balanced")),
                swing_entry_max=float(body.get("swing_entry_max", 0.40)),
                exit_target=float(body.get("exit_target", 0.50)),
                trade_amount_usd=float(body.get("trade_amount_usd", 10.0)),
                take_profit_pct=float(body.get("take_profit_pct", 0.0)),
                min_window_secs_remaining=int(body.get("min_window_secs_remaining", 60)),
                builder_api_key=str(body.get("builder_api_key", "")).strip(),
                builder_api_secret=str(body.get("builder_api_secret", "")).strip(),
                builder_api_passphrase=str(body.get("builder_api_passphrase", "")).strip(),
                auto_redeem=bool(body.get("auto_redeem", False)),
                dry_run=bool(body.get("dry_run", True)),
                market_timeframe=adv_tf,
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"status": "error", "message": f"Invalid config: {exc}"}), 400

        err = config.validate()
        if err:
            return jsonify({"status": "error", "message": err}), 400

        _advanced = AdvancedTrader(config=config, log_callback=broadcaster.broadcast)
        _advanced_thread = threading.Thread(
            target=_advanced.run, name="AdvancedTrader", daemon=True
        )
        _advanced_thread.start()
        logger.info("AdvancedTrader started | %s", config.safe_repr())

    return jsonify({"status": "started", "dry_run": config.dry_run})


@app.route("/api/stop-advanced", methods=["POST"])
def stop_advanced():
    global _advanced
    with _advanced_lock:
        if _advanced and _advanced.running:
            _advanced.stop()
            logger.info("AdvancedTrader stop requested via API")
            return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})


@app.route("/api/status-advanced")
def status_advanced():
    with _advanced_lock:
        if _advanced and _advanced.running:
            return jsonify({
                "running": True,
                "dry_run": _advanced.config.dry_run,
                "stats":                    dict(_advanced.stats),
                "aggression":               _advanced.config.aggression,
                "swing_entry_max":          _advanced.config.swing_entry_max,
                "exit_target":              _advanced.config.exit_target,
                "trade_amount_usd":         _advanced.config.trade_amount_usd,
            })
    return jsonify({"running": False})


@app.route("/api/price")
def get_price():
    """Live BTC price + price-to-beat with 5 s cache. Fast endpoint for UI ticker."""
    now = _time.time()
    with _price_lock:
        if now - _price_cache["ts"] < _PRICE_TTL and _price_cache["price"] is not None:
            return jsonify(_price_cache)

    try:
        r = _requests.get(
            "https://polymarket.com/api/chainlink-candles",
            params={"symbol": "BTC", "interval": "5m"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        r.raise_for_status()
        candles = r.json().get("candles", [])
        if not candles:
            raise ValueError("empty response")

        latest = candles[-1]
        price_to_beat = _fetch_ptb_authoritative(now, _selected_market["timeframe"])

        result = {
            "price":         round(float(latest["close"]), 2),
            "price_to_beat": price_to_beat,
            "source":        "polymarket_chainlink",
            "ts":            now,
        }
        with _price_lock:
            _price_cache.update(result)
        return jsonify(result)

    except Exception:
        with _price_lock:
            if _price_cache["price"] is not None:
                return jsonify(_price_cache)
        return jsonify({"price": None, "price_to_beat": None}), 503


def _rpc_call(rpc_url: str, method: str, params: list):
    """Single JSON-RPC call. Returns result value or raises on error."""
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = _requests.post(rpc_url, json=body, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data.get("result", "0x0") or "0x0"


def _erc20_balance(rpc_url: str, contract: str, wallet: str) -> int:
    """Call balanceOf(address) on an ERC-20 contract. Returns raw integer."""
    addr_hex = wallet.lower().replace("0x", "").zfill(64)
    hex_val = _rpc_call(rpc_url, "eth_call",
                        [{"to": contract, "data": "0x70a08231" + addr_hex}, "latest"])
    return int(hex_val, 16)


def _fetch_wallet_balances(wallet: str, rpc_url: str = _POLYGON_RPC) -> dict:
    """
    Fetch USDC (native + USDC.e) and POL balances for `wallet`.

    Checks both Polygon USDC contracts because Polymarket accepts both.
    Uses separate RPC calls to avoid silent batch failures.
    Returns {"wallet", "usdc", "usdc_native", "usdc_e", "pol", "error"}.
    """
    key = wallet.lower()
    now = _time.time()
    with _balance_lock:
        cached = _balance_cache.get(key)
        if cached and now - cached["_ts"] < _BALANCE_TTL:
            return {k: v for k, v in cached.items() if k != "_ts"}

    result = {"wallet": wallet, "usdc": 0.0, "usdc_native": 0.0,
              "usdc_e": 0.0, "pol": 0.0, "error": None}
    errors = []

    # Native USDC (Circle, Polygon)
    try:
        raw = _erc20_balance(rpc_url, _USDC_NATIVE, wallet)
        result["usdc_native"] = round(raw / 1e6, 2)
    except Exception as exc:
        errors.append(f"native USDC: {exc}")
        logger.warning("Balance fetch native USDC failed: %s", exc)

    # USDC.e (bridged)
    try:
        raw = _erc20_balance(rpc_url, _USDC_E, wallet)
        result["usdc_e"] = round(raw / 1e6, 2)
    except Exception as exc:
        errors.append(f"USDC.e: {exc}")
        logger.warning("Balance fetch USDC.e failed: %s", exc)

    # POL (gas token)
    try:
        hex_val = _rpc_call(rpc_url, "eth_getBalance", [wallet, "latest"])
        result["pol"] = round(int(hex_val, 16) / 1e18, 4)
    except Exception as exc:
        errors.append(f"POL: {exc}")
        logger.warning("Balance fetch POL failed: %s", exc)

    result["usdc"] = round(result["usdc_native"] + result["usdc_e"], 2)
    if errors:
        result["error"] = "; ".join(errors)

    with _balance_lock:
        _balance_cache[key] = {**result, "_ts": now}
    return result


@app.route("/api/balance")
def get_balance():
    """
    GET /api/balance?wallet=0x...
    Returns USDC (native + USDC.e) and POL balances.
    Also checks funder_address separately when it differs from wallet
    (proxy wallet users hold USDC in the funder/proxy contract, not the EOA).
    """
    wallet      = request.args.get("wallet", "").strip()
    funder      = request.args.get("funder", "").strip()
    rpc_url     = _POLYGON_RPC

    with _bot_lock:
        if _bot:
            rpc_url = _bot.config.polygon_rpc_url or _POLYGON_RPC
            if not wallet:
                wallet = _bot.config.my_wallet or ""
            # Bot config funder overrides query param
            funder = (_bot.config.funder_address or "").strip() or funder

    if not wallet:
        return jsonify({"error": "no wallet address provided or bot running"}), 400

    # Allow cache-bust via ?refresh=1
    if request.args.get("refresh"):
        with _balance_lock:
            _balance_cache.pop(wallet.lower(), None)
            if funder and funder.lower() != wallet.lower():
                _balance_cache.pop(funder.lower(), None)

    result = _fetch_wallet_balances(wallet, rpc_url)

    # For proxy/email wallets the USDC lives in the funder address, not the EOA
    if funder and funder.lower() != wallet.lower():
        funder_result = _fetch_wallet_balances(funder, rpc_url)
        result["funder_address"] = funder
        result["funder_usdc"]    = funder_result["usdc"]
        result["funder_pol"]     = funder_result["pol"]
        result["funder_usdc_native"] = funder_result["usdc_native"]
        result["funder_usdc_e"]      = funder_result["usdc_e"]
        # Total is whichever address holds more (usually the funder for proxy wallets)
        result["usdc_combined"] = round(result["usdc"] + funder_result["usdc"], 2)
    else:
        result["usdc_combined"] = result["usdc"]

    logger.info(
        "Balance | wallet=%s usdc=%.2f pol=%.4f funder=%s funder_usdc=%.2f",
        wallet[:10], result["usdc"], result["pol"],
        funder[:10] if funder else "—",
        result.get("funder_usdc", 0),
    )
    return jsonify(result)


@app.route("/api/verbose")
def get_verbose():
    """Return whether verbose/test mode is enabled."""
    return jsonify({"verbose": _verbose})


@app.route("/api/positions")
def get_positions():
    """
    GET /api/positions
    Returns current Up/Down share counts for the active BTC window
    by querying the Gamma API positions endpoint for the configured wallet.
    Tries funder_address (POLY_PROXY) and data-api as fallbacks.
    """
    # Explicit wallet from query string always takes priority
    qs_wallet = request.args.get("wallet", "").strip()
    wallets_to_try: list = []
    if qs_wallet:
        wallets_to_try = [qs_wallet]
    else:
        with _bot_lock:
            if _bot:
                if _bot.config.my_wallet:
                    wallets_to_try.append(_bot.config.my_wallet)
                if _bot.config.funder_address:
                    wallets_to_try.append(_bot.config.funder_address)
        if not wallets_to_try:
            with _predictor_lock:
                if _predictor:
                    if _predictor.config.my_wallet:
                        wallets_to_try.append(_predictor.config.my_wallet)
                    if _predictor.config.funder_address:
                        wallets_to_try.append(_predictor.config.funder_address)
    if not wallets_to_try:
        return jsonify({"error": "no wallet configured", "up_shares": 0, "down_shares": 0, "positions": [], "queried_wallet": ""})

    def _fetch(wallet: str) -> list:
        for base_url in ("https://gamma-api.polymarket.com/positions",
                         "https://data-api.polymarket.com/positions"):
            try:
                resp = _requests.get(
                    base_url,
                    params={"user": wallet, "sizeThreshold": "0.0001"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("data", data.get("positions", data.get("results", [])))
                if isinstance(data, list) and data:
                    return data
            except Exception:
                continue
        return []

    try:
        all_positions: list = []
        used_wallet: str = ""
        for w in wallets_to_try:
            positions = _fetch(w)
            if positions:
                all_positions = positions
                used_wallet = w
                break

        result = {"up_shares": 0.0, "down_shares": 0.0, "positions": [], "queried_wallet": used_wallet}
        for pos in all_positions:
            title = pos.get("title", "").lower()
            outcome = (pos.get("outcome") or pos.get("side") or "").lower()
            size = float(pos.get("size") or 0)
            if size <= 0:
                continue
            # Skip closed or redeemed markets — shares there can't be traded, only redeemed
            if pos.get("closed") or pos.get("redeemed"):
                continue
            # Filter to BTC Up/Down markets (all timeframes)
            if not (("bitcoin" in title or "btc" in title) and ("up" in title or "down" in title)):
                continue
            cur_price = float(pos.get("curPrice") or pos.get("currentPrice") or pos.get("avgPrice") or 0)
            entry = {
                "title": pos.get("title", ""),
                "outcome": pos.get("outcome", ""),
                "size": round(size, 4),
                "curPrice": round(cur_price, 4),
                "value": round(size * cur_price, 4),
            }
            result["positions"].append(entry)
            if "up" in outcome:
                result["up_shares"] += size
            elif "down" in outcome:
                result["down_shares"] += size

        result["up_shares"] = round(result["up_shares"], 4)
        result["down_shares"] = round(result["down_shares"], 4)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "up_shares": 0, "down_shares": 0, "positions": [], "queried_wallet": ""})


@app.route("/api/market-spread")
def get_market_spread():
    """
    Return current BTC 5m market bid/ask + spread for Up/Down outcomes.
    Cached briefly (2s) for UI polling.
    """
    now = _time.time()
    with _spread_lock:
        if now - _spread_cache["ts"] < _SPREAD_TTL and _spread_cache["data"] is not None:
            return jsonify(_spread_cache["data"])

    try:
        market = _get_current_btc_5m_market()
        slug = market["slug"]
        out = market["outcomes"]
        up_token = out.get("up")
        down_token = out.get("down")
        if not up_token or not down_token:
            return jsonify({"error": f"Could not resolve Up/Down token IDs for {slug}"}), 404

        up_book = _get_best_book_prices(up_token)
        down_book = _get_best_book_prices(down_token)

        result = {
            "slug": slug,
            "timeframe": market.get("timeframe", _selected_market.get("timeframe", "5m")),
            "window_start": market["window_start"],
            "up": {
                "token_id": up_token,
                "best_bid": up_book["best_bid"],
                "best_ask": up_book["best_ask"],
                "spread": up_book["spread"],
            },
            "down": {
                "token_id": down_token,
                "best_bid": down_book["best_bid"],
                "best_ask": down_book["best_ask"],
                "spread": down_book["spread"],
            },
            "timestamp": _utc_now(),
        }
        with _spread_lock:
            _spread_cache["data"] = result
            _spread_cache["ts"] = now
        return jsonify(result)
    except Exception as exc:
        logger.debug("market-spread: %s", exc)
        return jsonify({"error": str(exc)}), 200  # 200 so UI handles gracefully; not a server fault


@app.route("/api/test-trade", methods=["POST"])
def test_trade():
    """
    Manually place a single test order against the current BTC 5m window.
    Only meaningful when the server is started with -v / --verbose.
    Requires a running predictor or bot (so CLOB credentials are loaded).
    Body: { outcome: "Up"|"Down", side: "BUY"|"SELL", amount_usd: float, shares: float (optional, overrides amount_usd for SELL) }
    """
    body = request.get_json(force=True) or {}
    outcome    = str(body.get("outcome", "Up")).strip()
    side       = str(body.get("side", "BUY")).strip().upper()
    amount_usd = float(body.get("amount_usd", 5.0))
    shares_override = body.get("shares")  # optional: caller already knows share count

    # Grab executor from whichever bot is running
    executor = None
    with _predictor_lock:
        if _predictor:
            executor = _predictor.executor
    if not executor:
        with _bot_lock:
            if _bot:
                executor = _bot.executor
    if not executor:
        logger.warning("test-trade: no executor found (bot_running=%s predictor_running=%s)",
                       _bot is not None, _predictor is not None)
        return jsonify({"error": "No bot or predictor is running — start one first"}), 400
    if not executor.client:
        dry     = getattr(executor.config, "dry_run", True)
        has_key = bool(getattr(executor.config, "private_key", ""))
        if dry and not has_key:
            reason = "dry-run mode with no private key — enter your key and restart in LIVE mode"
        elif dry:
            reason = "dry-run mode is on — switch to LIVE mode so the CLOB client initialises"
        elif not has_key:
            reason = "no private key configured"
        else:
            init_err = getattr(executor, "init_error", "")
            reason = init_err if init_err else "CLOB client failed to initialise — unknown error"
        logger.warning("test-trade: executor found but client=None  dry=%s has_key=%s", dry, has_key)
        return jsonify({"error": f"CLOB init failed: {reason}"}), 400

    # Current 5-minute window token lookup
    try:
        market = _get_current_btc_5m_market()
        slug = market["slug"]
        key = outcome.strip().lower()
        token_id = market["outcomes"].get(key)
        if not token_id:
            return jsonify({"error": f"Token '{outcome}' not found in {slug}"}), 404
    except Exception as exc:
        return jsonify({"error": f"Token lookup failed: {exc}"}), 500

    # Mid-price — use ClobClient.get_midpoint() directly
    try:
        mid_resp  = executor.client.get_midpoint(token_id)
        mid_price = float(mid_resp.get("mid", 0))
        if not mid_price or mid_price <= 0:
            return jsonify({"error": f"Could not fetch mid-price from CLOB (got {mid_resp})"}), 500
    except Exception as exc:
        return jsonify({"error": f"Midpoint fetch failed: {exc}"}), 500

    if shares_override is not None:
        shares = round(float(shares_override), 2)
        amount_usd = round(shares * mid_price, 2)
    else:
        shares = round(amount_usd / mid_price, 2)
    if shares <= 0:
        return jsonify({"error": "Calculated shares <= 0"}), 400

    # Place order
    try:
        response = executor.place_order(token_id, side, shares, mid_price)
        trade_entry = _build_manual_trade_entry(
            slug=slug,
            token_id=token_id,
            outcome=outcome,
            side=side,
            shares=shares,
            mid_price=mid_price,
            amount_usd=amount_usd,
            dry_run=bool(getattr(executor.config, "dry_run", True)),
            response=response,
        )
        with _manual_trade_lock:
            _manual_trade_journal.append(trade_entry)
            if len(_manual_trade_journal) > 500:
                del _manual_trade_journal[:-500]

        broadcaster.broadcast({
            "type": "copy",
            "timestamp": _utc_now(),
            "message": (
                f"[VERBOSE TEST] {side} {outcome} | {shares} shares "
                f"@ ~${mid_price:.4f} = ~${amount_usd:.2f} | {response}"
            ),
            "data": {
                "trade_source": "self_test",
                "manual_trade": True,
                "token_id": token_id,
                "slug": slug,
            },
        })
        return jsonify({
            "ok": True, "slug": slug, "token_id": token_id,
            "shares": shares, "mid_price": mid_price, "response": response,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/redeem", methods=["POST"])
def redeem_positions():
    """
    Trigger on-chain redemption of all winning positions via poly-web3 relayer.
    Requires Builder API credentials from polymarket.com/settings?tab=builder.
    Creds can be passed in the request body or are read from the running bot config.
    """
    body = request.get_json(force=True, silent=True) or {}

    def _cred(key: str) -> str:
        val = str(body.get(key, "")).strip()
        if val:
            return val
        for lock, ref in [(_bot_lock, "_bot"), (_predictor_lock, "_predictor"), (_advanced_lock, "_advanced")]:
            with lock:
                obj = globals().get(ref)
                if obj:
                    v = getattr(getattr(obj, "config", None), key, "")
                    if v:
                        return str(v)
        return ""

    private_key            = _cred("private_key")
    funder_address         = _cred("funder_address")
    builder_api_key        = _cred("builder_api_key")
    builder_api_secret     = _cred("builder_api_secret")
    builder_api_passphrase = _cred("builder_api_passphrase")

    # signature_type: prefer request body, then running bot
    sig_type = body.get("signature_type")
    if sig_type is None:
        with _bot_lock:
            if _bot:
                sig_type = _bot.config.signature_type
    sig_type = int(sig_type or 0)

    if not builder_api_key or not builder_api_secret or not builder_api_passphrase:
        return jsonify({"status": "error", "message": "Builder API credentials required (key, secret, passphrase)"}), 400
    if not private_key:
        return jsonify({"status": "error", "message": "Private key required for redemption"}), 400

    try:
        from py_builder_relayer_client.client import RelayClient          # type: ignore
        from py_builder_signing_sdk.config import BuilderConfig           # type: ignore
        from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds  # type: ignore
        from py_clob_client.client import ClobClient                     # type: ignore
        from poly_web3 import RELAYER_URL, PolyWeb3Service               # type: ignore
    except ImportError as exc:
        return jsonify({"status": "error", "message": f"poly-web3 not installed: {exc}. Run: pip install poly-web3"}), 500

    try:
        clob_client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=sig_type,
            funder=funder_address or None,
        )
        clob_client.set_api_creds(clob_client.create_or_derive_api_creds())

        relayer_client = RelayClient(
            RELAYER_URL,
            137,
            private_key,
            BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=builder_api_key,
                    secret=builder_api_secret,
                    passphrase=builder_api_passphrase,
                )
            ),
        )

        service = PolyWeb3Service(
            clob_client=clob_client,
            relayer_client=relayer_client,
        )

        results  = service.redeem_all(batch_size=10)
        redeemed = [r for r in (results or []) if r is not None]
        skipped  = len(results or []) - len(redeemed)
        msg = (
            f"[REDEEM] redeem_all() done — {len(redeemed)} tx(s) submitted"
            + (f", {skipped} position(s) skipped/failed" if skipped else "")
        )
        broadcaster.broadcast({"type": "copy", "timestamp": _utc_now(), "message": msg})
        logger.info(msg)
        return jsonify({"status": "ok", "redeemed": len(redeemed), "skipped": skipped, "txs": redeemed})

    except Exception as exc:
        err = f"[REDEEM] Error: {exc}"
        broadcaster.broadcast({"type": "error", "timestamp": _utc_now(), "message": err})
        logger.error(err)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/stream")
def stream():
    """
    Server-Sent Events endpoint.
    Browser opens EventSource('/api/stream') and receives JSON log entries.
    """
    q = broadcaster.subscribe()

    def generate():
        # Initial connection confirmation
        yield _sse({"type": "connected", "timestamp": _utc_now()})
        try:
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield _sse(msg)
                except queue.Empty:
                    # Keep-alive ping so the connection doesn't time out
                    yield _sse({"type": "ping", "timestamp": _utc_now()})
        except GeneratorExit:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slug_for_window(tf: str, window_start: int) -> list:
    """Return list of slugs to try for the given timeframe + window start.
    Returns multiple candidates for 1h (date-based slug with EDT/EST fallback).
    """
    from datetime import datetime, timezone, timedelta
    if tf == "1h":
        def _fmt(dt, include_year: bool = True):
            h12 = dt.hour % 12 or 12
            ampm = "am" if dt.hour < 12 else "pm"
            if include_year:
                return f"bitcoin-up-or-down-{dt.strftime('%B').lower()}-{dt.day}-{dt.year}-{h12}{ampm}-et"
            return f"bitcoin-up-or-down-{dt.strftime('%B').lower()}-{dt.day}-{h12}{ampm}-et"
        try:
            from zoneinfo import ZoneInfo
            dt_et = datetime.fromtimestamp(window_start, tz=ZoneInfo("America/New_York"))
            slugs = [_fmt(dt_et, True), _fmt(dt_et, False)]
        except Exception:
            slugs = []
        # Always add both EDT (-4) and EST (-5) candidates as fallbacks
        for offset in (-4, -5):
            dt = datetime.fromtimestamp(window_start, tz=timezone.utc) + timedelta(hours=offset)
            for inc_year in (True, False):
                s = _fmt(dt, inc_year)
                if s not in slugs:
                    slugs.append(s)
        return slugs
    elif tf == "15m":
        return [f"btc-updown-15m-{window_start}"]
    else:
        return [f"btc-updown-5m-{window_start}"]


def _get_current_btc_5m_market(tf: str = "") -> dict:
    """Resolve the current market for the selected timeframe and return token IDs for Up/Down outcomes."""
    if not tf:
        tf = _selected_market.get("timeframe", "5m")
    cfg = TIMEFRAME_CONFIG.get(tf, TIMEFRAME_CONFIG["5m"])
    win_sec = cfg["window_sec"]
    window_start = (int(_time.time()) // win_sec) * win_sec
    # Also try previous window as fallback — new markets can take ~30 s to appear in gamma
    slugs = _slug_for_window(tf, window_start) + _slug_for_window(tf, window_start - win_sec)

    data = None
    chosen_slug = slugs[0]
    for slug in slugs:
        try:
            r = _requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                timeout=8,
            )
            r.raise_for_status()
            candidate = r.json()
            if candidate:
                data = candidate
                chosen_slug = slug
                break
        except Exception:
            continue

    if not data:
        raise RuntimeError(f"Market not listed yet (tried {len(slugs)} slugs)")
    market = data[0] if isinstance(data, list) else data

    raw_outcomes = market.get("outcomes", "[]")
    raw_ids = market.get("clobTokenIds", "[]")
    if isinstance(raw_outcomes, str):
        raw_outcomes = json.loads(raw_outcomes)
    if isinstance(raw_ids, str):
        raw_ids = json.loads(raw_ids)

    outcome_map: dict[str, str] = {}
    for i, outcome in enumerate(raw_outcomes):
        if i >= len(raw_ids):
            continue
        o = str(outcome).strip().lower()
        if o == "up":
            outcome_map["up"] = str(raw_ids[i])
        elif o == "down":
            outcome_map["down"] = str(raw_ids[i])

    return {"slug": chosen_slug, "window_start": window_start, "outcomes": outcome_map, "timeframe": tf}


def _get_best_book_prices(token_id: str) -> dict:
    """Fetch best bid/ask from CLOB order book for one token."""
    resp = _requests.get(f"{_CLOB_HOST}/book", params={"token_id": token_id}, timeout=8)
    resp.raise_for_status()
    data = resp.json() or {}

    bids = data.get("bids") or []
    asks = data.get("asks") or []

    best_bid = max((_safe_float(b.get("price"), 0.0) for b in bids), default=0.0)
    ask_candidates = [_safe_float(a.get("price"), 0.0) for a in asks]
    ask_candidates = [p for p in ask_candidates if p > 0]
    best_ask = min(ask_candidates) if ask_candidates else 0.0

    spread = round(best_ask - best_bid, 4) if best_bid > 0 and best_ask > 0 else 0.0
    return {
        "best_bid": round(best_bid, 4),
        "best_ask": round(best_ask, 4),
        "spread": spread,
    }


def _build_manual_trade_entry(
    *,
    slug: str,
    token_id: str,
    outcome: str,
    side: str,
    shares: float,
    mid_price: float,
    amount_usd: float,
    dry_run: bool,
    response: dict,
) -> dict:
    """Create a trade-journal entry for /api/test-order manual trades."""
    trade_id = str(response.get("orderID") or response.get("id") or response.get("hash") or "")
    if not trade_id:
        trade_id = f"manual-{int(_time.time() * 1000)}"

    # Keep fields aligned with CopyTradingBot journal schema used by the UI.
    return {
        "id": trade_id,
        "entry_time": _utc_now(),
        "market": "Manual Verbose Test",
        "slug": slug,
        "outcome": outcome,
        "side": side,
        "their_size": 0.0,
        "their_cost": 0.0,
        "their_pnl": 0.0,
        "our_size": round(float(shares), 4),
        "entry_price": round(float(mid_price), 4),
        "cost_usd": round(float(amount_usd), 4),
        "token_id": token_id,
        "status": "SIMULATED" if dry_run else "OPEN",
        "cur_price": round(float(mid_price), 4),
        "pnl_usd": 0.0,
        "dry_run": dry_run,
        "source": "self_test",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--port", type=int, default=5002)
    _args = _parser.parse_args()
    print("=" * 60)
    print("  Polymarket Copy Bot — Web UI")
    print(f"  Open http://localhost:{_args.port} in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host="0.0.0.0", port=_args.port, debug=False, threaded=True)
