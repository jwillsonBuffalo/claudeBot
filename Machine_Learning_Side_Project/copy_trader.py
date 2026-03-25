"""
copy_trader.py — Core engine for the Polymarket BTC 5-Minute Copy Trading Bot.

Classes
-------
Notifier        — sends alerts via Telegram, Slack, or stdout only
TradeMonitor    — polls data-api.polymarket.com for new target trades
RiskManager     — validates risk limits and calculates scaled trade sizes
TradeExecutor   — places orders via py-clob-client (CLOB API)
CopyTradingBot  — orchestrates the polling loop
PredictionTrader — autonomous prediction-based order placer with take-profit
AdvancedTrader  — both-sides arbitrage and spread trading strategies

To extend to multiple target wallets:
  • Change `config.target_wallet` to a list[str]
  • In `CopyTradingBot.run()`, instantiate one `TradeMonitor` per wallet
    and iterate over monitors in the main loop.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import AdvancedTraderConfig, BotConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Create a requests.Session with automatic retry + backoff."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "polymarket-copy-bot/1.0",
        "Accept": "application/json",
    })
    return session


def _make_rpc_session() -> requests.Session:
    """
    Session for Polygon JSON-RPC calls — NO automatic retries.

    ChainMonitor has its own URL-rotation logic; letting urllib3 retry
    before that fires causes 15s×4 = 60 s stalls that destroy detection lag.
    Fail fast here so ChainMonitor can immediately switch to the next RPC.
    """
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "polymarket-copy-bot/1.0",
        "Content-Type": "application/json",
    })
    return session


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_trade_time(trade: dict) -> Optional[datetime]:
    """
    Extract the trade execution timestamp from various field names the
    Polymarket data API may return.  Always returns UTC-aware datetime or None.
    """
    for field in ("match_time", "created_at", "last_update"):
        raw = trade.get(field)
        if raw:
            try:
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                pass
    # Fallback: numeric unix timestamp (seconds)
    ts = trade.get("timestamp")
    if ts:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
    return None


def _round_to_tick(price: float, tick: float, round_up: bool = False) -> float:
    """Round a price to the nearest tick boundary."""
    if tick <= 0:
        return round(price, 4)
    if round_up:
        return math.ceil(price / tick) * tick
    return math.floor(price / tick) * tick


# ─────────────────────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────────────────────

class Notifier:
    """
    Sends human-readable notifications via Telegram or Slack using raw HTTP
    (no external bot SDK — only `requests`).
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._session = _make_session()

    def send(self, message: str) -> None:
        method = self.config.notification_method.lower()
        if method == "telegram":
            self._telegram(message)
        elif method == "slack":
            self._slack(message)
        # Always falls through so callers don't need to check

    def _telegram(self, message: str) -> None:
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            logger.warning("Telegram creds not set — skipping notification")
            return
        url = (
            f"https://api.telegram.org/bot{self.config.telegram_bot_token}"
            f"/sendMessage"
        )
        try:
            resp = self._session.post(
                url,
                params={
                    "chat_id": self.config.telegram_chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if not resp.ok:
                logger.warning(
                    "Telegram notification failed (%s): %s", resp.status_code, resp.text[:120]
                )
        except requests.RequestException as exc:
            logger.warning("Telegram request error: %s", exc)

    def _slack(self, message: str) -> None:
        if not self.config.slack_webhook_url:
            logger.warning("Slack webhook not set — skipping notification")
            return
        try:
            resp = self._session.post(
                self.config.slack_webhook_url,
                json={"text": message},
                timeout=10,
            )
            if not resp.ok:
                logger.warning(
                    "Slack notification failed (%s): %s", resp.status_code, resp.text[:120]
                )
        except requests.RequestException as exc:
            logger.warning("Slack request error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# TradeMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TradeMonitor:
    """
    Polls data-api.polymarket.com for the target wallet's recent trades
    and surfaces only NEW trades in BTC 5-minute markets.

    First poll seeds the seen-IDs set (no historical replay).
    Subsequent polls return only unseen trade objects.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._session = _make_session()
        self._seen_ids: Set[str] = set()
        self._initialized = False

    # ── Public ───────────────────────────────────────────────────────────────

    def is_tracked_market(self, trade: dict) -> bool:
        slug = (trade.get("slug") or trade.get("eventSlug") or "").lower()
        return any(p.lower() in slug for p in self.config.market_slug_patterns)

    def get_new_trades(self) -> list:
        """Return list of new BTC-5min trades since last call. Empty on first call."""
        raw = self._fetch_raw_trades()
        if not raw:
            return []

        # First call → seed seen set, return nothing
        if not self._initialized:
            for t in raw:
                self._seen_ids.add(t.get("transactionHash", ""))
            self._initialized = True
            logger.info("TradeMonitor seeded %d existing trade IDs — no replay", len(self._seen_ids))
            return []

        new_trades = []
        for trade in raw:
            trade_id = trade.get("transactionHash", "")
            if not trade_id or trade_id in self._seen_ids:
                continue
            self._seen_ids.add(trade_id)
            if getattr(self.config, "copy_all_markets", False) or self.is_tracked_market(trade):
                new_trades.append(trade)
            else:
                slug = trade.get("slug") or trade.get("eventSlug") or trade.get("title", "")
                logger.debug("Untracked market skipped: %s", str(slug)[:70])

        # Bound seen-IDs to avoid unbounded memory growth
        if len(self._seen_ids) > self.config.max_seen_ids:
            overflow = list(self._seen_ids)
            self._seen_ids = set(overflow[-(self.config.max_seen_ids // 2):])

        return new_trades

    # ── Private ──────────────────────────────────────────────────────────────

    def _fetch_raw_trades(self) -> list:
        url = f"{self.config.data_api_base}/trades"
        params = {"user": self.config.target_wallet, "limit": 100}
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Handle both direct list and wrapped {"data": [...]} shapes
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                return data["data"]
            logger.warning("Unexpected trades response shape: %s", str(data)[:100])
            return []
        except requests.RequestException as exc:
            logger.error("TradeMonitor fetch error: %s", exc)
            return []
        except ValueError as exc:
            logger.error("TradeMonitor JSON parse error: %s", exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# ChainMonitor — blockchain event-based trade detection (~2-4 s lag)
# ─────────────────────────────────────────────────────────────────────────────

# keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
_ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"


class ChainMonitor:
    """
    Detects target wallet trades by polling Polygon eth_getLogs every ~2 seconds.

    Replaces the HTTP data-api poller which has 4+ minute indexing lag.
    Detection lag drops to 2-6 seconds (one Polygon block ≈ 2.1 s).

    Maintains an internal metadata cache (token_id → slug/title/outcome)
    refreshed from data-api every 60 s — used for display and slug filtering.
    Unknown tokens are passed through so no trade is ever silently dropped.
    """

    def __init__(self, config: BotConfig, log_fn: Optional[Callable] = None) -> None:
        self.config = config
        self._log_fn = log_fn  # optional SSE broadcaster — None means terminal only
        self._rpc_session = _make_rpc_session()   # no retries — own URL rotation handles fallback
        self._meta_session = _make_session()
        self._last_block: Optional[int] = None
        self._metadata_cache: Dict[str, dict] = {}
        self._last_metadata_refresh: float = 0.0
        self._last_heartbeat: float = 0.0  # for periodic block-progress logs
        # Ordered list of RPCs: primary first, then fallbacks
        self._rpc_urls: list = [config.polygon_rpc_url] + list(config.polygon_rpc_fallbacks)
        self._rpc_index: int = 0  # index into _rpc_urls for current active RPC
        # monotonic time of last failure per RPC index — failed RPCs are skipped for 5 min
        self._rpc_fail_ts: Dict[int, float] = {}
        # Pad target wallet to 32-byte topic for eth_getLogs filter
        self._maker_topic = "0x" + "0" * 24 + config.target_wallet[2:].lower()

    # ── Internal log helper ───────────────────────────────────────────────────

    def _emit(self, level: str, msg: str) -> None:
        """Emit a log entry to the SSE broadcaster (if wired up) and Python logger."""
        if self._log_fn is not None:
            self._log_fn(level, msg)
        else:
            logger.info("[ChainMonitor] %s", msg)

    # ── Public ───────────────────────────────────────────────────────────────

    def get_new_trades(self) -> list:
        """
        Return list of new trades since last call. Empty on first call (seeding).
        Refreshes the metadata cache once per minute in the background.
        """
        self._maybe_refresh_metadata()

        latest_block = self._get_latest_block()
        if latest_block is None:
            raise RuntimeError("ChainMonitor: eth_blockNumber RPC call failed")

        if self._last_block is None:
            self._last_block = latest_block
            self._emit("info",
                f"ChainMonitor: seeded at block {latest_block} | "
                f"metadata cache={len(self._metadata_cache)} entries | "
                f"RPC={self._rpc_urls[self._rpc_index % len(self._rpc_urls)]}")
            return []

        if latest_block <= self._last_block:
            # Periodic heartbeat every 30 s so user knows the chain poller is alive
            if time.monotonic() - self._last_heartbeat > 30:
                self._last_heartbeat = time.monotonic()
                rpc_url = self._rpc_urls[self._rpc_index % len(self._rpc_urls)]
                self._emit("info",
                    f"ChainMonitor: at block {latest_block} | "
                    f"cache={len(self._metadata_cache)} slugs | RPC={rpc_url}")
            return []  # no new blocks yet

        # Cap the block range to avoid "invalid block range params" on free-tier RPCs
        # publicnode's actual limit is <200 blocks; 1rpc.io allows ~1000. Use 150 to be safe.
        _MAX_BLOCK_RANGE = 150
        from_block = max(self._last_block + 1, latest_block - _MAX_BLOCK_RANGE)
        if from_block > self._last_block + 1:
            skipped = from_block - (self._last_block + 1)
            self._emit("warn",
                f"ChainMonitor: block gap {latest_block - self._last_block} > {_MAX_BLOCK_RANGE} "
                f"— skipping oldest {skipped} blocks to stay within RPC limit")

        logs = self._get_logs(from_block, latest_block)
        self._last_block = latest_block

        # Heartbeat on every new block batch (throttled to 30 s)
        if time.monotonic() - self._last_heartbeat > 30:
            self._last_heartbeat = time.monotonic()
            rpc_url = self._rpc_urls[self._rpc_index % len(self._rpc_urls)]
            self._emit("info",
                f"ChainMonitor: block {from_block}→{latest_block} | "
                f"{len(logs)} raw log(s) | cache={len(self._metadata_cache)} slugs | RPC={rpc_url}")

        trades = []
        unknown_count = 0
        for log in logs:
            trade = self._decode_log(log, latest_block)
            if trade is None:
                continue
            self._enrich(trade)
            copy_all = getattr(self.config, "copy_all_markets", False)
            # Skip if slug is unknown — metadata cache refreshes every 60 s.
            # In copy_all mode we pass through anyway (user wants everything).
            if not trade["slug"]:
                if copy_all:
                    trade["slug"] = ""   # will render as "Unknown" in the UI
                    trade["title"] = trade.get("title") or f"Unknown market ({trade['asset'][:16]}…)"
                else:
                    unknown_count += 1
                    logger.debug("ChainMonitor: skipped trade with unknown slug (token=%s)", trade["asset"][:16])
                    continue
            if not copy_all and not any(
                p.lower() in trade["slug"].lower()
                for p in self.config.market_slug_patterns
            ):
                self._emit("skip",
                    f"ChainMonitor: non-tracked market '{trade['slug']}' — not in slug patterns")
                continue
            trades.append(trade)

        if unknown_count:
            self._emit("warn",
                f"ChainMonitor: {unknown_count} trade(s) skipped — token_id not in metadata cache "
                f"(cache has {len(self._metadata_cache)} entries; refresh runs every 60 s)")

        return trades

    # ── Private ──────────────────────────────────────────────────────────────

    def _get_latest_block(self) -> Optional[int]:
        result = self._rpc("eth_blockNumber", [])
        if result is None:
            return None
        return int(result, 16)

    def _get_logs(self, from_block: int, to_block: int) -> list:
        result = self._rpc("eth_getLogs", [{
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "address": self.config.ctf_exchange_addresses,
            "topics": [_ORDER_FILLED_TOPIC, None, self._maker_topic],
        }])
        return result if isinstance(result, list) else []

    def _decode_log(self, log: dict, latest_block: int) -> Optional[dict]:
        try:
            data = bytes.fromhex(log["data"][2:])
            if len(data) < 160:
                return None
            maker_asset = int.from_bytes(data[0:32], "big")
            taker_asset = int.from_bytes(data[32:64], "big")
            maker_amt   = int.from_bytes(data[64:96], "big")
            taker_amt   = int.from_bytes(data[96:128], "big")
            if maker_amt == 0 or taker_amt == 0:
                return None

            if maker_asset == 0:         # BUY: maker pays USDC, receives shares
                side     = "BUY"
                token_id = str(taker_asset)
                size     = taker_amt / 1e6
                price    = maker_amt / taker_amt
            else:                        # SELL: maker pays shares, receives USDC
                side     = "SELL"
                token_id = str(maker_asset)
                size     = maker_amt / 1e6
                price    = taker_amt / maker_amt

            # Estimate block timestamp (Polygon ~2.1 s/block)
            blocks_ago   = latest_block - int(log["blockNumber"], 16)
            estimated_ts = int(time.time() - blocks_ago * 2.1)

            return {
                "asset":           token_id,
                "side":            side,
                "size":            round(size, 6),
                "price":           round(price, 6),
                "transactionHash": log["transactionHash"],
                "timestamp":       estimated_ts,
                "proxyWallet":     self.config.target_wallet,
                "slug":            "",
                "eventSlug":       "",
                "title":           "",
                "outcome":         "?",
            }
        except Exception as exc:
            logger.debug("ChainMonitor._decode_log error: %s", exc)
            return None

    def _enrich(self, trade: dict) -> None:
        """Populate slug/title/outcome from the metadata cache if available."""
        meta = self._metadata_cache.get(trade["asset"])
        if meta:
            trade["slug"]      = meta.get("slug", "")
            trade["eventSlug"] = meta.get("slug", "")
            trade["title"]     = meta.get("title", "")
            trade["outcome"]   = meta.get("outcome", "?")

    def _maybe_refresh_metadata(self) -> None:
        """
        Refresh token_id → market_info cache at most once/60 s.

        Two sources:
        1. data-api /trades for target wallet — covers markets they've traded before
        2. gamma-api /markets for each tracked slug pattern at the CURRENT 5-min window
           — this is the critical one: data-api has 4+ min indexing lag so brand-new
             window markets won't appear there until after the window is half-over.
        """
        import json as _json
        if time.monotonic() - self._last_metadata_refresh < 60:
            return
        self._last_metadata_refresh = time.monotonic()

        added = 0

        # ── Source 1: target wallet's past trades (data-api) ─────────────────
        try:
            resp = self._meta_session.get(
                f"{self.config.data_api_base}/trades",
                params={"user": self.config.target_wallet, "limit": 100},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for t in data:
                    tid  = t.get("asset", "")
                    slug = t.get("slug") or t.get("eventSlug", "")
                    if tid and slug and tid not in self._metadata_cache:
                        self._metadata_cache[tid] = {
                            "slug":    slug,
                            "title":   t.get("title", ""),
                            "outcome": t.get("outcome", "?"),
                        }
                        added += 1
        except Exception as exc:
            logger.warning("ChainMonitor metadata (data-api) refresh failed: %s", exc)

        # ── Source 2: current window markets for each tracked slug pattern ────
        # data-api lags 4+ minutes — we must proactively seed the cache with the
        # current 5-minute window's token IDs so new-window trades aren't dropped.
        _TF_WINDOWS = {"5m": 300, "15m": 900, "1h": 3600}
        _tf = getattr(self.config, "market_timeframe", "5m")
        _WINDOW_SEC = _TF_WINDOWS.get(_tf, 300)
        window_start = (int(time.time()) // _WINDOW_SEC) * _WINDOW_SEC
        gamma_base = self.config.gamma_api_base.rstrip("/")
        # When copy_all_markets is on, seed ALL common crypto updown markets so
        # their token IDs are cached before any trade arrives.
        _copy_all = getattr(self.config, "copy_all_markets", False)
        _ALL_CRYPTO = ["btc", "eth", "sol", "bnb", "xrp", "doge", "avax", "link", "ada"]
        if _copy_all:
            _patterns = [f"{a}-updown-{_tf}" for a in _ALL_CRYPTO]
        else:
            _patterns = list(self.config.market_slug_patterns)
        for pattern in _patterns:
            # Build slug list: for 1h markets Polymarket uses a date-based slug
            if _tf == "1h":
                from datetime import datetime, timezone, timedelta
                slugs = [PredictionTrader._1h_slug(window_start)]
                dt_est = datetime.fromtimestamp(window_start, tz=timezone.utc) + timedelta(hours=-5)
                h12 = dt_est.hour % 12 or 12
                ampm = "am" if dt_est.hour < 12 else "pm"
                slug_est = f"bitcoin-up-or-down-{dt_est.strftime('%B').lower()}-{dt_est.day}-{h12}{ampm}-et"
                if slug_est != slugs[0]:
                    slugs.append(slug_est)
            else:
                slugs = [f"{pattern}-{window_start}"]
            slug = slugs[0]
            markets = None
            for try_slug in slugs:
                try:
                    resp = self._meta_session.get(
                        f"{gamma_base}/markets",
                        params={"slug": try_slug},
                        timeout=8,
                    )
                    resp.raise_for_status()
                    candidate = resp.json()
                    if candidate:
                        markets = candidate
                        slug = try_slug
                        break
                except Exception as exc:
                    logger.debug("ChainMonitor metadata (gamma-api %s) failed: %s", try_slug, exc)
            if not markets:
                continue
            try:
                market = markets[0] if isinstance(markets, list) else markets
                market_slug = market.get("slug", "") or slug
                raw_outcomes = market.get("outcomes", "[]")
                raw_ids      = market.get("clobTokenIds", "[]")
                if isinstance(raw_outcomes, str):
                    raw_outcomes = _json.loads(raw_outcomes)
                if isinstance(raw_ids, str):
                    raw_ids = _json.loads(raw_ids)
                title = market.get("question", market.get("title", slug))
                for i, outcome in enumerate(raw_outcomes):
                    if i >= len(raw_ids):
                        continue
                    tid = str(raw_ids[i])
                    if tid and tid not in self._metadata_cache:
                        self._metadata_cache[tid] = {
                            "slug":    market_slug,
                            "title":   f"{title} — {outcome}",
                            "outcome": str(outcome),
                        }
                        added += 1
            except Exception as exc:
                logger.debug("ChainMonitor metadata (gamma-api %s) failed: %s", slug, exc)

        if added:
            self._emit("info",
                f"ChainMonitor: metadata cache refreshed — {added} new entries, "
                f"{len(self._metadata_cache)} total")
        else:
            logger.debug("ChainMonitor metadata cache: %d entries (no new)", len(self._metadata_cache))

    def _rpc(self, method: str, params: list) -> Any:
        """
        JSON-RPC call with automatic fallback rotation and per-RPC cooldown.

        Failing RPCs are penalised for _RPC_COOLDOWN_SEC (5 min) so they are
        skipped on subsequent calls instead of being retried every poll cycle.
        This suppresses repeated timeout warnings from persistently dead endpoints.
        Raises RuntimeError only if every URL fails.
        """
        _RPC_COOLDOWN_SEC = 300  # 5 minutes
        num_urls = len(self._rpc_urls)
        now = time.monotonic()

        # Build an ordered candidate list: start at _rpc_index, skip cooled-down RPCs.
        # If ALL are on cooldown (unlikely), fall back to trying them all anyway.
        ordered = [(self._rpc_index + i) % num_urls for i in range(num_urls)]
        candidates = [i for i in ordered if now - self._rpc_fail_ts.get(i, 0) >= _RPC_COOLDOWN_SEC]
        if not candidates:
            candidates = ordered  # last resort — all are struggling

        last_exc: Optional[Exception] = None
        for idx in candidates:
            url = self._rpc_urls[idx]
            try:
                resp = self._rpc_session.post(
                    url,
                    json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                    timeout=5,   # fail fast — own URL rotation is faster than urllib3 retries
                )
                resp.raise_for_status()
                body = resp.json()
                if "error" in body:
                    raise RuntimeError(f"RPC error: {body['error']}")
                self._rpc_index = idx  # stick with the working RPC
                return body.get("result")
            except Exception as exc:
                last_exc = exc
                self._rpc_fail_ts[idx] = now  # penalise this RPC
                next_candidates = [i for i in ordered
                                   if i != idx and now - self._rpc_fail_ts.get(i, 0) >= _RPC_COOLDOWN_SEC]
                next_url = self._rpc_urls[next_candidates[0]] if next_candidates else "(all failing)"
                logger.warning("RPC %s failed (%s) — switching to %s", url, exc, next_url)
        raise RuntimeError(f"All RPC endpoints failed. Last error: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# RiskManager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Evaluates whether a detected trade should be copied and at what size.

    Decision flow
    -------------
    1. Is their USD value ≥ min_usd_to_copy?          → else SKIP
    2. Scale size = their_size × multiplier, capped at max_usd_per_trade
    3. Would copying exceed max_total_exposure?        → reduce or SKIP
    4. Return (should_copy, copy_size, reason)
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._session = _make_session()
        self._exposure_cache_value: float = 0.0
        self._exposure_cache_ts: float = 0.0
        self._exposure_ttl_sec: float = 5.0
        self._last_exposure_warn_ts: float = 0.0

    # ── Public ───────────────────────────────────────────────────────────────

    def evaluate(self, trade: dict) -> tuple[bool, float, str, float]:
        """
        Returns
        -------
        (should_copy, copy_size_shares, reason_string, their_usd_value)
        """
        their_size = _safe_float(trade.get("size", 0))
        their_price = _safe_float(trade.get("price", 0))
        their_usd = their_size * their_price

        # 1. Minimum threshold
        if their_usd < self.config.min_usd_to_copy:
            return (
                False,
                0.0,
                f"trade USD ${their_usd:.2f} < min ${self.config.min_usd_to_copy:.2f}",
                their_usd,
            )

        # 2. Scale size
        copy_usd = their_usd * self.config.copy_multiplier
        copy_usd = min(copy_usd, self.config.max_usd_per_trade)
        if their_price <= 0:
            return False, 0.0, "target price is 0 (cannot compute size)", their_usd
        copy_size = copy_usd / their_price

        # 3. Exposure check
        current_exposure = self.get_current_exposure()
        headroom = self.config.max_total_exposure_usd - current_exposure

        if headroom <= 0:
            return (
                False,
                0.0,
                f"max exposure reached (current ${current_exposure:.2f} / ${self.config.max_total_exposure_usd:.2f})",
                their_usd,
            )

        reason = "ok"
        if copy_usd > headroom:
            reduced_usd = headroom
            copy_size = reduced_usd / their_price
            copy_usd = reduced_usd
            reason = f"size reduced to fit headroom (${headroom:.2f} remaining)"
            if copy_usd < self.config.min_usd_to_copy:
                return (
                    False,
                    0.0,
                    f"remaining headroom ${headroom:.2f} < min ${self.config.min_usd_to_copy:.2f}",
                    their_usd,
                )

        return True, copy_size, reason, their_usd

    def get_current_exposure(self) -> float:
        """
        Fetches open positions from Gamma API and sums USD value for
        BTC-5min markets only.
        """
        # For POLY_PROXY / Gnosis Safe (sig_type 1/2), positions belong to the
        # proxy/funder wallet, not the signing EOA.  Fall back to my_wallet.
        funder = getattr(self.config, "funder_address", "")
        sig_type = getattr(self.config, "signature_type", 0)
        if sig_type in (1, 2) and funder:
            wallet = funder
        else:
            wallet = self.config.my_wallet
        if not wallet:
            return 0.0

        now = time.time()
        cache_ts = float(getattr(self, "_exposure_cache_ts", 0.0))
        cache_val = float(getattr(self, "_exposure_cache_value", 0.0))
        cache_ttl = float(getattr(self, "_exposure_ttl_sec", 5.0))
        last_warn_ts = float(getattr(self, "_last_exposure_warn_ts", 0.0))

        if now - cache_ts <= cache_ttl:
            return cache_val

        try:
            positions = self._fetch_positions(wallet)
            total = 0.0
            for pos in positions:
                if not self._is_tracked_position(pos.get("title", "")):
                    continue
                size = _safe_float(pos.get("size", 0))
                # Prefer currentValue if available, otherwise size × curPrice
                current_value = _safe_float(pos.get("currentValue", 0))
                if current_value:
                    total += current_value
                else:
                    cur_price = _safe_float(pos.get("curPrice", pos.get("avgPrice", 0.5)))
                    total += size * cur_price
            self._exposure_cache_value = total
            self._exposure_cache_ts = now
            return total
        except Exception as exc:
            # Non-fatal: never block copy flow because exposure endpoint is flaky.
            # Reuse last good value and throttle warning noise.
            if now - last_warn_ts > 60:
                logger.warning(
                    "RiskManager exposure lookup failed (%s). Using cached exposure %.2f",
                    exc,
                    cache_val,
                )
                self._last_exposure_warn_ts = now
            return cache_val

    def _fetch_positions(self, wallet: str) -> list:
        """
        Gamma API shape has changed historically; try a few variants.
        """
        base = self.config.gamma_api_base.rstrip("/")
        attempts = [
            (f"{base}/positions", {"user": wallet}),
            (f"{base}/positions", {"userAddress": wallet}),
            (f"{base}/users/{wallet}/positions", None),
        ]

        last_exc: Optional[Exception] = None
        for url, params in attempts:
            try:
                resp = self._session.get(url, params=params, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    if isinstance(data.get("data"), list):
                        return data["data"]
                    if isinstance(data.get("positions"), list):
                        return data["positions"]
                return []
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(last_exc or "positions lookup failed")

    def _is_tracked_position(self, title: str) -> bool:
        t = title.lower()
        return any(p.lower() in t for p in self.config.market_slug_patterns)


# ─────────────────────────────────────────────────────────────────────────────
# TradeExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TradeExecutor:
    """
    Wraps py-clob-client to place FOK limit orders on Polymarket CLOB.

    Fetches and caches `tick_size` and `neg_risk` per token_id to avoid
    redundant CLOB API calls on hot trading paths.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._clob_session = _make_session()
        self._tick_cache: Dict[str, str] = {}    # token_id → "0.01"
        self._neg_risk_cache: Dict[str, bool] = {}  # token_id → bool
        self.client = None
        self.init_error: str = ""   # set when _init_client fails — surfaced in API responses
        # Skip CLOB init entirely in dry-run — no key needed, no orders placed
        if config.private_key:
            self._init_client()
        else:
            logger.info("TradeExecutor: no private key supplied — CLOB client skipped (dry-run only)")

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_client(self) -> None:
        try:
            from py_clob_client.client import ClobClient  # type: ignore

            kwargs: Dict[str, Any] = {
                "host": self.config.clob_host,
                "key": self.config.private_key,
                "chain_id": self.config.chain_id,
                "signature_type": self.config.signature_type,
            }
            # Docs require funder even for EOA (type 0). Default to derived wallet.
            funder = self.config.funder_address
            if not funder and self.config.signature_type == 0:
                funder = self.derive_wallet_address() or ""
            if funder:
                kwargs["funder"] = funder

            self.client = ClobClient(**kwargs)

            eoa = self.derive_wallet_address() or "unknown"
            logger.info(
                "TradeExecutor: operator EOA=%s  funder/proxy=%s  sig_type=%d",
                eoa[:10], (funder or "—")[:10], self.config.signature_type,
            )

            # Prefer derive over create to avoid key proliferation
            creds = self.client.create_or_derive_api_creds()
            if not creds:
                raise RuntimeError("create_or_derive_api_creds returned None — check private key / funder address")
            self.client.set_api_creds(creds)
            logger.info("TradeExecutor: CLOB client initialised (sig_type=%d, api_key=%s…)",
                        self.config.signature_type, creds.api_key[:8] if creds.api_key else "?")
        except ImportError:
            msg = "py-clob-client not installed — run: pip install py-clob-client"
            logger.error("TradeExecutor: %s", msg)
            self.init_error = msg
            self.client = None
        except Exception as exc:
            import traceback as _tb
            tb = _tb.format_exc()
            logger.error("TradeExecutor init failed: %s\n%s", exc, tb)
            self.init_error = f"{exc}\n{tb}"
            self.client = None

    def derive_wallet_address(self) -> Optional[str]:
        """Derive the EOA address from the configured private key. Returns None in dry-run with no key."""
        if not self.config.private_key:
            return None
        try:
            from eth_account import Account  # type: ignore
            account = Account.from_key(self.config.private_key)
            return account.address
        except Exception as exc:
            logger.error("Cannot derive wallet address: %s", exc)
            return None

    # ── Ordering ─────────────────────────────────────────────────────────────

    def place_order(
        self,
        token_id: str,
        side: str,          # "BUY" | "SELL"
        size: float,        # shares to trade
        ref_price: float,   # target's fill price (our reference)
        use_gtc: bool = False,  # True → GTC limit order (sits on book); False → FAK
    ) -> Dict[str, Any]:
        """
        Place a FAK order and retry any unfilled remainder.
        If use_gtc=True, places a single GTC limit order at ref_price instead.

        Adds a slippage buffer to the price so the order is likely to fill
        immediately:
          • BUY  → price * (1 + slippage), capped at 0.99
          • SELL → price * (1 - slippage), floored at 0.01

        Raises RuntimeError if the CLOB client is not ready.
        """
        if not self.client:
            raise RuntimeError("CLOB client not initialised — check credentials")

        from py_clob_client.clob_types import (
            OrderArgs,
            OrderType,
            CreateOrderOptions,
            MarketOrderArgs,
        )  # type: ignore
        from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore

        order_side = BUY if side.upper() == "BUY" else SELL
        tick_str  = self._get_tick_size(token_id)   # keep as string to avoid float drift
        tick      = float(tick_str)
        neg_risk  = self._get_neg_risk(token_id)
        base_slip = max(0.0, float(getattr(self.config, "slippage_pct", 0.01)))
        retries = max(0, int(getattr(self.config, "order_retries", 2)))
        slip_step = max(0.0, float(getattr(self.config, "retry_slippage_step_pct", 0.005)))
        max_slip = max(base_slip, float(getattr(self.config, "max_slippage_pct", 0.05)))

        from decimal import Decimal, ROUND_DOWN  # stdlib — no new dep
        D = Decimal
        remaining_size = D(str(size)).quantize(D("0.0001"), rounding=ROUND_DOWN)
        if remaining_size <= 0:
            raise RuntimeError(f"Invalid size after rounding: {size}")

        options = CreateOrderOptions(
            tick_size=tick_str,
            neg_risk=neg_risk,
        )

        # ── GTC path: bid at live best ask so we cross the spread immediately ──
        if use_gtc:
            # Fetch live best ask — if empty book, bail out now rather than posting
            live_ask = 0.0
            try:
                ask_resp = self.client.get_price(token_id, "buy")
                live_ask = float(ask_resp.get("price", 0) or 0)
            except Exception:
                pass
            if live_ask == 0.0:
                raise RuntimeError("Empty order book — no sell orders found, skipping")
            order_price = min(_round_to_tick(live_ask, tick, round_up=True), 0.99)
            p_int = round(float(order_price) * 100)
            s_max_int = math.floor(float(remaining_size) * 100)
            gcd_val = math.gcd(p_int, 100) if p_int > 0 else 1
            step = 100 // gcd_val
            s_int = (s_max_int // step) * step
            if s_int <= 0:
                raise RuntimeError(f"GTC size rounds to zero at price {order_price}")
            size_d = D(s_int) / D(100)
            order_args = OrderArgs(
                token_id=token_id,
                price=float(order_price),
                size=float(size_d),
                side=order_side,
            )
            try:
                signed = self.client.create_order(order_args, options)
            except TypeError:
                signed = self.client.create_order(order_args)
            response = self.client.post_order(signed, OrderType.GTC)
            response_d = response if isinstance(response, dict) else {"raw": str(response)}
            return {"success": True, "mode": "GTC", "price": float(order_price), "size": float(size_d), "response": response_d}
        # ── End GTC path ────────────────────────────────────────────────────────

        attempts: list[dict] = []
        filled_total = D("0")
        is_buy = order_side == BUY

        for attempt_idx in range(retries + 1):
            if remaining_size <= D("0.0001"):
                break

            if is_buy:
                # Fetch the live best ask on every attempt.
                # The target's fill price (ref_price) is stale — their order likely
                # cleared the book and asks have refilled at a higher price.
                # Bidding at the current ask (not ref_price + %) ensures we always
                # cross the spread regardless of how much the market has moved.
                live_ask = 0.0
                try:
                    ask_resp = self.client.get_price(token_id, "buy")
                    live_ask = float(ask_resp.get("price", 0) or 0)
                except Exception:
                    pass
                slip = min(base_slip + (attempt_idx * slip_step), max_slip)
                if live_ask > 0:
                    # Bid at live ask + escalating slippage so each retry is more
                    # aggressive. Attempt 0 = ask+1%, attempt 1 = ask+1.5%, etc.
                    raw_price = live_ask * (1 + slip)
                else:
                    # Book empty — escalate from ref_price while waiting for refill
                    raw_price = ref_price * (1 + slip)
                order_price = min(_round_to_tick(raw_price, tick, round_up=True), 0.99)
            else:
                slip = min(base_slip + (attempt_idx * slip_step), max_slip)
                raw_price = ref_price * (1 - slip)
                order_price = max(_round_to_tick(raw_price, tick, round_up=False), 0.01)

            price_d = D(str(order_price))
            if price_d <= 0:
                break

            if is_buy:
                # Use limit OrderArgs for BUY (same as SELL path).
                # MarketOrderArgs with amount=size×stale_price fails with
                # "not enough balance/allowance" when current ask > target fill price,
                # because the CLOB validates amount >= current_ask × size.
                # OrderArgs(size, price) avoids this — CLOB validates size×price <= balance.
                #
                # API constraint: maker_amount (size×price USDC) must have ≤ 2dp.
                # The py-clob-client library (RoundConfig for tick 0.01) rounds size to 2dp
                # then multiplies — product can be 3-4dp which the API rejects.
                #
                # Fix using integer arithmetic:
                #   price in hundredths: p_int = round(price × 100)
                #   valid sizes (in hundredths): s must satisfy  s × p_int ≡ 0 (mod 100)
                #   → s must be a multiple of  step = 100 / gcd(p_int, 100)
                #   → take largest such s ≤ floor(remaining_size × 100)
                p_int = round(float(price_d) * 100)
                s_max_int = math.floor(float(remaining_size) * 100)
                gcd_val = math.gcd(p_int, 100) if p_int > 0 else 1
                step = 100 // gcd_val
                s_int = (s_max_int // step) * step
                if s_int <= 0:
                    break
                size_d = D(s_int) / D(100)
                order_args = OrderArgs(
                    token_id=token_id,
                    price=float(price_d),
                    size=float(size_d),
                    side=order_side,
                )
                try:
                    signed = self.client.create_order(order_args, options)
                except TypeError:
                    signed = self.client.create_order(order_args)
            else:
                # SELL path: maker=shares (2dp), taker=USDC (≤4dp). Library handles this fine.
                size_d = remaining_size.quantize(D("0.01"), rounding=ROUND_DOWN)
                if size_d <= 0:
                    break
                order_args = OrderArgs(
                    token_id=token_id,
                    price=float(price_d),
                    size=float(size_d),
                    side=order_side,
                )
                try:
                    signed = self.client.create_order(order_args, options)
                except TypeError:
                    signed = self.client.create_order(order_args)

            try:
                response = self.client.post_order(signed, OrderType.FAK)
                response_d = response if isinstance(response, dict) else {"raw": str(response)}
            except Exception as exc:
                # Common FAK miss: HTTP 400 "no orders found to match with FAK order"
                # Treat as zero-fill and continue retrying with wider slippage.
                response_d = {"success": False, "error": str(exc)}

                # Only swallow known liquidity/IOC miss cases and transient network errors.
                err_l = str(exc).lower()
                retryable = (
                    "no orders found to match with fak order" in err_l
                    or "fak orders are partially filled or killed" in err_l
                    or "couldn't be fully filled" in err_l
                    # Network-level failures (status_code=None) — safe to retry for FAK
                    # since the order was never sent or was killed server-side.
                    or "request exception" in err_l
                    or "connection" in err_l
                    or "timeout" in err_l
                )
                if not retryable:
                    raise

                # When the book is momentarily empty (target's order just cleared
                # all liquidity), pause and re-fetch the live ask so the next
                # attempt starts from a fresh price.  The book typically refills
                # within 1-3 seconds.
                is_empty_book = (
                    "no orders found to match with fak order" in err_l
                    or "fak orders are partially filled or killed" in err_l
                )
                if is_buy and is_empty_book:
                    time.sleep(1.5)
                    pass  # next attempt will re-fetch the live ask at top of loop

            filled_shares = self._extract_filled_shares(response_d, is_buy)

            # If SDK response omits amount fields but indicates a clean match, treat as full attempt fill.
            if filled_shares <= 0 and response_d.get("success") and str(response_d.get("status", "")).lower() in ("matched", "filled"):
                filled_shares = float(remaining_size)

            filled_d = D(str(max(0.0, filled_shares))).quantize(D("0.0001"), rounding=ROUND_DOWN)
            if filled_d > remaining_size:
                filled_d = remaining_size

            remaining_size -= filled_d
            filled_total += filled_d
            attempts.append({
                "attempt": attempt_idx + 1,
                "slippage_pct": round(slip, 6),
                "requested_shares": float((filled_d + remaining_size).quantize(D("0.0001"), rounding=ROUND_DOWN)),
                "filled_shares": float(filled_d),
                "response": response_d,
            })

        if filled_total <= 0:
            last = attempts[-1]["response"] if attempts else {"error": "no attempt made"}
            raise RuntimeError(
                f"No fill after {retries + 1} FAK attempt(s). Last response: {last}"
            )

        partial = remaining_size > D("0.0001")
        return {
            "success": True,
            "mode": "FAK_RETRY",
            "requested_shares": float(D(str(size)).quantize(D("0.0001"), rounding=ROUND_DOWN)),
            "filled_shares": float(filled_total),
            "remaining_shares": float(remaining_size if remaining_size > 0 else D('0')),
            "partial_fill": partial,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }

    @staticmethod
    def _extract_filled_shares(response: Dict[str, Any], is_buy: bool) -> float:
        """
        Best-effort share fill parsing from CLOB response.
        BUY  => takingAmount is shares.
        SELL => makingAmount is shares.
        """
        if not isinstance(response, dict):
            return 0.0
        key = "takingAmount" if is_buy else "makingAmount"
        filled = _safe_float(response.get(key), 0.0)
        return max(0.0, filled)

    # ── Token metadata (cached) ───────────────────────────────────────────────

    def _get_tick_size(self, token_id: str) -> str:
        if token_id not in self._tick_cache:
            try:
                resp = self._clob_session.get(
                    f"{self.config.clob_host}/tick-size",
                    params={"token_id": token_id},
                    timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()
                self._tick_cache[token_id] = str(
                    data.get("minimum_tick_size", data.get("tick_size", "0.01"))
                )
            except Exception:
                self._tick_cache[token_id] = "0.01"  # safe default
        return self._tick_cache[token_id]

    def _get_neg_risk(self, token_id: str) -> bool:
        if token_id not in self._neg_risk_cache:
            try:
                resp = self._clob_session.get(
                    f"{self.config.clob_host}/neg-risk",
                    params={"token_id": token_id},
                    timeout=8,
                )
                resp.raise_for_status()
                self._neg_risk_cache[token_id] = bool(
                    resp.json().get("neg_risk", False)
                )
            except Exception:
                self._neg_risk_cache[token_id] = False
        return self._neg_risk_cache[token_id]


# ─────────────────────────────────────────────────────────────────────────────
# PredictionTrader — autonomous prediction-based order placer
# ─────────────────────────────────────────────────────────────────────────────

class PredictionTrader:
    """
    Polls MarketAnalyzer every 30 s. When confidence >= min_confidence_pct and
    prediction is UP or DOWN, looks up the current BTC 5-minute market token
    via gamma-api and places one BUY order per 5-minute window.

    Reuses TradeExecutor for all CLOB interaction.
    PredictionTraderConfig satisfies TradeExecutor's interface directly.
    """

    POLL_INTERVAL_SEC = 30
    # Seconds after window open in which the predictor is allowed to fire.
    # Scaled to 20% of the window so 5m=60s, 15m=180s, 1h=720s.
    _TF_OPEN_SEC      = {"5m": 60, "15m": 180, "1h": 720}
    _TF_WINDOWS       = {"5m": 300, "15m": 900, "1h": 3600}

    def __init__(
        self,
        config: Any,       # PredictionTraderConfig
        analyzer: Any,     # MarketAnalyzer singleton — injected from app.py
        log_callback: Optional[Callable] = None,
    ) -> None:
        self.config   = config
        self.analyzer = analyzer
        self._log_cb  = log_callback or (lambda _: None)
        self.running  = False
        _tf = getattr(config, "market_timeframe", "5m")
        self.WINDOW_SEC     = self._TF_WINDOWS.get(_tf, 300)
        self.WINDOW_OPEN_SEC = self._TF_OPEN_SEC.get(_tf, 60)
        self._stop_event = threading.Event()
        self._last_traded_window: Optional[int] = None
        self._session = _make_session()
        self._pnl_session = _make_session()
        self.executor = TradeExecutor(config)
        # Trade journal — tracks positions for P/L display and take-profit exits
        self.trade_journal: list = []
        self._journal_lock = threading.Lock()
        self._last_monitor_ts: float = 0.0

    def _position_wallets(self) -> list[str]:
        wallets: list[str] = []
        if self.config.signature_type in (1, 2) and self.config.funder_address:
            wallets.append(self.config.funder_address.strip())
        if self.config.my_wallet:
            wallets.append(self.config.my_wallet.strip())
        try:
            derived = self.executor.derive_wallet_address()
            if derived:
                wallets.append(derived.strip())
        except Exception:
            pass
        out: list[str] = []
        for w in wallets:
            if w and w not in out:
                out.append(w)
        return out

    def _fetch_positions(self, wallet: str) -> list:
        """
        Try gamma-api and data-api for current positions.
        Returns a list of position dicts.
        """
        gamma_base = self.config.gamma_api_base.rstrip("/")
        data_base  = "https://data-api.polymarket.com"
        attempts = [
            (f"{gamma_base}/positions", {"user": wallet, "sizeThreshold": "0"}),
            (f"{gamma_base}/positions", {"userAddress": wallet, "sizeThreshold": "0"}),
            (f"{data_base}/positions",  {"user": wallet}),
            (f"{data_base}/positions",  {"userAddress": wallet}),
        ]

        last_exc: Optional[Exception] = None
        for url, params in attempts:
            try:
                resp = self._session.get(url, params=params, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and data:
                    return data
                if isinstance(data, dict):
                    for key in ("data", "positions", "results"):
                        if isinstance(data.get(key), list) and data[key]:
                            return data[key]
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(last_exc or "positions lookup failed")

    @staticmethod
    def _pos_token_id(pos: dict) -> str:
        for k in ("clobTokenId", "clobTokenID", "tokenId", "token_id", "tokenID", "token", "asset", "conditionId"):
            v = pos.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    @staticmethod
    def _pos_avg_price(pos: dict) -> float:
        for k in ("avgPrice", "avg_price", "averagePrice", "avgCost", "entryPrice"):
            v = pos.get(k)
            if v is not None:
                return _safe_float(v, 0.0)
        return 0.0

    @staticmethod
    def _pos_size(pos: dict) -> float:
        for k in ("size", "shares", "quantity", "position", "balance"):
            v = pos.get(k)
            if v is not None:
                return _safe_float(v, 0.0)
        return 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        self._stop_event.clear()
        mode = "DRY RUN" if self.config.dry_run else "LIVE"
        self._log("info", f"[PREDICTOR] Started | mode={mode} | {self.config.safe_repr()}")
        if self.config.dry_run:
            self._log("info", "[PREDICTOR] DRY RUN active — predictions logged, no orders placed.")
        if self.config.auto_redeem:
            threading.Thread(
                target=_auto_redeem_loop,
                args=(self.config, self._log, self._stop_event),
                name="PredictorAutoRedeem",
                daemon=True,
            ).start()
            self._log("info", "[PREDICTOR] Auto-redeem background thread started")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self._log("error", f"[PREDICTOR] Unexpected error: {exc}")
            self._stop_event.wait(self.POLL_INTERVAL_SEC)
        self.running = False
        self._log("info", "[PREDICTOR] Stopped.")

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False

    def get_trade_journal(self) -> list:
        """Return a snapshot of the trade journal (thread-safe copy)."""
        with self._journal_lock:
            return list(self.trade_journal)

    # ── Core tick ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now           = int(time.time())
        window_start  = (now // self.WINDOW_SEC) * self.WINDOW_SEC
        secs_into     = now - window_start
        secs_left     = self.WINDOW_SEC - secs_into

        # Monitor open positions for take-profit exits (every tick, lightweight)
        if time.monotonic() - self._last_monitor_ts > self.POLL_INTERVAL_SEC:
            try:
                self._monitor_positions()
            except Exception as exc:
                self._log("warn", f"[PREDICTOR] Position monitor error: {exc}")
            self._last_monitor_ts = time.monotonic()

        _tf         = getattr(self.config, "market_timeframe", "5m")
        analysis    = self.analyzer.get_analysis(_tf)
        prediction  = analysis.get("prediction", "NEUTRAL")
        confidence  = int(analysis.get("confidence") or 0)
        error       = analysis.get("error")

        _src = analysis.get("source", "?")
        self._log(
            "info",
            f"[PREDICTOR] tick | window={window_start} | "
            f"pred={prediction} conf={confidence}% | src={_src} | {secs_left}s left in window",
        )

        if error:
            self._log("warn", f"[PREDICTOR] Analysis error — skip: {error}")
            return

        # Gate 1: window-open timing
        if self.config.trade_at_window_open_only and secs_into > self.WINDOW_OPEN_SEC:
            self._log(
                "skip",
                f"[PREDICTOR] SKIP: {secs_into}s into window (limit={self.WINDOW_OPEN_SEC}s). "
                "Waiting for next window.",
            )
            return

        # Gate 2: per-window dedup
        if self._last_traded_window == window_start:
            self._log("skip", f"[PREDICTOR] SKIP: already traded window {window_start}.")
            return

        # Gate 3: direction + confidence
        if prediction == "NEUTRAL":
            self._log("skip", "[PREDICTOR] SKIP: prediction is NEUTRAL.")
            return
        if confidence < self.config.min_confidence_pct:
            self._log(
                "skip",
                f"[PREDICTOR] SKIP: confidence {confidence}% < min {self.config.min_confidence_pct}%.",
            )
            return

        # Token discovery
        token_id, outcome_label = self._lookup_token(window_start, prediction)
        if not token_id:
            self._log("error", f"[PREDICTOR] Cannot find token for window {window_start}. Skipping.")
            return

        # Mid-price and share size
        mid_price = self._get_midpoint(token_id)
        if not mid_price or mid_price <= 0:
            self._log("error", f"[PREDICTOR] Cannot get midpoint for token {token_id[:16]}. Skipping.")
            return

        shares   = round(self.config.auto_trade_usd / mid_price, 2)
        cost_usd = round(shares * mid_price, 4)

        self._execute_trade(
            token_id=token_id, prediction=prediction, outcome_label=outcome_label,
            shares=shares, mid_price=mid_price, cost_usd=cost_usd,
            confidence=confidence, window_start=window_start,
        )

    # ── Order execution ───────────────────────────────────────────────────────

    def _execute_trade(
        self, token_id: str, prediction: str, outcome_label: str,
        shares: float, mid_price: float, cost_usd: float,
        confidence: int, window_start: int,
    ) -> None:
        summary = (
            f"{prediction} {outcome_label} | {shares:.2f} shares @ "
            f"~${mid_price:.4f} = ~${cost_usd:.2f} | conf={confidence}% | "
            f"window={window_start}"
        )
        tf = getattr(self.config, "market_timeframe", "5m")
        slug = self._1h_slug(window_start) if tf == "1h" else f"btc-updown-{tf}-{window_start}"
        if self.config.dry_run:
            self._log(
                "dry_run",
                f"[DRY RUN] Would BUY {summary}",
                {"token_id": token_id, "prediction": prediction, "outcome": outcome_label,
                 "shares": shares, "price": mid_price, "cost_usd": cost_usd,
                 "window_start": window_start},
            )
            self._last_traded_window = window_start
            self._append_journal(slug, token_id, outcome_label, "BUY",
                                 shares, mid_price, cost_usd, dry_run=True,
                                 confidence=confidence, window_start=window_start)
            return

        # Live execution
        try:
            self._log("info", f"[PREDICTOR] Placing order: BUY {summary}")
            response = self.executor.place_order(token_id, "BUY", shares, mid_price)
            self._last_traded_window = window_start
            self._log(
                "copy",
                f"[PREDICTOR] Order placed: BUY {summary}",
                {"token_id": token_id, "clob_response": response},
            )
            self._append_journal(slug, token_id, outcome_label, "BUY",
                                 shares, mid_price, cost_usd, dry_run=False,
                                 confidence=confidence, window_start=window_start)
        except Exception as exc:
            self._log("error", f"[PREDICTOR] Order failed: {exc}", {"token_id": token_id})
            # Do NOT mark window as traded on failure — allow retry next tick

    # ── Token lookup ──────────────────────────────────────────────────────────

    @staticmethod
    def _1h_slug(window_start: int) -> str:
        """Generate the date-based slug Polymarket uses for hourly BTC markets.
        E.g. window_start at 2026-03-14 13:00 UTC → 'bitcoin-up-or-down-march-14-9am-et'
        """
        from datetime import datetime, timezone, timedelta
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(window_start, tz=ZoneInfo("America/New_York"))
        except Exception:
            # Approximate: try both EDT (-4) and EST (-5); caller tries both slugs
            dt_utc = datetime.fromtimestamp(window_start, tz=timezone.utc)
            dt = dt_utc + timedelta(hours=-4)  # default EDT
        h12 = dt.hour % 12 or 12
        ampm = "am" if dt.hour < 12 else "pm"
        return f"bitcoin-up-or-down-{dt.strftime('%B').lower()}-{dt.day}-{h12}{ampm}-et"

    def _lookup_token(self, window_start: int, prediction: str):
        """Query gamma-api for the current market window token_id.

        gamma-api returns outcomes and clobTokenIds as JSON-encoded strings,
        e.g. outcomes='["Up","Down"]', clobTokenIds='["123...", "456..."]'.
        They map 1:1 by index — tokens[] is always empty.
        """
        import json as _json
        tf = getattr(self.config, "market_timeframe", "5m")
        _TF_SLUG_PREFIX = {"5m": "btc-updown-5m", "15m": "btc-updown-15m"}
        slug_prefix = _TF_SLUG_PREFIX.get(tf)
        if slug_prefix:
            slugs_to_try = [f"{slug_prefix}-{window_start}"]
        else:
            # 1h: Polymarket uses a date-based slug format; try both EDT and EST offsets
            from datetime import datetime, timezone, timedelta
            slugs_to_try = [self._1h_slug(window_start)]
            # Also try EST offset in case EDT detection failed
            dt_est = datetime.fromtimestamp(window_start, tz=timezone.utc) + timedelta(hours=-5)
            h12_est = dt_est.hour % 12 or 12
            ampm_est = "am" if dt_est.hour < 12 else "pm"
            slug_est = f"bitcoin-up-or-down-{dt_est.strftime('%B').lower()}-{dt_est.day}-{h12_est}{ampm_est}-et"
            if slug_est != slugs_to_try[0]:
                slugs_to_try.append(slug_est)
        slug = slugs_to_try[0]  # used for logging/journal
        target = "Up" if prediction == "UP" else "Down"
        for try_slug in slugs_to_try:
            try:
                resp = self._session.get(
                    f"{self.config.gamma_api_base}/markets",
                    params={"slug": try_slug},
                    timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    continue
                market = data[0] if isinstance(data, list) else data
                slug = try_slug  # update for logging

                # Parse the JSON-encoded string arrays from gamma-api
                raw_outcomes = market.get("outcomes", "[]")
                raw_ids      = market.get("clobTokenIds", "[]")
                if isinstance(raw_outcomes, str):
                    raw_outcomes = _json.loads(raw_outcomes)
                if isinstance(raw_ids, str):
                    raw_ids = _json.loads(raw_ids)

                for i, outcome in enumerate(raw_outcomes):
                    if outcome.strip().lower() == target.lower() and i < len(raw_ids):
                        return raw_ids[i], target

                self._log("warn",
                    f"[PREDICTOR] No '{target}' outcome in {try_slug}. "
                    f"outcomes={raw_outcomes} ids={raw_ids}")
                return None, None
            except Exception:
                continue
        self._log("warn", f"[PREDICTOR] Market not yet listed for {slugs_to_try} — will retry.")
        return None, None

    # ── Midpoint price ────────────────────────────────────────────────────────

    def _get_midpoint(self, token_id: str) -> Optional[float]:
        """GET /midpoint?token_id=... → {"mid": "0.54"}"""
        try:
            resp = self._session.get(
                f"{self.config.clob_host}/midpoint",
                params={"token_id": token_id},
                timeout=8,
            )
            resp.raise_for_status()
            mid = _safe_float(resp.json().get("mid"), 0.0)
            return mid if mid > 0 else None
        except Exception as exc:
            self._log("error", f"[PREDICTOR] Midpoint fetch failed: {exc}")
            return None

    # ── Trade journal & take-profit ───────────────────────────────────────────

    def _append_journal(
        self, slug: str, token_id: str, outcome: str, side: str,
        shares: float, entry_price: float, cost_usd: float, dry_run: bool,
        confidence: int = 0, window_start: int = 0,
    ) -> None:
        _TF_WINDOWS = {"5m": 300, "15m": 900, "1h": 3600}
        _tf = getattr(self.config, "market_timeframe", "5m")
        entry = {
            "id":           f"pred-{int(time.time() * 1000)}",
            "entry_time":   _now_iso(),
            "market":       slug,
            "slug":         slug,
            "outcome":      outcome,
            "side":         side,
            "their_size":   0.0,
            "their_cost":   0.0,
            "their_pnl":    0.0,
            "our_size":     round(shares, 4),
            "entry_price":  round(entry_price, 4),
            "cost_usd":     round(cost_usd, 4),
            "token_id":     token_id,
            "status":       "SIMULATED" if dry_run else "OPEN",
            "cur_price":    round(entry_price, 4),
            "pnl_usd":      0.0,
            "dry_run":      dry_run,
            "source":       "predictor",
            "confidence":   confidence,
            "window_start": window_start,
            "window_sec":   _TF_WINDOWS.get(_tf, 300),
        }
        with self._journal_lock:
            self.trade_journal.append(entry)
            if len(self.trade_journal) > 500:
                self.trade_journal = self.trade_journal[-500:]

    def _monitor_positions(self) -> None:
        """Poll midpoints for open predictor positions; trigger take-profit exits."""
        take_profit = getattr(self.config, "take_profit_pct", 0.0)
        with self._journal_lock:
            active = [e for e in self.trade_journal if e["status"] in ("OPEN", "SIMULATED")]
        if not active:
            return

        is_dry = self.config.dry_run

        # Build gamma-API position map (live mode only — dry-run has no real positions)
        pos_by_token: dict = {}
        pos_wallet = ""
        if not is_dry:
            for w in self._position_wallets():
                try:
                    raw = self._fetch_positions(w)
                    if raw:
                        for p in raw:
                            tok = self._pos_token_id(p)
                            if tok:
                                pos_by_token[tok] = p
                        pos_wallet = w
                        break
                except Exception:
                    continue
            if not pos_by_token:
                self._log("debug", "[PREDICTOR] Position lookup empty — using journal data for P/L")

        for entry in active:
            token_id = entry.get("token_id", "")
            if not token_id:
                continue
            try:
                resp = self._pnl_session.get(
                    f"{self.config.clob_host}/midpoint",
                    params={"token_id": token_id},
                    timeout=6,
                )
                if not resp.ok:
                    continue
                mid = _safe_float(resp.json().get("mid"), 0.0)
                if mid <= 0:
                    # CLOB has no live mid — try gamma settlement for expired windows
                    self._try_settle_from_gamma(entry)
                    continue

                is_sim = entry.get("dry_run", self.config.dry_run)
                is_sell = entry.get("side", "BUY").upper() == "SELL"
                sign = -1 if is_sell else 1

                # Use real wallet position if available, else fall back to journal data
                pos = pos_by_token.get(str(token_id))
                if pos:
                    wallet_size  = self._pos_size(pos)
                    wallet_entry = self._pos_avg_price(pos) or _safe_float(entry.get("entry_price"), 0.0)
                else:
                    wallet_size  = _safe_float(entry.get("our_size"), 0.0)
                    wallet_entry = _safe_float(entry.get("entry_price"), 0.0)

                if wallet_size <= 0 or wallet_entry <= 0:
                    continue

                pnl = sign * (mid - wallet_entry) * wallet_size
                won  = (mid <= 0.01) if is_sell else (mid >= 0.99)
                lost = (mid >= 0.99) if is_sell else (mid <= 0.01)
                status = ("SIM_WON" if is_sim else "WON") if won else \
                         ("SIM_LOST" if is_sim else "LOST") if lost else \
                         ("SIMULATED" if is_sim else "OPEN")
                with self._journal_lock:
                    entry["cur_price"] = round(mid, 4)
                    entry["pnl_usd"]   = round(pnl, 4)
                    entry["status"]    = status
                    if pos_wallet:
                        entry["wallet_size"]   = round(wallet_size, 4)
                        entry["wallet_entry"]  = round(wallet_entry, 4)
                        entry["wallet_source"] = pos_wallet
                # Take-profit gate
                if take_profit > 0 and status in ("OPEN", "SIMULATED"):
                    roi = (mid - wallet_entry) / wallet_entry \
                          if not is_sell \
                          else (wallet_entry - mid) / wallet_entry
                    if roi >= take_profit:
                        entry["our_size"]    = round(wallet_size, 4)
                        entry["entry_price"] = round(wallet_entry, 4)
                        self._predictor_take_profit(entry, mid, roi)
            except Exception as exc:
                logger.debug("PredictionTrader monitor error: %s", exc)

    def _predictor_take_profit(self, entry: dict, cur_price: float, roi: float) -> None:
        roi_pct = round(roi * 100, 1)
        token_id = entry.get("token_id", "")
        our_size = entry.get("our_size", 0.0)
        exit_side = "SELL" if entry.get("side", "BUY").upper() == "BUY" else "BUY"
        msg = (
            f"[PREDICTOR PROFIT] {entry.get('outcome','?')} {our_size:.2f}sh "
            f"entry={entry['entry_price']:.4f} now={cur_price:.4f} ROI={roi_pct}%"
        )
        if entry.get("dry_run", self.config.dry_run):
            with self._journal_lock:
                entry["status"] = "SIM_PROFIT_TAKEN"
            self._log("dry_run", f"[DRY PROFIT] {msg}")
            return
        try:
            self.executor.place_order(token_id, exit_side, our_size, cur_price)
            with self._journal_lock:
                entry["status"] = "PROFIT_TAKEN"
            self._log("copy", f"[PREDICTOR PROFIT] {msg} — exit placed")
        except Exception as exc:
            self._log("error", f"[PREDICTOR PROFIT] Exit failed: {exc}")

    def _try_settle_from_gamma(self, entry: dict) -> None:
        """
        When CLOB has no live mid (expired/settled market), query Gamma API
        outcomePrices to determine the final settlement result.
        Called for both predictor and copy-trader journal entries.
        """
        slug = entry.get("slug", "")
        outcome = (entry.get("outcome") or "").lower().strip()
        if not slug or not outcome:
            return
        try:
            resp = self._pnl_session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            if not resp.ok:
                return
            data = resp.json()
            market = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if not market:
                return
            raw_outcomes = [o.lower().strip() for o in market.get("outcomes", [])]
            raw_prices   = market.get("outcomePrices", [])
            if len(raw_outcomes) != len(raw_prices):
                return
            prices = [_safe_float(p, 0.5) for p in raw_prices]
            idx = raw_outcomes.index(outcome) if outcome in raw_outcomes else -1
            if idx < 0:
                return
            p = prices[idx]
            if p < 0.05 or p > 0.95:  # settled
                is_sim  = entry.get("dry_run", self.config.dry_run)
                is_sell = entry.get("side", "BUY").upper() == "SELL"
                won  = (p <= 0.05) if is_sell else (p >= 0.95)
                lost = (p >= 0.95) if is_sell else (p <= 0.05)
                status = ("SIM_WON" if is_sim else "WON") if won else \
                         ("SIM_LOST" if is_sim else "LOST") if lost else None
                if status:
                    wallet_size  = _safe_float(entry.get("our_size") or entry.get("wallet_size"), 0.0)
                    wallet_entry = _safe_float(entry.get("entry_price") or entry.get("wallet_entry"), 0.0)
                    sign = -1 if is_sell else 1
                    pnl  = sign * (p - wallet_entry) * wallet_size if wallet_size > 0 and wallet_entry > 0 else 0.0
                    with self._journal_lock:
                        entry["cur_price"] = round(p, 4)
                        entry["status"]    = status
                        if pnl != 0.0:
                            entry["pnl_usd"] = round(pnl, 4)
        except Exception as exc:
            logger.debug("Gamma settlement lookup failed for %s: %s", slug, exc)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, log_type: str, message: str, data: Optional[dict] = None) -> None:
        entry = {
            "type":      log_type,
            "timestamp": _now_iso(),
            "message":   message,
            "data":      data or {},
        }
        {
            "error": logger.error,
            "warn":  logger.warning,
            "skip":  logger.warning,
        }.get(log_type, logger.info)(message)
        try:
            self._log_cb(entry)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared auto-redeem helpers (used by CopyTradingBot, PredictionTrader,
# and AdvancedTrader — all three have auto_redeem in their configs)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_redeem_loop(config, log_fn, stop_event) -> None:
    """
    Background thread body: calls redeem_all() once immediately on start
    (catches anything already redeemable), then 30 s after every 5-minute
    window boundary thereafter.
    """
    from datetime import datetime, timezone as _tz

    _do_redeem(config, log_fn)

    while not stop_event.is_set():
        now = datetime.now(_tz.utc)
        secs_past = (now.minute % 5) * 60 + now.second
        secs_to_next_boundary = (5 * 60) - secs_past
        wait = secs_to_next_boundary + 30  # 30 s grace after window close
        if stop_event.wait(wait):
            break
        _do_redeem(config, log_fn)


def _do_redeem(config, log_fn) -> None:
    """Call poly_web3 redeem_all() using bot credentials. Logs result via SSE."""
    if not config.builder_api_key or not config.builder_api_secret or not config.builder_api_passphrase:
        log_fn("warn", "[AUTO-REDEEM] Skipped — Builder API credentials not configured")
        return
    if not config.private_key:
        log_fn("warn", "[AUTO-REDEEM] Skipped — no private key")
        return
    try:
        from py_builder_relayer_client.client import RelayClient          # type: ignore
        from py_builder_signing_sdk.config import BuilderConfig           # type: ignore
        from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds  # type: ignore
        from py_clob_client.client import ClobClient                     # type: ignore
        from poly_web3 import RELAYER_URL, PolyWeb3Service               # type: ignore
    except ImportError as exc:
        log_fn("warn", f"[AUTO-REDEEM] poly-web3 not installed ({exc}) — auto-redeem disabled")
        return
    try:
        clob_client = ClobClient(
            host="https://clob.polymarket.com",
            key=config.private_key,
            chain_id=137,
            signature_type=config.signature_type,
            funder=config.funder_address or None,
        )
        clob_client.set_api_creds(clob_client.create_or_derive_api_creds())
        relayer_client = RelayClient(
            RELAYER_URL,
            137,
            config.private_key,
            BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=config.builder_api_key,
                    secret=config.builder_api_secret,
                    passphrase=config.builder_api_passphrase,
                )
            ),
        )
        service = PolyWeb3Service(clob_client=clob_client, relayer_client=relayer_client)
        results  = service.redeem_all(batch_size=10)
        redeemed = [r for r in (results or []) if r is not None]
        skipped  = len(results or []) - len(redeemed)
        if redeemed or skipped:
            msg = (
                f"[AUTO-REDEEM] {len(redeemed)} tx(s) submitted"
                + (f", {skipped} position(s) skipped" if skipped else "")
            )
            log_fn("copy" if redeemed else "info", msg)
    except Exception as exc:
        log_fn("error", f"[AUTO-REDEEM] Failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CopyTradingBot — main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class CopyTradingBot:
    """
    Runs the copy-trading polling loop in a thread.

    Parameters
    ----------
    config : BotConfig
    log_callback : callable receiving dict log entries (for SSE streaming)

    Thread-safe: all public methods (run, stop) are safe to call from any thread.
    """

    def __init__(
        self,
        config: BotConfig,
        log_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self._log_cb = log_callback or (lambda _: None)
        self.running = False
        self._stop_event = threading.Event()

        self.monitor = TradeMonitor(config)        # kept as fallback
        self.chain_monitor = ChainMonitor(config, log_fn=self._log)  # primary: blockchain events
        self.risk = RiskManager(config)
        self.executor = TradeExecutor(config)
        self.notifier = Notifier(config)

        # Trade journal — in-memory list of all copied trades with PnL tracking
        self.trade_journal: list = []
        self._journal_lock = threading.Lock()
        self._pnl_session = _make_session()

        self.stats: Dict[str, Any] = {
            "trades_detected": 0,
            "trades_copied": 0,
            "trades_skipped": 0,
            "total_usd_copied": 0.0,
            "errors": 0,
            "started_at": None,
            # Poll cycle timing
            "last_poll_ms": 0.0,
            # Latency tracking (ms)
            "latency_samples": 0,
            "avg_detection_ms": 0.0,   # trade match_time → bot detects it
            "min_detection_ms": float("inf"),
            "max_detection_ms": 0.0,
            "avg_copy_ms": 0.0,        # trade match_time → order submitted
            "min_copy_ms": float("inf"),
            "max_copy_ms": 0.0,
        }

        # Auto-derive wallet address if not provided (skipped in dry-run without key)
        if not self.config.my_wallet and self.config.private_key:
            addr = self.executor.derive_wallet_address()
            if addr:
                self.config.my_wallet = addr
                self._log("info", f"Wallet derived from private key: {addr}")
        if not self.config.my_wallet and self.config.dry_run:
            self._log("warn", "No wallet address — exposure check skipped (dry-run only)")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Entry point (called in a daemon thread by app.py).

        Architecture: producer-consumer with a shared queue.

          Thread A  —  _poll_loop()   : HTTP poller, runs as fast as network
                                        allows, enqueues raw trade dicts.
                                        NEVER blocked by order execution.

          Thread B  —  this method    : dequeues trades, evaluates risk,
                                        places CLOB orders (up to 700 ms each
                                        in live mode). While placing an order,
                                        Thread A keeps polling uninterrupted.

        In dry-run the processor is near-instant, but the separation is still
        correct and costs nothing.
        """
        self.running = True
        self._stop_event.clear()
        self.stats["started_at"] = _now_iso()

        mode_label = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        self._log("info", f"Bot started | mode={mode_label} | target={self.config.target_wallet}")
        self._log("info", self.config.safe_repr())
        self._log("info", "Producer-consumer threads active: chain poller (blockchain events) + processor running independently")

        if self.config.dry_run:
            self._log("info", "DRY RUN active — all orders are simulated. No funds will move.")

        # Bounded queue — if the processor falls far behind, old trades are dropped
        trade_queue: queue.Queue = queue.Queue(maxsize=500)

        # Thread A — HTTP poller (producer)
        poller = threading.Thread(
            target=self._poll_loop,
            args=(trade_queue,),
            name="TradePoller",
            daemon=True,
        )
        poller.start()

        # Thread C — auto-redeemer (optional, only when auto_redeem=True)
        if self.config.auto_redeem:
            redeemer = threading.Thread(
                target=self._auto_redeem_loop,
                name="AutoRedeem",
                daemon=True,
            )
            redeemer.start()
            self._log("info", "[AUTO-REDEEM] Background redeemer started — will run every 5 min after each window close")

        # Thread B — trade processor (consumer, runs in the calling thread)
        _last_pnl_refresh = 0.0
        while not self._stop_event.is_set():
            try:
                trade = trade_queue.get(timeout=1.0)
            except queue.Empty:
                # Periodically refresh PnL for live open positions
                if time.monotonic() - _last_pnl_refresh > 30:
                    self._refresh_journal_pnl()
                    _last_pnl_refresh = time.monotonic()
                continue  # re-check stop_event
            try:
                self.stats["trades_detected"] += 1
                self._process_trade(trade)
            except Exception as exc:
                self.stats["errors"] += 1
                self._log("error", f"Trade processing error: {exc}")

        self.running = False
        self._log("info", f"Bot stopped | stats={self.stats}")

    def _poll_loop(self, trade_queue: queue.Queue) -> None:
        """
        Thread A — HTTP poller (producer).

        Polls data-api.polymarket.com as fast as the network allows and puts
        new BTC-5min trades onto trade_queue.  Never touches the CLOB or does
        any order logic, so it is never stalled by order execution latency.
        """
        consecutive_errors = 0
        while not self._stop_event.is_set():
            poll_start = time.monotonic()
            try:
                new_trades = self.chain_monitor.get_new_trades()
                for trade in new_trades:
                    try:
                        trade_queue.put_nowait(trade)
                    except queue.Full:
                        self._log("warn", "Trade queue full — oldest unprocessed trade dropped")
                consecutive_errors = 0

            except Exception as exc:
                consecutive_errors += 1
                self.stats["errors"] += 1
                self._log("error", f"Poll error #{consecutive_errors}: {exc}")
                backoff = min(2 ** consecutive_errors, 60)
                self._log("warn", f"Poller backing off {backoff}s after error")
                self._stop_event.wait(backoff)
                continue

            # Track poll cycle time (HTTP round-trip only, no order latency)
            cycle_ms = (time.monotonic() - poll_start) * 1000
            self.stats["last_poll_ms"] = round(cycle_ms, 1)

            # Pace polls to ~1 per block (Polygon ≈ 2.1 s) to avoid RPC quota errors
            elapsed = time.monotonic() - poll_start
            wait = max(0.0, self.config.chain_poll_interval - elapsed)
            if wait > 0:
                self._stop_event.wait(wait)

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False

    # ── Auto-redeem ───────────────────────────────────────────────────────────

    def _auto_redeem_loop(self) -> None:
        _auto_redeem_loop(self.config, self._log, self._stop_event)

    def _do_redeem(self) -> None:
        _do_redeem(self.config, self._log)

    # ── Latency helpers ───────────────────────────────────────────────────────

    def _ms_since(self, t: Optional[datetime]) -> Optional[float]:
        """Milliseconds elapsed since a UTC datetime, or None."""
        if t is None:
            return None
        return (datetime.now(timezone.utc) - t).total_seconds() * 1000

    def _record_latency(self, key_prefix: str, ms: float) -> None:
        """Update rolling min/max/avg for a latency bucket."""
        n = self.stats["latency_samples"]
        avg_key = f"avg_{key_prefix}_ms"
        min_key = f"min_{key_prefix}_ms"
        max_key = f"max_{key_prefix}_ms"
        # Welford running average
        self.stats[avg_key] += (ms - self.stats[avg_key]) / max(n, 1)
        self.stats[min_key] = min(self.stats[min_key], ms)
        self.stats[max_key] = max(self.stats[max_key], ms)

    # ── Trade journal ──────────────────────────────────────────────────────────

    def get_trade_journal(self) -> list:
        """Return a snapshot of the trade journal (thread-safe copy)."""
        with self._journal_lock:
            return list(self.trade_journal)

    def _append_journal(
        self,
        trade: dict,
        side: str,
        copy_size: float,
        entry_price: float,
        cost_usd: float,
    ) -> None:
        """Record a copied trade in the journal."""
        their_size = _safe_float(trade.get("size", 0))
        their_cost = round(their_size * entry_price, 4)
        entry = {
            "id":           trade.get("transactionHash", ""),
            "entry_time":   _now_iso(),
            "market":       trade.get("title", "Unknown"),
            "slug":         trade.get("slug") or trade.get("eventSlug", ""),
            "outcome":      trade.get("outcome", "?"),
            "side":         side,
            # ── Target's trade ───────────────────────────────────────────────
            "their_size":   round(their_size, 4),
            "their_cost":   their_cost,
            "their_pnl":    0.0,
            # ── Our copy trade ───────────────────────────────────────────────
            "our_size":     round(copy_size, 4),
            "entry_price":  round(entry_price, 4),
            "cost_usd":     round(cost_usd, 4),
            "token_id":     trade.get("asset", ""),
            "status":       "SIMULATED" if self.config.dry_run else "OPEN",
            "cur_price":    round(entry_price, 4),
            "pnl_usd":      0.0,
            "dry_run":      self.config.dry_run,
            "source":       "copy_bot",
        }
        with self._journal_lock:
            self.trade_journal.append(entry)
            # Cap at 500 entries to avoid unbounded growth
            if len(self.trade_journal) > 500:
                self.trade_journal = self.trade_journal[-500:]

    def _refresh_journal_pnl(self) -> None:
        """
        Update cur_price and P/L for all live (OPEN) and simulated (SIMULATED)
        journal entries by querying the CLOB midpoint endpoint.
        Called every ~30 s from the processor loop.

        Status transitions:
          OPEN      → WON / LOST / OPEN            (live)
          SIMULATED → SIM_WON / SIM_LOST / SIMULATED  (dry-run)
        """
        # Retroactively fill in market titles for entries that were journaled
        # before their token_id appeared in the metadata cache.
        meta_cache = self.chain_monitor._metadata_cache
        with self._journal_lock:
            for e in self.trade_journal:
                if e.get("market") in ("Unknown", "Unknown Market", "") and e.get("token_id"):
                    meta = meta_cache.get(e["token_id"])
                    if meta:
                        e["market"]  = meta.get("title", e["market"])
                        e["slug"]    = meta.get("slug",  e["slug"])
                        e["outcome"] = meta.get("outcome", e["outcome"])

        with self._journal_lock:
            active = [e for e in self.trade_journal
                      if e["status"] in ("OPEN", "SIMULATED")]
        if not active:
            return
        for entry in active:
            token_id = entry["token_id"]
            if not token_id:
                continue
            try:
                resp = self._pnl_session.get(
                    f"{self.config.clob_host}/midpoint",
                    params={"token_id": token_id},
                    timeout=8,
                )
                if resp.ok:
                    # Use 0.0 as fallback — do NOT fall back to entry["cur_price"]
                    # because that hides settled markets (mid=null → stays OPEN forever)
                    mid = _safe_float(resp.json().get("mid"), 0.0)
                    if mid <= 0:
                        # CLOB has no live quote — market may be expired; try gamma settlement
                        self._try_settle_from_gamma(entry)
                        continue
                    is_sim = entry.get("dry_run", False)
                    is_sell = entry.get("side", "BUY").upper() == "SELL"
                    # BUY profits when price rises; SELL profits when price falls
                    sign      = -1 if is_sell else 1
                    pnl       = sign * (mid - entry["entry_price"]) * entry["our_size"]
                    their_pnl = sign * (mid - entry["entry_price"]) * entry.get("their_size", 0)
                    # WIN condition is also inverted for SELLs
                    won  = (mid <= 0.01) if is_sell else (mid >= 0.99)
                    lost = (mid >= 0.99) if is_sell else (mid <= 0.01)
                    if won:
                        status = "SIM_WON" if is_sim else "WON"
                    elif lost:
                        status = "SIM_LOST" if is_sim else "LOST"
                    else:
                        status = "SIMULATED" if is_sim else "OPEN"
                    with self._journal_lock:
                        entry["cur_price"] = round(mid, 4)
                        entry["pnl_usd"]   = round(pnl, 4)
                        entry["their_pnl"] = round(their_pnl, 4)
                        entry["status"]    = status

                    # ── Take-profit check ─────────────────────────────────────
                    take_profit = getattr(self.config, "take_profit_pct", 0.0)
                    if take_profit > 0 and status in ("OPEN", "SIMULATED"):
                        roi = (mid - entry["entry_price"]) / entry["entry_price"] \
                              if not is_sell \
                              else (entry["entry_price"] - mid) / entry["entry_price"]
                        if roi >= take_profit:
                            self._take_profit_exit(entry, mid, roi)
            except Exception as exc:
                logger.debug("PnL refresh error for %s: %s", token_id[:16], exc)

    def _take_profit_exit(self, entry: dict, cur_price: float, roi: float) -> None:
        """Place an exit order when take-profit ROI threshold is reached."""
        roi_pct = round(roi * 100, 1)
        token_id = entry.get("token_id", "")
        our_size = entry.get("our_size", 0.0)
        side = entry.get("side", "BUY").upper()
        exit_side = "SELL" if side == "BUY" else "BUY"
        msg = (
            f"[TAKE PROFIT] {entry.get('outcome','?')} {our_size:.2f}sh "
            f"entry={entry['entry_price']:.4f} now={cur_price:.4f} ROI={roi_pct}%"
        )
        is_sim = entry.get("dry_run", self.config.dry_run)
        if is_sim:
            with self._journal_lock:
                entry["status"] = "SIM_PROFIT_TAKEN"
            self._log("dry_run", f"[DRY PROFIT] {msg}")
            return
        try:
            self.executor.place_order(token_id, exit_side, our_size, cur_price)
            with self._journal_lock:
                entry["status"] = "PROFIT_TAKEN"
            self._log("copy", f"[TAKE PROFIT] {msg} — exit placed")
            self.notifier.send(msg)
        except Exception as exc:
            self._log("error", f"[TAKE PROFIT] Exit order failed: {exc}")

    def _try_settle_from_gamma(self, entry: dict) -> None:
        """
        When CLOB has no live mid (expired/settled market), query Gamma API
        outcomePrices to determine the final settlement result.
        """
        slug = entry.get("slug", "")
        outcome = (entry.get("outcome") or "").lower().strip()
        if not slug or not outcome:
            return
        try:
            resp = self._pnl_session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            if not resp.ok:
                return
            data = resp.json()
            market = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if not market:
                return
            raw_outcomes = [o.lower().strip() for o in market.get("outcomes", [])]
            raw_prices   = market.get("outcomePrices", [])
            if len(raw_outcomes) != len(raw_prices):
                return
            prices = [_safe_float(p, 0.5) for p in raw_prices]
            idx = raw_outcomes.index(outcome) if outcome in raw_outcomes else -1
            if idx < 0:
                return
            p = prices[idx]
            if p < 0.05 or p > 0.95:  # settled
                is_sim  = entry.get("dry_run", False)
                is_sell = entry.get("side", "BUY").upper() == "SELL"
                won  = (p <= 0.05) if is_sell else (p >= 0.95)
                lost = (p >= 0.95) if is_sell else (p <= 0.05)
                status = ("SIM_WON" if is_sim else "WON") if won else \
                         ("SIM_LOST" if is_sim else "LOST") if lost else None
                if status:
                    sign = -1 if is_sell else 1
                    pnl  = sign * (p - entry["entry_price"]) * entry["our_size"] \
                           if entry.get("entry_price") and entry.get("our_size") else 0.0
                    their_pnl = sign * (p - entry["entry_price"]) * entry.get("their_size", 0) \
                                if entry.get("entry_price") else 0.0
                    with self._journal_lock:
                        entry["cur_price"] = round(p, 4)
                        entry["status"]    = status
                        if pnl != 0.0:
                            entry["pnl_usd"]   = round(pnl, 4)
                            entry["their_pnl"] = round(their_pnl, 4)
        except Exception as exc:
            logger.debug("Gamma settlement lookup failed for %s: %s", slug, exc)

    # ── Trade processing ──────────────────────────────────────────────────────

    def _process_trade(self, trade: dict) -> None:
        detected_at = datetime.now(timezone.utc)   # stamp immediately on entry
        token_id = trade.get("asset", "")
        side = trade.get("side", "BUY").upper()
        their_size = _safe_float(trade.get("size", 0))
        their_price = _safe_float(trade.get("price", 0))
        their_usd = their_size * their_price
        outcome = trade.get("outcome", "?")
        title = trade.get("title", "Unknown")[:55]

        # ── Compute detection latency ─────────────────────────────────────────
        trade_time = _parse_trade_time(trade)
        detection_ms = (
            (detected_at - trade_time).total_seconds() * 1000
            if trade_time else None
        )
        if detection_ms is not None:
            self.stats["latency_samples"] += 1
            self._record_latency("detection", detection_ms)

        lat_str = f" | detect_lag={detection_ms:.0f}ms" if detection_ms is not None else ""

        self._log(
            "trade",
            f"[DETECTED] Target {side} {their_size:.0f} {outcome} @ ${their_price:.4f}"
            f" → ~${their_usd:.2f} | {title}{lat_str}",
            {
                "trade_id": trade.get("transactionHash", ""),
                "token_id": token_id[:20],
                "detection_ms": detection_ms,
                "trade_matched_at": trade_time.isoformat() if trade_time else None,
            },
        )

        # ── Risk gate ────────────────────────────────────────────────────────
        should_copy, copy_size, reason, _ = self.risk.evaluate(trade)
        copy_usd = copy_size * their_price

        if not should_copy:
            self.stats["trades_skipped"] += 1
            skip_msg = (
                f"[SKIP] Target {side} {their_size:.0f} {outcome} @ ${their_price:.4f}"
                f" → ~${their_usd:.2f}{lat_str} | Reason: {reason}"
            )
            self._log("skip", skip_msg, {"reason": reason})
            self.notifier.send(skip_msg)
            return

        # ── Build copy summary ────────────────────────────────────────────────
        pct = self.config.copy_multiplier * 100
        copy_msg = (
            f"[{'DRY_RUN' if self.config.dry_run else 'COPY'}]"
            f" Target {side} {their_size:.0f} {outcome} @ ${their_price:.4f} → ~${their_usd:.2f}\n"
            f"  ↳ side={side}, token={token_id[:16]}…, "
            f"size={copy_size:.2f} shares, est_USD=${copy_usd:.2f} ({pct:.0f}% scale)"
        )
        if reason != "ok":
            copy_msg += f"\n  ⚠ Adjusted: {reason}"
        if detection_ms is not None:
            copy_msg += f"\n  ⏱ detect_lag={detection_ms:.0f}ms"

        order_detail = {
            "token_id": token_id,
            "side": side,
            "size": copy_size,
            "ref_price": their_price,
            "est_usd": copy_usd,
            "outcome": outcome,
            "market_title": trade.get("title", ""),
            "detection_ms": detection_ms,
        }

        # ── Dry run path ──────────────────────────────────────────────────────
        if self.config.dry_run:
            # In dry-run: copy_ms = time from trade_match to "would submit"
            copy_ms = self._ms_since(trade_time)
            if copy_ms is not None:
                self._record_latency("copy", copy_ms)
                copy_msg += f" | copy_lag={copy_ms:.0f}ms"
                order_detail["copy_ms"] = copy_ms
            self._log("dry_run", copy_msg, order_detail)
            self.notifier.send(f"[DRY_RUN] {copy_msg}")
            self.stats["trades_copied"] += 1
            self.stats["total_usd_copied"] += copy_usd
            self._append_journal(trade, side, copy_size, their_price, copy_usd)
            return

        # ── Live execution ────────────────────────────────────────────────────
        try:
            response = self.executor.place_order(token_id, side, copy_size, their_price, use_gtc=True)
            # copy_ms measured after order is submitted to CLOB
            copy_ms = self._ms_since(trade_time)
            if copy_ms is not None:
                self._record_latency("copy", copy_ms)
                copy_msg += f"\n  ⏱ copy_lag={copy_ms:.0f}ms"
                order_detail["copy_ms"] = copy_ms
            self.stats["trades_copied"] += 1
            self.stats["total_usd_copied"] += copy_usd
            self._append_journal(trade, side, copy_size, their_price, copy_usd)
            self._log(
                "copy",
                f"{copy_msg}\n  ✓ CLOB response: {response}",
                {**order_detail, "clob_response": response},
            )
            self.notifier.send(copy_msg)
        except Exception as exc:
            self.stats["errors"] += 1
            _eoa    = self.executor.derive_wallet_address() or "?"
            _funder = getattr(self.executor.config, "funder_address", "") or _eoa
            _sigt   = getattr(self.executor.config, "signature_type", "?")
            err_msg = (
                f"[ERROR] Order placement failed for {token_id[:16]}…: {exc}"
                f" | attempted: {copy_size:.2f}sh × ${their_price:.4f} ≈ ${copy_usd:.2f} USDC"
                f" | sig_type={_sigt} eoa={_eoa[:10]}… funder={_funder[:10]}…"
            )
            self._log("error", err_msg, {"token_id": token_id, "error": str(exc),
                                          "attempted_usdc": round(copy_usd, 2),
                                          "attempted_shares": round(copy_size, 4)})
            self.notifier.send(err_msg)

    # ── Logging helper ────────────────────────────────────────────────────────

    def _log(
        self,
        log_type: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        entry = {
            "type": log_type,
            "timestamp": _now_iso(),
            "message": message,
            "data": data or {},
        }
        # Route to Python's stdlib logger
        {
            "error": logger.error,
            "skip": logger.warning,
            "warn": logger.warning,
        }.get(log_type, logger.info)(message)

        # Route to web UI / SSE callback
        try:
            self._log_cb(entry)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# AdvancedTrader — spread trading
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedTrader:
    """
    Swing reversal strategy for BTC 5-minute markets.

    SWING REVERSAL:
        Every second, fetch the current BTC price vs window open (price-to-beat).
        When a reversal pattern is detected on 1m Binance candles, buy the OPPOSITE
        side cheap — betting BTC reverts before the window closes.
        Position monitor auto-SELLs when mid reaches exit_target.
        Example: BTC up 0.3% from window open → buy Down@0.30 → auto-exit at 0.50 → 67% ROI.
    """

    POLL_INTERVAL_SEC    = 1    # scan for opportunities
    MONITOR_INTERVAL_SEC = 3    # check open positions for exits
    _TF_WINDOWS          = {"5m": 300, "15m": 900, "1h": 3600}

    def __init__(
        self,
        config: AdvancedTraderConfig,
        log_callback: Optional[Callable] = None,
    ) -> None:
        self.config        = config
        self._log_cb       = log_callback or (lambda _: None)
        self.running       = False
        self.WINDOW_SEC    = self._TF_WINDOWS.get(getattr(config, "market_timeframe", "5m"), 300)
        self._stop_event   = threading.Event()
        self._session      = _make_session()
        self._pnl_session  = _make_session()
        self.executor      = TradeExecutor(config)
        self.trade_journal: list = []
        self._journal_lock = threading.Lock()
        self._last_monitor_ts: float = 0.0
        # Cache current market for the lifetime of a window (avoids gamma-api hit every tick)
        self._market_cache: Optional[dict] = None
        self._market_cache_window: Optional[int] = None
        # One swing-reversal entry per window
        self._last_reversal_window: Optional[int] = None
        # 5-second cache for intra-window BTC price fetch
        self._btc_price_cache: dict = {"data": None, "ts": 0.0}
        self.stats = {"trades_entered": 0, "exits_taken": 0, "errors": 0}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        self._stop_event.clear()
        mode = "DRY RUN" if self.config.dry_run else "LIVE"
        self._log("info",
            f"[ADVANCED] Started | strategy=swing_reversal | mode={mode} | "
            f"{self.config.safe_repr()}"
        )
        if self.config.dry_run:
            self._log("info", "[ADVANCED] DRY RUN — no orders will be placed.")
        if self.config.auto_redeem:
            threading.Thread(
                target=_auto_redeem_loop,
                args=(self.config, self._log, self._stop_event),
                name="AdvancedAutoRedeem",
                daemon=True,
            ).start()
            self._log("info", "[ADVANCED] Auto-redeem background thread started")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self.stats["errors"] += 1
                self._log("error", f"[ADVANCED] Tick error: {exc}")
            if time.monotonic() - self._last_monitor_ts > self.MONITOR_INTERVAL_SEC:
                try:
                    self._monitor_positions()
                except Exception as exc:
                    self._log("error", f"[ADVANCED] Monitor error: {exc}")
                self._last_monitor_ts = time.monotonic()
            self._stop_event.wait(self.POLL_INTERVAL_SEC)
        self.running = False
        self._log("info", f"[ADVANCED] Stopped. Stats: {self.stats}")

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False

    def get_trade_journal(self) -> list:
        with self._journal_lock:
            return list(self.trade_journal)

    # ── Core tick ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = int(time.time())
        window_start = (now // self.WINDOW_SEC) * self.WINDOW_SEC
        secs_remaining = self.WINDOW_SEC - (now - window_start)

        if secs_remaining < self.config.min_window_secs_remaining:
            self._log("skip",
                f"[ADVANCED] SKIP: {secs_remaining}s left < min {self.config.min_window_secs_remaining}s")
            return

        self._try_swing_reversal(window_start)

    # ── Position monitor (take-profit + exit target) ───────────────────────────

    def _monitor_positions(self) -> None:
        """Check all OPEN/SIMULATED journal entries; place exits when targets are hit."""
        with self._journal_lock:
            active = [e for e in self.trade_journal
                      if e["status"] in ("OPEN", "SIMULATED")]
        for entry in active:
            token_id = entry.get("token_id", "")
            if not token_id:
                continue
            try:
                resp = self._pnl_session.get(
                    f"{self.config.clob_host}/midpoint",
                    params={"token_id": token_id},
                    timeout=6,
                )
                if not resp.ok:
                    continue
                mid = _safe_float(resp.json().get("mid"), 0.0)
                if mid <= 0:
                    self._try_settle_from_gamma(entry)
                    continue

                roi = (mid - entry["entry_price"]) / entry["entry_price"] \
                      if entry["entry_price"] > 0 else 0.0
                pnl = (mid - entry["entry_price"]) * entry["our_size"]
                won  = mid >= 0.99
                lost = mid <= 0.01
                is_sim = entry.get("dry_run", self.config.dry_run)
                status = ("SIM_WON" if is_sim else "WON") if won else \
                         ("SIM_LOST" if is_sim else "LOST") if lost else \
                         ("SIMULATED" if is_sim else "OPEN")
                with self._journal_lock:
                    entry["cur_price"] = round(mid, 4)
                    entry["pnl_usd"]   = round(pnl, 4)
                    entry["status"]    = status

                # Exit target hit
                exit_target = entry.get("exit_target", 0.0)
                if exit_target > 0 and mid >= exit_target and status in ("OPEN", "SIMULATED"):
                    self._exit_position(entry, mid, f"exit target {exit_target:.4f} reached")
                    continue

                # Generic take-profit exit
                tp = self.config.take_profit_pct
                if tp > 0 and roi >= tp and status in ("OPEN", "SIMULATED"):
                    self._exit_position(entry, mid, f"take-profit {tp*100:.1f}% reached")
            except Exception as exc:
                logger.debug("AdvancedTrader monitor error for %s: %s", token_id[:16], exc)

    def _exit_position(self, entry: dict, cur_price: float, reason: str) -> None:
        """Place SELL exit; update journal status."""
        roi_pct = round((cur_price - entry["entry_price"]) / entry["entry_price"] * 100, 1) \
                  if entry["entry_price"] > 0 else 0.0
        msg = (
            f"[ADV EXIT] {entry.get('outcome','?')} {entry['our_size']:.2f}sh "
            f"entry={entry['entry_price']:.4f} now={cur_price:.4f} ROI={roi_pct}% | {reason}"
        )
        is_sim = entry.get("dry_run", self.config.dry_run)
        if is_sim:
            self._log("dry_run", f"[DRY EXIT] {msg}")
            with self._journal_lock:
                entry["status"] = "SIM_PROFIT_TAKEN"
            self.stats["exits_taken"] += 1
            return
        try:
            self.executor.place_order(entry["token_id"], "SELL", entry["our_size"], cur_price)
            with self._journal_lock:
                entry["status"] = "PROFIT_TAKEN"
            self.stats["exits_taken"] += 1
            self._log("copy", f"[ADV EXIT] {msg} — SELL placed")
        except Exception as exc:
            self._log("error", f"[ADV EXIT] SELL failed: {exc}")

    def _try_settle_from_gamma(self, entry: dict) -> None:
        """When CLOB has no live mid, query Gamma API outcomePrices for settlement."""
        slug = entry.get("slug", "")
        outcome = (entry.get("outcome") or "").lower().strip()
        if not slug or not outcome:
            return
        try:
            resp = self._pnl_session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            if not resp.ok:
                return
            data = resp.json()
            market = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
            if not market:
                return
            raw_outcomes = [o.lower().strip() for o in market.get("outcomes", [])]
            raw_prices   = market.get("outcomePrices", [])
            if len(raw_outcomes) != len(raw_prices):
                return
            prices = [_safe_float(p, 0.5) for p in raw_prices]
            idx = raw_outcomes.index(outcome) if outcome in raw_outcomes else -1
            if idx < 0:
                return
            p = prices[idx]
            if p < 0.05 or p > 0.95:
                is_sim = entry.get("dry_run", self.config.dry_run)
                won  = p >= 0.95
                lost = p <= 0.05
                status = ("SIM_WON" if is_sim else "WON") if won else \
                         ("SIM_LOST" if is_sim else "LOST") if lost else None
                if status:
                    pnl = (p - entry["entry_price"]) * entry["our_size"] \
                          if entry.get("entry_price") and entry.get("our_size") else 0.0
                    with self._journal_lock:
                        entry["cur_price"] = round(p, 4)
                        entry["status"]    = status
                        if pnl != 0.0:
                            entry["pnl_usd"] = round(pnl, 4)
        except Exception as exc:
            logger.debug("Gamma settlement lookup failed for %s: %s", slug, exc)

    # ── Market data helpers ────────────────────────────────────────────────────

    def _get_current_market(self) -> dict:
        """Resolve current window token IDs from gamma-api (cached per window)."""
        import json as _json
        from datetime import datetime, timezone, timedelta
        window_start = (int(time.time()) // self.WINDOW_SEC) * self.WINDOW_SEC
        if self._market_cache is not None and self._market_cache_window == window_start:
            return self._market_cache
        tf = getattr(self.config, "market_timeframe", "5m")
        if tf == "1h":
            slugs_to_try = [PredictionTrader._1h_slug(window_start)]
            dt_est = datetime.fromtimestamp(window_start, tz=timezone.utc) + timedelta(hours=-5)
            h12 = dt_est.hour % 12 or 12
            ampm = "am" if dt_est.hour < 12 else "pm"
            slug_est = f"bitcoin-up-or-down-{dt_est.strftime('%B').lower()}-{dt_est.day}-{h12}{ampm}-et"
            if slug_est != slugs_to_try[0]:
                slugs_to_try.append(slug_est)
        else:
            slugs_to_try = [f"btc-updown-{tf}-{window_start}"]
        slug = slugs_to_try[0]
        data = None
        for try_slug in slugs_to_try:
            try:
                resp = self._session.get(
                    f"{self.config.gamma_api_base}/markets",
                    params={"slug": try_slug},
                    timeout=8,
                )
                resp.raise_for_status()
                candidate = resp.json()
                if candidate:
                    data = candidate
                    slug = try_slug
                    break
            except Exception:
                continue
        if not data:
            raise RuntimeError(f"Market not listed yet: {slugs_to_try}")
        market = data[0] if isinstance(data, list) else data
        raw_outcomes = market.get("outcomes", "[]")
        raw_ids      = market.get("clobTokenIds", "[]")
        if isinstance(raw_outcomes, str):
            raw_outcomes = _json.loads(raw_outcomes)
        if isinstance(raw_ids, str):
            raw_ids = _json.loads(raw_ids)
        outcome_map: Dict[str, str] = {}
        for i, outcome in enumerate(raw_outcomes):
            if i >= len(raw_ids):
                continue
            o = str(outcome).strip().lower()
            if o in ("up", "down"):
                outcome_map[o] = str(raw_ids[i])
        result = {"slug": slug, "window_start": window_start, "outcomes": outcome_map}
        self._market_cache = result
        self._market_cache_window = window_start
        return result

    def _get_book_prices(self, token_id: str) -> dict:
        """Fetch best bid/ask from CLOB order book."""
        resp = self._session.get(
            f"{self.config.clob_host}/book",
            params={"token_id": token_id},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        best_bid = max((_safe_float(b.get("price"), 0.0) for b in bids), default=0.0)
        ask_prices = [_safe_float(a.get("price"), 0.0) for a in asks if _safe_float(a.get("price"), 0.0) > 0]
        best_ask = min(ask_prices) if ask_prices else 0.0
        spread = round(best_ask - best_bid, 4) if best_bid > 0 and best_ask > 0 else 0.0
        return {"best_bid": round(best_bid, 4), "best_ask": round(best_ask, 4), "spread": spread}

    # ── Candlestick pattern detection ─────────────────────────────────────────

    def _fetch_btc_1m_candles(self) -> Optional[list]:
        """
        Fetch last ~20 completed 1-minute BTC/USDT candles from Binance.
        Caches result for 10 seconds (candles only change every 60s).
        Returns list of dicts (oldest → newest, last element is latest COMPLETE candle).
        """
        now_mono = time.monotonic()
        cached = self._btc_price_cache
        if now_mono - cached["ts"] < 10.0 and cached["data"] is not None:
            return cached["data"]

        candles = None

        def _parse_binance(rows):
            # rows[-1] is the still-open current candle — exclude it
            result = []
            for r in rows[:-1]:
                result.append({
                    "open_time": int(r[0]),
                    "open":  float(r[1]), "high": float(r[2]),
                    "low":   float(r[3]), "close": float(r[4]),
                    "volume": float(r[5]),
                })
            return result or None

        # Source 1: Binance global
        try:
            r = self._session.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": "22"},
                timeout=5,
            )
            r.raise_for_status()
            candles = _parse_binance(r.json())
        except Exception:
            pass

        # Source 2: Binance US
        if candles is None:
            try:
                r = self._session.get(
                    "https://api.binance.us/api/v3/klines",
                    params={"symbol": "BTCUSDT", "interval": "1m", "limit": "22"},
                    timeout=5,
                )
                r.raise_for_status()
                candles = _parse_binance(r.json())
            except Exception:
                pass

        # Source 3: Bybit (returns newest first → reverse)
        if candles is None:
            try:
                r = self._session.get(
                    "https://api.bybit.com/v5/market/kline",
                    params={"category": "spot", "symbol": "BTCUSDT",
                            "interval": "1", "limit": "22"},
                    timeout=5,
                )
                r.raise_for_status()
                rows = r.json().get("result", {}).get("list", [])
                if rows and len(rows) > 1:
                    rows = list(reversed(rows[1:]))  # skip current open candle, oldest first
                    candles = [{"open_time": int(rw[0]), "open": float(rw[1]),
                                "high": float(rw[2]), "low": float(rw[3]),
                                "close": float(rw[4]), "volume": float(rw[5])}
                               for rw in rows]
            except Exception:
                pass

        self._btc_price_cache = {"data": candles, "ts": now_mono}
        return candles

    def _compute_atr14(self, candles: list) -> float:
        """Average True Range over last 14 candles. Returns 0 if insufficient data."""
        if len(candles) < 2:
            return 0.0
        tr_values = []
        for i in range(1, len(candles)):
            h  = candles[i]["high"]
            lo = candles[i]["low"]
            pc = candles[i - 1]["close"]
            tr_values.append(max(h - lo, abs(h - pc), abs(lo - pc)))
        period = min(14, len(tr_values))
        return sum(tr_values[-period:]) / period

    def _compute_rsi14(self, candles: list) -> Optional[float]:
        """RSI(14) using Wilder's smoothed average. Returns None if < 15 candles."""
        if len(candles) < 15:
            return None
        closes = [c["close"] for c in candles[-15:]]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    def _score_signal(
        self,
        pattern: dict,
        candles: list,
        atr: float,
        rsi: Optional[float],
    ) -> "tuple[int, list[str]]":
        """
        Score a detected pattern 0–100 across 5 signals:
        pattern quality, ATR size, RSI extreme, volume spike, window alignment.
        Returns (score, reasons) for logging.
        """
        score = 0
        reasons: list = []

        # ── Pattern base score ─────────────────────────────────────────────────
        strength = pattern.get("strength", 1)
        if strength >= 3:
            score += 40
            reasons.append(f"{pattern['pattern']} str=3 +40")
        elif strength >= 2:
            score += 25
            reasons.append(f"{pattern['pattern']} str=2 +25")

        # ── ATR size contribution ──────────────────────────────────────────────
        c2 = pattern.get("candle", candles[-1])
        candle_range = c2["high"] - c2["low"]
        if atr > 0:
            ratio = candle_range / atr
            if ratio >= 1.5:
                score += 20
                reasons.append(f"range={ratio:.1f}xATR +20")
            elif ratio >= 1.0:
                score += 10
                reasons.append(f"range={ratio:.1f}xATR +10")

        # ── RSI contribution ───────────────────────────────────────────────────
        signal = pattern["signal"]
        if rsi is not None:
            if signal == "bearish":
                if rsi > 70:
                    score += 25
                    reasons.append(f"RSI={rsi:.1f} overbought +25")
                elif rsi >= 65:
                    score += 15
                    reasons.append(f"RSI={rsi:.1f} near-overbought +15")
            else:
                if rsi < 30:
                    score += 25
                    reasons.append(f"RSI={rsi:.1f} oversold +25")
                elif rsi <= 35:
                    score += 15
                    reasons.append(f"RSI={rsi:.1f} near-oversold +15")

        # ── Volume contribution ────────────────────────────────────────────────
        if len(candles) >= 7:
            recent_vol = c2.get("volume", 0)
            prior_vols = [c.get("volume", 0) for c in candles[-7:-2]]
            avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio >= 1.5:
                    score += 15
                    reasons.append(f"vol={vol_ratio:.1f}x +15")
                elif vol_ratio >= 1.2:
                    score += 8
                    reasons.append(f"vol={vol_ratio:.1f}x +8")

        # ── Window alignment contribution ──────────────────────────────────────
        window_open = self._get_window_open_price(candles)
        if window_open and candles:
            current  = candles[-1]["close"]
            btc_pct  = (current - window_open) / window_open * 100
            btc_up   = current > window_open
            aligned  = (signal == "bearish" and btc_up) or (signal == "bullish" and not btc_up)
            if aligned:
                score += 15
                reasons.append(f"BTC{btc_pct:+.2f}% aligned +15")
            else:
                score -= 10
                reasons.append(f"BTC{btc_pct:+.2f}% against -10")

        # ── London open penalty ────────────────────────────────────────────────
        import datetime as _dt
        utc_hour = _dt.datetime.utcnow().hour
        if 8 <= utc_hour < 12:
            score -= 20
            reasons.append(f"London h={utc_hour}UTC -20")

        return max(0, min(score, 100)), reasons

    def _get_window_open_price(self, candles: list) -> Optional[float]:
        """Return the 1m candle open that started the current 5-min window."""
        window_start_ms = ((int(time.time()) // self.WINDOW_SEC) * self.WINDOW_SEC) * 1000
        for c in candles:
            if c["open_time"] >= window_start_ms:
                return c["open"]
        return candles[-1]["open"] if candles else None

    def _detect_reversal_pattern(self, candles: list, atr: float) -> Optional[dict]:
        """
        Check last 2 completed 1m candles for candlestick reversal patterns.
        Returns {"signal": "bearish"|"bullish", "pattern": str, "strength": 1-3} or None.
        Strength: 3=Engulfing, 2=Star/Doji/DarkCloud, 1=(unused/future).
        """
        if len(candles) < 2:
            return None
        c1 = candles[-2]   # previous completed candle
        c2 = candles[-1]   # most recently completed candle

        def body(c):    return abs(c["close"] - c["open"])
        def rng(c):     return c["high"] - c["low"]
        def is_bull(c): return c["close"] > c["open"]
        def is_bear(c): return c["close"] < c["open"]
        def upper_wick(c): return c["high"] - max(c["open"], c["close"])
        def lower_wick(c): return min(c["open"], c["close"]) - c["low"]
        def atr_ok(c, mult): return rng(c) >= atr * mult if atr > 0 else True
        atr_mult = 0.0  # ATR gate is now handled by the confidence scorer

        # ── Bearish Engulfing (strength 3) ─────────────────────────────────────
        if (is_bull(c1) and is_bear(c2)
                and c2["open"] >= c1["close"]
                and c2["close"] <= c1["open"]
                and body(c1) >= 0.3 * atr
                and body(c2) >= 0.5 * atr
                and atr_ok(c2, atr_mult)):
            return {"signal": "bearish", "pattern": "Bearish Engulfing", "strength": 3,
                    "candle": c2}

        # ── Bullish Engulfing (strength 3) ─────────────────────────────────────
        if (is_bear(c1) and is_bull(c2)
                and c2["open"] <= c1["close"]
                and c2["close"] >= c1["open"]
                and body(c1) >= 0.3 * atr
                and body(c2) >= 0.5 * atr
                and atr_ok(c2, atr_mult)):
            return {"signal": "bullish", "pattern": "Bullish Engulfing", "strength": 3,
                    "candle": c2}

        # ── Shooting Star (bearish, strength 2) ────────────────────────────────
        b2 = body(c2); r2 = rng(c2); uw2 = upper_wick(c2); lw2 = lower_wick(c2)
        if (r2 > 0 and b2 > 0
                and uw2 >= 2.0 * b2
                and lw2 <= 0.1 * b2
                and max(c2["open"], c2["close"]) < c2["low"] + 0.4 * r2
                and is_bear(c2)
                and atr_ok(c2, atr_mult)):
            return {"signal": "bearish", "pattern": "Shooting Star", "strength": 2,
                    "candle": c2}

        # ── Hammer (bullish, strength 2) ───────────────────────────────────────
        if (r2 > 0 and b2 > 0
                and lw2 >= 2.0 * b2
                and uw2 <= 0.1 * b2
                and min(c2["open"], c2["close"]) > c2["low"] + 0.6 * r2
                and is_bull(c2)
                and atr_ok(c2, atr_mult)):
            return {"signal": "bullish", "pattern": "Hammer", "strength": 2,
                    "candle": c2}

        # ── Gravestone Doji (bearish, strength 2) ──────────────────────────────
        if (r2 > 0
                and b2 / r2 <= 0.05
                and uw2 >= 2.0 * max(b2, 0.001)
                and lw2 <= 0.05 * r2
                and atr_ok(c2, max(atr_mult * 0.8, 0))):
            return {"signal": "bearish", "pattern": "Gravestone Doji", "strength": 2,
                    "candle": c2}

        # ── Dragonfly Doji (bullish, strength 2) ───────────────────────────────
        if (r2 > 0
                and b2 / r2 <= 0.05
                and lw2 >= 2.0 * max(b2, 0.001)
                and uw2 <= 0.05 * r2
                and atr_ok(c2, max(atr_mult * 0.8, 0))):
            return {"signal": "bullish", "pattern": "Dragonfly Doji", "strength": 2,
                    "candle": c2}

        # ── Dark Cloud Cover (bearish, strength 2) ─────────────────────────────
        midpoint_c1 = (c1["open"] + c1["close"]) / 2
        if (is_bull(c1) and is_bear(c2)
                and c2["open"] > c1["high"]
                and c2["close"] < midpoint_c1
                and c2["close"] > c1["open"]
                and body(c1) >= 0.5 * atr
                and body(c2) >= 0.4 * atr
                and atr_ok(c2, atr_mult)):
            return {"signal": "bearish", "pattern": "Dark Cloud Cover", "strength": 2,
                    "candle": c2}

        # ── Piercing Line (bullish, strength 2) ────────────────────────────────
        midpoint_c1b = (c1["open"] + c1["close"]) / 2
        if (is_bear(c1) and is_bull(c2)
                and c2["open"] < c1["low"]
                and c2["close"] > midpoint_c1b
                and c2["close"] < c1["open"]
                and body(c1) >= 0.5 * atr
                and body(c2) >= 0.4 * atr
                and atr_ok(c2, atr_mult)):
            return {"signal": "bullish", "pattern": "Piercing Line", "strength": 2,
                    "candle": c2}

        return None

    def _try_swing_reversal(self, window_start: int) -> None:
        """
        Entry triggered by BTC 1-minute candlestick reversal patterns.
        Each setup is scored 0-100 across pattern quality, ATR size, RSI, volume,
        and window alignment. Only fires when confidence >= aggression threshold.
        One entry per window.
        """
        if self._last_reversal_window == window_start:
            return

        # Fetch 1m candles
        candles = self._fetch_btc_1m_candles()
        if not candles or len(candles) < 2:
            return

        # Compute indicators
        atr = self._compute_atr14(candles)
        rsi = self._compute_rsi14(candles)

        # Shape-only pattern detection (ATR gate handled in scorer)
        pattern = self._detect_reversal_pattern(candles, atr)
        if pattern is None:
            return  # no pattern — silent skip to avoid log spam

        # Confidence scoring
        confidence, reasons = self._score_signal(pattern, candles, atr, rsi)

        # Aggression → minimum confidence threshold
        _THRESHOLDS = {"conservative": 80, "balanced": 65, "aggressive": 45}
        min_conf = _THRESHOLDS.get(self.config.aggression, 65)
        reasons_str = " | ".join(reasons)

        if confidence < min_conf:
            self._log("skip",
                f"[SWING] {pattern['pattern']} conf={confidence}/100 < min={min_conf} "
                f"({self.config.aggression}) — skip [{reasons_str}]")
            return

        signal = pattern["signal"]
        target_side = "down" if signal == "bearish" else "up"

        try:
            market = self._get_current_market()
        except Exception as exc:
            self._log("warn", f"[SWING] Market lookup failed: {exc}")
            return

        token_id = market["outcomes"].get(target_side)
        if not token_id:
            self._log("warn", f"[SWING] No token for side={target_side}")
            return

        try:
            book = self._get_book_prices(token_id)
        except Exception as exc:
            self._log("warn", f"[SWING] Book fetch failed: {exc}")
            return

        ask = book["best_ask"]
        entry_max = self.config.swing_entry_max
        if ask <= 0 or ask > entry_max:
            self._log("skip",
                f"[SWING] {pattern['pattern']} → {target_side.upper()} but ask={ask:.4f} > "
                f"entry_max={entry_max:.4f} — market already priced move")
            return

        slug = market["slug"]
        shares = round(self.config.trade_amount_usd / ask, 4)
        roi_potential = round((self.config.exit_target / ask - 1) * 100, 1)
        candle_pct = round(
            (pattern["candle"]["close"] - pattern["candle"]["open"])
            / pattern["candle"]["open"] * 100, 3
        ) if pattern["candle"]["open"] else 0

        self._log("info",
            f"[SWING] {pattern['pattern']} conf={confidence}/100 on 1m BTC "
            f"({candle_pct:+.3f}%) | {reasons_str} "
            f"| buying {target_side.upper()} | ask={ask:.4f} → exit={self.config.exit_target:.4f} "
            f"(~{roi_potential}% ROI)"
        )

        if self.config.dry_run:
            self._log("dry_run",
                f"[DRY SWING] Would BUY {target_side.upper()} {shares:.2f}sh@{ask:.4f} "
                f"| exit={self.config.exit_target:.4f} | {pattern['pattern']} conf={confidence}"
            )
            self._record_swing(slug, token_id, target_side.upper(), shares, ask,
                               self.config.exit_target, window_start,
                               pattern["pattern"], pattern["strength"], candle_pct,
                               confidence, dry_run=True)
        else:
            try:
                resp = self.executor.place_order(token_id, "BUY", shares, ask)
                self._record_swing(slug, token_id, target_side.upper(), shares, ask,
                                   self.config.exit_target, window_start,
                                   pattern["pattern"], pattern["strength"], candle_pct,
                                   confidence, dry_run=False)
                self._log("copy",
                    f"[SWING] BUY {target_side.upper()} {shares:.2f}sh@{ask:.4f} filled "
                    f"| {pattern['pattern']} conf={confidence} | exit={self.config.exit_target:.4f} | {resp}"
                )
            except Exception as exc:
                self._log("error", f"[SWING] Order failed: {exc}")
                return

        self.stats["trades_entered"] += 1
        self._last_reversal_window = window_start

    def _record_swing(
        self, slug: str, token_id: str, outcome: str,
        shares: float, entry_price: float, exit_target: float,
        window_start: int, pattern_name: str, pattern_strength: int,
        candle_pct: float, confidence: int, dry_run: bool,
    ) -> None:
        ts = int(time.time() * 1000)
        entry = {
            "id":               f"swing-{window_start}-{outcome.lower()}-{ts}",
            "entry_time":       _now_iso(),
            "market":           slug,
            "slug":             slug,
            "outcome":          outcome,
            "side":             "BUY",
            "their_size":       0.0,
            "their_cost":       0.0,
            "their_pnl":        0.0,
            "our_size":         round(shares, 4),
            "entry_price":      round(entry_price, 4),
            "cost_usd":         round(shares * entry_price, 4),
            "token_id":         token_id,
            "status":           "SIMULATED" if dry_run else "OPEN",
            "cur_price":        round(entry_price, 4),
            "pnl_usd":          0.0,
            "dry_run":          dry_run,
            "source":           "advanced",
            "strategy":         "swing",
            "exit_target":      round(exit_target, 4),
            "pattern_name":     pattern_name,
            "pattern_strength": pattern_strength,
            "trigger_candle_pct": candle_pct,
            "confidence":       confidence,
        }
        with self._journal_lock:
            self.trade_journal.append(entry)
            if len(self.trade_journal) > 500:
                self.trade_journal = self.trade_journal[-500:]

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log(self, log_type: str, message: str, data: Optional[dict] = None) -> None:
        entry = {
            "type":      log_type,
            "timestamp": _now_iso(),
            "message":   message,
            "data":      data or {},
        }
        {
            "error": logger.error,
            "warn":  logger.warning,
            "skip":  logger.warning,
        }.get(log_type, logger.info)(message)
        try:
            self._log_cb(entry)
        except Exception:
            pass


# (HedgeTrader removed)
