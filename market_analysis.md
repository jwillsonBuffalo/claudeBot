# market_analysis.py — Reference Document

**File:** `market_analysis.py`
**Purpose:** BTC 5-minute directional predictor for the Polymarket copy-trading bot.

---

## Overview

Fetches BTC price data, computes multi-signal technical analysis, makes a directional prediction (UP / DOWN / NEUTRAL), tracks whether past predictions were correct, and adaptively adjusts signal weights over time to improve accuracy.

### Data Sources (in priority order)

| # | Source | Method | Notes |
|---|--------|--------|-------|
| 1 | **Chainlink on-chain** | `eth_call` via Polygon JSON-RPC | Free, no API key. Contract `0xc907E116054Ad103354f2D350FD2514433D57F6f` |
| 2 | Bybit | REST `GET /v5/market/kline` | Spot 5m OHLCV |
| 3 | Kraken | REST `GET /0/public/OHLC` | 5m OHLCV |
| 4 | Binance.US | REST `GET /api/v3/klines` | 5m OHLCV |

---

## Constants

```python
CACHE_TTL_SECONDS   = 30    # analysis cached for 30s
FETCH_TIMEOUT_SECONDS = 10  # HTTP/RPC timeout

_CL_BTC_USD   = "0xc907E116054Ad103354f2D350FD2514433D57F6f"  # Chainlink BTC/USD Polygon
_CL_DECIMALS  = 8           # answer / 1e8 = USD price
_LATEST_SEL   = "0xfeaf968c"  # latestRoundData() selector
_ROUND_SEL    = "0x9a6fc8f5"  # getRoundData(uint80) selector
_CL_BATCH_ROUNDS = 80       # historical rounds fetched per refresh
_CANDLE_SEC   = 300         # 5-minute OHLCV bucket size
```

---

## Classes

### `PredictionTracker`

Tracks one pending prediction per 5-minute window. When the clock crosses into a new window, resolves the previous prediction by comparing BTC price at window-end vs window-start.

**Constants:**
- `MAX_RECORDS = 20` — rolling history size
- `WINDOW_SEC = 300` — 5-minute window

**Methods:**

| Method | Description |
|--------|-------------|
| `record(btc_price, prediction, signal_directions, unix_ts)` | Registers result for the current window. Returns a resolved `dict` if the previous window just closed, else `None`. |
| `get_stats()` | Returns `{total, correct, accuracy_pct, last_outcome, last_prediction}` |
| `_resolve(pending, close_price)` *(static)* | Determines correctness: UP correct if `close > start`, DOWN correct if `close < start`, NEUTRAL unscored. |

**Resolved record fields:**

```python
{
    "window_start":      int,    # unix timestamp floored to 300s
    "price_at_start":    float,
    "prediction":        str,    # "UP" | "DOWN" | "NEUTRAL"
    "signal_directions": dict,   # {"momentum": "UP", "rsi": "DOWN", ...}
    "resolved":          True,
    "correct":           bool | None,   # None for NEUTRAL predictions
    "price_at_close":    float,
}
```

---

### `AdaptiveWeights`

Holds the four signal weights and adjusts them after each resolved prediction.

**Default weights:** `{ema: 3.0, momentum: 2.0, rsi: 2.0, volume: 1.0}`

**Bounds:** all weights clamped to `[0.5, 5.0]`; volume additionally capped at `1.5`

**Learning rates:**

| Case | Delta |
|------|-------|
| Signal agreed with prediction AND prediction correct | `+0.15` |
| Signal disagreed with prediction AND prediction correct | `+0.05` |
| Signal agreed with prediction AND prediction wrong | `−0.15` |
| Signal disagreed with prediction AND prediction wrong | `−0.05` |

**Volume signal mapping:**
- `HIGH` → agrees with UP prediction
- `LOW` → agrees with DOWN prediction
- `NORMAL` → neutral (no update)

**Methods:**

| Method | Description |
|--------|-------------|
| `get()` | Returns copy of current weights dict. Thread-safe. |
| `update(resolved_record)` | Applies reward/punishment. Returns new weights dict. No-op if `correct is None`. |
| `_agrees(sig_val, prediction)` *(static)* | Returns `True` (agreed), `False` (disagreed), or `None` (neutral). |

---

### `MarketAnalyzer`

Main class. Thread-safe, lazily-refreshed. Owns a `PredictionTracker` and `AdaptiveWeights` instance.

**`get_analysis() → dict`**

Public entry point. Returns cached result if < 30s old, otherwise calls `_safe_refresh()`. Never raises.

**`_safe_refresh() → dict`**

Calls `_fetch_candles()` then `_analyze()`. On any exception returns a safe error dict with all keys present.

**`_fetch_candles(limit=20) → (candles, source_name)`**

Tries each source in order. Sets `self._last_points = None` first (only Chainlink path sets it). Returns first source with ≥ 15 candles.

**`_analyze(candles, source) → dict`**

1. Computes all 4 signals from closes/volumes
2. Gets current adaptive weights from `_weights_aw`
3. Calls `_composite()` with live weights
4. Calls `_tracker.record()` — if window just flipped, gets resolved record back
5. If resolved record: calls `_weights_aw.update()` and logs the outcome
6. Buckets `_last_points` at 30s for chart candles (Chainlink only)
7. Returns full dict

**`_chainlink_via_rpc(rpc_url, limit) → List[dict]`**

1. Single `eth_call` → `latestRoundData()` → current price + round ID
2. Builds batch of up to 80 `getRoundData(roundId)` calls
3. Sends batch in one HTTP POST
4. Decodes all valid responses into `{price, ts}` points
5. Stores raw points in `self._last_points`
6. Buckets at 300s → returns 5-min candles

**`_decode_round(hex_data) → (round_id, price_usd, updated_at)`** *(static)*

ABI-decodes 160-byte response: 5 × uint256 words.
- Word 0: round_id
- Word 1: answer (int256, price × 1e8)
- Word 3: updatedAt (unix timestamp)

**`_composite(mom_sig, rsi_sig, ema_sig, vol_sig, weights=None) → (prediction, confidence)`**

Weighted directional vote:
- `total = score(ema)*w_ema + score(mom)*w_mom + score(rsi)*w_rsi + vol_bonus`
- `max_possible = sum(weights.values())`
- `confidence = clamp(40 + (|total| / max_possible) * 55, max=95)`
- Returns `("UP"|"DOWN"|"NEUTRAL", int)`

---

## Signals

### Momentum
- Looks at last 3 **completed** candles (`closes[-4:-1]`)
- Counts up-moves vs down-moves between consecutive closes
- Majority direction wins; tie → NEUTRAL

### RSI(14) — Wilder
| RSI Range | Signal |
|-----------|--------|
| > 70 | DOWN (overbought) |
| 55–70 | UP (mild bullish) |
| 45–55 | NEUTRAL |
| 30–45 | DOWN (mild bearish) |
| < 30 | UP (oversold) |

Uses SMA seed for first 14 bars, then Wilder smoothing: `avg = (avg * 13 + new) / 14`

### EMA Cross
- EMA(5) vs EMA(20): both use SMA seed
- EMA5 > EMA20 → UP; EMA5 < EMA20 → DOWN; equal → NEUTRAL

### Volume
- Compares `volumes[-2]` (last completed candle) to 5-candle rolling average
- `ratio >= 1.3` → HIGH; `ratio <= 0.7` → LOW; else → NORMAL
- **Not used when source = chainlink** (no trade volume on-chain; always set to NORMAL)

---

## API Response Structure

```json
{
  "btc_price":     95432.10,
  "prediction":    "UP",
  "confidence":    72,
  "source":        "chainlink",
  "signals": {
    "momentum":  { "signal": "UP",     "detail": "2/2 recent candles bullish" },
    "rsi":       { "signal": "UP",     "value": 58.2 },
    "ema_cross": { "signal": "UP",     "ema5": 95400.12, "ema20": 95350.44 },
    "volume":    { "signal": "NORMAL", "ratio": 1.0 }
  },
  "weights": {
    "ema": 3.15, "momentum": 1.85, "rsi": 2.10, "volume": 0.95
  },
  "accuracy": {
    "total": 12, "correct": 8,
    "accuracy_pct": 66.7,
    "last_outcome": true,
    "last_prediction": "UP"
  },
  "candles_30s": [
    { "open_time": 1740995400000, "open": 95410.0, "high": 95440.0,
      "low": 95400.0, "close": 95432.1, "volume": 3.0 }
  ],
  "interval_label": "30s",
  "updated_at": "2026-03-03T15:00:00+00:00",
  "error": null
}
```

`candles_30s` is `null` when Chainlink is unavailable and an exchange fallback is used.
`error` is `null` on success, or a string describing the failure.

---

## Helper Functions (module-level)

**`_points_to_candles(points, interval_sec) → List[dict]`**

Groups `[{price, ts}, ...]` into OHLCV buckets.
- `open_time` = bucket_start × 1000 (milliseconds)
- `volume` = count of Chainlink price updates in bucket (not trade volume)
- Returns sorted oldest-first

**`_utc_now() → str`**

Returns current UTC time as ISO 8601 string with seconds precision.

---

## Polygon RPC Endpoints

```
https://polygon-bor-rpc.publicnode.com  (primary)
https://polygon.drpc.org                (fallback 1)
https://polygon-mainnet.public.blastapi.io  (fallback 2)
```

Same endpoints used by `ChainMonitor` in `copy_trader.py`.

---

## Threading Model

- `MarketAnalyzer._lock` — protects `_last_result` and `_last_fetched` (cache)
- `PredictionTracker._lock` — protects `_records` and `_pending`
- `AdaptiveWeights._lock` — protects `_weights`
- Network calls happen **outside** all locks to avoid blocking Flask SSE threads
- `_last_points` is written by `_chainlink_via_rpc` and read by `_analyze`, both called sequentially within `_safe_refresh` — no race condition in normal operation
