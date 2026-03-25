# Polymarket Copy Bot

A web-UI-driven copy-trading bot for **BTC 5-minute Up/Down markets** on Polymarket.
Monitor a target whale wallet and automatically mirror their trades at a scaled-down size — with configurable risk controls, a full dry-run simulation mode, and a real-time log feed.

---

## Architecture

```
app.py               Flask web server (UI + SSE log streaming)
copy_trader.py       Core engine
  ├── TradeMonitor   Polls data-api.polymarket.com for target trades
  ├── RiskManager    Validates risk limits, scales sizes
  ├── TradeExecutor  Places orders via py-clob-client (Polymarket CLOB)
  └── Notifier       Telegram / Slack alerts (Sprint 2)
config.py            BotConfig dataclass with all settings
templates/
  └── index.html     Single-page web UI
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A funded Polygon wallet with USDC on Polymarket
- A Polymarket account (trade at least once so the CLOB recognises your address)

### 2. Install

```bash
cd polymarket-copy-bot
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Then open **http://localhost:5002** in your browser.

> **Production / remote server:** run behind HTTPS (nginx + Let's Encrypt or Cloudflare tunnel) so the private key is never sent in plaintext.

---

## Web UI Walkthrough

| Field | Description |
|-------|-------------|
| **Target Wallet** | The 0x-address of the whale you want to copy |
| **Private Key** | Your EOA private key — used only in-memory, never stored |
| **My Wallet** | Leave blank to auto-derive from the private key |
| **Signature type** | `0` = standard MetaMask/hardware wallet (default) |
| **Copy multiplier** | Fraction of their trade size to copy (slider, 1–100%) |
| **Max USD / trade** | Hard cap on each individual copied order |
| **Min USD to copy** | Ignore tiny trades below this threshold |
| **Max total exposure** | Maximum cumulative open USD in BTC-5min positions |
| **Poll interval** | How often to query the target's trade history (seconds) |
| **DRY RUN toggle** | ON = simulate only; OFF = real orders (requires confirmation) |

### Mode indicators

| Mode | Visual |
|------|--------|
| **Dry Run** | Entire UI turns amber / dark yellow, persistent yellow banner |
| **Live Trading** | UI turns dark red, persistent red warning banner |
| **Idle** | Default dark theme, no banner |

### Log feed

- Colour-coded by event type: **COPY** (green), **DRY_RUN** (amber), **SKIP** (orange), **ERROR** (red), **INFO** (blue)
- Filter chips to show/hide event types
- Auto-scroll (toggle with the "Auto" button)
- Clear button

---

## Signature Types

| Value | Wallet type | When to use |
|-------|-------------|-------------|
| `0` | EOA (standard) | MetaMask, Ledger, Trezor — most users |
| `1` | Magic / Email wallet | Polymarket email login |
| `2` | Proxy contract | Polymarket browser wallet (created via UI) |

If you use type `1` or `2`, also fill in the **Funder / Proxy address** field.

---

## Risk Controls Explained

```
their_usd    = their_size  × their_price
copy_usd     = min(their_usd × COPY_MULTIPLIER, MAX_USD_PER_TRADE)
copy_size    = copy_usd / their_price          # shares to order

current_exp  = sum of open BTC-5min positions  (from gamma-api.polymarket.com)
headroom     = MAX_TOTAL_EXPOSURE_USD − current_exp
copy_usd     = min(copy_usd, headroom)         # reduced if needed
```

**Skip conditions:**

| Reason | What triggers it |
|--------|-----------------|
| `trade USD < min` | Their trade value < MIN_USD_TO_COPY |
| `max exposure reached` | current_exposure ≥ MAX_TOTAL_EXPOSURE_USD |
| `reduced size < min` | After reducing for headroom, copy_usd < MIN_USD_TO_COPY |
| `price is 0` | Malformed trade data |

---

## Notification Setup (Sprint 2)

### Telegram

1. Message **@BotFather** on Telegram → `/newbot` → follow prompts → copy the token.
2. Add your bot to a channel/group **or** start a private chat.
3. Get your Chat ID: message **@userinfobot** or use `https://api.telegram.org/bot<TOKEN>/getUpdates`.
4. Paste token + chat ID into the UI or `.env`.

### Slack

1. Go to **api.slack.com/apps** → Create app → Incoming Webhooks → activate → Add webhook to channel.
2. Copy the Webhook URL and paste into the UI or `.env`.

---

## Headless / Env-Var Mode

For server deployments without a browser:

```bash
cp .env.example .env
# edit .env with your values
python -c "
from dotenv import load_dotenv; load_dotenv()
from config import BotConfig
from copy_trader import CopyTradingBot
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
bot = CopyTradingBot(BotConfig.from_env())
bot.run()
"
```

---

## USDC Approval (first-time only)

Before live trading your wallet must approve the Polymarket CLOB contracts to spend USDC and Conditional Tokens.
The py-clob-client exposes a helper — run once:

```python
from py_clob_client.client import ClobClient
client = ClobClient("https://clob.polymarket.com", key="0xYOUR_KEY", chain_id=137)
client.approve_conditional_tokens()   # approve Conditional Token ERC-1155
client.approve_usdc()                 # approve USDC spend
```

---

## Extending to Multiple Wallets

In `copy_trader.py`, `CopyTradingBot` currently uses a single `TradeMonitor`.
To track several wallets simultaneously:

```python
# In CopyTradingBot.__init__:
wallets = ["0xWHALE1", "0xWHALE2"]   # replace config.target_wallet with a list
self.monitors = [TradeMonitor(BotConfig(**{**vars(config), 'target_wallet': w}))
                 for w in wallets]

# In CopyTradingBot.run — replace:
#   new_trades = self.monitor.get_new_trades()
# with:
#   new_trades = []
#   for m in self.monitors:
#       new_trades.extend(m.get_new_trades())
```

---

## Security Notes

- Private key is **never** written to disk, logged, or included in exception messages.
- The `.env` file (if used) should be in `.gitignore`.
- Run the UI on `localhost` or behind HTTPS — never expose port 5000 to the internet over plain HTTP.
- The `Notifier` uses raw `requests` only — no third-party bot SDKs.

---

## File Map

```
polymarket-copy-bot/
├── app.py                 Flask server + SSE broadcaster
├── copy_trader.py         Bot engine (all 5 classes)
├── config.py              BotConfig dataclass
├── templates/
│   └── index.html         Web UI (single file, no build step)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CLOB client not initialised` | Check private key format (must start with `0x` or be bare hex) |
| Orders rejected | Ensure USDC approval is done (see above) |
| No trades detected | Verify target wallet address; check BTC-5min patterns in `config.py` |
| `Telegram notification failed 401` | Bot token is wrong or bot is not added to the chat |
| High latency | Lower `polling_interval` to `0.5`; run on a VPS close to Polygon RPCs |
| `py-clob-client` import error | `pip install py-clob-client` and confirm Python ≥ 3.11 |
