"""
diagnose.py — run this to see why trades aren't being tracked
Usage:  python diagnose.py 0xWALLET_ADDRESS_HERE
"""
import sys
import json
import requests

TARGET = sys.argv[1] if len(sys.argv) > 1 else None
if not TARGET:
    print("Usage: python diagnose.py 0xYOUR_TARGET_WALLET")
    sys.exit(1)

SLUG_PATTERNS = ["btc-updown-5m", "eth-updown-5m", "sol-updown-5m", "xrp-updown-5m"]
DATA_API = "https://data-api.polymarket.com"

print(f"\n{'='*60}")
print(f"Diagnosing: {TARGET}")
print(f"{'='*60}\n")

# ── 1. Data-API trades ──────────────────────────────────────────────────────
print("[ 1 ] Fetching recent trades from data-api...")
try:
    r = requests.get(f"{DATA_API}/trades", params={"user": TARGET, "limit": 20}, timeout=15)
    r.raise_for_status()
    trades = r.json()
    if isinstance(trades, dict):
        trades = trades.get("data", [])
except Exception as e:
    print(f"  ERROR: {e}")
    trades = []

if not trades:
    print("  ❌  data-api returned 0 trades for this wallet.")
    print("      → Wallet address is probably wrong, or user has no recent trades.")
else:
    print(f"  ✅  Got {len(trades)} trades. Slug breakdown:\n")
    passed = 0
    for t in trades:
        slug = t.get("slug") or t.get("eventSlug") or "(no slug)"
        side = t.get("side", "?")
        size = t.get("size", "?")
        price = t.get("price", "?")
        matches = any(p in slug.lower() for p in SLUG_PATTERNS)
        status = "✅ TRACKED" if matches else "❌ FILTERED"
        if matches:
            passed += 1
        print(f"  {status}  {slug:<45}  {side} {size} @ {price}")

    print(f"\n  Summary: {passed}/{len(trades)} trades would pass the slug filter.")
    if passed == 0:
        print("\n  ⚠️  ALL trades are being filtered out.")
        print("  → This wallet doesn't trade btc/eth/sol/xrp-updown-5m markets.")
        print("  → OR their recent trades are in different markets.")
        unique_slugs = sorted(set(
            (t.get("slug") or t.get("eventSlug") or "?")[:60] for t in trades
        ))
        print(f"\n  Actual slugs seen:")
        for s in unique_slugs:
            print(f"    {s}")

# ── 2. Blockchain check ─────────────────────────────────────────────────────
print(f"\n[ 2 ] Checking on-chain logs for wallet {TARGET}...")
POLYGON_RPC = "https://polygon.drpc.org"
ORDER_FILLED = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
CTF_ADDRS    = ["0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
                "0xC5d563A36AE78145C45a50134d48A1215220f80a"]
maker_topic  = "0x" + "0" * 24 + TARGET[2:].lower()

try:
    # Get latest block
    r2 = requests.post(POLYGON_RPC, json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}, timeout=10)
    latest_hex = r2.json()["result"]
    latest = int(latest_hex, 16)
    from_block = latest - 2000  # ~70 minutes of blocks

    r3 = requests.post(POLYGON_RPC, json={
        "jsonrpc": "2.0", "method": "eth_getLogs", "id": 2,
        "params": [{
            "fromBlock": hex(from_block),
            "toBlock":   latest_hex,
            "address":   CTF_ADDRS,
            "topics":    [ORDER_FILLED, None, maker_topic],
        }]
    }, timeout=15)
    logs = r3.json().get("result", [])
    if isinstance(logs, list):
        print(f"  ✅  Found {len(logs)} on-chain OrderFilled events in last ~2000 blocks (~70 min)")
        if len(logs) == 0:
            print("  ⚠️  No on-chain events found for this wallet as MAKER.")
            print("      → Check if their proxy wallet address is correct.")
            print("      → Or they haven't traded in the last 70 minutes.")
    else:
        print(f"  ❌  eth_getLogs error: {r3.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\n{'='*60}\n")
