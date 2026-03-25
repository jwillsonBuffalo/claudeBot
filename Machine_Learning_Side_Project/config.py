"""
config.py — BotConfig dataclass for the Polymarket Copy Trading Bot.

All sensitive values (private keys) live in-memory only and are NEVER
written to disk, logged, or included in any exception message.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BotConfig:
    # ─── Target wallet (the whale we follow) ─────────────────────────────────
    target_wallet: str = ""

    # ─── My credentials — NEVER log private_key ───────────────────────────────
    private_key: str = ""      # raw hex private key (0x-prefixed or bare)
    my_wallet: str = ""        # auto-derived from private_key if left empty
    signature_type: int = 0    # 0 = EOA, 1 = POLY_PROXY (Polymarket browser wallet), 2 = Gnosis Safe
    funder_address: str = ""   # required for sig_type 1 (POLY_PROXY) and 2 (Gnosis Safe)

    # ─── Risk controls ────────────────────────────────────────────────────────
    copy_multiplier: float = 0.10          # fraction of their size to copy
    max_usd_per_trade: float = 50.0        # hard cap per copied trade (USD)
    min_usd_to_copy: float = 5.0           # skip if their trade < this (USD)
    max_total_exposure_usd: float = 500.0  # max open USD in BTC-5min markets

    # ─── Mode ─────────────────────────────────────────────────────────────────
    dry_run: bool = True    # True → log only, never place real orders
    take_profit_pct: float = 0.0  # exit when position ROI ≥ this (0 = hold to settlement)

    # ─── Builder API (for on-chain redemption via relayer) ────────────────────
    builder_api_key: str = ""           # from polymarket.com/settings?tab=builder
    builder_api_secret: str = ""
    builder_api_passphrase: str = ""
    auto_redeem: bool = False           # call redeem_all() when a WON trade is detected

    # ─── Notifications (Sprint 2) ─────────────────────────────────────────────
    notification_method: str = "none"   # "telegram" | "slack" | "none"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    slack_webhook_url: str = ""

    # ─── API endpoints ────────────────────────────────────────────────────────
    data_api_base: str = "https://data-api.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"   # positions
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137  # Polygon mainnet

    # ─── Blockchain event monitoring (ChainMonitor) ───────────────────────────
    # Primary Polygon JSON-RPC. Override with POLYGON_RPC_URL env var to use
    # a private Alchemy/Infura key: https://polygon-mainnet.g.alchemy.com/v2/KEY
    # NOTE: polygon-rpc.com shut down free access Feb 16 2026 (now 401).
    #       ankr now requires API key; llamarpc domain gone; polygon-rpc.drpc.org is wrong URL.
    polygon_rpc_url: str = "https://polygon.drpc.org"
    # Fallback RPCs tried in order when primary fails (no API key required)
    polygon_rpc_fallbacks: List[str] = field(default_factory=lambda: [
        "https://polygon.publicnode.com",
        "https://polygon-public.nodies.app",
        "https://polygon.api.onfinality.io/public",
        "https://polygon-bor-rpc.publicnode.com",
        "https://polygon-mainnet.public.blastapi.io",
        "https://polygon.blockpi.network/v1/rpc/public",
        "https://polygon.meowrpc.com",
        "https://gateway.tenderly.co/public/polygon",
    ])
    # Seconds to wait between ChainMonitor polls (Polygon block ≈ 2.1 s).
    # 0.25 s gives ~0.125 s avg wait after block mine vs ~1 s at 2.0 s.
    chain_poll_interval: float = 0.25
    # Polymarket exchange contracts that emit OrderFilled events
    ctf_exchange_addresses: List[str] = field(default_factory=lambda: [
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # Standard CTF Exchange
        "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # NegRisk CTF Exchange
    ])

    # ─── Market timeframe ─────────────────────────────────────────────────────
    # "5m" (300s) | "15m" (900s) | "1h" (3600s)
    market_timeframe: str = "5m"

    # ─── Market slug patterns (substring match against trade's slug field) ─────
    # Real slugs look like: btc-updown-5m-1772163300, eth-updown-5m-…, etc.
    # When market_timeframe is set via the UI, this is overridden at start time.
    market_slug_patterns: List[str] = field(default_factory=lambda: [
        "btc-updown-5m",
    ])

    # When True, copy every trade the target wallet makes regardless of slug patterns.
    copy_all_markets: bool = False

    # ─── Order execution ──────────────────────────────────────────────────────
    slippage_pct: float = 0.01   # 1 % price buffer when placing limit/FOK order
    order_timeout_sec: int = 10  # seconds to wait for CLOB response
    order_retries: int = 4       # additional retries for unfilled remainder (FAK)
    retry_slippage_step_pct: float = 0.005  # +0.5% slippage per retry
    max_slippage_pct: float = 0.05          # hard cap across retries

    # ─── Internal ─────────────────────────────────────────────────────────────
    max_seen_ids: int = 10_000   # cap on the dedup set size

    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load from environment variables with sane defaults."""
        return cls(
            target_wallet=os.getenv("TARGET_WALLET", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            my_wallet=os.getenv("MY_WALLET", ""),
            signature_type=int(os.getenv("SIGNATURE_TYPE", "0")),
            funder_address=os.getenv("FUNDER_ADDRESS", ""),
            copy_multiplier=float(os.getenv("COPY_MULTIPLIER", "0.10")),
            max_usd_per_trade=float(os.getenv("MAX_USD_PER_TRADE", "50.0")),
            min_usd_to_copy=float(os.getenv("MIN_USD_TO_COPY", "5.0")),
            max_total_exposure_usd=float(os.getenv("MAX_TOTAL_EXPOSURE_USD", "500.0")),
            dry_run=os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no"),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.0")),
            polygon_rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon.drpc.org"),
            chain_poll_interval=float(os.getenv("CHAIN_POLL_INTERVAL", "0.25")),
            notification_method=os.getenv("NOTIFICATION_METHOD", "none"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            order_retries=int(os.getenv("ORDER_RETRIES", "4")),
            retry_slippage_step_pct=float(os.getenv("RETRY_SLIPPAGE_STEP_PCT", "0.005")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.05")),
            builder_api_key=os.getenv("BUILDER_API_KEY", ""),
            builder_api_secret=os.getenv("BUILDER_API_SECRET", ""),
            builder_api_passphrase=os.getenv("BUILDER_API_PASSPHRASE", ""),
            auto_redeem=os.getenv("AUTO_REDEEM", "false").lower() not in ("false", "0", "no"),
        )

    def validate(self) -> Optional[str]:
        """Return a human-readable error string, or None if config is valid."""
        if not self.target_wallet:
            return "Target wallet address is required"
        if not self.target_wallet.startswith("0x") or len(self.target_wallet) != 42:
            return "Target wallet must be a valid 0x-prefixed Ethereum address (42 chars)"
        if not self.dry_run and not self.private_key:
            return "Private key is required for live trading"
        if self.copy_multiplier <= 0 or self.copy_multiplier > 10:
            return "copy_multiplier must be between 0 (exclusive) and 10"
        if self.max_usd_per_trade <= 0:
            return "max_usd_per_trade must be positive"
        if self.min_usd_to_copy < 0:
            return "min_usd_to_copy must be ≥ 0"
        if self.max_total_exposure_usd <= 0:
            return "max_total_exposure_usd must be positive"
        if self.signature_type not in (0, 1, 2):
            return "signature_type must be 0 (EOA), 1 (POLY_PROXY), or 2 (Gnosis Safe)"
        if self.notification_method not in ("telegram", "slack", "none"):
            return "notification_method must be 'telegram', 'slack', or 'none'"
        return None

    def safe_repr(self) -> str:
        """String representation that omits the private key."""
        return (
            f"BotConfig(target={self.target_wallet}, my_wallet={self.my_wallet or '(auto)'}, "
            f"multiplier={self.copy_multiplier}, max/trade=${self.max_usd_per_trade}, "
            f"min=${self.min_usd_to_copy}, max_exp=${self.max_total_exposure_usd}, "
            f"dry_run={self.dry_run}, "
            f"notify={self.notification_method})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PredictionTraderConfig — autonomous prediction-based order placer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionTraderConfig:
    """
    Configuration for the Prediction Auto-Trader.

    Credentials fields match what TradeExecutor.__init__ reads, so
    TradeExecutor(pred_config) works without any changes to TradeExecutor.
    """

    # ─── Credentials ──────────────────────────────────────────────────────────
    private_key: str = ""
    my_wallet: str = ""
    signature_type: int = 0
    funder_address: str = ""

    # ─── Trade sizing ─────────────────────────────────────────────────────────
    auto_trade_usd: float = 10.0       # fixed USD per prediction trade

    # ─── Market timeframe ─────────────────────────────────────────────────────
    market_timeframe: str = "5m"   # "5m" | "15m" | "1h"

    # ─── Signal thresholds ────────────────────────────────────────────────────
    min_confidence_pct: int = 65       # skip if confidence < this
    trade_at_window_open_only: bool = True  # only fire in first 60 s of each window
    take_profit_pct: float = 0.0       # exit when position ROI ≥ this (0 = hold to settlement)

    # ─── Builder API (for on-chain redemption via relayer) ────────────────────
    builder_api_key: str = ""
    builder_api_secret: str = ""
    builder_api_passphrase: str = ""
    auto_redeem: bool = False

    # ─── Mode ─────────────────────────────────────────────────────────────────
    dry_run: bool = True

    # ─── API endpoints (same defaults as BotConfig) ───────────────────────────
    clob_host: str = "https://clob.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    slippage_pct: float = 0.01
    order_timeout_sec: int = 10
    order_retries: int = 4
    retry_slippage_step_pct: float = 0.005
    max_slippage_pct: float = 0.05

    def validate(self) -> Optional[str]:
        if not self.dry_run and not self.private_key:
            return "Private key is required for live prediction trading"
        if self.auto_trade_usd <= 0:
            return "auto_trade_usd must be positive"
        if not (0 <= self.min_confidence_pct <= 100):
            return "min_confidence_pct must be 0–100"
        if self.signature_type not in (0, 1, 2):
            return "signature_type must be 0 (EOA), 1 (POLY_PROXY), or 2 (Gnosis Safe)"
        return None

    def safe_repr(self) -> str:
        tp = f", take_profit={self.take_profit_pct}%" if self.take_profit_pct else ""
        return (
            f"PredictionTraderConfig(wallet={self.my_wallet or '(auto)'}, "
            f"usd=${self.auto_trade_usd}, min_conf={self.min_confidence_pct}%, "
            f"window_open_only={self.trade_at_window_open_only}{tp}, dry_run={self.dry_run})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AdvancedTraderConfig — both-sides arbitrage and spread trading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdvancedTraderConfig:
    """
    Configuration for the Advanced Trader (candlestick pattern reversal strategy).

    Watches BTC 1-minute candles from Binance for specific reversal patterns
    (Shooting Star, Bearish/Bullish Engulfing, Gravestone/Dragonfly Doji, etc.).
    When a pattern is detected, buys the OPPOSITE side on Polymarket and
    auto-exits when mid-price hits exit_target.

    Example: Bearish Engulfing on 1m BTC → buy Down token at ask=0.30 →
             auto-exit at mid=0.50 → ~67% ROI if BTC reverts before window close.

    Credentials fields match TradeExecutor's interface so TradeExecutor(adv_config) works.
    """

    # ─── Credentials ──────────────────────────────────────────────────────────
    private_key: str = ""
    my_wallet: str = ""
    signature_type: int = 0
    funder_address: str = ""

    # ─── Aggression / confidence threshold ────────────────────────────────────
    aggression: str = "balanced"        # "conservative" (80) | "balanced" (65) | "aggressive" (45)

    # ─── Trade sizing & exit ──────────────────────────────────────────────────
    swing_entry_max: float = 0.40       # max ask to enter (skip if market already priced the move)
    exit_target: float = 0.50           # mid-price at which to auto-SELL for profit
    trade_amount_usd: float = 10.0      # USDC per trade

    # ─── Market timeframe ─────────────────────────────────────────────────────
    market_timeframe: str = "5m"   # "5m" | "15m" | "1h"

    # ─── Builder API (for on-chain redemption via relayer) ────────────────────
    builder_api_key: str = ""
    builder_api_secret: str = ""
    builder_api_passphrase: str = ""
    auto_redeem: bool = False

    # ─── Risk controls ────────────────────────────────────────────────────────
    take_profit_pct: float = 0.0        # additional ROI exit on top of exit_target (0 = disabled)
    min_window_secs_remaining: int = 60 # skip entry if < this many seconds left in window
    dry_run: bool = True

    # ─── API endpoints (same defaults as BotConfig) ───────────────────────────
    clob_host: str = "https://clob.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    slippage_pct: float = 0.01
    order_timeout_sec: int = 10
    order_retries: int = 2
    retry_slippage_step_pct: float = 0.005
    max_slippage_pct: float = 0.05

    def validate(self) -> Optional[str]:
        if not self.dry_run and not self.private_key:
            return "private_key required for live advanced trading"
        if self.aggression not in ("conservative", "balanced", "aggressive"):
            return "aggression must be 'conservative', 'balanced', or 'aggressive'"
        if not (0 < self.swing_entry_max < self.exit_target <= 1.0):
            return "swing_entry_max must be < exit_target ≤ 1.0"
        if self.trade_amount_usd <= 0:
            return "trade_amount_usd must be positive"
        if self.min_window_secs_remaining < 0:
            return "min_window_secs_remaining must be ≥ 0"
        if self.signature_type not in (0, 1, 2):
            return "signature_type must be 0 (EOA), 1 (POLY_PROXY), or 2 (Gnosis Safe)"
        return None

    def safe_repr(self) -> str:
        return (
            f"AdvancedTraderConfig(aggression={self.aggression}, "
            f"entry_max={self.swing_entry_max}, "
            f"exit={self.exit_target}, usd=${self.trade_amount_usd}, dry_run={self.dry_run})"
        )


@dataclass
class MLTraderConfig:
    """
    Configuration for the ML XGBoost Predictor (BTC 5m and 15m markets).

    Credential fields satisfy TradeExecutor's duck-typed interface so
    TradeExecutor(ml_config) works without modification.
    """

    # ─── Market timeframe ─────────────────────────────────────────────────────
    timeframe:        str = "5m"    # "5m" or "15m"

    # ─── Credentials (never log private_key) ─────────────────────────────────
    private_key:      str = ""
    my_wallet:        str = ""
    signature_type:   int = 0       # 0=EOA, 1=POLY_PROXY, 2=Gnosis Safe
    funder_address:   str = ""

    # ─── Trade sizing ─────────────────────────────────────────────────────────
    trade_amount_usd: float = 10.0

    # ─── Asset ────────────────────────────────────────────────────────────────
    asset: str = "btc"   # "btc" | "bnb"

    # ─── Model selection ──────────────────────────────────────────────────────
    model_type: str = "xgboost"   # "xgboost" | "random_forest" | "extra_trees" | "ensemble"

    # ─── Signal thresholds ────────────────────────────────────────────────────
    min_confidence_pct:           int   = 0      # skip trade if confidence < this (0 = off)
    trade_at_window_open_only:    bool  = True   # only fire in first N secs of window
    window_open_sec:              int   = 60     # gate: secs after window open
    require_analysis_agreement:   bool  = False  # skip trade if Analysis engine disagrees

    # ─── Online retraining ────────────────────────────────────────────────────
    retrain_threshold:   int = 10    # retrain after this many new resolved outcomes
    retrain_window_size: int = 500   # use last N outcomes for each retrain

    # ─── Risk controls ────────────────────────────────────────────────────────
    take_profit_pct:           float = 0.0
    min_window_secs_remaining: int   = 60

    # ─── Builder API (for auto-redeem) ────────────────────────────────────────
    builder_api_key:        str  = ""
    builder_api_secret:     str  = ""
    builder_api_passphrase: str  = ""
    auto_redeem:            bool = False

    # ─── Mode ─────────────────────────────────────────────────────────────────
    dry_run:      bool = True
    predict_only: bool = False  # True = track predictions silently, never trade

    # ─── API endpoints ────────────────────────────────────────────────────────
    clob_host:          str   = "https://clob.polymarket.com"
    gamma_api_base:     str   = "https://gamma-api.polymarket.com"
    chain_id:           int   = 137
    slippage_pct:       float = 0.02
    order_timeout_sec:  int   = 10
    order_retries:      int   = 5
    retry_slippage_step_pct: float = 0.01
    max_slippage_pct:   float = 0.08

    def validate(self) -> Optional[str]:
        if not self.dry_run and not self.predict_only and not self.private_key:
            return "private_key required for live ML trading"
        if self.trade_amount_usd <= 0:
            return "trade_amount_usd must be positive"
        if self.signature_type not in (0, 1, 2):
            return "signature_type must be 0 (EOA), 1 (POLY_PROXY), or 2 (Gnosis Safe)"
        return None

    def safe_repr(self) -> str:
        return (
            f"MLTraderConfig(usd=${self.trade_amount_usd}, "
            f"window_only={self.trade_at_window_open_only}, dry_run={self.dry_run})"
        )

