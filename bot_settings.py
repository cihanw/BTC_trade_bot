"""Editable runtime settings for the live BTC demo trade bot."""

from __future__ import annotations

import os


# Fill these in or set the matching environment variables before trading.
BINANCE_DEMO_API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "")
BINANCE_DEMO_API_SECRET = os.getenv("BINANCE_DEMO_API_SECRET", "")

# Trading settings.
SYMBOL = os.getenv("BTC_BOT_SYMBOL", "BTCUSDT")
ACCOUNT_RISK_PER_TRADE = float(os.getenv("BTC_BOT_ACCOUNT_RISK_PER_TRADE", "0.03"))
STOP_LOSS_FACTOR = float(os.getenv("BTC_BOT_STOP_LOSS_FACTOR", "0.75"))
LEVERAGE = int(os.getenv("BTC_BOT_LEVERAGE", "5"))

# Market data uses mainnet public endpoints by default.
PUBLIC_MARKET_DATA_URL = os.getenv("BTC_BOT_PUBLIC_MARKET_DATA_URL", "https://fapi.binance.com")
BINANCE_DEMO_BASE_URL = os.getenv("BINANCE_DEMO_BASE_URL", "https://testnet.binancefuture.com")

# Runtime settings.
POLL_INTERVAL_SECONDS = float(os.getenv("BTC_BOT_POLL_INTERVAL_SECONDS", "15"))
KLINE_HISTORY_BARS = int(os.getenv("BTC_BOT_KLINE_HISTORY_BARS", "3400"))
ORDER_CLIENT_PREFIX = os.getenv("BTC_BOT_ORDER_PREFIX", "btcbot")
RECV_WINDOW_MS = int(os.getenv("BTC_BOT_RECV_WINDOW_MS", "5000"))
