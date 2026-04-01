"""Editable runtime settings for the live BTC demo trade bot."""

from __future__ import annotations

import os


# Fill these in or set the matching environment variables before trading.
BINANCE_DEMO_API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "")
BINANCE_DEMO_API_SECRET = os.getenv("BINANCE_DEMO_API_SECRET", "")
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "")

# Trading settings.
SYMBOL = os.getenv("BTC_BOT_SYMBOL", "BTCUSDT")
ACCOUNT_RISK_PER_TRADE = float(os.getenv("BTC_BOT_ACCOUNT_RISK_PER_TRADE", "0.10"))
STOP_LOSS_FACTOR = float(os.getenv("BTC_BOT_STOP_LOSS_FACTOR", "0.75"))
LEVERAGE = int(os.getenv("BTC_BOT_LEVERAGE", "5"))
TRADE_SIGNAL_MAX_FLAT_PROBABILITY = float(os.getenv("BTC_BOT_TRADE_SIGNAL_MAX_FLAT_PROBABILITY", "0.65"))
TRADE_SIGNAL_SIDE_RATIO_THRESHOLD = float(os.getenv("BTC_BOT_TRADE_SIGNAL_SIDE_RATIO_THRESHOLD", "2.0"))

# Market data uses mainnet public endpoints by default.
PUBLIC_MARKET_DATA_URL = os.getenv("BTC_BOT_PUBLIC_MARKET_DATA_URL", "https://fapi.binance.com")
BINANCE_FUTURES_PUBLIC_URL = os.getenv("BTC_BOT_BINANCE_FUTURES_PUBLIC_URL", "https://fapi.binance.com")
BINANCE_SPOT_PUBLIC_URL = os.getenv("BTC_BOT_BINANCE_SPOT_PUBLIC_URL", "https://api.binance.com")
COINBASE_PUBLIC_URL = os.getenv("BTC_BOT_COINBASE_PUBLIC_URL", "https://api.exchange.coinbase.com")
BYBIT_PUBLIC_URL = os.getenv("BTC_BOT_BYBIT_PUBLIC_URL", "https://api.bybit.com")
DATABENTO_CME_DATASET = os.getenv("BTC_BOT_DATABENTO_CME_DATASET", "GLBX.MDP3")
DATABENTO_CME_SYMBOL = os.getenv("BTC_BOT_DATABENTO_CME_SYMBOL", "BTC.v.0")
DATABENTO_CME_STYPE_IN = os.getenv("BTC_BOT_DATABENTO_CME_STYPE_IN", "continuous")
DATABENTO_CME_LOOKBACK_HOURS = float(os.getenv("BTC_BOT_DATABENTO_CME_LOOKBACK_HOURS", "24"))
BINANCE_DEMO_BASE_URL = os.getenv("BINANCE_DEMO_BASE_URL", "https://demo-fapi.binance.com")

# Runtime settings.
POLL_INTERVAL_SECONDS = float(os.getenv("BTC_BOT_POLL_INTERVAL_SECONDS", "15"))
BAR_INTERVAL_MINUTES = int(os.getenv("BTC_BOT_BAR_INTERVAL_MINUTES", "30"))
BAR_CLOSE_BUFFER_SECONDS = float(os.getenv("BTC_BOT_BAR_CLOSE_BUFFER_SECONDS", "2"))
ERROR_RETRY_SECONDS = float(os.getenv("BTC_BOT_ERROR_RETRY_SECONDS", "10"))
KLINE_HISTORY_BARS = int(os.getenv("BTC_BOT_KLINE_HISTORY_BARS", "3400"))
ORDER_CLIENT_PREFIX = os.getenv("BTC_BOT_ORDER_PREFIX", "btcbot")
RECV_WINDOW_MS = int(os.getenv("BTC_BOT_RECV_WINDOW_MS", "5000"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("BTC_BOT_HTTP_TIMEOUT_SECONDS", "20"))
PUBLIC_API_MAX_RETRIES = int(os.getenv("BTC_BOT_PUBLIC_API_MAX_RETRIES", "2"))
PUBLIC_API_RETRY_BACKOFF_SECONDS = float(os.getenv("BTC_BOT_PUBLIC_API_RETRY_BACKOFF_SECONDS", "1.5"))
LIVE_RECENT_30M_BARS = int(os.getenv("BTC_BOT_LIVE_RECENT_30M_BARS", "128"))
COINBASE_RECENT_15M_BARS = int(os.getenv("BTC_BOT_COINBASE_RECENT_15M_BARS", "260"))
