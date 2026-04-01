# Raw Data Layout

This folder now keeps raw downloads grouped by source and market.

Expected 30m downloader outputs:

- `binance/spot/BTCUSDT/30m/`
- `binance/futures/BTCUSDT/30m/`
- `bybit/futures/BTCUSDT/`
- `coinbase/spot/BTC-USD/30m/`
- `cme/databento/BTC_v_0/`

Existing 1d/raw macro and derivatives folders kept for now:

- `fear-greed/`
- `liquidations-OI/`
- `netLiq/`

Legacy folders removed during downloader refactor:

- `cme/` old yfinance file layout
- `klines/`
- `fundingRate/`
