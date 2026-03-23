# BTC Trade Bot

A Python-based data collection and merging project. Binance Futures data and additional market metrics are collected and merged into a single processed dataset.

## Current Status

- Binance **30m Kline** data has been collected.
- Binance **Funding Rate** data has been collected.
- **Open Interest** and **Long/Short Ratio** data has been added.
- **CME BTC Futures** data has been integrated into the project.
- All sources were merged, and processed outputs were generated under `data/processed/`.

## Project Structure

- `klines_fundingRate_30min.py`: Kline and funding rate download operations
- `openInterest.py`: Open interest processing
- `CME.py`: CME data fetching and saving
- `preprocess1.py`, `preprocess2.py`, `preprocess3.py`: Data cleaning and merge steps
- `data/raw/`: Raw data files
- `data/processed/`: Merged/processed outputs

## Usage

1. Install dependencies:
   ```bash
   pip install pandas requests
   ```
2. Run data collection scripts:
   ```bash
   python klines_fundingRate_30min.py
   python openInterest.py
   python CME.py
   ```
3. Run preprocessing and merge steps:
   ```bash
   python preprocess1.py
   python preprocess2.py
   python preprocess3.py
   ```

## Live Demo Bot

- Edit `bot_settings.py` and place your Binance Global demo API key/secret there.
- Smoke test the live inference pipeline without sending orders:
  ```bash
  python live_trading_bot.py --smoke-test
  ```
- Start the UI:
  ```bash
  python live_trading_bot.py
  ```
- Run headless for 24/7 server usage:
  ```bash
  python live_trading_bot.py --headless --risk low
  ```

Running `live_trading_bot.py` now starts a local browser dashboard. You can enter the Binance demo API key/secret directly in the page or leave them blank and use `bot_settings.py` as fallback. The bot rebuilds the training-time scaler from `data/processed/final.csv`, fetches the latest 30m Binance Futures data, derives the model decision, and applies the requested position-management rules on the Binance demo account.

Sizing and exits:
- `TP` distance uses the model-driven threshold (`barrier_width`).
- `SL` distance uses `barrier_width * 0.75`.
- New positions are sized from account equity so that a stop-out targets roughly `3%` account loss, capped by the configured leverage.
