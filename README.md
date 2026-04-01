# BTC Trade Bot

Multi-source BTC market-data pipeline plus a local Binance demo trading bot driven by the trained dual-input model in `model/model.pt`.

## Repository Notes

- Large raw and processed datasets are intentionally **not versioned**.
- Prepare the local datasets under `data/raw/` and `data/processed/` before running the notebooks.
- The live bot now keeps its own runtime cache under `data/liveData/` and no longer depends on `data/processed/`.
- `data/raw/README.md` is the only file kept under `data/` in git so the expected folder structure stays documented.

## Main Components

- `dataDownloaders/`: source-specific downloaders for Binance, Coinbase, Bybit, Databento CME, Fear & Greed, net liquidity, and open interest.
- `preprocess/`: preprocessing scripts that normalize each source and build merged 30m / 1d model tables.
- `trainScript.ipynb`: training notebook for the current multi-timescale model.
- `live_model_runtime.py`: bootstraps the latest model inputs from live/public APIs and stores a runtime cache in `data/liveData/`.
- `live_trading_bot.py`: local dashboard and Binance demo execution loop.

## Data Pipeline

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the required downloaders in `dataDownloaders/` for the sources you want to refresh.
3. Run the preprocessing scripts in `preprocess/` to rebuild the processed model inputs under `data/processed/`.

## Live Demo Bot

- Edit `bot_settings.py` and place your Binance demo API key/secret there if you do not want to enter them from the UI.
- For live feature bootstrap, provide:
  - `DATABENTO_API_KEY` for CME
  - `FRED_API_KEY` for net liquidity
  - `COINALYZE_API_KEY` for daily open interest / long-short ratio
- Smoke test the live inference pipeline without sending orders:
  ```bash
  .venv\Scripts\python.exe live_trading_bot.py --smoke-test
  ```
- Start the local UI:
  ```bash
  .venv\Scripts\python.exe live_trading_bot.py
  ```
- Process only the latest closed candle once and exit:
  ```bash
  .venv\Scripts\python.exe live_trading_bot.py --run-once
  ```

Running `live_trading_bot.py` now starts a local browser dashboard on `127.0.0.1`. You can enter the Binance demo API key/secret plus the Databento, FRED, and Coinalyze API keys directly in the page or leave any field blank and use environment variables / `bot_settings.py` as fallback. On startup the bot bootstraps the minimum required history into `data/liveData/`, then refreshes only the most recent data on each cycle. It loads the dual-input model from `model/model.pt`, rebuilds the latest feature windows, derives the notebook-aligned trade decision, and applies the position-management rules on the Binance demo account.

Sizing and exits:
- `TP` distance uses the model-driven threshold (`barrier_width`).
- `SL` distance uses `barrier_width * 0.75`.
- New positions are sized from account equity so that a stop-out targets roughly `10%` account loss, capped by the configured leverage.
