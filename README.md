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
