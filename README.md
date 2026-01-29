# BTC Trade Bot

A Python-based tool for downloading and processing historical cryptocurrency data from Binance Futures.

## Features

- Downloads **Klines (Price & Volume)** data (30m interval).
- Downloads **Funding Rate** data.
- Handles Monthly Zip file downloads from Binance Vision.

## Usage

1.  Install dependencies:
    ```bash
    pip install pandas requests
    ```
2.  Run the downloader script:
    ```bash
    python klines_fundingRate_30min.py
    ```

## Roadmap

- [x] Klines Download
- [x] Funding Rate Download
- [ ] **Open Interest (OI) Data Download** (Next Step)
    - *Note: Historical OI data is not directly available via simple file download and requires alternative sourcing (e.g., Kaggle or Paid APIs).*
