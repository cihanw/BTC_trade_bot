import os
import requests
import pandas as pd
from datetime import datetime

# --- SETTINGS ---
SYMBOL = "BTCUSDT"          # Pair to download
START_DATE = "2020-01-01"   # Start date (Year-Month-Day)
END_DATE = "2026-01-29"     # End date
INTERVAL = "30m"            # Timeframe for klines
BASE_DIR = r"C:\Users\Bilge\OneDrive\Masaüstü\BTC\data\raw" # Main directory for downloads

# Binance Vision Base URL (Futures - USDT Margined)
BASE_URL = "https://data.binance.vision/data/futures/um/monthly"

def create_directory_structure():
    """Creates the folder structure for data types."""
    subdirs = ["klines", "fundingRate"]
    for sub in subdirs:
        path = os.path.join(BASE_DIR, sub)
        if not os.path.exists(path):
            os.makedirs(path)
    print(f"📂 Folder structure created under '{BASE_DIR}'.")

def download_file(url, save_path):
    """Downloads file from the given URL and saves it."""
    if os.path.exists(save_path):
        print(f"✅ Already exists: {os.path.basename(save_path)}")
        return

    print(f"⬇️ Downloading: {url} ...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {os.path.basename(save_path)}")
        elif response.status_code == 404:
            print(f"❌ File Not Found (404): {url}")
        else:
            print(f"⚠️ Error ({response.status_code}): {url}")
    except Exception as e:
        print(f"💥 Download error: {e}")

def main():
    create_directory_structure()
    
    # Create date range monthly
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    print(f"\n🚀 Starting data download for {SYMBOL}...\n")

    for date in date_range:
        year = date.year
        month = f"{date.month:02d}" # 1 -> 01
        
        # --- 1. KLINES (Price and Volume - 30m) ---
        # URL Format: /klines/BTCUSDT/30m/BTCUSDT-30m-2023-01.zip
        file_name_kline = f"{SYMBOL}-{INTERVAL}-{year}-{month}.zip"
        url_kline = f"{BASE_URL}/klines/{SYMBOL}/{INTERVAL}/{file_name_kline}"
        path_kline = os.path.join(BASE_DIR, "klines", file_name_kline)
        download_file(url_kline, path_kline)




        # --- 3. FUNDING RATE ---
        # URL Format: /fundingRate/BTCUSDT/BTCUSDT-fundingRate-2023-01.zip
        file_name_fund = f"{SYMBOL}-fundingRate-{year}-{month}.zip"
        url_fund = f"{BASE_URL}/fundingRate/{SYMBOL}/{file_name_fund}"
        path_fund = os.path.join(BASE_DIR, "fundingRate", file_name_fund)
        download_file(url_fund, path_fund)

    print(f"\n✨ Operation complete. Files are in '{BASE_DIR}'.")

if __name__ == "__main__":
    # Ensure required libraries are installed:
    # pip install requests pandas
    main()