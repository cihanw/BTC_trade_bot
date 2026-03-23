import os
import yfinance as yf
import pandas as pd

# --- SETTINGS ---
SYMBOL = "BTC=F"            # CME Bitcoin Futures
START_DATE = "2019-12-31"   # Start date (Year-Month-Day)
END_DATE = "2026-03-15"     # End date
BASE_DIR = r"C:\Users\cihan\441\BTC_trade_bot\data\raw" # Main directory for downloads

def create_directory_structure():
    """Creates the folder structure for data types."""
    # Creating a specific folder for CME data
    path = os.path.join(BASE_DIR, "cme")
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"📂 Folder structure created under '{BASE_DIR}'.")
    return path

def main():
    save_dir = create_directory_structure()
    
    print(f"\n🚀 Starting data download for {SYMBOL} (CME Bitcoin Futures)...\n")
    
    try:
        # Fetch historical data using yfinance
        cme_btc = yf.Ticker(SYMBOL)
        
        print(f"⬇️ Downloading data from {START_DATE} to {END_DATE}...")
        # Note: yfinance can download 1d interval data for long periods. 
        # Intraday data (like 30m) is only available for the last 60 days in yfinance.
        df = cme_btc.history(start=START_DATE, end=END_DATE, interval="1d")
        
        if df.empty:
            print(f"⚠️ No data found for {SYMBOL} in the given date range.")
            return
            
        # File name and path
        file_name = f"CME_{SYMBOL.replace('=', '_')}_{START_DATE}_to_{END_DATE}.csv"
        save_path = os.path.join(save_dir, file_name)
        
        # Save to CSV
        df.to_csv(save_path)
        print(f"✅ Downloaded and saved: {file_name}")
        
    except Exception as e:
        print(f"💥 Download error: {e}")

    print(f"\n✨ Operation complete. Files are in '{save_dir}'.")

if __name__ == "__main__":
    # Ensure required libraries are installed:
    # pip install yfinance pandas
    main()
