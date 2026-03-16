import pandas as pd
import sys

file_path = r"c:\Users\cihan\441\BTC_trade_bot\data\raw\cme\CME_BTC_F_2020-01-01_to_2026-03-15.csv"

def main():
    try:
        # Read csv
        df = pd.read_csv(file_path)

        # Drop Dividends and Stock Splits
        df.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')

        # Convert Date to datetime and keep only date part
        # Note: the date string is like "2020-01-02 00:00:00-05:00"
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        
        # Set Date as datetime index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Create a full date range from the min date to max date
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

        # Reindex to add missing dates
        df = df.reindex(full_date_range)

        # Fill forward the O, H, L, C columns
        cols_to_ffill = ['Open', 'High', 'Low', 'Close']
        df[cols_to_ffill] = df[cols_to_ffill].ffill()

        # Fill 0 for Volume column
        df['Volume'] = df['Volume'].fillna(0)

        # Reset index to make Date a column again
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        # Format Date to string '%Y-%m-%d'
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # Save to the same file
        df.to_csv(file_path, index=False)
        print("Successfully processed the CME file! Added missing days, forward filled price data, set 0 volume and fixed Date format.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
