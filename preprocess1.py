import pandas as pd

# Define file paths
cme_file = "data/raw/cme/CME_BTC_F_2020-01-01_to_2026-03-15.csv"
lsr_file = "data/raw/coinalyze_long_short_ratio_1d.csv"
oi_file = "data/raw/coinalyze_open_interest_1d.csv"

# Load datasets
cme_df = pd.read_csv(cme_file)
# Load Coinalyze datasets
lsr_df = pd.read_csv(lsr_file)
oi_df = pd.read_csv(oi_file)

# 1. CME Processing (from process_cme.py)
if 'Dividends' in cme_df.columns:
    cme_df = cme_df.drop('Dividends', axis=1)
if 'Stock Splits' in cme_df.columns:
    cme_df = cme_df.drop('Stock Splits', axis=1)

cme_df['Date'] = pd.to_datetime(cme_df['Date'], utc=True).dt.tz_convert('America/New_York')
cme_df = cme_df.set_index('Date')
full_date_range = pd.date_range(start=cme_df.index.min(), end=cme_df.index.max(), freq='D')
cme_df = cme_df.reindex(full_date_range)
cme_df[['Open', 'High', 'Low', 'Close']] = cme_df[['Open', 'High', 'Low', 'Close']].ffill()
cme_df['Volume'] = cme_df['Volume'].fillna(0)
cme_df = cme_df.reset_index().rename(columns={'index': 'Date'})
cme_df['Date'] = cme_df['Date'].dt.strftime('%Y-%m-%d')
# Set Date as index for merging
cme_df = cme_df.set_index('Date')
cme_df.index.name = 'Date'

# 2. Coinalyze Long/Short Ratio Processing
lsr_df['datetime'] = pd.to_datetime(lsr_df['datetime']).dt.strftime('%Y-%m-%d')
# Select relevant columns and rename datetime to Date for joining
lsr_df = lsr_df[['datetime', 'long_short_ratio', 'long_ratio', 'short_ratio']]
lsr_df = lsr_df.rename(columns={'datetime': 'Date'})
lsr_df = lsr_df.set_index('Date')

# 3. Coinalyze Open Interest Processing
oi_df['datetime'] = pd.to_datetime(oi_df['datetime']).dt.strftime('%Y-%m-%d')
# Select relevant columns and rename datetime to Date for joining
oi_df = oi_df[['datetime', 'oi_open', 'oi_high', 'oi_low', 'oi_close']]
oi_df = oi_df.rename(columns={'datetime': 'Date'})
oi_df = oi_df.set_index('Date')

# 4. Join all three datasets based on Date
# Using an outer join to keep all dates from all datasets
merged_df = cme_df.join([lsr_df, oi_df], how='outer')

# Reset index to make Date a column again
merged_df = merged_df.reset_index()

# Save merged dataset 
output_file = "data/processed/merged_data.csv"
import os
os.makedirs("data/processed", exist_ok=True)
merged_df.to_csv(output_file, index=False)
print(f"Data processed and merged successfully. Saved to {output_file}")