from __future__ import annotations

from pathlib import Path
import pandas as pd


RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def resolve_time_column(df: pd.DataFrame) -> str:
    for candidate in ("time", "date", "Date"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No datetime column found in merged_klines_funding_30m.csv (`time`, `date`, or `Date`).")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing OHLC columns for indicator calculation: {missing}")

    out = df.copy()
    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    close = out["close"]
    high = out["high"]
    low = out["low"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    out["rsi_14"] = rsi

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()

    bb_mid = close.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean()
    bb_std = close.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    bb_upper = bb_mid + (BB_STD * bb_std)
    bb_lower = bb_mid - (BB_STD * bb_std)
    bb_range = bb_upper - bb_lower
    out["bb_position_20_2"] = ((close - bb_lower) / bb_range).where(bb_range != 0)

    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL,
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    return out


def main() -> None:
    project_root = Path(__file__).resolve().parent
    klines_path = project_root / "data" / "processed" / "merged_klines_funding_30m.csv"
    daily_path = project_root / "data" / "processed" / "merged_data.csv"
    mergedx_path = project_root / "data" / "processed" / "mergedX.csv"
    output_path = project_root / "data" / "processed" / "merged.csv"

    if not klines_path.exists():
        raise FileNotFoundError(f"Missing file: {klines_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing file: {daily_path}")
    if not mergedx_path.exists():
        raise FileNotFoundError(f"Missing file: {mergedx_path}")

    klines_df = pd.read_csv(klines_path)
    daily_df = pd.read_csv(daily_path)
    mergedx_df = pd.read_csv(mergedx_path)

    klines_time_col = resolve_time_column(klines_df)
    if "Date" not in daily_df.columns:
        raise KeyError("`Date` column was not found in merged_data.csv")
    if "Date" not in mergedx_df.columns:
        raise KeyError("`Date` column was not found in mergedX.csv")

    klines_df[klines_time_col] = pd.to_datetime(klines_df[klines_time_col], errors="coerce")
    daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")
    mergedx_df["Date"] = pd.to_datetime(mergedx_df["Date"], errors="coerce")
    for col in ("fed_net_liquidity", "fearGreed"):
        if col not in mergedx_df.columns:
            raise KeyError(f"`{col}` column was not found in mergedX.csv")
        mergedx_df[col] = pd.to_numeric(mergedx_df[col], errors="coerce")

    mergedx_df = mergedx_df[["Date", "fed_net_liquidity", "fearGreed"]].copy()
    mergedx_df = mergedx_df.drop_duplicates(subset="Date", keep="last")
    daily_df = daily_df.merge(mergedx_df, on="Date", how="left")

    klines_df = add_technical_indicators(klines_df)

    cme_rename_map = {
        "Date": "CME_Date",
        "Open": "CME_Open",
        "High": "CME_High",
        "Low": "CME_Low",
        "Close": "CME_Close",
        "Volume": "CME_Volume",
    }
    daily_df = daily_df.rename(columns={k: v for k, v in cme_rename_map.items() if k in daily_df.columns})

    bad_30m_time = int(klines_df[klines_time_col].isna().sum())
    bad_1d_time = int(daily_df["CME_Date"].isna().sum())
    if bad_30m_time:
        klines_df = klines_df.dropna(subset=[klines_time_col]).copy()
    if bad_1d_time:
        daily_df = daily_df.dropna(subset=["CME_Date"]).copy()

    klines_df["merge_date"] = klines_df[klines_time_col].dt.normalize()
    # Shift daily data by +1 day so a daily row (e.g. 2020-06-01)
    # is merged into the next day's 30m rows (e.g. 2020-06-02).
    daily_df["merge_date"] = daily_df["CME_Date"].dt.normalize() + pd.Timedelta(days=1)

    merged = klines_df.merge(daily_df, on="merge_date", how="left")
    merged = merged.drop(columns=["merge_date"])
    merged = merged.sort_values(klines_time_col).reset_index(drop=True)

    # First 48 rows naturally have empty daily data due to +1 day shift; drop them.
    if len(merged) < 48:
        raise ValueError("Merged data has fewer than 48 rows; cannot drop first 48 rows.")
    merged = merged.iloc[48:].reset_index(drop=True)

    # Keep a single datetime column in output and preserve the 30m timestamp.
    merged["Date"] = merged[klines_time_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove timestamp-like columns and extra daily date column.
    drop_columns = [
        klines_time_col,
        "close_time",
        "funding_ts",
        "timestamp",
        "calc_time",
        "CME_Date",
    ]
    merged = merged.drop(columns=[c for c in drop_columns if c in merged.columns])

    # Put Date as the first column.
    first_cols = ["Date"]
    merged = merged[first_cols + [c for c in merged.columns if c not in first_cols]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    rows_30m = len(klines_df)
    rows_1d = len(daily_df)
    rows_merged = len(merged)
    missing_daily = int(merged["CME_Open"].isna().sum()) if "CME_Open" in merged.columns else -1
    avg_rows_per_day = klines_df.groupby(klines_df[klines_time_col].dt.date).size().mean()

    print(f"30m rows read: {rows_30m}")
    print(f"1d rows read: {rows_1d}")
    print("1d enrichment source: merged_data.csv + mergedX.csv")
    print(f"Merged rows (after dropping first 48): {rows_merged}")
    print(f"Rows with missing 1d data (Open is NaN): {missing_daily}")
    print(f"Invalid 30m time rows dropped: {bad_30m_time}")
    print(f"Invalid 1d date rows dropped: {bad_1d_time}")
    print(f"Average 30m rows per day: {avg_rows_per_day:.2f}")
    print("Added columns: rsi_14, atr_14, bb_position_20_2, macd, macd_signal, macd_hist")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
