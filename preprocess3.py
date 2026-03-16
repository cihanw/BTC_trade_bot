from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parent
    klines_path = project_root / "data" / "processed" / "merged_klines_funding_30m.csv"
    daily_path = project_root / "data" / "processed" / "merged_data.csv"
    output_path = project_root / "data" / "processed" / "merged.csv"

    if not klines_path.exists():
        raise FileNotFoundError(f"Missing file: {klines_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing file: {daily_path}")

    klines_df = pd.read_csv(klines_path)
    daily_df = pd.read_csv(daily_path)

    if "time" not in klines_df.columns:
        raise KeyError("`time` column was not found in merged_klines_funding_30m.csv")
    if "Date" not in daily_df.columns:
        raise KeyError("`Date` column was not found in merged_data.csv")

    klines_df["time"] = pd.to_datetime(klines_df["time"], errors="coerce")
    daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")

    cme_rename_map = {
        "Date": "CME_Date",
        "Open": "CME_Open",
        "High": "CME_High",
        "Low": "CME_Low",
        "Close": "CME_Close",
        "Volume": "CME_Volume",
    }
    daily_df = daily_df.rename(columns={k: v for k, v in cme_rename_map.items() if k in daily_df.columns})

    bad_30m_time = int(klines_df["time"].isna().sum())
    bad_1d_time = int(daily_df["CME_Date"].isna().sum())
    if bad_30m_time:
        klines_df = klines_df.dropna(subset=["time"]).copy()
    if bad_1d_time:
        daily_df = daily_df.dropna(subset=["CME_Date"]).copy()

    klines_df["merge_date"] = klines_df["time"].dt.normalize()
    daily_df["merge_date"] = daily_df["CME_Date"].dt.normalize()

    merged = klines_df.merge(daily_df, on="merge_date", how="left")
    merged = merged.drop(columns=["merge_date"])
    merged = merged.sort_values("time").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    rows_30m = len(klines_df)
    rows_1d = len(daily_df)
    rows_merged = len(merged)
    missing_daily = int(merged["CME_Open"].isna().sum()) if "CME_Open" in merged.columns else -1
    avg_rows_per_day = klines_df.groupby(klines_df["time"].dt.date).size().mean()

    print(f"30m rows read: {rows_30m}")
    print(f"1d rows read: {rows_1d}")
    print(f"Merged rows: {rows_merged}")
    print(f"Rows with missing 1d data (Open is NaN): {missing_daily}")
    print(f"Invalid 30m time rows dropped: {bad_30m_time}")
    print(f"Invalid 1d date rows dropped: {bad_1d_time}")
    print(f"Average 30m rows per day: {avg_rows_per_day:.2f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
