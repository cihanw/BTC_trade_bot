from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000


def read_kline_zip(zip_path: str) -> pd.DataFrame:
    # Some monthly files are headerless while newer ones include a header row.
    df = pd.read_csv(zip_path, compression="zip")
    if not set(KLINE_COLUMNS).issubset(df.columns):
        df = pd.read_csv(zip_path, compression="zip", header=None, names=KLINE_COLUMNS)
    return df[KLINE_COLUMNS]


def read_funding_zip(zip_path: str) -> pd.DataFrame:
    df = pd.read_csv(zip_path, compression="zip")
    if not set(FUNDING_COLUMNS).issubset(df.columns):
        df = pd.read_csv(zip_path, compression="zip", header=None, names=FUNDING_COLUMNS)
    return df[FUNDING_COLUMNS]


def load_all_klines(klines_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(klines_glob))
    if not files:
        raise FileNotFoundError(f"No kline zip files found: {klines_glob}")
    parts = [read_kline_zip(path) for path in files]
    out = pd.concat(parts, ignore_index=True)
    out["open_time"] = pd.to_numeric(out["open_time"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["open_time"]).copy()
    out["open_time"] = out["open_time"].astype("int64")
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return out.reset_index(drop=True)


def load_all_funding(funding_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(funding_glob))
    if not files:
        raise FileNotFoundError(f"No funding zip files found: {funding_glob}")
    parts = [read_funding_zip(path) for path in files]
    out = pd.concat(parts, ignore_index=True)
    out["calc_time"] = pd.to_numeric(out["calc_time"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["calc_time"]).copy()
    out["calc_time"] = out["calc_time"].astype("int64")
    out["funding_rate_8h"] = pd.to_numeric(out["last_funding_rate"], errors="coerce")
    out["funding_ts"] = (out["calc_time"] // EIGHT_HOURS_MS) * EIGHT_HOURS_MS
    out = out.sort_values(["funding_ts", "calc_time"]).drop_duplicates(
        subset=["funding_ts"], keep="last"
    )
    return out[["funding_ts", "funding_rate_8h"]].reset_index(drop=True)


def merge_klines_with_funding(klines_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        klines_df.sort_values("open_time"),
        funding_df.sort_values("funding_ts"),
        left_on="open_time",
        right_on="funding_ts",
        direction="backward",
    )
    return merged


def convert_open_time_to_datetime(merged_df: pd.DataFrame) -> pd.DataFrame:
    # Convert milliseconds timestamp to readable UTC datetime and rename to `time`.
    merged_df["time"] = pd.to_datetime(merged_df["open_time"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    merged_df = merged_df.drop(columns=["open_time"])
    first_cols = ["time"]
    other_cols = [c for c in merged_df.columns if c not in first_cols]
    return merged_df[first_cols + other_cols]


def main() -> None:
    project_root = Path(__file__).resolve().parent
    klines_glob = str(project_root / "data" / "raw" / "klines" / "*.zip")
    funding_glob = str(project_root / "data" / "raw" / "fundingRate" / "*.zip")
    output_path = project_root / "data" / "processed" / "merged_klines_funding_30m.csv"

    klines_df = load_all_klines(klines_glob)
    funding_df = load_all_funding(funding_glob)
    merged_df = merge_klines_with_funding(klines_df, funding_df)
    merged_df = convert_open_time_to_datetime(merged_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    total_rows = len(merged_df)
    missing_funding = int(merged_df["funding_rate_8h"].isna().sum())
    missing_ratio = (missing_funding / total_rows * 100.0) if total_rows else 0.0

    print(f"Klines rows read: {len(klines_df)}")
    print(f"Funding rows read: {len(funding_df)}")
    print(f"Merged rows: {total_rows}")
    print(f"Rows with missing funding_rate_8h: {missing_funding} ({missing_ratio:.4f}%)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
