from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
import requests

from _common import (
    END_DATE,
    END_EXCLUSIVE_DATE,
    RAW_ROOT,
    START_DATE,
    create_session,
    download_file,
    ensure_dir,
    month_starts,
    request_with_retries,
)


SYMBOL = "BTCUSDT"
INTERVAL = "30m"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
REST_API_URL = "https://api.binance.com/api/v3/klines"
BAR_MS = 30 * 60 * 1000
API_LIMIT = 1000

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

OUTPUT_DIR = RAW_ROOT / "binance" / "spot" / SYMBOL / INTERVAL
ARCHIVE_DIR = OUTPUT_DIR / "monthly_zips"
MERGED_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_{INTERVAL}_{START_DATE}_to_{END_DATE}.csv"


def _normalize_epoch_series_to_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    microsecond_mask = numeric.abs() >= 1e15
    if microsecond_mask.any():
        numeric.loc[microsecond_mask] = numeric.loc[microsecond_mask] // 1000
    return numeric


def _read_monthly_archive(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="zip")
    if not set(KLINE_COLUMNS).issubset(df.columns):
        df = pd.read_csv(path, compression="zip", header=None, names=KLINE_COLUMNS)
    return df[KLINE_COLUMNS]


def download_monthly_archives() -> list[Path]:
    ensure_dir(ARCHIVE_DIR)
    session = create_session()
    paths: list[Path] = []
    for month_start in month_starts(START_DATE, END_DATE):
        file_name = f"{SYMBOL}-{INTERVAL}-{month_start:%Y-%m}.zip"
        url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{file_name}"
        destination = ARCHIVE_DIR / file_name
        try:
            status = download_file(session, url, destination)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                status = "missing"
            else:
                raise
        print(f"[{status}] {destination.name}")
        if status != "missing":
            paths.append(destination)
    return paths


def fetch_api_klines(start_ms: int, end_exclusive_ms: int) -> pd.DataFrame:
    if start_ms >= end_exclusive_ms:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    session = create_session()
    rows: list[list] = []
    cursor_ms = start_ms

    while cursor_ms < end_exclusive_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": cursor_ms,
            "endTime": end_exclusive_ms - 1,
            "limit": API_LIMIT,
        }
        response = request_with_retries(session, "GET", REST_API_URL, params=params)
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + BAR_MS
        if next_cursor <= cursor_ms:
            raise RuntimeError("Binance spot API pagination did not advance.")
        cursor_ms = next_cursor
        sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    api_df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return api_df[KLINE_COLUMNS]


def merge_archives(paths: list[Path]) -> pd.DataFrame:
    parts = [_read_monthly_archive(path) for path in paths if path.exists()]
    if not parts:
        merged_raw = pd.DataFrame(columns=KLINE_COLUMNS)
    else:
        merged_raw = pd.concat(parts, ignore_index=True)

    start_ts = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)

    if not merged_raw.empty:
        merged_raw["open_time"] = _normalize_epoch_series_to_ms(merged_raw["open_time"])
        merged_raw["close_time"] = _normalize_epoch_series_to_ms(merged_raw["close_time"])
        merged_raw = merged_raw.dropna(subset=["open_time"]).copy()
        merged_raw["open_time"] = merged_raw["open_time"].astype("int64")
        if merged_raw["close_time"].notna().any():
            merged_raw["close_time"] = merged_raw["close_time"].astype("Int64")

    api_start_ts = start_ts
    if not merged_raw.empty:
        api_start_ts = int(merged_raw["open_time"].max()) + BAR_MS
    api_start_ts = max(api_start_ts, start_ts)

    api_df = fetch_api_klines(api_start_ts, end_ts)
    if not api_df.empty:
        api_df["open_time"] = _normalize_epoch_series_to_ms(api_df["open_time"])
        api_df["close_time"] = _normalize_epoch_series_to_ms(api_df["close_time"])
        api_df = api_df.dropna(subset=["open_time"]).copy()
        api_df["open_time"] = api_df["open_time"].astype("int64")
        if api_df["close_time"].notna().any():
            api_df["close_time"] = api_df["close_time"].astype("Int64")

    if merged_raw.empty and api_df.empty:
        raise FileNotFoundError("No Binance spot data was available from archives or the REST API.")

    df = pd.concat([part for part in [merged_raw, api_df] if not part.empty], ignore_index=True)
    df = df.dropna(subset=["open_time"]).copy()

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["open_time"] >= start_ts) & (df["open_time"] < end_ts)].copy()
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["exchange"] = "binance"
    df["market_type"] = "spot"
    df["symbol"] = SYMBOL
    first_cols = ["time", "exchange", "market_type", "symbol"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    return df[ordered].reset_index(drop=True)


def main() -> None:
    paths = download_monthly_archives()
    merged = merge_archives(paths)

    ensure_dir(OUTPUT_DIR)
    merged.to_csv(MERGED_CSV_PATH, index=False)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved: {MERGED_CSV_PATH}")


if __name__ == "__main__":
    main()
