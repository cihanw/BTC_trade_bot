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
BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
KLINE_API_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINE_BAR_MS = 30 * 60 * 1000
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

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

OUTPUT_DIR = RAW_ROOT / "binance" / "futures" / SYMBOL / INTERVAL
KLINE_ARCHIVE_DIR = OUTPUT_DIR / "klines_monthly_zips"
FUNDING_ARCHIVE_DIR = OUTPUT_DIR / "funding_monthly_zips"
KLINE_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_ohlcv_30m_{START_DATE}_to_{END_DATE}.csv"
FUNDING_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_funding_8h_{START_DATE}_to_{END_DATE}.csv"
MERGED_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_30m_with_funding_{START_DATE}_to_{END_DATE}.csv"


def _read_zip(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, compression="zip")
    if not set(columns).issubset(df.columns):
        df = pd.read_csv(path, compression="zip", header=None, names=columns)
    return df[columns]


def _download_monthly_archives(kind: str, archive_dir: Path) -> list[Path]:
    session = create_session()
    ensure_dir(archive_dir)
    paths: list[Path] = []

    for month_start in month_starts(START_DATE, END_DATE):
        if kind == "klines":
            file_name = f"{SYMBOL}-{INTERVAL}-{month_start:%Y-%m}.zip"
            url = f"{BASE_URL}/klines/{SYMBOL}/{INTERVAL}/{file_name}"
        elif kind == "fundingRate":
            file_name = f"{SYMBOL}-fundingRate-{month_start:%Y-%m}.zip"
            url = f"{BASE_URL}/fundingRate/{SYMBOL}/{file_name}"
        else:
            raise ValueError(f"Unsupported monthly archive kind: {kind}")

        destination = archive_dir / file_name
        try:
            status = download_file(session, url, destination)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                status = "missing"
            else:
                raise
        print(f"[{kind}:{status}] {destination.name}")
        if status != "missing":
            paths.append(destination)

    return paths


def _fetch_kline_api_tail(start_ms: int, end_exclusive_ms: int) -> pd.DataFrame:
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
        response = request_with_retries(session, "GET", KLINE_API_URL, params=params)
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + KLINE_BAR_MS
        if next_cursor <= cursor_ms:
            raise RuntimeError("Binance futures kline API pagination did not advance.")
        cursor_ms = next_cursor
        sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    return pd.DataFrame(rows, columns=KLINE_COLUMNS)[KLINE_COLUMNS]


def _fetch_funding_api_tail(start_ms: int, end_exclusive_ms: int) -> pd.DataFrame:
    if start_ms >= end_exclusive_ms:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    session = create_session()
    rows: list[dict] = []
    cursor_ms = start_ms

    while cursor_ms < end_exclusive_ms:
        params = {
            "symbol": SYMBOL,
            "startTime": cursor_ms,
            "endTime": end_exclusive_ms - 1,
            "limit": API_LIMIT,
        }
        response = request_with_retries(session, "GET", FUNDING_API_URL, params=params)
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        next_cursor = last_time + (8 * 60 * 60 * 1000)
        if next_cursor <= cursor_ms:
            raise RuntimeError("Binance futures funding API pagination did not advance.")
        cursor_ms = next_cursor
        sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    api_df = pd.DataFrame(rows)
    api_df = api_df.rename(
        columns={
            "fundingTime": "calc_time",
            "fundingRate": "last_funding_rate",
        }
    )
    api_df["funding_interval_hours"] = 8
    return api_df[FUNDING_COLUMNS]


def _merge_klines(paths: list[Path]) -> pd.DataFrame:
    parts = [_read_zip(path, KLINE_COLUMNS) for path in paths if path.exists()]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KLINE_COLUMNS)

    numeric_cols = KLINE_COLUMNS[:-1]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    start_ts = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)

    api_start_ts = start_ts
    if not df.empty:
        api_start_ts = int(pd.to_numeric(df["open_time"], errors="coerce").dropna().max()) + KLINE_BAR_MS
    api_start_ts = max(api_start_ts, start_ts)

    api_df = _fetch_kline_api_tail(api_start_ts, end_ts)
    if not api_df.empty:
        for col in numeric_cols:
            api_df[col] = pd.to_numeric(api_df[col], errors="coerce")
        df = pd.concat([part for part in [df, api_df] if not part.empty], ignore_index=True)

    if df.empty:
        raise FileNotFoundError("No Binance futures kline data was available from archives or the REST API.")

    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    if df["close_time"].notna().any():
        df["close_time"] = df["close_time"].astype("Int64")
    df = df[(df["open_time"] >= start_ts) & (df["open_time"] < end_ts)].copy()
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["exchange"] = "binance"
    df["market_type"] = "futures"
    df["symbol"] = SYMBOL
    first_cols = ["time", "exchange", "market_type", "symbol"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    return df[ordered].reset_index(drop=True)


def _merge_funding(paths: list[Path]) -> pd.DataFrame:
    parts = [_read_zip(path, FUNDING_COLUMNS) for path in paths if path.exists()]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FUNDING_COLUMNS)

    df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
    df["funding_interval_hours"] = pd.to_numeric(df["funding_interval_hours"], errors="coerce")
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")

    start_ts = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)

    api_start_ts = start_ts
    if not df.empty:
        api_start_ts = int(df["calc_time"].dropna().max()) + (8 * 60 * 60 * 1000)
    api_start_ts = max(api_start_ts, start_ts)

    api_df = _fetch_funding_api_tail(api_start_ts, end_ts)
    if not api_df.empty:
        api_df["calc_time"] = pd.to_numeric(api_df["calc_time"], errors="coerce")
        api_df["funding_interval_hours"] = pd.to_numeric(api_df["funding_interval_hours"], errors="coerce")
        api_df["last_funding_rate"] = pd.to_numeric(api_df["last_funding_rate"], errors="coerce")
        df = pd.concat([part for part in [df, api_df] if not part.empty], ignore_index=True)

    if df.empty:
        raise FileNotFoundError("No Binance futures funding data was available from archives or the REST API.")

    df = df.dropna(subset=["calc_time"]).copy()
    df["calc_time"] = df["calc_time"].astype("int64")
    df = df[(df["calc_time"] >= start_ts) & (df["calc_time"] < end_ts)].copy()
    df = df.sort_values("calc_time").drop_duplicates(subset=["calc_time"], keep="last")

    df["time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["exchange"] = "binance"
    df["market_type"] = "futures"
    df["symbol"] = SYMBOL
    first_cols = ["time", "exchange", "market_type", "symbol"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    return df[ordered].reset_index(drop=True)


def _merge_klines_and_funding(kline_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    left = kline_df.copy().sort_values("open_time")
    right = funding_df.copy().sort_values("calc_time")
    merged = pd.merge_asof(
        left,
        right[["calc_time", "last_funding_rate"]],
        left_on="open_time",
        right_on="calc_time",
        direction="backward",
    )
    merged = merged.rename(columns={"last_funding_rate": "funding_rate_8h"})
    return merged


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    kline_paths = _download_monthly_archives("klines", KLINE_ARCHIVE_DIR)
    funding_paths = _download_monthly_archives("fundingRate", FUNDING_ARCHIVE_DIR)

    kline_df = _merge_klines(kline_paths)
    funding_df = _merge_funding(funding_paths)
    merged_df = _merge_klines_and_funding(kline_df, funding_df)

    kline_df.to_csv(KLINE_CSV_PATH, index=False)
    funding_df.to_csv(FUNDING_CSV_PATH, index=False)
    merged_df.to_csv(MERGED_CSV_PATH, index=False)

    print(f"Kline rows: {len(kline_df)}")
    print(f"Funding rows: {len(funding_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Saved: {KLINE_CSV_PATH}")
    print(f"Saved: {FUNDING_CSV_PATH}")
    print(f"Saved: {MERGED_CSV_PATH}")


if __name__ == "__main__":
    main()
