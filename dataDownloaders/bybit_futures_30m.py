from __future__ import annotations

from datetime import date, datetime, timedelta
import os
from pathlib import Path
import re
from time import sleep

import pandas as pd

from _common import (
    END_DATE,
    END_EXCLUSIVE_DATE,
    RAW_ROOT,
    START_DATE,
    create_session,
    download_file,
    ensure_dir,
    iter_days,
    request_with_retries,
)


SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "30"
KLINE_LIMIT = 1000
FUNDING_LIMIT = 200
KLINE_BAR_MS = 30 * 60 * 1000
FUNDING_BAR_MS = 8 * 60 * 60 * 1000

KLINE_URL = "https://api.bybit.com/v5/market/kline"
FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"
TRADES_ARCHIVE_BASE_URL = f"https://public.bybit.com/trading/{SYMBOL}"
DOWNLOAD_TRADE_ARCHIVES = os.getenv("BYBIT_DOWNLOAD_TRADE_ARCHIVES", "0") == "1"
TRADE_ARCHIVE_ONLY = os.getenv("BYBIT_TRADE_ARCHIVE_ONLY", "0") == "1"
TRADE_ARCHIVE_START_DATE = os.getenv("BYBIT_TRADE_ARCHIVE_START_DATE")
TRADE_ARCHIVE_MIN_DATE = os.getenv("BYBIT_TRADE_ARCHIVE_MIN_DATE")
TRADE_ARCHIVE_MAX_BACKFILL_MONTHS = int(os.getenv("BYBIT_TRADE_ARCHIVE_MAX_BACKFILL_MONTHS", "4"))

OUTPUT_DIR = RAW_ROOT / "bybit" / "futures" / SYMBOL
KLINE_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_ohlcv_30m_{START_DATE}_to_{END_DATE}.csv"
FUNDING_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_funding_8h_{START_DATE}_to_{END_DATE}.csv"
MERGED_CSV_PATH = OUTPUT_DIR / f"{SYMBOL}_30m_with_funding_{START_DATE}_to_{END_DATE}.csv"
TRADES_ARCHIVE_DIR = OUTPUT_DIR / "trades_daily"

DATE_RANGE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$")
TRADE_FILE_RE = re.compile(rf"{SYMBOL}(?P<day>\d{{4}}-\d{{2}}-\d{{2}})\.csv\.gz$")


def _first_day_of_month(day: date) -> date:
    return date(day.year, day.month, 1)


def _subtract_months(month_start: date, months: int) -> date:
    if months <= 0:
        return month_start

    month_index = month_start.month - months
    year = month_start.year
    while month_index <= 0:
        month_index += 12
        year -= 1
    return date(year, month_index, 1)


def _default_trade_archive_floor_date() -> date:
    anchor = _first_day_of_month(END_DATE)
    return _subtract_months(anchor, TRADE_ARCHIVE_MAX_BACKFILL_MONTHS)


def _trade_archive_floor_date() -> date:
    if TRADE_ARCHIVE_MIN_DATE:
        return datetime.strptime(TRADE_ARCHIVE_MIN_DATE, "%Y-%m-%d").date()
    return _default_trade_archive_floor_date()


def _response_list(payload: dict) -> list:
    if payload.get("retCode") not in (0, None):
        raise ValueError(f"Bybit API error: {payload}")
    result = payload.get("result", {})
    return result.get("list", [])


def _path_end_date(path: Path) -> date | None:
    match = DATE_RANGE_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(2), "%Y-%m-%d").date()


def _resolve_latest_csv(prefix: str) -> Path | None:
    if not OUTPUT_DIR.exists():
        return None

    candidates: list[tuple[date, Path]] = []
    for path in OUTPUT_DIR.glob(f"{prefix}_*.csv"):
        end_date = _path_end_date(path)
        if end_date is not None:
            candidates.append((end_date, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def _load_existing_klines(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    numeric_cols = ["open_time", "open", "high", "low", "close", "volume", "turnover", "quote_volume"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return df


def _load_existing_funding(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for column in ("funding_time", "funding_rate"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["funding_time"]).copy()
    df["funding_time"] = df["funding_time"].astype("int64")
    df = df.sort_values("funding_time").drop_duplicates(subset=["funding_time"], keep="last").reset_index(drop=True)
    return df


def _combine_frames(existing_df: pd.DataFrame, new_df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    if existing_df.empty and new_df.empty:
        return pd.DataFrame()

    combined = pd.concat([frame for frame in [existing_df, new_df] if not frame.empty], ignore_index=True)
    combined[key_column] = pd.to_numeric(combined[key_column], errors="coerce")
    combined = combined.dropna(subset=[key_column]).copy()
    combined[key_column] = combined[key_column].astype("int64")
    combined = combined.sort_values(key_column).drop_duplicates(subset=[key_column], keep="last")
    return combined.reset_index(drop=True)


def fetch_klines(*, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    start_ms = start_ms or int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ms = end_ms or int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)
    if start_ms >= end_ms:
        return pd.DataFrame()

    session = create_session()
    cursor_ms = start_ms
    rows: list[list[str]] = []

    while cursor_ms < end_ms:
        window_end = min(cursor_ms + (KLINE_LIMIT * KLINE_BAR_MS) - 1, end_ms - 1)
        params = {
            "category": CATEGORY,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "start": cursor_ms,
            "end": window_end,
            "limit": KLINE_LIMIT,
        }
        response = request_with_retries(session, "GET", KLINE_URL, params=params)
        payload = response.json()
        batch = _response_list(payload)
        if not batch:
            cursor_ms = window_end + 1
            continue

        rows.extend(batch)
        batch_start_times = sorted(int(item[0]) for item in batch)
        cursor_ms = batch_start_times[-1] + KLINE_BAR_MS
        sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )
    numeric_cols = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] < end_ms)].copy()
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["quote_volume"] = df["turnover"]
    df["exchange"] = "bybit"
    df["market_type"] = "futures"
    df["symbol"] = SYMBOL
    first_cols = ["time", "exchange", "market_type", "symbol"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    return df[ordered].reset_index(drop=True)


def fetch_funding_history(*, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    start_ms = start_ms or int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ms = end_ms or int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)
    if start_ms >= end_ms:
        return pd.DataFrame()

    session = create_session()
    cursor_ms = start_ms
    rows: list[dict] = []

    while cursor_ms < end_ms:
        window_end = min(cursor_ms + (FUNDING_LIMIT * FUNDING_BAR_MS) - 1, end_ms - 1)
        params = {
            "category": CATEGORY,
            "symbol": SYMBOL,
            "startTime": cursor_ms,
            "endTime": window_end,
            "limit": FUNDING_LIMIT,
        }
        response = request_with_retries(session, "GET", FUNDING_URL, params=params)
        payload = response.json()
        batch = _response_list(payload)
        if not batch:
            cursor_ms = window_end + 1
            continue

        rows.extend(batch)
        batch_times = sorted(int(item["fundingRateTimestamp"]) for item in batch)
        cursor_ms = batch_times[-1] + FUNDING_BAR_MS
        sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_numeric(df["fundingRateTimestamp"], errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["funding_time"]).copy()
    df["funding_time"] = df["funding_time"].astype("int64")
    df = df.sort_values("funding_time").drop_duplicates(subset=["funding_time"], keep="last")
    df = df[(df["funding_time"] >= start_ms) & (df["funding_time"] < end_ms)].copy()
    df["time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["exchange"] = "bybit"
    df["market_type"] = "futures"
    df["symbol"] = SYMBOL
    first_cols = ["time", "exchange", "market_type", "symbol"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    return df[ordered].reset_index(drop=True)


def merge_klines_and_funding(kline_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    left = kline_df.copy().sort_values("open_time")
    right = funding_df.copy().sort_values("funding_time")
    merged = pd.merge_asof(
        left,
        right[["funding_time", "funding_rate"]],
        left_on="open_time",
        right_on="funding_time",
        direction="backward",
    )
    return merged


def _latest_trade_archive_date() -> date | None:
    if not TRADES_ARCHIVE_DIR.exists():
        return None

    latest_day: date | None = None
    for path in TRADES_ARCHIVE_DIR.glob(f"{SYMBOL}*.csv.gz"):
        match = TRADE_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        day = datetime.strptime(match.group("day"), "%Y-%m-%d").date()
        if latest_day is None or day > latest_day:
            latest_day = day
    return latest_day


def _resolve_trade_archive_start_date(existing_end_date: date | None = None) -> date:
    if TRADE_ARCHIVE_START_DATE:
        start_date = datetime.strptime(TRADE_ARCHIVE_START_DATE, "%Y-%m-%d").date()
    else:
        latest_trade_day = _latest_trade_archive_date()
        if latest_trade_day is not None:
            start_date = latest_trade_day + timedelta(days=1)
        elif existing_end_date is not None:
            start_date = existing_end_date + timedelta(days=1)
        else:
            start_date = START_DATE

    floor_date = _trade_archive_floor_date()
    if start_date < floor_date:
        print(
            "Trade archive start date was clamped to avoid full-history redownload: "
            f"{start_date} -> {floor_date}"
        )
        return floor_date
    return start_date


def download_trade_archives(existing_end_date: date | None = None) -> tuple[int, int]:
    session = create_session()
    ensure_dir(TRADES_ARCHIVE_DIR)
    downloaded = 0
    skipped_existing = 0
    start_date = _resolve_trade_archive_start_date(existing_end_date=existing_end_date)
    floor_date = _trade_archive_floor_date()
    print(
        f"Trade archive download window: {start_date} -> {END_DATE} "
        f"(minimum backfill date: {floor_date})"
    )
    if start_date > END_DATE:
        return downloaded, skipped_existing

    for day in iter_days(start_date, END_DATE):
        file_name = f"{SYMBOL}{day:%Y-%m-%d}.csv.gz"
        destination = TRADES_ARCHIVE_DIR / file_name
        url = f"{TRADES_ARCHIVE_BASE_URL}/{file_name}"
        try:
            status = download_file(session, url, destination)
            if status == "downloaded":
                downloaded += 1
            else:
                skipped_existing += 1
            print(f"[{status}] {file_name}")
        except Exception as exc:
            print(f"[skip] {file_name} -> {exc}")
        sleep(0.02)
    return downloaded, skipped_existing


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    existing_kline_path = _resolve_latest_csv(f"{SYMBOL}_ohlcv_30m")
    existing_funding_path = _resolve_latest_csv(f"{SYMBOL}_funding_8h")
    existing_merged_path = _resolve_latest_csv(f"{SYMBOL}_30m_with_funding")
    existing_merged_end = _path_end_date(existing_merged_path) if existing_merged_path is not None else None

    if TRADE_ARCHIVE_ONLY:
        downloaded, skipped_existing = download_trade_archives(existing_end_date=existing_merged_end)
        print("Trade archive only mode enabled.")
        print(f"Trade archives downloaded: {downloaded}")
        print(f"Trade archives already present: {skipped_existing}")
        print(f"Saved under: {TRADES_ARCHIVE_DIR}")
        return

    existing_kline_df = _load_existing_klines(existing_kline_path)
    existing_funding_df = _load_existing_funding(existing_funding_path)

    end_ms = int(pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC").timestamp() * 1000)
    kline_start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    if not existing_kline_df.empty:
        kline_start_ms = int(existing_kline_df["open_time"].max()) + KLINE_BAR_MS

    funding_start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    if not existing_funding_df.empty:
        funding_start_ms = int(existing_funding_df["funding_time"].max()) + FUNDING_BAR_MS

    new_kline_df = fetch_klines(start_ms=kline_start_ms, end_ms=end_ms)
    new_funding_df = fetch_funding_history(start_ms=funding_start_ms, end_ms=end_ms)

    kline_df = _combine_frames(existing_kline_df, new_kline_df, "open_time")
    funding_df = _combine_frames(existing_funding_df, new_funding_df, "funding_time")
    if kline_df.empty:
        raise ValueError("Bybit kline data is empty after incremental refresh.")
    if funding_df.empty:
        raise ValueError("Bybit funding data is empty after incremental refresh.")

    merged_df = merge_klines_and_funding(kline_df, funding_df)

    kline_df.to_csv(KLINE_CSV_PATH, index=False)
    funding_df.to_csv(FUNDING_CSV_PATH, index=False)
    merged_df.to_csv(MERGED_CSV_PATH, index=False)

    downloaded = 0
    skipped_existing = 0
    if DOWNLOAD_TRADE_ARCHIVES:
        downloaded, skipped_existing = download_trade_archives(existing_end_date=existing_merged_end)
    else:
        print("Trade archive download skipped. Set BYBIT_DOWNLOAD_TRADE_ARCHIVES=1 to enable it.")

    print(f"Kline rows: {len(kline_df)}")
    print(f"Funding rows: {len(funding_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Trade archives downloaded: {downloaded}")
    print(f"Trade archives already present: {skipped_existing}")
    print(f"Saved: {KLINE_CSV_PATH}")
    print(f"Saved: {FUNDING_CSV_PATH}")
    print(f"Saved: {MERGED_CSV_PATH}")


if __name__ == "__main__":
    main()
