from __future__ import annotations

from datetime import timedelta
from time import sleep

import pandas as pd

try:
    from _common import (
        END_DATE,
        END_EXCLUSIVE_DATE,
        RAW_ROOT,
        START_DATE,
        create_session,
        ensure_dir,
        isoformat_z,
        request_with_retries,
        utc_start_of_day,
    )
except ModuleNotFoundError:
    from dataDownloaders._common import (
        END_DATE,
        END_EXCLUSIVE_DATE,
        RAW_ROOT,
        START_DATE,
        create_session,
        ensure_dir,
        isoformat_z,
        request_with_retries,
        utc_start_of_day,
    )


PRODUCT_ID = "BTC-USD"
GRANULARITY_SECONDS = 900
MAX_CANDLES_PER_REQUEST = 300
API_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"
OUTPUT_DIR = RAW_ROOT / "coinbase" / "spot" / PRODUCT_ID / "candles"
RAW_15M_CSV_PATH = OUTPUT_DIR / f"{PRODUCT_ID}_candles_15m_{START_DATE}_to_{END_DATE}.csv"


def fetch_15m_candles() -> pd.DataFrame:
    session = create_session()
    start_dt = utc_start_of_day(START_DATE)
    end_exclusive_dt = utc_start_of_day(END_EXCLUSIVE_DATE)
    step = timedelta(seconds=GRANULARITY_SECONDS * (MAX_CANDLES_PER_REQUEST - 1))
    cursor = start_dt
    rows: list[list[float]] = []

    while cursor < end_exclusive_dt:
        window_end = min(cursor + step, end_exclusive_dt)
        params = {
            "start": isoformat_z(cursor),
            "end": isoformat_z(window_end),
            "granularity": GRANULARITY_SECONDS,
        }
        response = request_with_retries(session, "GET", API_URL, params=params)
        batch = response.json()
        if batch:
            rows.extend(batch)
        cursor = window_end
        sleep(0.12)

    if not rows:
        raise ValueError("Coinbase candles endpoint returned no rows for the requested range.")

    df = pd.DataFrame(
        rows,
        columns=["epoch_seconds", "low", "high", "open", "close", "volume"],
    )
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["epoch_seconds"]).copy()
    df["epoch_seconds"] = df["epoch_seconds"].astype("int64")
    df["time"] = pd.to_datetime(df["epoch_seconds"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    df = df[
        (df["time"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (df["time"] < pd.Timestamp(END_EXCLUSIVE_DATE, tz="UTC"))
    ].copy()
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ordered = ["time", "epoch_seconds", "low", "high", "open", "close", "volume"]
    return df[ordered].reset_index(drop=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    raw_15m = fetch_15m_candles()
    raw_15m.to_csv(RAW_15M_CSV_PATH, index=False)

    print(f"15m rows: {len(raw_15m)}")
    print(f"Saved: {RAW_15M_CSV_PATH}")


if __name__ == "__main__":
    main()
