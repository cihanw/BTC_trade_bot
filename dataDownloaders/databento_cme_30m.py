from __future__ import annotations

import csv
from datetime import datetime, timedelta
import os
from pathlib import Path

import databento as db

from _common import END_DATE, END_EXCLUSIVE_DATE, RAW_ROOT, START_DATE, ensure_dir, chunk_datetimes, utc_start_of_day


API_KEY = os.getenv("DATABENTO_API_KEY", "")
DATASET = "GLBX.MDP3"
SCHEMA = "trades"
SYMBOL = "BTC.v.0"
STYPE_IN = "continuous"
CHUNK_DAYS = 31

OUTPUT_DIR = RAW_ROOT / "cme" / "databento" / SYMBOL.replace(".", "_") / SCHEMA
MANIFEST_PATH = OUTPUT_DIR / f"{SYMBOL.replace('.', '_')}_{SCHEMA}_manifest_{START_DATE}_to_{END_DATE}.csv"


def _chunk_output_path(chunk_start: datetime, chunk_end: datetime) -> Path:
    inclusive_end = (chunk_end - timedelta(microseconds=1)).date()
    filename = f"{SYMBOL.replace('.', '_')}_{SCHEMA}_{chunk_start.date()}_to_{inclusive_end}.dbn.zst"
    return OUTPUT_DIR / filename


def _write_manifest(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "dataset",
        "schema",
        "requested_symbol",
        "stype_in",
        "chunk_start_utc",
        "chunk_end_exclusive_utc",
        "status",
        "file_path",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_trade_chunks() -> list[dict[str, str]]:
    if not API_KEY or API_KEY == "YOUR_DATABENTO_API_KEY":
        raise ValueError("Set DATABENTO_API_KEY in the environment before running this script.")

    ensure_dir(OUTPUT_DIR)
    client = db.Historical(API_KEY)
    start_dt = utc_start_of_day(START_DATE)
    end_exclusive_dt = utc_start_of_day(END_EXCLUSIVE_DATE)

    manifest_rows: list[dict[str, str]] = []
    for chunk_start, chunk_end in chunk_datetimes(
        start=start_dt,
        end_exclusive=end_exclusive_dt,
        chunk_size=timedelta(days=CHUNK_DAYS),
    ):
        chunk_path = _chunk_output_path(chunk_start, chunk_end)
        status = "exists"
        if not chunk_path.exists():
            client.timeseries.get_range(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=[SYMBOL],
                stype_in=STYPE_IN,
                start=chunk_start,
                end=chunk_end,
                path=chunk_path,
            )
            status = "downloaded"

        manifest_rows.append(
            {
                "dataset": DATASET,
                "schema": SCHEMA,
                "requested_symbol": SYMBOL,
                "stype_in": STYPE_IN,
                "chunk_start_utc": chunk_start.isoformat(),
                "chunk_end_exclusive_utc": chunk_end.isoformat(),
                "status": status,
                "file_path": str(chunk_path),
            }
        )
        print(f"{status.upper()}: {chunk_start} -> {chunk_end} :: {chunk_path.name}")

    return manifest_rows


def main() -> None:
    manifest_rows = download_trade_chunks()
    _write_manifest(manifest_rows)

    downloaded_count = sum(row["status"] == "downloaded" for row in manifest_rows)
    existing_count = sum(row["status"] == "exists" for row in manifest_rows)

    print(f"Chunk files tracked: {len(manifest_rows)}")
    print(f"New chunks downloaded: {downloaded_count}")
    print(f"Existing chunks reused: {existing_count}")
    print(f"Manifest saved: {MANIFEST_PATH}")
    print("Saved raw Databento trades only; derive OHLCV/count/delta/cvd in a later preprocessing step.")
    print("Databento trades do not include a direct USD quote-volume field, so later notional volume should be derived from raw price, size, and contract specs if needed.")


if __name__ == "__main__":
    main()
