from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time
from pathlib import Path
import os
import re
import warnings

import databento as db
import numpy as np
import pandas as pd


RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATE_RANGE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$")
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cme" / "databento" / "BTC_v_0"
TRADES_DIR = INPUT_DIR / "trades"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cmeProcessed.csv"
TRADE_AGG_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "cme_trade_metrics_30m.csv"

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
TRADE_COLUMNS = ["count", "delta", "trade_volume"]
INDICATOR_COLUMNS = [
    "rsi_14",
    "atr_14",
    "bb_position_20_2",
    "macd",
    "macd_signal",
    "macd_hist",
]

DAILY_CLOSE_TIMES = {time(20, 30), time(21, 30)}
SESSION_OPEN_TIMES = {time(22, 0), time(23, 0), time(0, 0)}
EARLY_CLOSE_TIMES = {
    time(13, 0),
    time(14, 0),
    time(14, 30),
    time(15, 30),
    time(16, 30),
    time(17, 30),
    time(18, 30),
}

DEFAULT_WORKERS = min(4, os.cpu_count() or 4)
MAX_WORKERS = int(os.getenv("CME_PREPROCESS_WORKERS", str(DEFAULT_WORKERS)))
USE_TRADE_CACHE = os.getenv("CME_USE_TRADE_CACHE", "1") == "1"


def _resolve_latest_csv(directory: Path, prefix: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for path in directory.glob(f"{prefix}_*.csv"):
        match = DATE_RANGE_RE.search(path.name)
        if match:
            candidates.append((match.group(2), path))
    if not candidates:
        raise FileNotFoundError(f"No input CSV matched `{prefix}_*.csv` under {directory}")
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close"):
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
            high - low,
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


def load_cme_30m() -> pd.DataFrame:
    input_path = _resolve_latest_csv(INPUT_DIR, "BTC_v_0_30m")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing CME input file: {input_path}")

    df = pd.read_csv(input_path)
    required = [
        "time",
        "exchange",
        "vendor",
        "market_type",
        "continuous_symbol",
        "ts_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "instrument_id",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required CME columns in {input_path.name}: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "instrument_id"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "count", "delta", "trade_volume"])


def _read_trade_file(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = db.DBNStore.from_bytes(path.read_bytes())
        trades = store.to_df(schema="trades")

    real_warnings = [str(w.message) for w in caught if "unclosed file" not in str(w.message)]
    truncated_warnings = [
        message
        for message in real_warnings
        if "truncated or contains an incomplete record" in message
    ]
    if truncated_warnings:
        raise ValueError(f"Corrupted Databento trade chunk detected: {path}")

    return trades


def _aggregate_trade_file(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    trades = _read_trade_file(path)
    if trades.empty:
        return _empty_trade_frame()

    trades = trades[["ts_event", "size", "side"]].copy()
    trades["ts_event"] = pd.to_datetime(trades["ts_event"], utc=True, errors="coerce")
    trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    trades["side"] = trades["side"].astype("string").str.upper()
    trades = trades.dropna(subset=["ts_event", "size"]).copy()
    if trades.empty:
        return _empty_trade_frame()

    trades["size"] = trades["size"].astype("int64")
    trades["time"] = trades["ts_event"].dt.floor("30min")
    trades["signed_size"] = np.select(
        [trades["side"] == "B", trades["side"] == "A"],
        [trades["size"], -trades["size"]],
        default=0,
    )

    aggregated = trades.groupby("time", as_index=False).agg(
        count=("size", "count"),
        delta=("signed_size", "sum"),
        trade_volume=("size", "sum"),
    )
    for column in ("count", "delta", "trade_volume"):
        aggregated[column] = aggregated[column].astype("int64")
    return aggregated


def _trade_files() -> list[Path]:
    if not TRADES_DIR.exists():
        raise FileNotFoundError(f"Missing CME trades directory: {TRADES_DIR}")

    trade_files = sorted(TRADES_DIR.glob("BTC_v_0_trades_*.dbn.zst"))
    if not trade_files:
        raise FileNotFoundError(f"No CME trade archives found in: {TRADES_DIR}")
    return trade_files


def _trade_cache_is_fresh(trade_files: list[Path]) -> bool:
    if not TRADE_AGG_CACHE_PATH.exists():
        return False

    cache_mtime = TRADE_AGG_CACHE_PATH.stat().st_mtime
    latest_trade_mtime = max(path.stat().st_mtime for path in trade_files)
    return cache_mtime >= latest_trade_mtime


def aggregate_trade_archives() -> pd.DataFrame:
    trade_files = _trade_files()

    if USE_TRADE_CACHE and _trade_cache_is_fresh(trade_files):
        cached = pd.read_csv(TRADE_AGG_CACHE_PATH)
        cached["time"] = pd.to_datetime(cached["time"], utc=True, errors="coerce")
        for column in TRADE_COLUMNS:
            cached[column] = pd.to_numeric(cached[column], errors="coerce")
        cached = cached.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        print(f"Loaded cached trade metrics: {TRADE_AGG_CACHE_PATH}")
        return cached

    worker_count = max(1, MAX_WORKERS)
    print(f"Aggregating {len(trade_files)} CME trade archives with {worker_count} worker(s)...")

    aggregated_parts: list[pd.DataFrame] = []
    completed = 0
    total_files = len(trade_files)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_aggregate_trade_file, str(path)): path.name
            for path in trade_files
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if not result.empty:
                aggregated_parts.append(result)

            if completed == 1 or completed % 10 == 0 or completed == total_files:
                print(f"Aggregated trade archive {completed}/{total_files}: {futures[future]}")

    if not aggregated_parts:
        raise ValueError("CME trade archives were found, but no aggregated trade rows were produced.")

    out = pd.concat(aggregated_parts, ignore_index=True)
    out = out.groupby("time", as_index=False).agg(
        count=("count", "sum"),
        delta=("delta", "sum"),
        trade_volume=("trade_volume", "sum"),
    )
    out = out.sort_values("time").reset_index(drop=True)
    for column in TRADE_COLUMNS:
        out[column] = out[column].astype("int64")

    TRADE_AGG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_to_save = out.copy()
    out_to_save["time"] = out_to_save["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_to_save.to_csv(TRADE_AGG_CACHE_PATH, index=False)
    print(f"Saved trade metrics cache: {TRADE_AGG_CACHE_PATH}")
    return out


def _is_likely_scheduled_closure(
    prev_time: pd.Timestamp,
    next_time: pd.Timestamp,
    missing_count: int,
) -> bool:
    gap = next_time - prev_time

    if gap >= pd.Timedelta(hours=4):
        return True
    if prev_time.weekday() == 4 and next_time.weekday() in {6, 0}:
        return True
    if prev_time.time() in DAILY_CLOSE_TIMES and next_time.time() in SESSION_OPEN_TIMES:
        return True
    if prev_time.time() in EARLY_CLOSE_TIMES and missing_count >= 2:
        return True
    return False


def _linspace_between(start_value: float, end_value: float, points: int) -> np.ndarray:
    return np.linspace(start_value, end_value, points + 2, dtype="float64")[1:-1]


def build_filled_timeline(cme_df: pd.DataFrame, trades_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    full_time_index = pd.date_range(
        start=cme_df["time"].min(),
        end=cme_df["time"].max(),
        freq="30min",
        tz="UTC",
    )
    full_df = pd.DataFrame({"time": full_time_index})
    full_df = full_df.merge(cme_df, on="time", how="left")
    full_df = full_df.merge(trades_df, on="time", how="left")
    full_df["bar_missing"] = full_df["open"].isna()
    full_df["fill_type"] = np.where(full_df["bar_missing"], "unclassified", "observed")

    scheduled_rows = 0
    open_market_rows = 0
    open_market_rows_without_trades = 0

    missing_mask = full_df["bar_missing"].to_numpy()
    row_count = len(full_df)
    idx = 0
    while idx < row_count:
        if not missing_mask[idx]:
            idx += 1
            continue

        start_idx = idx
        while idx < row_count and missing_mask[idx]:
            idx += 1
        end_idx = idx - 1

        prev_idx = start_idx - 1
        next_idx = end_idx + 1
        if prev_idx < 0 or next_idx >= row_count:
            raise ValueError("Encountered a missing-bar block at the edge of the CME series.")

        block = full_df.iloc[start_idx : end_idx + 1]
        has_trade_activity = block["trade_volume"].fillna(0).gt(0).any()
        if has_trade_activity:
            fill_type = "open_market"
        elif _is_likely_scheduled_closure(
            prev_time=full_df.at[prev_idx, "time"],
            next_time=full_df.at[next_idx, "time"],
            missing_count=len(block),
        ):
            fill_type = "scheduled_closure"
        else:
            fill_type = "open_market"
            open_market_rows_without_trades += len(block)

        full_df.loc[start_idx : end_idx, "fill_type"] = fill_type
        if fill_type == "scheduled_closure":
            scheduled_rows += len(block)
            previous_values = full_df.loc[prev_idx, ["open", "high", "low", "close"]]
            for column in ("open", "high", "low", "close"):
                full_df.loc[start_idx : end_idx, column] = previous_values[column]
            full_df.loc[start_idx : end_idx, "volume"] = 0.0
        else:
            open_market_rows += len(block)
            point_count = len(block)
            for column in BAR_COLUMNS:
                start_value = float(full_df.at[prev_idx, column])
                end_value = float(full_df.at[next_idx, column])
                full_df.loc[start_idx : end_idx, column] = _linspace_between(
                    start_value,
                    end_value,
                    point_count,
                )

    non_closure_mask = full_df["fill_type"] != "scheduled_closure"
    trade_work = full_df.loc[non_closure_mask, ["time", "count", "delta"]].copy()
    actual_trade_rows = int(trade_work["count"].notna().sum())
    missing_trade_rows_for_interpolation = int(trade_work["count"].isna().sum())

    trade_work["count"] = pd.to_numeric(trade_work["count"], errors="coerce")
    trade_work["delta"] = pd.to_numeric(trade_work["delta"], errors="coerce")
    trade_work["count"] = trade_work["count"].interpolate(method="linear", limit_area="inside")
    trade_work["delta"] = trade_work["delta"].interpolate(method="linear", limit_area="inside")
    trade_work["count"] = trade_work["count"].fillna(0.0).round().clip(lower=0).astype("int64")
    trade_work["delta"] = trade_work["delta"].fillna(0.0).round().astype("int64")

    full_df = full_df.drop(columns=["count", "delta"])
    full_df = full_df.merge(trade_work, on="time", how="left")
    full_df.loc[full_df["fill_type"] == "scheduled_closure", "count"] = 0
    full_df.loc[full_df["fill_type"] == "scheduled_closure", "delta"] = 0
    full_df["count"] = full_df["count"].fillna(0).astype("int64")
    full_df["delta"] = full_df["delta"].fillna(0).astype("int64")
    full_df["trade_volume"] = pd.to_numeric(full_df["trade_volume"], errors="coerce").fillna(0).astype("int64")
    full_df["cvd"] = full_df["delta"].cumsum().astype("int64")
    full_df["open_time"] = full_df["time"].map(lambda ts: int(ts.timestamp())).astype("int64")

    indicator_input = full_df.loc[non_closure_mask, ["time", "open", "high", "low", "close"]].copy()
    indicator_output = add_technical_indicators(indicator_input)
    indicator_output = indicator_output[["time"] + INDICATOR_COLUMNS]
    full_df = full_df.merge(indicator_output, on="time", how="left")
    full_df[INDICATOR_COLUMNS] = full_df[INDICATOR_COLUMNS].ffill()

    stats = {
        "scheduled_closure_rows": scheduled_rows,
        "open_market_rows": open_market_rows,
        "open_market_rows_without_trades": open_market_rows_without_trades,
        "actual_trade_rows": actual_trade_rows,
        "interpolated_trade_rows": missing_trade_rows_for_interpolation,
    }
    return full_df, stats


def build_output(full_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    observed_mask = full_df["fill_type"] == "observed"
    volume_mismatch_count = int(
        (
            full_df.loc[observed_mask, "volume"].round().astype("int64")
            != full_df.loc[observed_mask, "trade_volume"]
        ).sum()
    )

    drop_columns = [
        "exchange",
        "vendor",
        "market_type",
        "continuous_symbol",
        "ts_event",
        "symbol",
        "instrument_id",
        "trade_volume",
        "bar_missing",
        "fill_type",
    ]
    output = full_df.drop(columns=[column for column in drop_columns if column in full_df.columns]).copy()
    output = output.sort_values("time").reset_index(drop=True)
    output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    prefixed = output.rename(columns={column: f"CME_{column}" for column in output.columns})
    first_columns = ["CME_time"]
    ordered = first_columns + [column for column in prefixed.columns if column not in first_columns]
    return prefixed[ordered], volume_mismatch_count


def main() -> None:
    cme_df = load_cme_30m()
    trades_df = aggregate_trade_archives()
    full_df, stats = build_filled_timeline(cme_df, trades_df)
    output_df, volume_mismatch_count = build_output(full_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Input raw 30m rows: {len(cme_df)}")
    print(f"Aggregated trade rows: {len(trades_df)}")
    print(f"Scheduled closure rows filled: {stats['scheduled_closure_rows']}")
    print(f"Unexpected market-open rows filled: {stats['open_market_rows']}")
    print(f"Unexpected market-open rows without direct trades: {stats['open_market_rows_without_trades']}")
    print(f"Rows with actual trade metrics before interpolation: {stats['actual_trade_rows']}")
    print(f"Rows with interpolated trade metrics: {stats['interpolated_trade_rows']}")
    print(f"Non-closure volume mismatches vs trade sums: {volume_mismatch_count}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
