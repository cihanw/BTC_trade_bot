from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import re

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
TRADE_FILE_RE = re.compile(r"BTCUSDT(?P<day>\d{4}-\d{2}-\d{2})\.csv\.gz$")
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bybit" / "futures" / "BTCUSDT"
TRADES_DIR = INPUT_DIR / "trades_daily"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_processed.csv"
TRADE_AGG_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_trade_metrics_30m.csv"

DEFAULT_WORKERS = os.cpu_count() or 4
MAX_WORKERS = int(os.getenv("BYBIT_PREPROCESS_WORKERS", str(DEFAULT_WORKERS)))
USE_TRADE_CACHE = os.getenv("BYBIT_USE_TRADE_CACHE", "1") == "1"


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


def _empty_trade_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "count", "delta", "cvd"])


def _normalize_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_trade_metrics()

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
    out = out.dropna(subset=["time", "count", "delta"]).copy()
    out["count"] = out["count"].round().astype("int64")
    out["delta"] = out["delta"].astype("float64")
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    out["cvd"] = out["delta"].cumsum()
    return out


def _save_trade_metrics_cache(df: pd.DataFrame) -> None:
    TRADE_AGG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(TRADE_AGG_CACHE_PATH, index=False)
    print(f"Saved trade metrics cache: {TRADE_AGG_CACHE_PATH}")


def _load_trade_metrics_cache() -> pd.DataFrame:
    if not TRADE_AGG_CACHE_PATH.exists():
        return _empty_trade_metrics()
    cached = pd.read_csv(TRADE_AGG_CACHE_PATH)
    return _normalize_trade_metrics(cached)


def _load_existing_processed_trade_metrics() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return _empty_trade_metrics()

    required = {"bybit_time", "bybit_count", "bybit_delta"}
    existing = pd.read_csv(OUTPUT_PATH, usecols=lambda column: column in required)
    if required.difference(existing.columns):
        return _empty_trade_metrics()

    existing = existing.rename(
        columns={
            "bybit_time": "time",
            "bybit_count": "count",
            "bybit_delta": "delta",
        }
    )
    return _normalize_trade_metrics(existing)


def _trade_file_day(path: Path) -> pd.Timestamp | None:
    match = TRADE_FILE_RE.fullmatch(path.name)
    if not match:
        return None
    return pd.Timestamp(match.group("day"))


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


def load_bybit_30m() -> pd.DataFrame:
    input_path = _resolve_latest_csv(INPUT_DIR, "BTCUSDT_30m_with_funding")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Bybit input file: {input_path}")

    df = pd.read_csv(input_path)
    required = [
        "time",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "quote_volume",
        "funding_time",
        "funding_rate",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required Bybit columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _aggregate_trade_file(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    trades = pd.read_csv(
        path,
        compression="gzip",
        usecols=["timestamp", "side", "size"],
        dtype={"timestamp": "float64", "side": "string", "size": "float64"},
    )
    trades = trades.dropna(subset=["timestamp", "side", "size"]).copy()
    if trades.empty:
        return pd.DataFrame(columns=["time", "count", "delta"])

    trades["time"] = pd.to_datetime(trades["timestamp"], unit="s", utc=True).dt.floor("30min")
    trades["delta"] = np.where(
        trades["side"].str.upper() == "BUY",
        trades["size"],
        -trades["size"],
    )

    aggregated = trades.groupby("time", as_index=False).agg(
        count=("delta", "count"),
        delta=("delta", "sum"),
    )
    return aggregated


def aggregate_trade_archives() -> pd.DataFrame:
    base_trade_df = _load_trade_metrics_cache() if USE_TRADE_CACHE else _empty_trade_metrics()
    if base_trade_df.empty:
        base_trade_df = _load_existing_processed_trade_metrics()

    trade_files = sorted(TRADES_DIR.glob("BTCUSDT*.csv.gz")) if TRADES_DIR.exists() else []
    if not trade_files:
        if base_trade_df.empty:
            raise FileNotFoundError(f"No Bybit trade archives found in: {TRADES_DIR}")
        print("Using existing processed Bybit trade metrics; no raw trade archives were found.")
        if USE_TRADE_CACHE and not TRADE_AGG_CACHE_PATH.exists():
            _save_trade_metrics_cache(base_trade_df)
        return base_trade_df

    cutoff_time = base_trade_df["time"].max() if not base_trade_df.empty else None
    eligible_files = trade_files
    if cutoff_time is not None:
        cutoff_day = cutoff_time.tz_convert(None).normalize()
        eligible_files = [
            path
            for path in trade_files
            if (_trade_file_day(path) is None) or (_trade_file_day(path) >= cutoff_day)
        ]

    if not eligible_files:
        print("Loaded cached or existing Bybit trade metrics; no newer trade archives needed.")
        return base_trade_df

    worker_count = max(1, MAX_WORKERS)
    print(f"Aggregating {len(eligible_files)} Bybit trade archives with {worker_count} worker(s)...")

    daily_parts: list[pd.DataFrame] = []
    completed = 0
    total_files = len(eligible_files)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_aggregate_trade_file, str(path)): path.name
            for path in eligible_files
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if not result.empty:
                daily_parts.append(result)

            if completed == 1 or completed % 100 == 0 or completed == total_files:
                print(f"Aggregated trade archive {completed}/{total_files}: {futures[future]}")

    if daily_parts:
        new_metrics = pd.concat(daily_parts, ignore_index=True)
        new_metrics = new_metrics.groupby("time", as_index=False).agg(
            count=("count", "sum"),
            delta=("delta", "sum"),
        )
        if cutoff_time is not None:
            new_metrics = new_metrics.loc[new_metrics["time"] > cutoff_time].copy()
    else:
        new_metrics = pd.DataFrame(columns=["time", "count", "delta"])

    combined = pd.concat(
        [
            base_trade_df[["time", "count", "delta"]] if not base_trade_df.empty else pd.DataFrame(columns=["time", "count", "delta"]),
            new_metrics,
        ],
        ignore_index=True,
    )
    combined = combined.groupby("time", as_index=False).agg(
        count=("count", "sum"),
        delta=("delta", "sum"),
    )
    combined = _normalize_trade_metrics(combined)
    if combined.empty:
        raise ValueError("Bybit trade metrics could not be built from the existing processed file or the available trade archives.")

    if USE_TRADE_CACHE:
        _save_trade_metrics_cache(combined)
    return combined


def build_output(bybit_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    merged = bybit_df.merge(trades_df, on="time", how="left")
    merged["count"] = merged["count"].fillna(0).astype("int64")
    merged["delta"] = merged["delta"].fillna(0.0)
    merged["cvd"] = merged["delta"].cumsum()

    merged = add_technical_indicators(merged)

    drop_columns = ["exchange", "market_type", "symbol", "turnover", "funding_time", "volume"]
    merged = merged.drop(columns=[col for col in drop_columns if col in merged.columns])

    merged["time"] = merged["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    prefixed = merged.rename(columns={col: f"bybit_{col}" for col in merged.columns})
    first_cols = ["bybit_time"]
    return prefixed[first_cols + [col for col in prefixed.columns if col not in first_cols]]


def main() -> None:
    bybit_df = load_bybit_30m()
    trades_df = aggregate_trade_archives()
    output_df = build_output(bybit_df, trades_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Input 30m rows: {len(bybit_df)}")
    print(f"Aggregated trade rows: {len(trades_df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
