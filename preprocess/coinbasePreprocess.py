from __future__ import annotations

from pathlib import Path
import re

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
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "coinbase" / "spot" / "BTC-USD" / "candles"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "coinbaseProcessed.csv"


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


def load_coinbase_15m() -> pd.DataFrame:
    input_path = _resolve_latest_csv(INPUT_DIR, "BTC-USD_candles_15m")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Coinbase input file: {input_path}")

    df = pd.read_csv(input_path)
    required = ["time", "epoch_seconds", "low", "high", "open", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required Coinbase columns in {input_path.name}: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    numeric_columns = ["epoch_seconds", "low", "high", "open", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df


def resample_to_30m(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    working = df.copy()
    working["quote_volume"] = working["volume"] * working["close"]

    aggregated = (
        working.set_index("time")
        .resample("30min", label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            quote_volume=("quote_volume", "sum"),
            source_rows=("close", "size"),
        )
    )

    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).copy()
    incomplete_rows = int((aggregated["source_rows"] != 2).sum())
    aggregated = aggregated.loc[aggregated["source_rows"] == 2].copy()
    aggregated = aggregated.reset_index()
    aggregated["open_time"] = aggregated["time"].map(lambda ts: int(ts.timestamp())).astype("int64")
    aggregated = add_technical_indicators(aggregated)
    aggregated = aggregated.drop(columns=["source_rows"])
    return aggregated, incomplete_rows


def build_output(df_30m: pd.DataFrame) -> pd.DataFrame:
    output = df_30m.copy()
    output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output["open_time"] = output["open_time"].astype("int64")

    prefixed = output.rename(columns={column: f"coinbase_{column}" for column in output.columns})
    first_columns = ["coinbase_time"]
    return prefixed[first_columns + [column for column in prefixed.columns if column not in first_columns]]


def main() -> None:
    raw_df = load_coinbase_15m()
    resampled_df, incomplete_rows = resample_to_30m(raw_df)
    output_df = build_output(resampled_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Input 15m rows: {len(raw_df)}")
    print(f"Dropped incomplete 30m rows: {incomplete_rows}")
    print(f"Output 30m rows: {len(output_df)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
