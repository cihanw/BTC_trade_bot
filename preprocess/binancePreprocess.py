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
FUTURES_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "binance" / "futures" / "BTCUSDT" / "30m"
SPOT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "binance" / "spot" / "BTCUSDT" / "30m"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "binance_processed.csv"

COMMON_KEYS = ["time", "open_time"]


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


def load_source_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Binance input file: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required Binance columns in {path.name}: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["time", "open_time"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def prepare_source(
    df: pd.DataFrame,
    prefix: str,
    drop_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "quote_volume",
        "count",
        "taker_buy_quote_volume",
        "funding_rate_8h",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["quote_volume"] = out["quote_volume"].fillna(0.0)
    out["taker_buy_quote_volume"] = out["taker_buy_quote_volume"].fillna(0.0)
    out["delta"] = (2 * out["taker_buy_quote_volume"]) - out["quote_volume"]
    out["cvd"] = out["delta"].cumsum()

    out = add_technical_indicators(out)

    to_drop = [column for column in drop_columns if column in out.columns]
    to_drop.extend([column for column in ["taker_buy_quote_volume"] if column in out.columns])
    out = out.drop(columns=to_drop)

    rename_map = {
        column: f"{prefix}{column}"
        for column in out.columns
        if column not in COMMON_KEYS
    }
    return out.rename(columns=rename_map)


def build_output(futures_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    futures_processed = prepare_source(
        df=futures_df,
        prefix="binanceFutures_",
        drop_columns=[
            "exchange",
            "market_type",
            "symbol",
            "close_time",
            "ignore",
            "calc_time",
            "volume",
            "taker_buy_volume",
        ],
    )
    spot_processed = prepare_source(
        df=spot_df,
        prefix="binanceSpot_",
        drop_columns=[
            "exchange",
            "market_type",
            "symbol",
            "close_time",
            "volume",
            "taker_buy_volume",
            "ignore",
        ],
    )

    merged = futures_processed.merge(
        spot_processed,
        on=COMMON_KEYS,
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("time").reset_index(drop=True)
    merged["time"] = merged["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged["open_time"] = merged["open_time"].astype("int64")
    return merged


def main() -> None:
    futures_df = load_source_csv(
        _resolve_latest_csv(FUTURES_INPUT_DIR, "BTCUSDT_30m_with_funding"),
        required_columns=[
            "time",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "quote_volume",
            "count",
            "taker_buy_quote_volume",
            "funding_rate_8h",
        ],
    )
    spot_df = load_source_csv(
        _resolve_latest_csv(SPOT_INPUT_DIR, "BTCUSDT_30m"),
        required_columns=[
            "time",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "quote_volume",
            "count",
            "taker_buy_quote_volume",
        ],
    )

    output_df = build_output(futures_df=futures_df, spot_df=spot_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Futures rows: {len(futures_df)}")
    print(f"Spot rows: {len(spot_df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
