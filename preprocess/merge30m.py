from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COINBASE_PATH = PROJECT_ROOT / "data" / "processed" / "coinbaseProcessed.csv"
BINANCE_PATH = PROJECT_ROOT / "data" / "processed" / "binance_processed.csv"
BYBIT_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_processed.csv"
CME_PATH = PROJECT_ROOT / "data" / "processed" / "cmeProcessed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged30m.csv"


def _load_csv(path: Path, time_column: str, rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed input file: {path}")

    df = pd.read_csv(path)
    if time_column not in df.columns:
        raise KeyError(f"{path.name}: missing time column `{time_column}`")

    df = df.rename(columns=rename_map or {})
    if time_column == "time":
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).copy()
    else:
        df["time"] = pd.to_datetime(df[time_column], errors="coerce")
        df = df.dropna(subset=["time"]).copy()
        df = df.drop(columns=[time_column])
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    return df


def load_coinbase() -> pd.DataFrame:
    return _load_csv(COINBASE_PATH, time_column="coinbase_time")


def load_binance() -> pd.DataFrame:
    return _load_csv(
        BINANCE_PATH,
        time_column="time",
        rename_map={"open_time": "binance_open_time"},
    )


def load_bybit() -> pd.DataFrame:
    return _load_csv(BYBIT_PATH, time_column="bybit_time")


def load_cme() -> pd.DataFrame:
    return _load_csv(CME_PATH, time_column="CME_time")


def _helper_columns_to_drop(columns: list[str]) -> list[str]:
    to_drop: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered == "open_time" or lowered.endswith("_open_time"):
            to_drop.append(column)
    return to_drop


def build_merged() -> tuple[pd.DataFrame, dict[str, int]]:
    coinbase_df = load_coinbase()
    binance_df = load_binance()
    bybit_df = load_bybit()
    cme_df = load_cme()

    row_counts = {
        "coinbase_rows": len(coinbase_df),
        "binance_rows": len(binance_df),
        "bybit_rows": len(bybit_df),
        "cme_rows": len(cme_df),
    }

    merged = coinbase_df.merge(binance_df, on="time", how="inner", validate="one_to_one")
    merged = merged.merge(bybit_df, on="time", how="inner", validate="one_to_one")
    merged = merged.merge(cme_df, on="time", how="inner", validate="one_to_one")
    merged = merged.sort_values("time").reset_index(drop=True)

    if merged.empty:
        raise ValueError("No common 30m timestamps were found across the four processed sources.")

    common_times = set(merged["time"])
    overlap_exclusions = {
        "coinbase_excluded": int((~coinbase_df["time"].isin(common_times)).sum()),
        "binance_excluded": int((~binance_df["time"].isin(common_times)).sum()),
        "bybit_excluded": int((~bybit_df["time"].isin(common_times)).sum()),
        "cme_excluded": int((~cme_df["time"].isin(common_times)).sum()),
    }
    row_counts.update(overlap_exclusions)

    dropped_helper_columns = _helper_columns_to_drop(list(merged.columns))
    merged = merged.drop(columns=dropped_helper_columns)
    row_counts["dropped_helper_columns"] = len(dropped_helper_columns)

    merged["time"] = merged["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ordered_columns = ["time"] + [column for column in merged.columns if column != "time"]
    return merged[ordered_columns], row_counts


def main() -> None:
    merged_df, stats = build_merged()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Coinbase rows read: {stats['coinbase_rows']}")
    print(f"Binance rows read: {stats['binance_rows']}")
    print(f"Bybit rows read: {stats['bybit_rows']}")
    print(f"CME rows read: {stats['cme_rows']}")
    print(f"Coinbase timestamps excluded by overlap: {stats['coinbase_excluded']}")
    print(f"Binance timestamps excluded by overlap: {stats['binance_excluded']}")
    print(f"Bybit timestamps excluded by overlap: {stats['bybit_excluded']}")
    print(f"CME timestamps excluded by overlap: {stats['cme_excluded']}")
    print(f"Helper columns dropped: {stats['dropped_helper_columns']}")
    print(f"Output rows: {len(merged_df)}")
    print(f"Date range: {merged_df['time'].iloc[0]} -> {merged_df['time'].iloc[-1]}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
