from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
FED_PATH = PROJECT_ROOT / "data" / "raw" / "netLiq" / "fed_net_liquidity_1d.csv"
FG_PATH = PROJECT_ROOT / "data" / "raw" / "fear-greed" / "alternative_fear_greed_1d.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mergedX.csv"


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns: {missing}")


def load_and_fill_fed(path: Path) -> pd.DataFrame:
    fed_df = pd.read_csv(path)
    _require_columns(fed_df, ["date", "fed_net_liquidity"], "fed_net_liquidity_1d.csv")

    fed_df["date"] = pd.to_datetime(fed_df["date"], errors="coerce")
    fed_df = fed_df.dropna(subset=["date"]).copy()
    fed_df = fed_df[["date", "fed_net_liquidity"]].sort_values("date")
    fed_df = fed_df.drop_duplicates(subset="date", keep="last")

    full_dates = pd.date_range(start=fed_df["date"].min(), end=fed_df["date"].max(), freq="D")
    fed_df = fed_df.set_index("date").reindex(full_dates)
    fed_df["fed_net_liquidity"] = fed_df["fed_net_liquidity"].ffill().bfill()
    fed_df = fed_df.reset_index().rename(columns={"index": "Date"})

    return fed_df


def load_fear_greed(path: Path) -> pd.DataFrame:
    fg_df = pd.read_csv(path)
    _require_columns(fg_df, ["date", "value"], "alternative_fear_greed_1d.csv")

    fg_df["date"] = pd.to_datetime(fg_df["date"], errors="coerce")
    fg_df["value"] = pd.to_numeric(fg_df["value"], errors="coerce")
    fg_df = fg_df.dropna(subset=["date", "value"]).copy()
    fg_df = fg_df[["date", "value"]].sort_values("date")
    fg_df = fg_df.drop_duplicates(subset="date", keep="last")
    fg_df = fg_df.rename(columns={"date": "Date", "value": "fearGreed"})

    return fg_df


def main() -> None:
    fed_df = load_and_fill_fed(FED_PATH)
    fg_df = load_fear_greed(FG_PATH)

    merged_df = fed_df.merge(fg_df, on="Date", how="inner")
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    merged_df["Date"] = merged_df["Date"].dt.strftime("%Y-%m-%d")
    merged_df = merged_df[["Date", "fed_net_liquidity", "fearGreed"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(merged_df)}")
    print(f"Date range: {merged_df['Date'].iloc[0]} -> {merged_df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
