from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FED_PATH = PROJECT_ROOT / "data" / "raw" / "netLiq" / "fed_net_liquidity_1d.csv"
OI_PATH = PROJECT_ROOT / "data" / "raw" / "liquidations-OI" / "coinalyze_open_interest_1d.csv"
LSR_PATH = PROJECT_ROOT / "data" / "raw" / "liquidations-OI" / "coinalyze_long_short_ratio_1d.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged1d.csv"


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns: {missing}")


def _normalize_date_column(series: pd.Series, utc: bool) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=utc)
    if utc:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize()


def load_fed() -> pd.DataFrame:
    df = pd.read_csv(FED_PATH)
    required = ["date", "fed_net_liquidity", "tga", "rrp"]
    _require_columns(df, required, FED_PATH.name)

    df["Date"] = _normalize_date_column(df["date"], utc=False)
    for column in ("fed_net_liquidity", "tga", "rrp"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "fed_net_liquidity", "tga", "rrp"]).copy()
    df = df[["Date", "fed_net_liquidity", "tga", "rrp"]]
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df


def load_open_interest() -> pd.DataFrame:
    df = pd.read_csv(OI_PATH)
    required = ["datetime", "oi_open", "oi_high", "oi_low", "oi_close"]
    _require_columns(df, required, OI_PATH.name)

    df["Date"] = _normalize_date_column(df["datetime"], utc=True)
    for column in ("oi_open", "oi_high", "oi_low", "oi_close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "oi_open", "oi_high", "oi_low", "oi_close"]).copy()
    df = df[["Date", "oi_open", "oi_high", "oi_low", "oi_close"]]
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df


def load_long_short_ratio() -> pd.DataFrame:
    df = pd.read_csv(LSR_PATH)
    required = ["datetime", "long_ratio", "short_ratio"]
    _require_columns(df, required, LSR_PATH.name)

    df["Date"] = _normalize_date_column(df["datetime"], utc=True)
    for column in ("long_ratio", "short_ratio"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "long_ratio", "short_ratio"]).copy()
    df = df[["Date", "long_ratio", "short_ratio"]]
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df


def missing_dates_within_range(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    full_range = pd.date_range(start=start, end=end, freq="D")
    present = set(dates.dropna())
    return [date for date in full_range if date not in present]


def _overlap_bounds(frames: list[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(frame["Date"].min() for frame in frames)
    end = min(frame["Date"].max() for frame in frames)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("The daily sources do not share a valid overlapping date range.")
    return start, end


def _reindex_and_ffill(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    full_range = pd.date_range(start=start, end=end, freq="D")
    indexed = df.set_index("Date").sort_index()
    fill_index = indexed.index.union(full_range)
    out = indexed.reindex(fill_index).sort_index().ffill().reindex(full_range)
    if out.isna().any().any():
        raise ValueError("Forward-fill left unresolved daily gaps at the start of the overlap range.")
    out = out.reset_index().rename(columns={"index": "Date"})
    return out


def build_merged() -> tuple[pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    fed_df = load_fed()
    oi_df = load_open_interest()
    lsr_df = load_long_short_ratio()

    start, end = _overlap_bounds([fed_df, oi_df, lsr_df])
    missing_info = {
        "fed": missing_dates_within_range(fed_df["Date"], start, end),
        "oi": missing_dates_within_range(oi_df["Date"], start, end),
        "lsr": missing_dates_within_range(lsr_df["Date"], start, end),
    }

    fed_filled = _reindex_and_ffill(fed_df, start, end)
    oi_filled = _reindex_and_ffill(oi_df, start, end)
    lsr_filled = _reindex_and_ffill(lsr_df, start, end)

    merged = fed_filled.merge(oi_filled, on="Date", how="inner", validate="one_to_one")
    merged = merged.merge(lsr_filled, on="Date", how="inner", validate="one_to_one")
    merged = merged.sort_values("Date").reset_index(drop=True)

    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    return merged, missing_info


def main() -> None:
    merged_df, missing_info = build_merged()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows written: {len(merged_df)}")
    print(f"Date range: {merged_df['Date'].iloc[0]} -> {merged_df['Date'].iloc[-1]}")
    print(f"FED gaps filled from previous row: {len(missing_info['fed'])}")
    print(f"OI gaps filled from previous row: {len(missing_info['oi'])}")
    print(f"LSR gaps filled from previous row: {len(missing_info['lsr'])}")
    if missing_info["fed"]:
        print(f"Sample FED filled dates: {[date.strftime('%Y-%m-%d') for date in missing_info['fed'][:10]]}")
    if missing_info["oi"]:
        print(f"Sample OI filled dates: {[date.strftime('%Y-%m-%d') for date in missing_info['oi'][:10]]}")
    if missing_info["lsr"]:
        print(f"Sample LSR filled dates: {[date.strftime('%Y-%m-%d') for date in missing_info['lsr'][:10]]}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
