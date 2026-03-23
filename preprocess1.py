from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_data.csv"

CME_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "cme" / "CME_BTC_F_2020-01-01_to_2026-03-15.csv",
    PROJECT_ROOT / "data" / "raw" / "cme" / "CME_BTC_F_2019-12-31_to_2026-03-15.csv",
]
OI_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "coinalyze_open_interest_1d.csv",
    PROJECT_ROOT / "data" / "raw" / "liquidations-OI" / "coinalyze_open_interest_1d.csv",
]
LSR_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "coinalyze_long_short_ratio_1d.csv",
    PROJECT_ROOT / "data" / "raw" / "liquidations-OI" / "coinalyze_long_short_ratio_1d.csv",
]


def resolve_input_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    choices = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"{label} file not found. Checked:\n - {choices}")


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns: {missing}")


def parse_datetime_to_date(series: pd.Series, label: str) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    bad_count = int(parsed.isna().sum())
    if bad_count:
        print(f"[WARN] {label}: dropped {bad_count} rows due to invalid datetime.")
    return parsed.dt.tz_convert(None).dt.normalize(), bad_count


def process_cme(path: Path) -> tuple[pd.DataFrame, dict]:
    cme_df = pd.read_csv(path)
    raw_rows = len(cme_df)

    # Drop the last two known columns if they exist.
    drop_cols = [c for c in ["Dividends", "Stock Splits"] if c in cme_df.columns]
    if drop_cols:
        cme_df = cme_df.drop(columns=drop_cols)

    ensure_columns(cme_df, ["Date", "Open", "High", "Low", "Close", "Volume"], "CME")

    cme_dates, bad_date_rows = parse_datetime_to_date(cme_df["Date"], "CME Date")
    cme_df = cme_df.assign(Date=cme_dates).dropna(subset=["Date"]).copy()
    cme_df = cme_df.sort_values("Date").reset_index(drop=True)

    if cme_df["Date"].duplicated().any():
        dup = int(cme_df["Date"].duplicated().sum())
        raise ValueError(f"CME has duplicate Date values after normalization: {dup}")

    existing_dates = set(cme_df["Date"].tolist())
    cme_df = cme_df.set_index("Date")

    full_date_range = pd.date_range(cme_df.index.min(), cme_df.index.max(), freq="D")
    cme_df = cme_df.reindex(full_date_range)
    inserted_mask = ~cme_df.index.isin(existing_dates)

    prev_close = cme_df["Close"].ffill()
    cme_df.loc[inserted_mask, "Open"] = prev_close.loc[inserted_mask]
    cme_df.loc[inserted_mask, "High"] = prev_close.loc[inserted_mask]
    cme_df.loc[inserted_mask, "Low"] = prev_close.loc[inserted_mask]
    cme_df.loc[inserted_mask, "Close"] = prev_close.loc[inserted_mask]
    cme_df.loc[inserted_mask, "Volume"] = 0

    cme_df = cme_df.reset_index().rename(columns={"index": "Date"})
    cme_df["Date"] = cme_df["Date"].dt.strftime("%Y-%m-%d")

    stats = {
        "source_path": str(path),
        "raw_rows": raw_rows,
        "bad_date_rows": bad_date_rows,
        "rows_after_processing": len(cme_df),
        "inserted_missing_dates": int(inserted_mask.sum()),
        "unique_dates": int(cme_df["Date"].nunique()),
    }
    return cme_df, stats


def process_coinalyze(path: Path, label: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    raw_rows = len(df)

    drop_cols = [c for c in ["symbol", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    ensure_columns(df, ["datetime"], label)

    dates, bad_date_rows = parse_datetime_to_date(df["datetime"], f"{label} datetime")
    df = df.assign(datetime=dates).dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"datetime": "Date"}).sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        dup = int(df["Date"].duplicated().sum())
        raise ValueError(f"{label} has duplicate Date values after normalization: {dup}")

    stats = {
        "source_path": str(path),
        "raw_rows": raw_rows,
        "bad_date_rows": bad_date_rows,
        "rows_after_processing": len(df),
        "unique_dates": int(df["Date"].nunique()),
    }
    return df, stats


def fill_lsr_missing_with_neighbor_avg(lsr_df: pd.DataFrame, target_dates: list[str]) -> tuple[pd.DataFrame, dict]:
    out = lsr_df.copy().set_index("Date").reindex(target_dates)
    value_cols = [c for c in out.columns]
    original_missing_rows = out[value_cols].isna().all(axis=1)
    inserted_rows = int(original_missing_rows.sum())

    for col in value_cols:
        prev_vals = out[col].ffill()
        next_vals = out[col].bfill()
        fill_vals = (prev_vals + next_vals) / 2.0
        out.loc[out[col].isna(), col] = fill_vals.loc[out[col].isna()]

    remaining_missing_rows = int(out[value_cols].isna().any(axis=1).sum())
    if remaining_missing_rows:
        raise ValueError(
            f"LSR fill failed for {remaining_missing_rows} rows. "
            "Missing values at range edges may not have neighbors on both sides."
        )

    out = out.reset_index().rename(columns={"index": "Date"})
    stats = {
        "rows_after_reindex": len(out),
        "inserted_rows_from_missing_dates": inserted_rows,
        "remaining_missing_rows_after_fill": remaining_missing_rows,
        "unique_dates": int(out["Date"].nunique()),
    }
    return out, stats


def print_stats(title: str, stats: dict) -> None:
    print(f"\n[{title}]")
    for key, value in stats.items():
        print(f" - {key}: {value}")


def main() -> None:
    cme_path = resolve_input_path(CME_CANDIDATES, "CME")
    oi_path = resolve_input_path(OI_CANDIDATES, "Open Interest")
    lsr_path = resolve_input_path(LSR_CANDIDATES, "Long/Short Ratio")

    cme_df, cme_stats = process_cme(cme_path)
    oi_df, oi_stats = process_coinalyze(oi_path, "Open Interest")
    lsr_df, lsr_stats = process_coinalyze(lsr_path, "Long/Short Ratio")
    lsr_df, lsr_fill_stats = fill_lsr_missing_with_neighbor_avg(lsr_df, cme_df["Date"].tolist())

    merged_df = cme_df.merge(oi_df, on="Date", how="left")
    merged_df = merged_df.merge(lsr_df, on="Date", how="left")
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    if merged_df["Date"].duplicated().any():
        dup = int(merged_df["Date"].duplicated().sum())
        raise ValueError(f"Merged data has duplicate Date values: {dup}")

    counts = {
        "CME_rows": len(cme_df),
        "OI_rows": len(oi_df),
        "LSR_rows": len(lsr_df),
        "Merged_rows": len(merged_df),
    }
    if len({counts["CME_rows"], counts["OI_rows"], counts["LSR_rows"]}) != 1:
        print(
            "[WARN] Source row counts are not equal: "
            f"CME={counts['CME_rows']}, OI={counts['OI_rows']}, LSR={counts['LSR_rows']}"
        )

    cme_cols = [c for c in cme_df.columns if c != "Date"]
    oi_cols = [c for c in oi_df.columns if c != "Date"]
    lsr_cols = [c for c in lsr_df.columns if c != "Date"]

    missing_cme_rows = int(merged_df[cme_cols].isna().all(axis=1).sum())
    missing_oi_rows = int(merged_df[oi_cols].isna().all(axis=1).sum())
    missing_lsr_rows = int(merged_df[lsr_cols].isna().all(axis=1).sum())

    if missing_cme_rows or missing_oi_rows or missing_lsr_rows:
        print(
            "[WARN] Rows with missing source blocks: "
            f"CME_missing_rows={missing_cme_rows}, "
            f"OI_missing_rows={missing_oi_rows}, "
            f"LSR_missing_rows={missing_lsr_rows}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print_stats("CME", cme_stats)
    print_stats("Open Interest", oi_stats)
    print_stats("Long/Short Ratio", lsr_stats)
    print_stats("Long/Short Fill", lsr_fill_stats)
    print("\n[MERGE]")
    for key, value in counts.items():
        print(f" - {key}: {value}")
    print(f" - merged_unique_dates: {merged_df['Date'].nunique()}")
    print(f" - output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
