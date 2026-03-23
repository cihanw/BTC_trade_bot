from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_KEY = "65fbb0f8ef9c5619a7d1f51c1ddf5e41"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Fed Net Likidite = Fed Toplam Bilanco - (TGA + Ters Repo)
SERIES_IDS = {
    "fed_total_assets": "WALCL",   # Federal Reserve total balance sheet
    "tga": "WTREGEN",              # Treasury General Account
    "rrp": "RRPONTSYD",            # Overnight reverse repo
}

START_DATE = "2020-01-01"
END_DATE = "2026-03-01"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "netLiq"
OUTPUT_CSV_PATH = OUTPUT_DIR / "fed_net_liquidity_1d.csv"


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"FRED seri verisi bos dondu: {series_id}")

    rows = []
    for obs in observations:
        raw_value = obs.get("value")
        if raw_value in (None, "", "."):
            continue
        rows.append(
            {
                "date": obs.get("date"),
                "value": float(raw_value),
            }
        )

    if not rows:
        raise ValueError(f"Kullanilabilir numeric veri yok: {series_id}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def build_net_liquidity() -> pd.DataFrame:
    fed_df = fetch_fred_series(SERIES_IDS["fed_total_assets"]).rename(
        columns={"value": "fed_total_assets"}
    )
    tga_df = fetch_fred_series(SERIES_IDS["tga"]).rename(columns={"value": "tga"})
    rrp_df = fetch_fred_series(SERIES_IDS["rrp"]).rename(columns={"value": "rrp"})

    merged = fed_df.merge(tga_df, on="date", how="outer").merge(rrp_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Farkli frekansli serileri ayni zaman ekseninde kullanabilmek icin son degeri ileri tasiyoruz.
    component_cols = ["fed_total_assets", "tga", "rrp"]
    merged[component_cols] = merged[component_cols].ffill()
    merged = merged.dropna(subset=component_cols).copy()

    merged["fed_net_liquidity"] = merged["fed_total_assets"] - (merged["tga"] + merged["rrp"])
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    return merged[["date", "fed_total_assets", "tga", "rrp", "fed_net_liquidity"]]


def main() -> None:
    if API_KEY == "x":
        raise ValueError("API_KEY su an 'x'. Lutfen kendi FRED API key degerini gir.")

    df = build_net_liquidity()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Kaydedildi: {OUTPUT_CSV_PATH}")
    print(f"Toplam satir: {len(df)}")
    print(f"Tarih araligi: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
