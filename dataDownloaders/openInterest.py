import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

API_KEY = os.getenv("COINALYZE_API_KEY", "")
BASE_URL = "https://api.coinalyze.net/v1"
HEADERS = {"api_key": API_KEY}

START_DATE = "2020-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "liquidations-OI"


def to_timestamp(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def safe_get(endpoint: str, params: dict | None = None):
    if not API_KEY:
        raise ValueError("COINALYZE_API_KEY environment variable is not set.")
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} for {url}")
        print(resp.text)
    resp.raise_for_status()
    return resp.json()


def get_future_markets() -> pd.DataFrame:
    data = safe_get("future-markets")
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("future-markets boş döndü.")

    return df


def find_btc_perp_markets(df: pd.DataFrame) -> pd.DataFrame:
    """
    BTC perpetual future marketlerini döndürür.
    """
    required_cols = ["symbol", "exchange", "symbol_on_exchange", "base_asset", "quote_asset", "is_perpetual"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen kolonlar yok: {missing}\nMevcut kolonlar: {list(df.columns)}")

    out = df[
        (df["base_asset"].astype(str).str.upper() == "BTC") &
        (df["is_perpetual"] == True)
    ].copy()

    out = out.sort_values(["exchange", "quote_asset", "symbol"]).reset_index(drop=True)
    return out


def fetch_open_interest_history(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    params = {
        "symbols": symbol,
        "interval": "daily",
        "from": start_ts,
        "to": end_ts,
    }
    data = safe_get("open-interest-history", params=params)

    rows = []
    for item in data:
        sym = item.get("symbol", symbol)
        for h in item.get("history", []):
            rows.append({
                "symbol": sym,
                "timestamp": h.get("t"),
                "oi_open": h.get("o"),
                "oi_high": h.get("h"),
                "oi_low": h.get("l"),
                "oi_close": h.get("c"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


def fetch_long_short_ratio_history(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    params = {
        "symbols": symbol,
        "interval": "daily",
        "from": start_ts,
        "to": end_ts,
    }
    data = safe_get("long-short-ratio-history", params=params)

    rows = []
    for item in data:
        sym = item.get("symbol", symbol)
        for h in item.get("history", []):
            rows.append({
                "symbol": sym,
                "timestamp": h.get("t"),
                "long_short_ratio": h.get("r"),
                "long_ratio": h.get("l"),
                "short_ratio": h.get("s"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    start_ts = to_timestamp(START_DATE)
    end_ts = to_timestamp(END_DATE)

    print("Market listesi çekiliyor...")
    markets_df = get_future_markets()

    print("\nKolonlar:")
    print(list(markets_df.columns))

    btc_perp_df = find_btc_perp_markets(markets_df)

    if btc_perp_df.empty:
        print("\nBTC perpetual market bulunamadı. İlk 20 satırı gösteriyorum:")
        print(markets_df.head(20).to_string(index=False))
        raise ValueError("BTC perpetual market bulunamadı.")

    print("\nBulunan BTC perpetual marketler:")
    display_cols = [c for c in [
        "symbol", "exchange", "symbol_on_exchange", "base_asset",
        "quote_asset", "is_perpetual", "has_long_short_ratio_data"
    ] if c in btc_perp_df.columns]
    print(btc_perp_df[display_cols].to_string(index=False))

    symbol = input("\nKullanmak istediğin exact 'symbol' değerini gir: ").strip()

    oi_df = fetch_open_interest_history(symbol, start_ts, end_ts)
    lsr_df = fetch_long_short_ratio_history(symbol, start_ts, end_ts)

    print(f"\nOI rows: {len(oi_df)}")
    print(f"Long/Short rows: {len(lsr_df)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oi_path = OUTPUT_DIR / "coinalyze_open_interest_1d.csv"
    lsr_path = OUTPUT_DIR / "coinalyze_long_short_ratio_1d.csv"

    oi_df.to_csv(oi_path, index=False)
    lsr_df.to_csv(lsr_path, index=False)

    print("\nKaydedildi:")
    print(f" - {oi_path}")
    print(f" - {lsr_path}")
