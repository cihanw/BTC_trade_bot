import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fear-greed"
CSV_PATH = OUTPUT_DIR / "alternative_fear_greed_1d.csv"
JSON_PATH = OUTPUT_DIR / "alternative_fear_greed_1d.json"


def fetch_fear_greed() -> list[dict]:
    with urlopen(API_URL, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", [])
    if not data:
        raise ValueError("Alternative.me API veri dondurmedi.")
    return data


def to_iso_date(unix_timestamp: str) -> str:
    dt = datetime.fromtimestamp(int(unix_timestamp), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def save_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "value",
                "value_classification",
                "time_until_update",
                "timestamp",
            ],
        )
        writer.writeheader()
        # API en yeni tarihi ilk verir; kronolojik olsun diye ters ceviriyoruz.
        for row in reversed(rows):
            writer.writerow(
                {
                    "date": to_iso_date(row["timestamp"]),
                    "value": row.get("value"),
                    "value_classification": row.get("value_classification"),
                    "time_until_update": row.get("time_until_update"),
                    "timestamp": row.get("timestamp"),
                }
            )


def save_json(rows: list[dict]) -> None:
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    rows = fetch_fear_greed()
    save_csv(rows)
    save_json(rows)
    print(f"Kaydedildi: {CSV_PATH}")
    print(f"Kaydedildi: {JSON_PATH}")
    print(f"Toplam satir: {len(rows)}")


if __name__ == "__main__":
    main()
