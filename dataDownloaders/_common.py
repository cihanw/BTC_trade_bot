from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
import time as time_module

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"


def _parse_env_date(name: str, default: date) -> date:
    raw = os.getenv(name)
    if not raw:
        return default
    return datetime.strptime(raw, "%Y-%m-%d").date()


START_DATE = _parse_env_date("BTC_DATA_START_DATE", date(2020, 6, 1))
END_DATE = _parse_env_date("BTC_DATA_END_DATE", date.today() - timedelta(days=1))
END_EXCLUSIVE_DATE = END_DATE + timedelta(days=1)

HTTP_TIMEOUT = 60
USER_AGENT = "BTC_trade_bot data downloader/1.0"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_start_of_day(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_starts(start: date, end_inclusive: date) -> list[date]:
    current = date(start.year, start.month, 1)
    end_month = date(end_inclusive.year, end_inclusive.month, 1)
    out: list[date] = []
    while current <= end_month:
        out.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return out


def iter_days(start: date, end_inclusive: date) -> list[date]:
    current = start
    out: list[date] = []
    while current <= end_inclusive:
        out.append(current)
        current += timedelta(days=1)
    return out


def chunk_datetimes(
    start: datetime,
    end_exclusive: datetime,
    chunk_size: timedelta,
) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end_exclusive:
        next_cursor = min(cursor + chunk_size, end_exclusive)
        out.append((cursor, next_cursor))
        cursor = next_cursor
    return out


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    stream: bool = False,
    timeout: int = HTTP_TIMEOUT,
    max_attempts: int = 5,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                stream=stream,
                timeout=timeout,
            )
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                sleep_seconds = min(2 ** (attempt - 1), 10)
                time_module.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                break
            sleep_seconds = min(2 ** (attempt - 1), 10)
            time_module.sleep(sleep_seconds)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = min(2 ** (attempt - 1), 10)
            time_module.sleep(sleep_seconds)
    if last_error is None:
        raise RuntimeError(f"HTTP request failed without a captured exception: {method} {url}")
    raise last_error


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: int = HTTP_TIMEOUT,
) -> str:
    if destination.exists():
        return "exists"

    ensure_dir(destination.parent)
    response = request_with_retries(
        session,
        "GET",
        url,
        stream=True,
        timeout=timeout,
        max_attempts=3,
    )
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp_path.replace(destination)
    return "downloaded"
