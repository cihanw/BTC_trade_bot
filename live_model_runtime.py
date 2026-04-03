from __future__ import annotations

import time
from datetime import date
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable
from zoneinfo import ZoneInfo

import databento as db
import databento_dbn as dbn
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

import bot_settings as settings
from preprocess.binancePreprocess import build_output as build_binance_processed_output
from preprocess.bybit_preprocess import add_technical_indicators as add_bybit_technical_indicators
from preprocess.cmePreprocess import add_technical_indicators as add_cme_technical_indicators
from preprocess.coinbasePreprocess import build_output as build_coinbase_processed_output
from preprocess.coinbasePreprocess import resample_to_30m as coinbase_resample_to_30m
from preprocess.labelgenerator import ATR_PERIOD, compute_effective_barrier_multiplier, compute_past_only_atr


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pt"
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "liveData"
LIVE_DAILY_PATH = LIVE_DATA_DIR / "merged1d.csv"
LIVE_COINBASE_PATH = LIVE_DATA_DIR / "coinbase_processed.csv"
LIVE_BINANCE_PATH = LIVE_DATA_DIR / "binance_processed.csv"
LIVE_BYBIT_PATH = LIVE_DATA_DIR / "bybit_processed.csv"
LIVE_CME_PATH = LIVE_DATA_DIR / "cme_processed.csv"

THIRTY_MINUTES = pd.Timedelta(minutes=30)
ONE_DAY = pd.Timedelta(days=1)
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
LIVE_REPLAY_LOOKBACK = pd.Timedelta(hours=24)
THIRTY_MINUTES_MS = int(THIRTY_MINUTES.total_seconds() * 1000)
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
COINALYZE_BASE_URL = "https://api.coinalyze.net/v1"
US_MARKET_TIMEZONE = ZoneInfo("America/New_York")
US_MARKET_CLOSE_HOUR = 16
US_MARKET_CLOSE_MINUTE = 5
CME_INCREMENTAL_FETCH_OVERLAP_BARS = 4
CME_INCREMENTAL_CACHE_WARMUP_BARS = 96
CME_SPECIAL_LAST_BAR_OPEN_UTC: dict[date, pd.Timestamp] = {
    date(2026, 4, 3): pd.Timestamp("2026-04-03 15:00:00"),
}
CME_SPECIAL_SESSION_REASON_BY_DATE: dict[date, str] = {
    date(2026, 4, 3): "CME Good Friday 2026 ozel kapanisi",
}
FRED_SERIES_IDS = {
    "fed_total_assets": "WALCL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
}


@dataclass(frozen=True)
class DecisionRule:
    max_flat_probability: float
    side_ratio_threshold: float
    ordering_max_flat_probability: float

    def derive_trade_label(self, p_buy: float, p_hold: float, p_sell: float) -> str:
        flat_gate = p_hold < self.max_flat_probability
        long_mask = flat_gate and (p_buy > self.side_ratio_threshold * p_sell)
        short_mask = flat_gate and (not long_mask) and (p_sell > self.side_ratio_threshold * p_buy)
        ordering_gate = p_hold < self.ordering_max_flat_probability
        ordering_long_mask = ordering_gate and (p_buy > p_hold > p_sell)
        ordering_short_mask = ordering_gate and (p_sell > p_hold > p_buy)

        if long_mask or ordering_long_mask:
            return "up"
        if short_mask or ordering_short_mask:
            return "down"
        return "flat"


@dataclass(frozen=True)
class PredictionResult:
    bar_time: pd.Timestamp
    close_price: float
    barrier_width: float
    probs: dict[str, float]
    notes: tuple[str, ...]


class LatestBarPendingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        expected_latest: pd.Timestamp | None = None,
        latest_available: pd.Timestamp | None = None,
        lagging_sources: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.expected_latest = expected_latest
        self.latest_available = latest_available
        self.lagging_sources = lagging_sources
        self.notes = notes


class EncoderOnlyTransformerBranch(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max positional length {self.pos_embedding.shape[1]}"
            )

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)

        h_mean = self.token_norm(h.mean(dim=1))
        h_last = self.token_norm(h[:, -1, :])
        h_repr = torch.cat([h_mean, h_last], dim=-1)
        return self.output_norm(h_repr)


class DailyGRUBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.output_norm(h_n[-1])


class MultiTimescaleFusionModel(nn.Module):
    def __init__(
        self,
        input_dim_30m: int,
        input_dim_1d: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        gru_dropout: float,
        fusion_hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.encoder_branch = EncoderOnlyTransformerBranch(
            input_dim=input_dim_30m,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.daily_branch = DailyGRUBranch(
            input_dim=input_dim_1d,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
        )

        fusion_dim = (d_model * 2) + gru_hidden_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(self, x30: torch.Tensor, x1d: torch.Tensor) -> torch.Tensor:
        enc_repr = self.encoder_branch(x30)
        gru_repr = self.daily_branch(x1d)
        fused = torch.cat([enc_repr, gru_repr], dim=-1)
        fused = self.fusion_norm(fused)
        return self.head(fused)


def _parse_utc_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def dedupe_preserve_order(column_names: list[str]) -> list[str]:
    return list(dict.fromkeys(column_names))


def trailing_mean_past_only(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def infer_activity_base_col(col_name: str, available_columns: list[str]) -> str | None:
    prefix = col_name.rsplit("_", 1)[0]
    candidates = [
        f"{prefix}_quote_volume",
        f"{prefix}_volume",
        f"{prefix}_count",
    ]
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


def infer_atr_base_col(col_name: str, available_columns: list[str]) -> str | None:
    suffix_map = {
        "_macd_signal": "_atr_14",
        "_macd_hist": "_atr_14",
        "_macd": "_atr_14",
    }
    for suffix, replacement in suffix_map.items():
        if col_name.endswith(suffix):
            candidate = f"{col_name[:-len(suffix)]}{replacement}"
            if candidate in available_columns:
                return candidate
    return None


def normalize_30m_frame(
    raw_df: pd.DataFrame,
    groups: dict[str, list[str]],
    level_window_30m: int,
    flow_activity_window_30m: int,
    cvd_detrend_window_30m: int,
) -> pd.DataFrame:
    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    norm_df = numeric_df.copy()

    for col in groups["price_cols"]:
        prev = numeric_df[col].shift(1)
        norm_df[col] = np.log(numeric_df[col] / prev)

    for col in groups["level_cols"]:
        baseline = trailing_mean_past_only(numeric_df[col], level_window_30m).replace(0, np.nan)
        norm_df[col] = numeric_df[col] / baseline

    for col in groups["macd_cols"]:
        atr_base_col = infer_atr_base_col(col, numeric_df.columns.tolist())
        if atr_base_col is None:
            continue
        atr_base = numeric_df[atr_base_col].replace(0, np.nan)
        norm_df[col] = numeric_df[col] / atr_base

    for col in groups["delta_cols"]:
        activity_col = infer_activity_base_col(col, numeric_df.columns.tolist())
        activity_source = numeric_df[activity_col].abs() if activity_col is not None else numeric_df[col].abs()
        activity_baseline = trailing_mean_past_only(activity_source, flow_activity_window_30m).replace(0, np.nan)
        norm_df[col] = numeric_df[col] / activity_baseline

    for col in groups["cvd_cols"]:
        activity_col = infer_activity_base_col(col, numeric_df.columns.tolist())
        activity_source = numeric_df[activity_col].abs() if activity_col is not None else numeric_df[col].abs()
        activity_baseline = trailing_mean_past_only(activity_source, flow_activity_window_30m).replace(0, np.nan)
        cvd_trend = trailing_mean_past_only(numeric_df[col], cvd_detrend_window_30m)
        cvd_residual = numeric_df[col] - cvd_trend
        norm_df[col] = cvd_residual / activity_baseline

    return norm_df.replace([np.inf, -np.inf], np.nan)


def normalize_1d_frame(raw_df: pd.DataFrame, groups: dict[str, list[str]], level_window_1d: int) -> pd.DataFrame:
    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    norm_df = numeric_df.copy()

    for col in groups["level_cols"]:
        baseline = trailing_mean_past_only(numeric_df[col], level_window_1d).replace(0, np.nan)
        norm_df[col] = numeric_df[col] / baseline

    return norm_df.replace([np.inf, -np.inf], np.nan)


def _manual_standard_scale(df: pd.DataFrame, columns: list[str], mean_values: list[float], scale_values: list[float]) -> pd.DataFrame:
    out = df.copy()
    if not columns:
        return out

    means = pd.Series(np.asarray(mean_values, dtype=np.float64), index=columns)
    scales = pd.Series(np.asarray(scale_values, dtype=np.float64), index=columns).replace(0.0, 1.0)

    valid_mask = out[columns].notna().all(axis=1)
    if valid_mask.any():
        out.loc[valid_mask, columns] = (out.loc[valid_mask, columns] - means) / scales
    return out


def _drop_helper_open_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        column
        for column in df.columns
        if not (column.lower() == "open_time" or column.lower().endswith("_open_time"))
    ]
    return df[keep_columns].copy()


def _standardize_source_frame(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    out = df.copy()
    if time_column != "time":
        out = out.rename(columns={time_column: "time"})
    out["time"] = _parse_utc_naive(out["time"])
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return out


def _overlay_processed_frames(base_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df.empty:
        return base_df.copy()

    all_columns = [
        "time"
    ] + [
        column for column in dedupe_preserve_order(base_df.columns.tolist() + live_df.columns.tolist()) if column != "time"
    ]
    base = base_df.copy().reindex(columns=all_columns).set_index("time")
    live = live_df.copy().reindex(columns=all_columns).set_index("time")
    combined = live.combine_first(base).sort_index().reset_index()
    return combined


def _merge_time_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, ignore_index=True, sort=False)
    if "time" in merged.columns:
        merged["time"] = _parse_utc_naive(merged["time"])
        merged = merged.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return merged


def _align_cvd_with_base(base_df: pd.DataFrame, live_df: pd.DataFrame, cvd_column: str, time_column: str = "time") -> pd.DataFrame:
    if live_df.empty or cvd_column not in live_df.columns or time_column not in live_df.columns:
        return live_df

    out = live_df.copy().sort_values(time_column).reset_index(drop=True)
    if time_column not in base_df.columns:
        return out

    base = base_df.sort_values(time_column).reset_index(drop=True)
    base_cvd_column = cvd_column if cvd_column in base.columns else None
    if base_cvd_column is None:
        suffix_match = next((column for column in base.columns if column.endswith(f"_{cvd_column}")), None)
        if suffix_match is None:
            return out
        base_cvd_column = suffix_match

    first_time = out[time_column].iloc[0]
    previous = base.loc[base[time_column] < first_time, base_cvd_column]
    base_offset = float(previous.iloc[-1]) if not previous.empty else 0.0
    out[cvd_column] = pd.to_numeric(out[cvd_column], errors="coerce") + base_offset
    return out


def _latest_closed_30m_open_time(now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    current_open = now_utc.floor("30min").tz_convert(None)
    return current_open - THIRTY_MINUTES


def _cme_special_last_bar_open_utc(bar_open_utc: pd.Timestamp | str) -> pd.Timestamp | None:
    normalized = _utc_naive(bar_open_utc)
    return CME_SPECIAL_LAST_BAR_OPEN_UTC.get(normalized.date())


def _cme_special_session_reason(bar_open_utc: pd.Timestamp | str) -> str | None:
    normalized = _utc_naive(bar_open_utc)
    special_last_bar = _cme_special_last_bar_open_utc(normalized)
    if special_last_bar is None or normalized < special_last_bar:
        return None
    return CME_SPECIAL_SESSION_REASON_BY_DATE.get(normalized.date())


def _is_cme_bitcoin_trading_open_time(bar_open_utc: pd.Timestamp) -> bool:
    normalized = _utc_naive(bar_open_utc)
    special_last_bar = _cme_special_last_bar_open_utc(normalized)
    if special_last_bar is not None and normalized > special_last_bar:
        return False

    market_time = normalized.tz_localize("UTC").tz_convert(US_MARKET_TIMEZONE)
    weekday = market_time.weekday()
    hour = market_time.hour

    if weekday == 5:
        return False
    if weekday == 4 and hour >= 17:
        return False
    if weekday == 6 and hour < 18:
        return False
    if weekday in {0, 1, 2, 3} and hour == 17:
        return False
    return True


def _latest_closed_cme_30m_open_time(now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    candidate = _latest_closed_30m_open_time(now_utc)
    while not _is_cme_bitcoin_trading_open_time(candidate):
        candidate -= THIRTY_MINUTES
    return candidate


def _latest_closed_30m_fetch_end(source_label: str, now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    latest_open = _latest_closed_cme_30m_open_time(now_utc) if source_label == "Databento CME" else _latest_closed_30m_open_time(now_utc)
    return latest_open + THIRTY_MINUTES


def _latest_expected_common_30m_open_time(now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    return min(_latest_closed_30m_open_time(now_utc), _latest_closed_cme_30m_open_time(now_utc))


def _synthesize_missing_cme_rows(
    cme_df: pd.DataFrame,
    target_times: pd.Series | pd.Index | list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    if cme_df.empty:
        return cme_df.copy(), []

    target_series = target_times if isinstance(target_times, pd.Series) else pd.Series(list(target_times), dtype="object")
    target_index = pd.Index(_parse_utc_naive(target_series).dropna().drop_duplicates().sort_values())
    if target_index.empty:
        return _standardize_source_frame(cme_df.copy(), time_column="time"), []

    cme = _standardize_source_frame(cme_df.copy(), time_column="time").set_index("time")
    aligned = cme.reindex(target_index)
    observed_mask = aligned.notna().any(axis=1)
    has_prior_row = observed_mask.cummax()
    synthetic_index = aligned.index[(~observed_mask) & has_prior_row]

    filled = aligned.ffill().loc[has_prior_row].copy()
    if len(synthetic_index) > 0:
        if "CME_volume" in filled.columns:
            filled.loc[synthetic_index, "CME_volume"] = 0.0
        if "CME_count" in filled.columns:
            filled.loc[synthetic_index, "CME_count"] = 0
        if "CME_delta" in filled.columns:
            filled.loc[synthetic_index, "CME_delta"] = 0
        if "CME_open_time" in filled.columns:
            filled.loc[synthetic_index, "CME_open_time"] = [int(pd.Timestamp(ts).timestamp()) for ts in synthetic_index]

    out = filled.rename_axis("time").reset_index()
    if "CME_count" in out.columns:
        out["CME_count"] = pd.to_numeric(out["CME_count"], errors="coerce").fillna(0).round().astype("int64")
    if "CME_open_time" in out.columns:
        out["CME_open_time"] = pd.to_numeric(out["CME_open_time"], errors="coerce").fillna(0).astype("int64")
    return _standardize_source_frame(out, time_column="time"), [pd.Timestamp(ts) for ts in synthetic_index]


def _utc_naive(ts: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def _to_utc_iso(ts: pd.Timestamp) -> str:
    normalized = _utc_naive(ts).tz_localize("UTC")
    return normalized.isoformat().replace("+00:00", "Z")


def _trim_time_window(df: pd.DataFrame, start_time: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df.copy()
    out = df.copy()
    out["time"] = _parse_utc_naive(out["time"])
    return out.loc[out["time"] >= _utc_naive(start_time)].sort_values("time").reset_index(drop=True)


def _has_large_30m_gap(df: pd.DataFrame, time_column: str = "time") -> bool:
    if df.empty or time_column not in df.columns:
        return False
    times = _parse_utc_naive(df[time_column]).sort_values().reset_index(drop=True)
    max_gap = times.diff().max()
    return pd.notna(max_gap) and max_gap > (THIRTY_MINUTES + pd.Timedelta(minutes=1))


def _trim_daily_window(df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    return out.loc[out["Date"] >= pd.Timestamp(start_date).normalize()].sort_values("Date").reset_index(drop=True)


def _save_time_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["time"] = _parse_utc_naive(out["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def _load_time_cache(path: Path, time_column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _standardize_source_frame(pd.read_csv(path), time_column=time_column)


def _overlay_daily_frames(base_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df.empty:
        return base_df.copy()
    all_columns = ["Date"] + [
        column
        for column in dedupe_preserve_order(base_df.columns.tolist() + live_df.columns.tolist())
        if column != "Date"
    ]
    base = base_df.copy().reindex(columns=all_columns)
    live = live_df.copy().reindex(columns=all_columns)
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.normalize()
    live["Date"] = pd.to_datetime(live["Date"], errors="coerce").dt.normalize()
    combined = live.set_index("Date").combine_first(base.set_index("Date")).sort_index().reset_index()
    return combined


def _save_daily_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _load_daily_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def _candle_imbalance_proxy(df: pd.DataFrame, activity_column: str, count_column: str) -> pd.DataFrame:
    out = df.copy()
    for column in ["open", "high", "low", "close", activity_column, count_column]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    price_range = (out["high"] - out["low"]).abs()
    body = out["close"] - out["open"]
    imbalance = (body / price_range.replace(0, np.nan)).clip(-1.0, 1.0)
    imbalance = imbalance.where(price_range > 0, np.sign(body)).fillna(0.0)

    activity = out[activity_column].fillna(0.0)
    count_activity = out[count_column].fillna(0.0)
    out["delta"] = activity * imbalance
    out["count"] = count_activity.round().clip(lower=0).astype("int64")
    out["cvd"] = pd.to_numeric(out["delta"], errors="coerce").fillna(0.0).cumsum()
    return out


def _align_prefixed_cvd_columns(base_df: pd.DataFrame, live_df: pd.DataFrame, cvd_columns: list[str]) -> pd.DataFrame:
    out = live_df.copy()
    for column in cvd_columns:
        out = _align_cvd_with_base(base_df, out, column)
    return out


def _daily_overlap_bounds(frames: list[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(frame["Date"].min() for frame in frames)
    end = min(frame["Date"].max() for frame in frames)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Daily live sources do not share a valid overlapping date range.")
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def _reindex_and_ffill_daily(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    full_range = pd.date_range(start=start, end=end, freq="D")
    indexed = df.set_index("Date").sort_index()
    fill_index = indexed.index.union(full_range)
    out = indexed.reindex(fill_index).sort_index().ffill().reindex(full_range)
    if out.isna().any().any():
        raise ValueError("Daily live bootstrap left unresolved gaps at the start of the overlap range.")
    return out.reset_index().rename(columns={"index": "Date"})


class RetryHttpClient:
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(self, timeout_seconds: float, max_retries: int, backoff_seconds: float, verify_ssl: bool = True) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.verify_ssl = verify_ssl
        self.session = requests.Session()

    def get_json(
        self,
        url: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict | list:
        attempts = self.max_retries + 1
        response: requests.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                if not response.text.strip():
                    return {}
                return response.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= attempts:
                    raise RuntimeError(f"HTTP GET failed after {attempt} attempts | url={url} | reason={exc}") from exc
                time.sleep(self.backoff_seconds * attempt)
            except requests.HTTPError as exc:
                status_code = response.status_code if response is not None else None
                if status_code in self.RETRYABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(self.backoff_seconds * attempt)
                    continue
                body = ""
                if response is not None:
                    body = response.text[:400].strip()
                raise RuntimeError(
                    f"HTTP GET failed | url={(response.url if response is not None else url)} | "
                    f"status={(status_code if status_code is not None else 'unknown')} | body={body}"
                ) from exc

        raise RuntimeError(f"HTTP GET exhausted retries unexpectedly | url={url}")


class BinanceLiveDataClient:
    FUTURES_KLINES_PATH = "/fapi/v1/klines"
    FUTURES_FUNDING_PATH = "/fapi/v1/fundingRate"
    SPOT_KLINES_PATH = "/api/v3/klines"
    MAX_KLINE_LIMIT = 1500
    MAX_FUNDING_LIMIT = 1000

    def __init__(self) -> None:
        self.futures_base_url = settings.BINANCE_FUTURES_PUBLIC_URL.rstrip("/")
        self.spot_base_url = settings.BINANCE_SPOT_PUBLIC_URL.rstrip("/")
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    def _fetch_klines_range(self, base_url: str, path: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        latest_closed = _latest_closed_30m_open_time()
        start_time = _utc_naive(start)
        end_time = min(_utc_naive(end), latest_closed + THIRTY_MINUTES)
        if end_time <= start_time:
            return pd.DataFrame()

        rows: list[list[object]] = []
        cursor_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        while cursor_ms < end_ms:
            payload = self.http.get_json(
                f"{base_url}{path}",
                params={
                    "symbol": symbol,
                    "interval": "30m",
                    "limit": self.MAX_KLINE_LIMIT,
                    "startTime": cursor_ms,
                    "endTime": end_ms - 1,
                },
            )
            if not isinstance(payload, list) or not payload:
                break

            rows.extend(payload)
            last_open = max(int(item[0]) for item in payload)
            next_cursor = last_open + THIRTY_MINUTES_MS
            if next_cursor <= cursor_ms:
                break
            cursor_ms = next_cursor

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )
        numeric_columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["open_time"]).copy()
        df["open_time"] = df["open_time"].astype("int64")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        df = df[(df["time"] >= start_time) & (df["time"] <= latest_closed)].copy()
        return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    def _fetch_funding_range(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_ms = max(0, int(_utc_naive(start).timestamp() * 1000))
        end_ms = int(_utc_naive(end).timestamp() * 1000)
        rows: list[dict[str, object]] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            payload = self.http.get_json(
                f"{self.futures_base_url}{self.FUTURES_FUNDING_PATH}",
                params={"symbol": symbol, "startTime": cursor_ms, "limit": self.MAX_FUNDING_LIMIT},
            )
            if not isinstance(payload, list) or not payload:
                break

            rows.extend(payload)
            last_time = max(int(item.get("fundingTime", 0)) for item in payload)
            if last_time >= end_ms:
                break
            next_cursor = last_time + 1
            if next_cursor <= cursor_ms or len(payload) < self.MAX_FUNDING_LIMIT:
                break
            cursor_ms = next_cursor

        if not rows:
            return pd.DataFrame(columns=["funding_time", "funding_rate_8h"])

        df = pd.DataFrame(rows)
        df["funding_time"] = pd.to_numeric(df.get("fundingTime"), errors="coerce")
        df["funding_rate_8h"] = pd.to_numeric(df.get("fundingRate"), errors="coerce")
        df = df.dropna(subset=["funding_time"]).copy()
        df["funding_time"] = df["funding_time"].astype("int64")
        df = df[df["funding_time"] <= end_ms].copy()
        return df.sort_values("funding_time").drop_duplicates("funding_time", keep="last").reset_index(drop=True)

    def fetch_processed_range(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        futures_df = self._fetch_klines_range(self.futures_base_url, self.FUTURES_KLINES_PATH, symbol=symbol, start=start, end=end)
        spot_df = self._fetch_klines_range(self.spot_base_url, self.SPOT_KLINES_PATH, symbol=symbol, start=start, end=end)
        if futures_df.empty or spot_df.empty:
            return pd.DataFrame()

        try:
            funding_df = self._fetch_funding_range(
                symbol=symbol,
                start=_utc_naive(start) - pd.Timedelta(milliseconds=EIGHT_HOURS_MS),
                end=end,
            )
        except Exception:
            funding_df = pd.DataFrame(columns=["funding_time", "funding_rate_8h"])
        if funding_df.empty:
            futures_df["funding_rate_8h"] = np.nan
        else:
            futures_df = pd.merge_asof(
                futures_df.sort_values("open_time"),
                funding_df.sort_values("funding_time"),
                left_on="open_time",
                right_on="funding_time",
                direction="backward",
            )
            futures_df["funding_rate_8h"] = futures_df["funding_rate_8h"].ffill().bfill()

        futures_df["exchange"] = "binance"
        futures_df["market_type"] = "futures"
        futures_df["symbol"] = symbol
        spot_df["exchange"] = "binance"
        spot_df["market_type"] = "spot"
        spot_df["symbol"] = symbol

        processed = _standardize_source_frame(
            build_binance_processed_output(futures_df=futures_df, spot_df=spot_df),
            time_column="time",
        )
        if base_df is not None and not base_df.empty:
            processed = _align_prefixed_cvd_columns(base_df, processed, ["binanceFutures_cvd", "binanceSpot_cvd"])
        return processed

    def fetch_recent_processed(self, symbol: str, bars: int, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        end = _latest_closed_30m_open_time() + THIRTY_MINUTES
        start = end - (THIRTY_MINUTES * max(int(bars), 2))
        return self.fetch_processed_range(symbol=symbol, start=start, end=end, base_df=base_df)


class CoinbaseLiveDataClient:
    MAX_BATCH = 300

    def __init__(self) -> None:
        self.base_url = settings.COINBASE_PUBLIC_URL.rstrip("/")
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    def fetch_processed_range(self, product_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_time = _utc_naive(start).floor("15min")
        latest_closed = _latest_closed_30m_open_time()
        end_time = min(_utc_naive(end).ceil("15min"), latest_closed + THIRTY_MINUTES)
        if end_time <= start_time:
            return pd.DataFrame()

        rows: list[list[object]] = []
        cursor = start_time
        while cursor < end_time:
            batch_end = min(cursor + pd.Timedelta(minutes=15 * self.MAX_BATCH), end_time)
            payload = self.http.get_json(
                f"{self.base_url}/products/{product_id}/candles",
                params={
                    "start": _to_utc_iso(cursor),
                    "end": _to_utc_iso(batch_end),
                    "granularity": 900,
                },
            )
            if isinstance(payload, list) and payload:
                rows.extend(payload)
            cursor = batch_end

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["epoch_seconds", "low", "high", "open", "close", "volume"])
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["epoch_seconds"]).copy()
        df["epoch_seconds"] = df["epoch_seconds"].astype("int64")
        df["time"] = pd.to_datetime(df["epoch_seconds"], unit="s", utc=True).dt.tz_convert(None)
        df = df[(df["time"] >= start_time) & (df["time"] <= latest_closed + THIRTY_MINUTES)].copy()
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        resampled, _ = coinbase_resample_to_30m(df)
        if resampled.empty:
            return pd.DataFrame()
        return _standardize_source_frame(build_coinbase_processed_output(resampled), time_column="coinbase_time")

    def fetch_recent_processed(self, product_id: str, fifteen_minute_bars: int) -> pd.DataFrame:
        end = _latest_closed_30m_open_time() + THIRTY_MINUTES
        start = end - pd.Timedelta(minutes=15 * max(int(fifteen_minute_bars), 2))
        return self.fetch_processed_range(product_id=product_id, start=start, end=end)


class BybitLiveDataClient:
    MAX_KLINE_LIMIT = 1000
    MAX_FUNDING_LIMIT = 200

    def __init__(self) -> None:
        self.base_url = settings.BYBIT_PUBLIC_URL.rstrip("/")
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    def _response_list(self, payload: dict) -> list:
        if payload.get("retCode") not in (0, None):
            raise RuntimeError(f"Bybit API error: {payload}")
        return payload.get("result", {}).get("list", [])

    def _fetch_kline_range(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_time = _utc_naive(start)
        latest_closed = _latest_closed_30m_open_time()
        end_time = min(_utc_naive(end), latest_closed + THIRTY_MINUTES)
        if end_time <= start_time:
            return pd.DataFrame()

        rows: list[list[object]] = []
        start_ms = int(start_time.timestamp() * 1000)
        cursor_end = int(end_time.timestamp() * 1000) - 1

        while cursor_end >= start_ms:
            payload = self.http.get_json(
                f"{self.base_url}/v5/market/kline",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "interval": "30",
                    "limit": self.MAX_KLINE_LIMIT,
                    "start": start_ms,
                    "end": cursor_end,
                },
            )
            kline_rows = self._response_list(payload if isinstance(payload, dict) else {})
            if not kline_rows:
                break

            rows.extend(kline_rows)
            earliest = min(int(row[0]) for row in kline_rows)
            if earliest <= start_ms or len(kline_rows) < self.MAX_KLINE_LIMIT:
                break
            next_cursor = earliest - 1
            if next_cursor >= cursor_end:
                break
            cursor_end = next_cursor

        if not rows:
            return pd.DataFrame()

        kline_df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
        for column in ["open_time", "open", "high", "low", "close", "volume", "turnover"]:
            kline_df[column] = pd.to_numeric(kline_df[column], errors="coerce")
        kline_df = kline_df.dropna(subset=["open_time"]).copy()
        kline_df["open_time"] = kline_df["open_time"].astype("int64")
        kline_df["time"] = pd.to_datetime(kline_df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        kline_df = kline_df[(kline_df["time"] >= start_time) & (kline_df["time"] <= latest_closed)].copy()
        return kline_df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    def _fetch_funding_range(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_ms = int(_utc_naive(start).timestamp() * 1000)
        cursor_end = int(_utc_naive(end).timestamp() * 1000)
        rows: list[dict[str, object]] = []

        while cursor_end >= start_ms:
            payload = self.http.get_json(
                f"{self.base_url}/v5/market/funding/history",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": cursor_end,
                    "limit": self.MAX_FUNDING_LIMIT,
                },
            )
            funding_rows = self._response_list(payload if isinstance(payload, dict) else {})
            if not funding_rows:
                break

            rows.extend(funding_rows)
            earliest = min(int(item.get("fundingRateTimestamp", 0)) for item in funding_rows)
            if earliest <= start_ms or len(funding_rows) < self.MAX_FUNDING_LIMIT:
                break
            next_cursor = earliest - 1
            if next_cursor >= cursor_end:
                break
            cursor_end = next_cursor

        if not rows:
            return pd.DataFrame(columns=["funding_time", "funding_rate"])

        funding_df = pd.DataFrame(rows)
        funding_df["funding_time"] = pd.to_numeric(funding_df.get("fundingRateTimestamp"), errors="coerce")
        funding_df["funding_rate"] = pd.to_numeric(funding_df.get("fundingRate"), errors="coerce")
        funding_df = funding_df.dropna(subset=["funding_time"]).copy()
        funding_df["funding_time"] = funding_df["funding_time"].astype("int64")
        return funding_df.sort_values("funding_time").drop_duplicates("funding_time", keep="last").reset_index(drop=True)

    def fetch_processed_range(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        kline_df = self._fetch_kline_range(symbol=symbol, start=start, end=end)
        if kline_df.empty:
            return pd.DataFrame()

        try:
            funding_df = self._fetch_funding_range(symbol=symbol, start=start - ONE_DAY, end=end)
        except Exception:
            funding_df = pd.DataFrame(columns=["funding_time", "funding_rate"])
        if funding_df.empty:
            kline_df["funding_rate"] = np.nan
        else:
            kline_df = pd.merge_asof(
                kline_df.sort_values("open_time"),
                funding_df[["funding_time", "funding_rate"]].sort_values("funding_time"),
                left_on="open_time",
                right_on="funding_time",
                direction="backward",
            )

        kline_df["quote_volume"] = pd.to_numeric(kline_df["turnover"], errors="coerce")

        # Bybit public REST does not offer deep historical trade pagination for this use case,
        # so the live bot derives flow features from candle imbalance instead of training files.
        kline_df = _candle_imbalance_proxy(kline_df, activity_column="volume", count_column="volume")
        if base_df is not None and not base_df.empty:
            base_for_cvd = base_df.rename(columns={"bybit_time": "time"}) if "bybit_time" in base_df.columns else base_df
            kline_df = _align_cvd_with_base(base_for_cvd, kline_df, "cvd")

        kline_df = add_bybit_technical_indicators(kline_df)
        output = kline_df.drop(columns=[column for column in ["turnover", "volume", "funding_time"] if column in kline_df.columns]).copy()
        output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        prefixed = output.rename(columns={column: f"bybit_{column}" for column in output.columns})
        ordered = ["bybit_time"] + [column for column in prefixed.columns if column != "bybit_time"]
        return _standardize_source_frame(prefixed[ordered], time_column="bybit_time")

    def fetch_recent_processed(self, symbol: str, bars: int, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        end = _latest_closed_30m_open_time() + THIRTY_MINUTES
        start = end - (THIRTY_MINUTES * max(int(bars), 2))
        return self.fetch_processed_range(symbol=symbol, start=start, end=end, base_df=base_df)


class DatabentoCmeClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or settings.DATABENTO_API_KEY).strip()
        self.dataset = settings.DATABENTO_CME_DATASET
        self.symbol = settings.DATABENTO_CME_SYMBOL
        self.stype_in = settings.DATABENTO_CME_STYPE_IN
        self.lookback_hours = max(float(settings.DATABENTO_CME_LOOKBACK_HOURS), 1.0)
        self.historical_client = db.Historical(self.api_key) if self.api_key else None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _ensure_ts_event_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "ts_event" in out.columns:
            return out
        if out.index.name == "ts_event" or isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "ts_event" not in out.columns:
                out = out.rename(columns={out.columns[0]: "ts_event"})
        return out

    @staticmethod
    def _extract_cached_warmup_frames(base_df: pd.DataFrame, start_time: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if base_df is None or base_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        cached = base_df.copy()
        cached["time"] = _parse_utc_naive(cached["time"])
        cached = cached.dropna(subset=["time"])
        cached = cached[cached["time"] >= _utc_naive(start_time)].sort_values("time").reset_index(drop=True)
        if cached.empty:
            return pd.DataFrame(), pd.DataFrame()

        bars = cached[
            ["time", "CME_open", "CME_high", "CME_low", "CME_close", "CME_volume"]
        ].rename(
            columns={
                "CME_open": "open",
                "CME_high": "high",
                "CME_low": "low",
                "CME_close": "close",
                "CME_volume": "volume",
            }
        )
        trades = cached[["time", "CME_count", "CME_delta"]].rename(
            columns={
                "CME_count": "count",
                "CME_delta": "delta",
            }
        )
        return bars.reset_index(drop=True), trades.reset_index(drop=True)

    @staticmethod
    def _parse_available_end_from_error(exc: Exception) -> pd.Timestamp | None:
        match = re.search(r"available up to '([^']+)'", str(exc))
        if match is None:
            return None
        parsed = pd.Timestamp(match.group(1))
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.tz_convert(None)

    def _get_range_historical(self, schema: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.historical_client is None:
            return pd.DataFrame()
        request_start = _utc_naive(start)
        request_end = _utc_naive(end)
        if request_end <= request_start:
            return pd.DataFrame()

        while True:
            try:
                store = self.historical_client.timeseries.get_range(
                    dataset=self.dataset,
                    start=request_start.tz_localize("UTC"),
                    end=request_end.tz_localize("UTC"),
                    symbols=[self.symbol],
                    schema=schema,
                    stype_in=self.stype_in,
                )
                return store.to_df(schema=schema, pretty_ts=True, map_symbols=False)
            except Exception as exc:
                available_end = self._parse_available_end_from_error(exc)
                if available_end is None:
                    raise
                adjusted_end = min(request_end, available_end)
                if adjusted_end >= request_end:
                    adjusted_end = request_end - pd.Timedelta(minutes=1)
                if adjusted_end <= request_start:
                    raise
                request_end = adjusted_end

    def _live_replay_start(self, start: pd.Timestamp) -> pd.Timestamp:
        now_utc = pd.Timestamp.now(tz="UTC").tz_convert(None)
        return max(_utc_naive(start), now_utc - LIVE_REPLAY_LOOKBACK)

    def _get_range_live(self, schema: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()

        replay_start = self._live_replay_start(start)
        replay_end = _utc_naive(end)
        if replay_end <= replay_start:
            return pd.DataFrame()

        client = db.Live(
            self.api_key,
            heartbeat_interval_s=5,
            reconnect_policy="none",
        )
        rows: list[dict[str, object]] = []
        replay_completed = False
        range_completed = False
        replay_start_utc = replay_start.tz_localize("UTC")
        replay_end_utc = replay_end.tz_localize("UTC")
        callback_error: Exception | None = None

        def on_record(record: object) -> None:
            nonlocal replay_completed, range_completed, callback_error
            if callback_error is not None:
                return

            if isinstance(record, dbn.ErrorMsg):
                callback_error = RuntimeError(f"Databento live error {record.code}: {record.err}")
                client.terminate()
                return

            if isinstance(record, dbn.SystemMsg):
                if getattr(record, "is_heartbeat", lambda: False)():
                    return
                if record.code == dbn.SystemCode.REPLAY_COMPLETED:
                    replay_completed = True
                    client.stop()
                return

            is_target_record = (
                (schema == "ohlcv-1m" and isinstance(record, dbn.OHLCVMsg))
                or (schema == "trades" and isinstance(record, dbn.TradeMsg))
            )
            if not is_target_record:
                return

            ts_event = pd.to_datetime(getattr(record, "ts_event", None), unit="ns", utc=True, errors="coerce")
            if pd.isna(ts_event):
                return
            if ts_event < replay_start_utc:
                return
            if ts_event > replay_end_utc:
                range_completed = True
                client.stop()
                return

            if schema == "ohlcv-1m" and isinstance(record, dbn.OHLCVMsg):
                rows.append(
                    {
                        "ts_event": ts_event,
                        "open": float(record.pretty_open),
                        "high": float(record.pretty_high),
                        "low": float(record.pretty_low),
                        "close": float(record.pretty_close),
                        "volume": float(record.volume),
                    }
                )
            elif schema == "trades" and isinstance(record, dbn.TradeMsg):
                side = getattr(record.side, "value", record.side)
                rows.append(
                    {
                        "ts_event": ts_event,
                        "size": float(record.size),
                        "side": str(side).upper(),
                    }
                )

        def on_callback_exception(exc: Exception) -> None:
            nonlocal callback_error
            callback_error = exc
            try:
                client.terminate()
            except Exception:
                pass

        client.add_callback(on_record, on_callback_exception)
        client.subscribe(
            dataset=self.dataset,
            schema=schema,
            symbols=[self.symbol],
            stype_in=self.stype_in,
            start=replay_start.tz_localize("UTC"),
        )

        try:
            client.start()
            client.block_for_close(timeout=max(settings.HTTP_TIMEOUT_SECONDS, 20.0))
            if callback_error is not None:
                raise callback_error
        finally:
            try:
                client.terminate()
            except Exception:
                pass

        if not replay_completed and not range_completed:
            raise TimeoutError(f"Databento live replay timeout for schema={schema}")

        if not rows:
            if replay_completed or range_completed:
                return pd.DataFrame()

        return pd.DataFrame(rows)

    def _fetch_ohlcv_30m(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ohlcv_df = self._ensure_ts_event_column(self._get_range_historical("ohlcv-1m", start=start, end=end))
        if ohlcv_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        required_columns = {"ts_event", "open", "high", "low", "close", "volume"}
        if not required_columns.issubset(ohlcv_df.columns):
            missing = sorted(required_columns - set(ohlcv_df.columns))
            raise RuntimeError(f"Databento CME OHLCV response is missing expected columns: {missing}")

        ohlcv_df["ts_event"] = pd.to_datetime(ohlcv_df["ts_event"], errors="coerce", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            ohlcv_df[column] = pd.to_numeric(ohlcv_df[column], errors="coerce")
        ohlcv_df = ohlcv_df.dropna(subset=["ts_event", "open", "high", "low", "close"]).copy()
        if ohlcv_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        ohlcv_df["time"] = ohlcv_df["ts_event"].dt.tz_convert(None).dt.floor("1min")
        aggregated = (
            ohlcv_df.sort_values("time")
            .groupby(pd.Grouper(key="time", freq="30min"), as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        )
        return aggregated.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)

    def _fetch_ohlcv_30m_live(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ohlcv_df = self._ensure_ts_event_column(self._get_range_live("ohlcv-1m", start=start, end=end))
        if ohlcv_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        required_columns = {"ts_event", "open", "high", "low", "close", "volume"}
        if not required_columns.issubset(ohlcv_df.columns):
            missing = sorted(required_columns - set(ohlcv_df.columns))
            raise RuntimeError(f"Databento CME live OHLCV response is missing expected columns: {missing}")

        ohlcv_df["ts_event"] = pd.to_datetime(ohlcv_df["ts_event"], errors="coerce", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            ohlcv_df[column] = pd.to_numeric(ohlcv_df[column], errors="coerce")
        ohlcv_df = ohlcv_df.dropna(subset=["ts_event", "open", "high", "low", "close"]).copy()
        if ohlcv_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        ohlcv_df["time"] = ohlcv_df["ts_event"].dt.tz_convert(None).dt.floor("1min")
        aggregated = (
            ohlcv_df.sort_values("time")
            .groupby(pd.Grouper(key="time", freq="30min"), as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        )
        return aggregated.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)

    def _fetch_trade_metrics_30m(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        trades_df = self._ensure_ts_event_column(self._get_range_historical("trades", start=start, end=end))
        if trades_df.empty:
            return pd.DataFrame(columns=["time", "count", "delta"])

        required_columns = {"ts_event", "size", "side"}
        if not required_columns.issubset(trades_df.columns):
            missing = sorted(required_columns - set(trades_df.columns))
            raise RuntimeError(f"Databento CME trades response is missing expected columns: {missing}")

        trades_df = trades_df[["ts_event", "size", "side"]].copy()
        trades_df["ts_event"] = pd.to_datetime(trades_df["ts_event"], errors="coerce", utc=True)
        trades_df["size"] = pd.to_numeric(trades_df["size"], errors="coerce")
        trades_df["side"] = trades_df["side"].astype("string").str.upper()
        trades_df = trades_df.dropna(subset=["ts_event", "size"]).copy()
        if trades_df.empty:
            return pd.DataFrame(columns=["time", "count", "delta"])

        trades_df["time"] = trades_df["ts_event"].dt.tz_convert(None).dt.floor("30min")
        trades_df["signed_size"] = np.select(
            [trades_df["side"] == "B", trades_df["side"] == "A"],
            [trades_df["size"], -trades_df["size"]],
            default=0.0,
        )
        aggregated = trades_df.groupby("time", as_index=False).agg(
            count=("size", "count"),
            delta=("signed_size", "sum"),
        )
        aggregated["count"] = aggregated["count"].round().astype("int64")
        aggregated["delta"] = aggregated["delta"].round().astype("int64")
        return aggregated

    def _fetch_trade_metrics_30m_live(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        trades_df = self._ensure_ts_event_column(self._get_range_live("trades", start=start, end=end))
        if trades_df.empty:
            return pd.DataFrame(columns=["time", "count", "delta"])

        required_columns = {"ts_event", "size", "side"}
        if not required_columns.issubset(trades_df.columns):
            missing = sorted(required_columns - set(trades_df.columns))
            raise RuntimeError(f"Databento CME live trades response is missing expected columns: {missing}")

        trades_df = trades_df[["ts_event", "size", "side"]].copy()
        trades_df["ts_event"] = pd.to_datetime(trades_df["ts_event"], errors="coerce", utc=True)
        trades_df["size"] = pd.to_numeric(trades_df["size"], errors="coerce")
        trades_df["side"] = trades_df["side"].astype("string").str.upper()
        trades_df = trades_df.dropna(subset=["ts_event", "size"]).copy()
        if trades_df.empty:
            return pd.DataFrame(columns=["time", "count", "delta"])

        trades_df["time"] = trades_df["ts_event"].dt.tz_convert(None).dt.floor("30min")
        trades_df["signed_size"] = np.select(
            [trades_df["side"] == "B", trades_df["side"] == "A"],
            [trades_df["size"], -trades_df["size"]],
            default=0.0,
        )
        aggregated = trades_df.groupby("time", as_index=False).agg(
            count=("size", "count"),
            delta=("signed_size", "sum"),
        )
        aggregated["count"] = aggregated["count"].round().astype("int64")
        aggregated["delta"] = aggregated["delta"].round().astype("int64")
        return aggregated

    def fetch_processed_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()

        latest_closed = _latest_closed_30m_open_time()
        end_time = min(_utc_naive(end), latest_closed + THIRTY_MINUTES)
        live_cutoff = pd.Timestamp.now(tz="UTC").tz_convert(None) - LIVE_REPLAY_LOOKBACK
        start_time = _utc_naive(start)
        source_start = start_time
        cached_bars = pd.DataFrame()
        cached_trades = pd.DataFrame()

        if base_df is not None and not base_df.empty:
            cached_times = _parse_utc_naive(base_df["time"])
            latest_cached = cached_times.max()
            if pd.notna(latest_cached):
                warmup_start = max(start_time, pd.Timestamp(latest_cached) - (THIRTY_MINUTES * CME_INCREMENTAL_CACHE_WARMUP_BARS))
                source_start = max(start_time, pd.Timestamp(latest_cached) - (THIRTY_MINUTES * CME_INCREMENTAL_FETCH_OVERLAP_BARS))
                cached_bars, cached_trades = self._extract_cached_warmup_frames(base_df, warmup_start)

        historical_bars = pd.DataFrame()
        if source_start < min(end_time, live_cutoff):
            try:
                historical_bars = self._fetch_ohlcv_30m(start=source_start, end=min(end_time, live_cutoff))
            except Exception as exc:
                raise RuntimeError(
                    f"Databento CME historical OHLCV fetch failed | start={source_start} | end={min(end_time, live_cutoff)} | reason={exc}"
                ) from exc

        recent_bars = pd.DataFrame()
        if end_time > live_cutoff:
            live_start = max(source_start, live_cutoff)
            try:
                recent_bars = self._fetch_ohlcv_30m_live(start=live_start, end=end_time)
            except Exception as exc:
                raise RuntimeError(
                    f"Databento CME live OHLCV fetch failed | start={live_start} | end={end_time} | reason={exc}"
                ) from exc
            if recent_bars.empty:
                try:
                    recent_bars = self._fetch_ohlcv_30m(start=live_start, end=end_time)
                except Exception as exc:
                    raise RuntimeError(
                        f"Databento CME historical OHLCV fallback failed | start={live_start} | end={end_time} | reason={exc}"
                    ) from exc

        bars_df = _merge_time_frames(cached_bars, historical_bars, recent_bars)
        if bars_df.empty:
            return pd.DataFrame()

        cme_df = _candle_imbalance_proxy(bars_df, activity_column="volume", count_column="volume")

        exact_start = max(source_start, end_time - pd.Timedelta(hours=self.lookback_hours))
        if exact_start < end_time:
            try:
                historical_trades = pd.DataFrame()
                if exact_start < min(end_time, live_cutoff):
                    historical_trades = self._fetch_trade_metrics_30m(start=exact_start, end=min(end_time, live_cutoff))

                recent_trades = pd.DataFrame()
                if end_time > live_cutoff:
                    trades_start = max(exact_start, live_cutoff)
                    recent_trades = self._fetch_trade_metrics_30m_live(start=trades_start, end=end_time)
                    if recent_trades.empty:
                        recent_trades = self._fetch_trade_metrics_30m(start=trades_start, end=end_time)

                trades_df = _merge_time_frames(cached_trades, historical_trades, recent_trades)
            except Exception:
                trades_df = cached_trades.copy() if not cached_trades.empty else pd.DataFrame(columns=["time", "count", "delta"])
            if not trades_df.empty:
                cme_df = cme_df.merge(trades_df, on="time", how="left", suffixes=("", "_exact"))
                cme_df["count"] = cme_df["count_exact"].fillna(cme_df["count"]).round().astype("int64")
                cme_df["delta"] = cme_df["delta_exact"].fillna(cme_df["delta"])
                cme_df = cme_df.drop(columns=[column for column in ["count_exact", "delta_exact"] if column in cme_df.columns])
                cme_df["cvd"] = pd.to_numeric(cme_df["delta"], errors="coerce").fillna(0.0).cumsum()

        cme_df = cme_df[cme_df["time"] <= latest_closed].copy()
        if cme_df.empty:
            return pd.DataFrame()

        cme_df = cme_df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        cme_df["open_time"] = cme_df["time"].map(lambda ts: int(pd.Timestamp(ts).timestamp())).astype("int64")

        if base_df is not None and not base_df.empty:
            base_for_cvd = base_df.rename(columns={"CME_time": "time"}) if "CME_time" in base_df.columns else base_df
            cme_df = _align_cvd_with_base(base_for_cvd, cme_df, "cvd")
        cme_df = add_cme_technical_indicators(cme_df)

        output = cme_df[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "count",
                "delta",
                "cvd",
                "open_time",
                "rsi_14",
                "atr_14",
                "bb_position_20_2",
                "macd",
                "macd_signal",
                "macd_hist",
            ]
        ].copy()
        output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        prefixed = output.rename(columns={column: f"CME_{column}" for column in output.columns})
        ordered = ["CME_time"] + [column for column in prefixed.columns if column != "CME_time"]
        return _standardize_source_frame(prefixed[ordered], time_column="CME_time")

    def fetch_recent_processed(self, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        end = _latest_closed_30m_fetch_end("Databento CME")
        start = end - pd.Timedelta(hours=self.lookback_hours)
        return self.fetch_processed_range(start=start, end=end, base_df=base_df)


class FredLiveDataClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or settings.FRED_API_KEY).strip()
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _fetch_series(self, series_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()
        padded_start = pd.Timestamp(start_date).normalize() - pd.Timedelta(days=30)
        try:
            payload = self.http.get_json(
                FRED_BASE_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": padded_start.strftime("%Y-%m-%d"),
                    "observation_end": pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d"),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"FRED series fetch failed | series_id={series_id} | reason={exc}") from exc
        observations = payload.get("observations", []) if isinstance(payload, dict) else []
        rows: list[dict[str, object]] = []
        for observation in observations:
            raw_value = observation.get("value")
            if raw_value in (None, "", "."):
                continue
            rows.append({"Date": observation.get("date"), "value": float(raw_value)})

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    def fetch_net_liquidity(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        fed_df = self._fetch_series(FRED_SERIES_IDS["fed_total_assets"], start_date, end_date).rename(columns={"value": "fed_total_assets"})
        tga_df = self._fetch_series(FRED_SERIES_IDS["tga"], start_date, end_date).rename(columns={"value": "tga"})
        rrp_df = self._fetch_series(FRED_SERIES_IDS["rrp"], start_date, end_date).rename(columns={"value": "rrp"})
        if fed_df.empty or tga_df.empty or rrp_df.empty:
            return pd.DataFrame()

        merged = fed_df.merge(tga_df, on="Date", how="outer").merge(rrp_df, on="Date", how="outer")
        merged = merged.sort_values("Date").reset_index(drop=True)
        merged[["fed_total_assets", "tga", "rrp"]] = merged[["fed_total_assets", "tga", "rrp"]].ffill()
        merged = merged.dropna(subset=["fed_total_assets", "tga", "rrp"]).copy()
        merged["fed_net_liquidity"] = merged["fed_total_assets"] - (merged["tga"] + merged["rrp"])
        merged = merged.loc[merged["Date"] >= pd.Timestamp(start_date).normalize()].copy()
        return merged[["Date", "fed_net_liquidity", "tga", "rrp"]].sort_values("Date").reset_index(drop=True)


class CoinalyzeLiveDataClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or settings.COINALYZE_API_KEY).strip()
        self.symbol = settings.COINALYZE_SYMBOL
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _fetch_history(self, endpoint: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()

        start_ts = int(pd.Timestamp(start_date).normalize().tz_localize("UTC").timestamp())
        end_ts = int((pd.Timestamp(end_date).normalize() + ONE_DAY).tz_localize("UTC").timestamp()) - 1
        payload = self.http.get_json(
            f"{COINALYZE_BASE_URL}/{endpoint}",
            params={"symbols": self.symbol, "interval": "daily", "from": start_ts, "to": end_ts},
            headers={"api_key": self.api_key},
        )
        if not isinstance(payload, list):
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for item in payload:
            for history_row in item.get("history", []):
                row = {"timestamp": history_row.get("t")}
                row.update(history_row)
                rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["Date"] = pd.to_datetime(pd.to_numeric(out["timestamp"], errors="coerce"), unit="s", utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    def fetch_open_interest(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self._fetch_history("open-interest-history", start_date, end_date)
        if df.empty:
            return df
        df = df.rename(columns={"o": "oi_open", "h": "oi_high", "l": "oi_low", "c": "oi_close"})
        for column in ["oi_open", "oi_high", "oi_low", "oi_close"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["Date", "oi_open", "oi_high", "oi_low", "oi_close"]).copy()
        return df[["Date", "oi_open", "oi_high", "oi_low", "oi_close"]].sort_values("Date").reset_index(drop=True)

    def fetch_long_short_ratio(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self._fetch_history("long-short-ratio-history", start_date, end_date)
        if df.empty:
            return df
        df = df.rename(columns={"l": "long_ratio", "s": "short_ratio"})
        for column in ["long_ratio", "short_ratio"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["Date", "long_ratio", "short_ratio"]).copy()
        return df[["Date", "long_ratio", "short_ratio"]].sort_values("Date").reset_index(drop=True)


class ModelSignalEngine:
    def __init__(
        self,
        databento_api_key: str | None = None,
        fred_api_key: str | None = None,
        coinalyze_api_key: str | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        self.config = checkpoint["config"]
        self.logger = logger or (lambda message: None)
        self.feature_cols_30m = list(self.config["feature_columns_30m"])
        self.feature_cols_1d = list(self.config["feature_columns_1d"])
        self.groups_30m = dict(self.config["groups_30m"])
        self.groups_1d = dict(self.config["groups_1d"])
        self.window_size_30m = int(self.config["window_size_30m"])
        self.window_size_1d = int(self.config["window_size_1d"])
        self.daily_feature_lag_days = int(self.config["daily_feature_lag_days"])
        self.level_window_30m = int(self.config["level_window_30m"])
        self.flow_activity_window_30m = int(self.config["flow_activity_window_30m"])
        self.cvd_detrend_window_30m = int(self.config["cvd_detrend_window_30m"])
        self.level_window_1d = int(self.config["level_window_1d"])
        self.bootstrap_30m_bars = max(
            self.level_window_30m,
            self.flow_activity_window_30m,
            self.cvd_detrend_window_30m,
            self.window_size_30m,
            96,
        ) + max(int(settings.LIVE_BOOTSTRAP_30M_BUFFER_BARS), 0)
        self.bootstrap_1d_days = max(
            self.level_window_1d,
            self.window_size_1d + self.daily_feature_lag_days,
            14,
        ) + max(int(settings.LIVE_BOOTSTRAP_1D_BUFFER_DAYS), 0)
        self.recent_30m_bars = max(int(settings.LIVE_RECENT_30M_BARS), 96)
        self.recent_1d_days = max(int(settings.LIVE_INCREMENTAL_1D_DAYS), 7)
        # Incremental source refreshes need extra pre-roll so per-source
        # indicators (RSI/ATR/BB/MACD) do not introduce NaNs at the refresh edge.
        self.recent_30m_warmup_bars = max(int(settings.LIVE_BOOTSTRAP_30M_BUFFER_BARS), 96)
        # Live trading thresholds are intentionally sourced from editable runtime
        # settings so the bot/UI can change behavior without re-exporting a model
        # checkpoint that may still carry older notebook defaults.
        self.decision_rule = DecisionRule(
            max_flat_probability=float(settings.TRADE_SIGNAL_MAX_FLAT_PROBABILITY),
            side_ratio_threshold=float(settings.TRADE_SIGNAL_SIDE_RATIO_THRESHOLD),
            ordering_max_flat_probability=float(settings.TRADE_SIGNAL_ORDERING_MAX_FLAT_PROBABILITY),
        )

        self.model = MultiTimescaleFusionModel(
            input_dim_30m=len(self.feature_cols_30m),
            input_dim_1d=len(self.feature_cols_1d),
            d_model=int(self.config["d_model"]),
            nhead=int(self.config["nhead"]),
            num_layers=int(self.config["num_layers"]),
            dim_feedforward=int(self.config["dim_feedforward"]),
            dropout=float(self.config["dropout"]),
            max_len=int(self.config["max_pos_embed_len"]),
            gru_hidden_dim=int(self.config["gru_hidden_dim"]),
            gru_num_layers=int(self.config["gru_num_layers"]),
            gru_dropout=float(self.config["gru_dropout"]),
            fusion_hidden_dim=int(self.config["fusion_hidden_dim"]),
            output_dim=int(self.config["output_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.binance_client = BinanceLiveDataClient()
        self.coinbase_client = CoinbaseLiveDataClient()
        self.bybit_client = BybitLiveDataClient()
        self.databento_client = DatabentoCmeClient(api_key=databento_api_key)
        self.fred_client = FredLiveDataClient(api_key=fred_api_key)
        self.coinalyze_client = CoinalyzeLiveDataClient(api_key=coinalyze_api_key)
        self.last_post_close_daily_refresh_date: str | None = None

        self.coinbase_base = self._bootstrap_30m_cache(
            label="Coinbase",
            cache_path=LIVE_COINBASE_PATH,
            fetch_fn=lambda start, end, base: self.coinbase_client.fetch_processed_range("BTC-USD", start=start, end=end),
        )
        self.binance_base = self._bootstrap_30m_cache(
            label="Binance",
            cache_path=LIVE_BINANCE_PATH,
            fetch_fn=lambda start, end, base: self.binance_client.fetch_processed_range(settings.SYMBOL, start=start, end=end, base_df=base),
        )
        self.bybit_base = self._bootstrap_30m_cache(
            label="Bybit",
            cache_path=LIVE_BYBIT_PATH,
            fetch_fn=lambda start, end, base: self.bybit_client.fetch_processed_range("BTCUSDT", start=start, end=end, base_df=base),
        )
        self.cme_base = self._bootstrap_30m_cache(
            label="Databento CME",
            cache_path=LIVE_CME_PATH,
            fetch_fn=lambda start, end, base: self.databento_client.fetch_processed_range(start=start, end=end, base_df=base),
        )
        self.daily_base = self._bootstrap_daily_cache()
        self.daily_base = self._run_daily_refresh([], log_label="Gunluk startup refresh")
        if self._is_after_us_market_close():
            self.last_post_close_daily_refresh_date = self._us_market_date_key()

    def _required_30m_history_start(self) -> pd.Timestamp:
        latest_closed = _latest_closed_30m_open_time()
        return latest_closed - (THIRTY_MINUTES * (self.bootstrap_30m_bars - 1))

    def _required_1d_history_start(self) -> pd.Timestamp:
        anchor_end = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize() - (ONE_DAY * self.daily_feature_lag_days)
        return anchor_end - pd.Timedelta(days=self.bootstrap_1d_days - 1)

    def _us_market_now(self, now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
        now_utc = now_utc or pd.Timestamp.now(tz="UTC")
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")
        return now_utc.tz_convert(US_MARKET_TIMEZONE)

    def _us_market_date_key(self, now_utc: pd.Timestamp | None = None) -> str:
        return self._us_market_now(now_utc).strftime("%Y-%m-%d")

    def _is_after_us_market_close(self, now_utc: pd.Timestamp | None = None) -> bool:
        market_now = self._us_market_now(now_utc)
        cutoff = market_now.normalize() + pd.Timedelta(hours=US_MARKET_CLOSE_HOUR, minutes=US_MARKET_CLOSE_MINUTE)
        return market_now >= cutoff

    def _log_note(self, note: str, notes: list[str]) -> None:
        if notes and notes[-1] == note:
            return
        notes.append(note)

    def _source_latest_times(self) -> dict[str, pd.Timestamp]:
        sources = {
            "Coinbase": self.coinbase_base,
            "Binance": self.binance_base,
            "Bybit": self.bybit_base,
            "Databento CME": self.cme_base,
        }
        latest_times: dict[str, pd.Timestamp] = {}
        for name, df in sources.items():
            if df is None or df.empty or "time" not in df.columns:
                latest_times[name] = pd.NaT
                continue
            latest_times[name] = _parse_utc_naive(df["time"]).max()
        return latest_times

    def _has_required_30m_history(
        self,
        df: pd.DataFrame,
        history_start: pd.Timestamp,
        *,
        require_contiguous_cache: bool,
    ) -> bool:
        return (
            not df.empty
            and df["time"].min() <= history_start
            and len(df) >= self.window_size_30m
            and (not require_contiguous_cache or not _has_large_30m_gap(df))
        )

    def _bootstrap_30m_cache(
        self,
        label: str,
        cache_path: Path,
        fetch_fn: Callable[[pd.Timestamp, pd.Timestamp, pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        history_start = self._required_30m_history_start()
        cached = _trim_time_window(_load_time_cache(cache_path, time_column="time"), history_start)
        require_contiguous_cache = label in {"Coinbase", "Binance", "Bybit"}
        has_required_history = self._has_required_30m_history(
            cached,
            history_start,
            require_contiguous_cache=require_contiguous_cache,
        )
        if has_required_history:
            _save_time_cache(cached, cache_path)
            return cached
        if label == "Databento CME" and cached.empty and not self.databento_client.configured:
            raise ValueError("Databento CME bootstrap icin Databento API key gerekli veya mevcut data/liveData CME cache'i bulunmali.")

        fetch_seed = cached if has_required_history else pd.DataFrame()
        fetched = _trim_time_window(
            fetch_fn(history_start, _latest_closed_30m_fetch_end(label), fetch_seed),
            history_start,
        )
        if fetched.empty:
            empty_note = (
                "Databento CME bootstrap bos dondu; API key mevcut ama Databento hic bar dondurmedi."
                if label == "Databento CME" and self.databento_client.configured
                else f"{label} bootstrap bos dondu; mevcut liveData cache korunuyor."
            )
            if not cached.empty:
                self.logger(empty_note + (" Mevcut liveData cache korunuyor." if label == "Databento CME" and self.databento_client.configured else ""))
                return cached
            if label == "Databento CME" and self.databento_client.configured:
                raise ValueError("Databento CME bootstrap bos dondu; API key mevcut ama Databento hic bar dondurmedi ve liveData cache bulunmuyor.")
            raise ValueError(f"{label} live bootstrap failed and no liveData cache exists.")

        _save_time_cache(fetched, cache_path)
        self.logger(f"{label} bootstrap tamamlandi | rows={len(fetched)}")
        return fetched

    def _fetch_daily_merged(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        fed_df = self.fred_client.fetch_net_liquidity(start_date=start_date, end_date=end_date)
        oi_df = self.coinalyze_client.fetch_open_interest(start_date=start_date, end_date=end_date)
        lsr_df = self.coinalyze_client.fetch_long_short_ratio(start_date=start_date, end_date=end_date)
        if fed_df.empty or oi_df.empty or lsr_df.empty:
            return pd.DataFrame()

        overlap_start, overlap_end = _daily_overlap_bounds([fed_df, oi_df, lsr_df])
        fed_filled = _reindex_and_ffill_daily(fed_df, overlap_start, overlap_end)
        oi_filled = _reindex_and_ffill_daily(oi_df, overlap_start, overlap_end)
        lsr_filled = _reindex_and_ffill_daily(lsr_df, overlap_start, overlap_end)
        merged = fed_filled.merge(oi_filled, on="Date", how="inner", validate="one_to_one")
        merged = merged.merge(lsr_filled, on="Date", how="inner", validate="one_to_one")
        return merged.sort_values("Date").reset_index(drop=True)

    def _bootstrap_daily_cache(self) -> pd.DataFrame:
        history_start = self._required_1d_history_start()
        cached = _trim_daily_window(_load_daily_cache(LIVE_DAILY_PATH), history_start)
        has_required_history = not cached.empty and cached["Date"].min() <= history_start and len(cached) >= self.window_size_1d
        if has_required_history:
            _save_daily_cache(cached, LIVE_DAILY_PATH)
            return cached
        if cached.empty and (not self.fred_client.configured or not self.coinalyze_client.configured):
            raise ValueError("Gunluk bootstrap icin hem FRED API key hem Coinalyze API key gerekli veya mevcut data/liveData daily cache'i bulunmali.")

        try:
            fetched = _trim_daily_window(
                self._fetch_daily_merged(
                    start_date=history_start,
                    end_date=pd.Timestamp.now(tz="UTC").tz_convert(None).normalize(),
                ),
                history_start,
            )
        except Exception as exc:
            if not cached.empty:
                self.logger(f"Gunluk bootstrap failed; mevcut liveData cache korunuyor. reason={exc}")
                return cached
            raise
        if fetched.empty:
            if not cached.empty:
                self.logger("Gunluk bootstrap bos dondu; mevcut liveData cache korunuyor.")
                return cached
            raise ValueError("Gunluk live bootstrap failed and no liveData daily cache exists.")

        _save_daily_cache(fetched, LIVE_DAILY_PATH)
        self.logger(f"Gunluk bootstrap tamamlandi | rows={len(fetched)}")
        return fetched

    def _refresh_30m_cache(
        self,
        label: str,
        base_df: pd.DataFrame,
        cache_path: Path,
        fetch_fn: Callable[[pd.Timestamp, pd.Timestamp, pd.DataFrame], pd.DataFrame],
        notes: list[str],
        *,
        is_source_configured: bool = True,
        missing_config_note: str | None = None,
    ) -> pd.DataFrame:
        history_start = self._required_30m_history_start()
        source_latest_closed = _latest_closed_cme_30m_open_time() if label == "Databento CME" else _latest_closed_30m_open_time()
        recent_start = max(history_start, source_latest_closed - (THIRTY_MINUTES * (self.recent_30m_bars - 1)))
        require_contiguous_cache = label in {"Coinbase", "Binance", "Bybit"}
        has_required_history = self._has_required_30m_history(
            _trim_time_window(base_df, history_start),
            history_start,
            require_contiguous_cache=require_contiguous_cache,
        )
        force_full_backfill = not has_required_history
        fetch_start = history_start if force_full_backfill else max(
            history_start,
            recent_start - (THIRTY_MINUTES * self.recent_30m_warmup_bars),
        )
        if not is_source_configured:
            self._log_note(
                missing_config_note or f"{label} recent update atlandi; kaynak konfiguru yok, liveData cache kullaniliyor.",
                notes,
            )
            return _trim_time_window(base_df, history_start)
        try:
            fetch_seed = base_df if has_required_history else pd.DataFrame()
            recent = fetch_fn(fetch_start, _latest_closed_30m_fetch_end(label), fetch_seed)
            recent = _trim_time_window(recent, history_start if force_full_backfill else recent_start)
            if recent.empty:
                if label == "Databento CME" and is_source_configured:
                    self._log_note(
                        "Databento CME recent update returned no closed bars despite configured API key; liveData cache kullaniliyor.",
                        notes,
                    )
                else:
                    self._log_note(f"{label} recent update returned no closed bars; liveData cache kullaniliyor.", notes)
                return _trim_time_window(base_df, history_start)

            updated = _trim_time_window(_overlay_processed_frames(base_df, recent), history_start)
            _save_time_cache(updated, cache_path)
            updated_latest = _parse_utc_naive(updated["time"]).max() if not updated.empty else pd.NaT
            expected_latest = source_latest_closed
            if label == "Databento CME" and (pd.isna(updated_latest) or updated_latest < expected_latest):
                self._log_note(
                    "Databento CME recent update son kapali mumu yetistiremedi; "
                    f"latest_utc={(updated_latest.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(updated_latest) else 'missing')} | "
                    f"expected_utc={expected_latest:%Y-%m-%d %H:%M:%S}",
                    notes,
                )
            return updated
        except Exception as exc:
            self._log_note(f"{label} recent update failed; liveData cache kullaniliyor. reason={exc}", notes)
            return _trim_time_window(base_df, history_start)

    def _run_daily_refresh(self, notes: list[str], *, log_label: str) -> pd.DataFrame:
        history_start = self._required_1d_history_start()
        base_daily = _trim_daily_window(self.daily_base, history_start)
        missing_sources: list[str] = []
        if not self.fred_client.configured:
            missing_sources.append("FRED")
        if not self.coinalyze_client.configured:
            missing_sources.append("Coinalyze")
        if missing_sources:
            self._log_note(
                f"{log_label} atlandi; eksik API key: {', '.join(missing_sources)}. liveData cache kullaniliyor.",
                notes,
            )
            return base_daily

        recent_start = max(history_start, pd.Timestamp.now(tz="UTC").tz_convert(None).normalize() - pd.Timedelta(days=self.recent_1d_days))
        try:
            recent = self._fetch_daily_merged(
                start_date=recent_start,
                end_date=pd.Timestamp.now(tz="UTC").tz_convert(None).normalize(),
            )
            if recent.empty:
                self._log_note(f"{log_label} bos dondu; liveData cache kullaniliyor.", notes)
                return base_daily

            updated = _trim_daily_window(_overlay_daily_frames(base_daily, recent), history_start)
            _save_daily_cache(updated, LIVE_DAILY_PATH)
            self.logger(f"{log_label} tamamlandi | rows={len(updated)}")
            return updated
        except Exception as exc:
            self._log_note(f"{log_label} failed; liveData cache kullaniliyor. reason={exc}", notes)
            return base_daily

    def _refresh_daily_cache_if_due(self, notes: list[str]) -> pd.DataFrame:
        history_start = self._required_1d_history_start()
        self.daily_base = _trim_daily_window(self.daily_base, history_start)
        now_utc = pd.Timestamp.now(tz="UTC")
        if not self._is_after_us_market_close(now_utc):
            return self.daily_base

        market_date = self._us_market_date_key(now_utc)
        if self.last_post_close_daily_refresh_date == market_date:
            return self.daily_base

        self.last_post_close_daily_refresh_date = market_date
        self.daily_base = self._run_daily_refresh(notes, log_label="Gunluk kapanis refresh")
        return self.daily_base

    def _build_updated_sources(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        notes: list[str] = []

        self.coinbase_base = self._refresh_30m_cache(
            label="Coinbase",
            base_df=self.coinbase_base,
            cache_path=LIVE_COINBASE_PATH,
            fetch_fn=lambda start, end, base: self.coinbase_client.fetch_processed_range("BTC-USD", start=start, end=end),
            notes=notes,
        )
        self.binance_base = self._refresh_30m_cache(
            label="Binance",
            base_df=self.binance_base,
            cache_path=LIVE_BINANCE_PATH,
            fetch_fn=lambda start, end, base: self.binance_client.fetch_processed_range(settings.SYMBOL, start=start, end=end, base_df=base),
            notes=notes,
        )
        self.bybit_base = self._refresh_30m_cache(
            label="Bybit",
            base_df=self.bybit_base,
            cache_path=LIVE_BYBIT_PATH,
            fetch_fn=lambda start, end, base: self.bybit_client.fetch_processed_range("BTCUSDT", start=start, end=end, base_df=base),
            notes=notes,
        )
        self.cme_base = self._refresh_30m_cache(
            label="Databento CME",
            base_df=self.cme_base,
            cache_path=LIVE_CME_PATH,
            fetch_fn=lambda start, end, base: self.databento_client.fetch_processed_range(start=start, end=end, base_df=base),
            notes=notes,
            is_source_configured=self.databento_client.configured,
            missing_config_note="Databento CME recent update atlandi; Databento API key yok, liveData cache kullaniliyor.",
        )
        self.daily_base = self._refresh_daily_cache_if_due(notes)

        merged_30m = self.coinbase_base.merge(self.binance_base, on="time", how="inner", validate="one_to_one")
        merged_30m = merged_30m.merge(self.bybit_base, on="time", how="inner", validate="one_to_one")

        cme_for_merge, synthetic_cme_times = _synthesize_missing_cme_rows(self.cme_base, merged_30m["time"])
        actual_cme_latest = _parse_utc_naive(self.cme_base["time"]).max() if not self.cme_base.empty else pd.NaT
        current_synthetic_cme_times = [
            ts for ts in synthetic_cme_times if pd.isna(actual_cme_latest) or ts > actual_cme_latest
        ]
        if current_synthetic_cme_times:
            special_reason = _cme_special_session_reason(actual_cme_latest) if pd.notna(actual_cme_latest) else None
            note_prefix = (
                f"Databento CME resmi seans kapanisi algilandi ({special_reason}); "
                if special_reason
                else "Databento CME eksik mumlar sentetik dolduruldu; "
            )
            self._log_note(
                note_prefix
                + 
                "fiyat kolonlari bir onceki CME satirindan kopyalandi, volume/count/delta=0 yapildi | "
                f"rows={len(current_synthetic_cme_times)} | "
                f"first_utc={current_synthetic_cme_times[0]:%Y-%m-%d %H:%M:%S} | "
                f"last_utc={current_synthetic_cme_times[-1]:%Y-%m-%d %H:%M:%S}",
                notes,
            )

        merged_30m = merged_30m.merge(cme_for_merge, on="time", how="inner", validate="one_to_one")
        merged_30m = _drop_helper_open_time_columns(merged_30m).sort_values("time").reset_index(drop=True)

        if merged_30m.empty:
            raise ValueError("No overlapping 30m rows remain after merging live API caches.")

        return merged_30m, self.daily_base.copy(), notes

    def _build_scaled_frames(self, merged_30m: pd.DataFrame, merged_1d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_30m = merged_30m[self.feature_cols_30m].copy()
        raw_1d = merged_1d[self.feature_cols_1d].copy()

        norm_30m = normalize_30m_frame(
            raw_30m,
            groups=self.groups_30m,
            level_window_30m=self.level_window_30m,
            flow_activity_window_30m=self.flow_activity_window_30m,
            cvd_detrend_window_30m=self.cvd_detrend_window_30m,
        )
        norm_1d = normalize_1d_frame(raw_1d, groups=self.groups_1d, level_window_1d=self.level_window_1d)

        scaled_30m = _manual_standard_scale(
            norm_30m,
            columns=list(self.groups_30m["standard_scale_cols"]),
            mean_values=list(self.config["scaler_30m_mean"]),
            scale_values=list(self.config["scaler_30m_scale"]),
        )
        scaled_1d = _manual_standard_scale(
            norm_1d,
            columns=list(self.groups_1d["standard_scale_cols"]),
            mean_values=list(self.config["scaler_1d_mean"]),
            scale_values=list(self.config["scaler_1d_scale"]),
        )
        return scaled_30m, scaled_1d

    def _compute_barrier_width(self, merged_30m: pd.DataFrame) -> pd.Series:
        timestamps = pd.to_datetime(merged_30m["time"], errors="coerce", utc=True)
        _, atr = compute_past_only_atr(
            high=pd.to_numeric(merged_30m["binanceFutures_high"], errors="coerce"),
            low=pd.to_numeric(merged_30m["binanceFutures_low"], errors="coerce"),
            close=pd.to_numeric(merged_30m["binanceFutures_close"], errors="coerce"),
            period=ATR_PERIOD,
        )
        effective_k, _ = compute_effective_barrier_multiplier(atr=atr, timestamps=timestamps)
        return effective_k * atr

    def predict_latest(self) -> PredictionResult:
        merged_30m, merged_1d, notes = self._build_updated_sources()
        merged_30m = merged_30m.copy()
        merged_30m["time"] = _parse_utc_naive(merged_30m["time"])
        merged_30m = merged_30m.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        merged_1d = merged_1d.copy()
        merged_1d["Date"] = pd.to_datetime(merged_1d["Date"], errors="coerce").dt.normalize()
        merged_1d = merged_1d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        latest_available = merged_30m["time"].max()
        expected_latest = _latest_expected_common_30m_open_time()
        if pd.isna(latest_available) or latest_available < expected_latest:
            source_latest_times = self._source_latest_times()
            lagging_sources = [
                f"{name} latest_utc={(latest.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(latest) else 'missing')}"
                for name, latest in source_latest_times.items()
                if pd.isna(latest) or latest < expected_latest
            ]
            if not lagging_sources:
                lagging_sources.append(
                    f"common_merge latest_utc={(pd.Timestamp(latest_available).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(latest_available) else 'missing')}"
                )

            reason_suffix = ""
            if notes:
                reason_suffix = f" | neden={notes[-1]}"

            raise LatestBarPendingError(
                "Son ortak mum icin veri eksik; sinyal uretilmedi | "
                f"hedef_mum_utc={expected_latest:%Y-%m-%d %H:%M:%S} | "
                f"ortak_son_mum_utc={(pd.Timestamp(latest_available).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(latest_available) else 'missing')} | "
                f"eksik_kaynaklar={', '.join(lagging_sources)}"
                f"{reason_suffix}",
                expected_latest=expected_latest,
                latest_available=(pd.Timestamp(latest_available) if pd.notna(latest_available) else None),
                lagging_sources=tuple(lagging_sources),
                notes=tuple(notes),
            )

        merged_30m["raw_30m_idx"] = np.arange(len(merged_30m), dtype=np.int64)
        merged_1d["daily_row_idx"] = np.arange(len(merged_1d), dtype=np.int64)
        merged_1d["available_time"] = merged_1d["Date"] + pd.to_timedelta(self.daily_feature_lag_days, unit="D")

        scaled_30m, scaled_1d = self._build_scaled_frames(merged_30m, merged_1d)
        barrier_width = self._compute_barrier_width(merged_30m)

        sample_df = pd.merge_asof(
            merged_30m[["time", "raw_30m_idx"]].sort_values("time"),
            merged_1d[["available_time", "daily_row_idx"]].sort_values("available_time"),
            left_on="time",
            right_on="available_time",
            direction="backward",
        )
        sample_df = sample_df.dropna(subset=["daily_row_idx"]).reset_index(drop=True)
        if sample_df.empty:
            raise ValueError("No inference samples have a completed daily anchor.")
        sample_df["daily_row_idx"] = sample_df["daily_row_idx"].astype(np.int64)

        for row in sample_df.iloc[::-1].itertuples(index=False):
            end_30m = int(row.raw_30m_idx)
            end_1d = int(row.daily_row_idx)

            if end_30m < self.window_size_30m - 1 or end_1d < self.window_size_1d - 1:
                continue

            window_30m = scaled_30m.iloc[end_30m - self.window_size_30m + 1 : end_30m + 1]
            window_1d = scaled_1d.iloc[end_1d - self.window_size_1d + 1 : end_1d + 1]
            if len(window_30m) != self.window_size_30m or len(window_1d) != self.window_size_1d:
                continue
            if not np.isfinite(window_30m.to_numpy(dtype=np.float32)).all():
                continue
            if not np.isfinite(window_1d.to_numpy(dtype=np.float32)).all():
                continue

            current_barrier_width = float(barrier_width.iloc[end_30m])
            if not np.isfinite(current_barrier_width) or current_barrier_width <= 0:
                continue

            x30 = torch.tensor(window_30m.to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            x1d = torch.tensor(window_1d.to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x30, x1d)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            return PredictionResult(
                bar_time=pd.Timestamp(row.time),
                close_price=float(pd.to_numeric(merged_30m.loc[end_30m, "binanceFutures_close"], errors="coerce")),
                barrier_width=current_barrier_width,
                probs={
                    "p_up": float(probs[0]),
                    "p_flat": float(probs[1]),
                    "p_down": float(probs[2]),
                },
                notes=tuple(notes),
            )

        raise ValueError("Could not find a valid latest inference window after live feature assembly.")
