from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import databento as db
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
MERGED_1D_PATH = PROJECT_ROOT / "data" / "processed" / "merged1d.csv"
COINBASE_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "coinbaseProcessed.csv"
BINANCE_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "binance_processed.csv"
BYBIT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_processed.csv"
CME_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cmeProcessed.csv"

THIRTY_MINUTES = pd.Timedelta(minutes=30)
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000


@dataclass(frozen=True)
class DecisionRule:
    max_flat_probability: float
    side_ratio_threshold: float

    def derive_trade_label(self, p_buy: float, p_hold: float, p_sell: float) -> str:
        flat_gate = p_hold < self.max_flat_probability
        long_mask = flat_gate and (p_buy > self.side_ratio_threshold * p_sell)
        short_mask = flat_gate and (not long_mask) and (p_sell > self.side_ratio_threshold * p_buy)

        if long_mask:
            return "up"
        if short_mask:
            return "down"
        return "flat"


@dataclass(frozen=True)
class PredictionResult:
    bar_time: pd.Timestamp
    close_price: float
    barrier_width: float
    probs: dict[str, float]
    notes: tuple[str, ...]


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


class RetryHttpClient:
    def __init__(self, timeout_seconds: float, max_retries: int, backoff_seconds: float, verify_ssl: bool = True) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.verify_ssl = verify_ssl
        self.session = requests.Session()

    def get_json(self, url: str, params: dict[str, object] | None = None) -> dict | list:
        attempts = self.max_retries + 1
        response: requests.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
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
                body = ""
                if response is not None:
                    body = response.text[:400].strip()
                raise RuntimeError(
                    f"HTTP GET failed | url={url} | status={(response.status_code if response is not None else 'unknown')} | body={body}"
                ) from exc

        raise RuntimeError(f"HTTP GET exhausted retries unexpectedly | url={url}")


class BinanceLiveDataClient:
    FUTURES_KLINES_PATH = "/fapi/v1/klines"
    FUTURES_FUNDING_PATH = "/fapi/v1/fundingRate"
    SPOT_KLINES_PATH = "/api/v3/klines"

    def __init__(self) -> None:
        self.futures_base_url = settings.BINANCE_FUTURES_PUBLIC_URL.rstrip("/")
        self.spot_base_url = settings.BINANCE_SPOT_PUBLIC_URL.rstrip("/")
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    def _fetch_klines(self, base_url: str, path: str, symbol: str, limit: int) -> pd.DataFrame:
        payload = self.http.get_json(
            f"{base_url}{path}",
            params={"symbol": symbol, "interval": "30m", "limit": int(limit)},
        )
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame()

        df = pd.DataFrame(
            payload,
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
        df = df[df["time"] <= _latest_closed_30m_open_time()].copy()
        df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        return df

    def _fetch_recent_funding(self, symbol: str, start_ms: int) -> pd.DataFrame:
        payload = self.http.get_json(
            f"{self.futures_base_url}{self.FUTURES_FUNDING_PATH}",
            params={"symbol": symbol, "startTime": int(start_ms), "limit": 1000},
        )
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame(columns=["funding_time", "funding_rate_8h"])

        df = pd.DataFrame(payload)
        df["funding_time"] = pd.to_numeric(df.get("fundingTime"), errors="coerce")
        df["funding_rate_8h"] = pd.to_numeric(df.get("fundingRate"), errors="coerce")
        df = df.dropna(subset=["funding_time"]).copy()
        df["funding_time"] = df["funding_time"].astype("int64")
        return df.sort_values("funding_time").drop_duplicates("funding_time", keep="last").reset_index(drop=True)

    def fetch_recent_processed(self, symbol: str, bars: int) -> pd.DataFrame:
        futures_df = self._fetch_klines(self.futures_base_url, self.FUTURES_KLINES_PATH, symbol=symbol, limit=bars)
        spot_df = self._fetch_klines(self.spot_base_url, self.SPOT_KLINES_PATH, symbol=symbol, limit=bars)
        if futures_df.empty or spot_df.empty:
            return pd.DataFrame()

        funding_start_ms = int(futures_df["open_time"].min()) - EIGHT_HOURS_MS
        funding_df = self._fetch_recent_funding(symbol=symbol, start_ms=funding_start_ms)
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

        processed = build_binance_processed_output(futures_df=futures_df, spot_df=spot_df)
        return _standardize_source_frame(processed, time_column="time")


class CoinbaseLiveDataClient:
    def __init__(self) -> None:
        self.base_url = settings.COINBASE_PUBLIC_URL.rstrip("/")
        self.http = RetryHttpClient(
            timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.PUBLIC_API_MAX_RETRIES,
            backoff_seconds=settings.PUBLIC_API_RETRY_BACKOFF_SECONDS,
        )

    def fetch_recent_processed(self, product_id: str, fifteen_minute_bars: int) -> pd.DataFrame:
        max_batch = 300
        remaining = int(max(fifteen_minute_bars, 2))
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        rows: list[list[float]] = []

        while remaining > 0:
            batch_size = min(max_batch, remaining)
            start = end - pd.Timedelta(minutes=15 * batch_size)
            payload = self.http.get_json(
                f"{self.base_url}/products/{product_id}/candles",
                params={
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                    "granularity": 900,
                },
            )
            if isinstance(payload, list) and payload:
                rows.extend(payload)
            end = start
            remaining -= batch_size

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["epoch_seconds", "low", "high", "open", "close", "volume"])
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["epoch_seconds"]).copy()
        df["epoch_seconds"] = df["epoch_seconds"].astype("int64")
        df["time"] = pd.to_datetime(df["epoch_seconds"], unit="s", utc=True).dt.tz_convert(None)
        df = df[df["time"] <= _latest_closed_30m_open_time() + THIRTY_MINUTES].copy()
        df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        resampled, _ = coinbase_resample_to_30m(df)
        processed = build_coinbase_processed_output(resampled)
        return _standardize_source_frame(processed, time_column="coinbase_time")


class BybitLiveDataClient:
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

    def fetch_recent_processed(self, symbol: str, bars: int, base_df: pd.DataFrame) -> pd.DataFrame:
        kline_payload = self.http.get_json(
            f"{self.base_url}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": "30",
                "limit": int(bars),
            },
        )
        kline_rows = self._response_list(kline_payload if isinstance(kline_payload, dict) else {})
        if not kline_rows:
            return pd.DataFrame()

        kline_df = pd.DataFrame(
            kline_rows,
            columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
        )
        for column in ["open_time", "open", "high", "low", "close", "volume", "turnover"]:
            kline_df[column] = pd.to_numeric(kline_df[column], errors="coerce")
        kline_df = kline_df.dropna(subset=["open_time"]).copy()
        kline_df["open_time"] = kline_df["open_time"].astype("int64")
        kline_df["time"] = pd.to_datetime(kline_df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        kline_df = kline_df[kline_df["time"] <= _latest_closed_30m_open_time()].copy()
        if kline_df.empty:
            return pd.DataFrame()

        funding_payload = self.http.get_json(
            f"{self.base_url}/v5/market/funding/history",
            params={
                "category": "linear",
                "symbol": symbol,
                "limit": 200,
            },
        )
        funding_rows = self._response_list(funding_payload if isinstance(funding_payload, dict) else {})
        funding_df = pd.DataFrame(funding_rows)
        if not funding_df.empty:
            funding_df["funding_time"] = pd.to_numeric(funding_df.get("fundingRateTimestamp"), errors="coerce")
            funding_df["funding_rate"] = pd.to_numeric(funding_df.get("fundingRate"), errors="coerce")
            funding_df = funding_df.dropna(subset=["funding_time"]).copy()
            funding_df["funding_time"] = funding_df["funding_time"].astype("int64")
            funding_df = funding_df.sort_values("funding_time").drop_duplicates("funding_time", keep="last")
            kline_df = pd.merge_asof(
                kline_df.sort_values("open_time"),
                funding_df[["funding_time", "funding_rate"]].sort_values("funding_time"),
                left_on="open_time",
                right_on="funding_time",
                direction="backward",
            )
        else:
            kline_df["funding_rate"] = np.nan

        kline_df["quote_volume"] = pd.to_numeric(kline_df["turnover"], errors="coerce")
        kline_df["count"] = 0
        kline_df["delta"] = 0.0

        if not base_df.empty:
            base_metrics = base_df[["time", "bybit_count", "bybit_delta"]].rename(
                columns={"bybit_count": "count", "bybit_delta": "delta"}
            )
            kline_df = kline_df.merge(base_metrics, on="time", how="left", suffixes=("", "_base"))
            kline_df["count"] = kline_df["count_base"].fillna(kline_df["count"]).round().astype("int64")
            kline_df["delta"] = kline_df["delta_base"].fillna(kline_df["delta"]).astype("float64")
            kline_df = kline_df.drop(columns=[column for column in ["count_base", "delta_base"] if column in kline_df.columns])

        kline_df["cvd"] = pd.to_numeric(kline_df["delta"], errors="coerce").fillna(0.0).cumsum()
        base_for_cvd = base_df.rename(columns={"bybit_time": "time"}) if "bybit_time" in base_df.columns else base_df
        kline_df = _align_cvd_with_base(base_for_cvd, kline_df, "cvd")
        kline_df = add_bybit_technical_indicators(kline_df)

        output = kline_df.drop(
            columns=[column for column in ["turnover", "volume", "funding_time"] if column in kline_df.columns]
        ).copy()
        output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        prefixed = output.rename(columns={column: f"bybit_{column}" for column in output.columns})
        ordered = ["bybit_time"] + [column for column in prefixed.columns if column != "bybit_time"]
        return _standardize_source_frame(prefixed[ordered], time_column="bybit_time")


class DatabentoCmeClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or settings.DATABENTO_API_KEY).strip()
        self.dataset = settings.DATABENTO_CME_DATASET
        self.symbol = settings.DATABENTO_CME_SYMBOL
        self.stype_in = settings.DATABENTO_CME_STYPE_IN
        self.lookback_hours = max(float(settings.DATABENTO_CME_LOOKBACK_HOURS), 1.0)
        self.client = db.Historical(self.api_key) if self.api_key else None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.client is not None)

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

    def _get_range(self, schema: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        store = self.client.timeseries.get_range(
            dataset=self.dataset,
            start=start,
            end=end,
            symbols=[self.symbol],
            schema=schema,
            stype_in=self.stype_in,
        )
        return store.to_df(schema=schema, pretty_ts=True, map_symbols=False)

    def _fetch_ohlcv_30m(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ohlcv_df = self._ensure_ts_event_column(self._get_range("ohlcv-1m", start=start, end=end))
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

    def _fetch_trade_metrics_30m(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        trades_df = self._ensure_ts_event_column(self._get_range("trades", start=start, end=end))
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

    def fetch_recent_processed(self, base_df: pd.DataFrame) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()

        end = pd.Timestamp.now(tz="UTC").ceil("1min")
        start = end - pd.Timedelta(hours=self.lookback_hours)
        latest_closed = _latest_closed_30m_open_time(end)

        bars_df = self._fetch_ohlcv_30m(start=start, end=end)
        if bars_df.empty:
            return pd.DataFrame()

        trades_df = self._fetch_trade_metrics_30m(start=start, end=end)
        cme_df = bars_df.merge(trades_df, on="time", how="left")
        cme_df["count"] = pd.to_numeric(cme_df["count"], errors="coerce").fillna(0).round().astype("int64")
        cme_df["delta"] = pd.to_numeric(cme_df["delta"], errors="coerce").fillna(0).round().astype("int64")
        cme_df = cme_df[cme_df["time"] <= latest_closed].copy()
        if cme_df.empty:
            return pd.DataFrame()

        cme_df = cme_df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        cme_df["open_time"] = cme_df["time"].map(lambda ts: int(pd.Timestamp(ts).timestamp())).astype("int64")

        if not base_df.empty:
            base_metrics = base_df[["time", "CME_count", "CME_delta"]].rename(
                columns={"CME_count": "count", "CME_delta": "delta"}
            )
            cme_df = cme_df.merge(base_metrics, on="time", how="left", suffixes=("", "_base"))
            cme_df["count"] = cme_df["count_base"].fillna(cme_df["count"]).round().astype("int64")
            cme_df["delta"] = cme_df["delta_base"].fillna(cme_df["delta"]).round().astype("int64")
            cme_df = cme_df.drop(columns=[column for column in ["count_base", "delta_base"] if column in cme_df.columns])

        cme_df["cvd"] = pd.to_numeric(cme_df["delta"], errors="coerce").fillna(0).cumsum().astype("int64")
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


class ModelSignalEngine:
    def __init__(
        self,
        databento_api_key: str | None = None,
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
        self.decision_rule = DecisionRule(
            max_flat_probability=float(
                self.config.get("trade_rule_max_flat_prob", settings.TRADE_SIGNAL_MAX_FLAT_PROBABILITY)
            ),
            side_ratio_threshold=float(
                self.config.get("trade_rule_side_ratio", settings.TRADE_SIGNAL_SIDE_RATIO_THRESHOLD)
            ),
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

        self.coinbase_base = _standardize_source_frame(pd.read_csv(COINBASE_PROCESSED_PATH), time_column="coinbase_time")
        self.binance_base = _standardize_source_frame(pd.read_csv(BINANCE_PROCESSED_PATH), time_column="time")
        self.bybit_base = _standardize_source_frame(pd.read_csv(BYBIT_PROCESSED_PATH), time_column="bybit_time")
        self.cme_base = _standardize_source_frame(pd.read_csv(CME_PROCESSED_PATH), time_column="CME_time")

        self.daily_base = pd.read_csv(MERGED_1D_PATH)
        self.daily_base["Date"] = pd.to_datetime(self.daily_base["Date"], errors="coerce").dt.normalize()
        self.daily_base = (
            self.daily_base.dropna(subset=["Date"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

        self.binance_client = BinanceLiveDataClient()
        self.coinbase_client = CoinbaseLiveDataClient()
        self.bybit_client = BybitLiveDataClient()
        self.databento_client = DatabentoCmeClient(api_key=databento_api_key)

    def _log_note(self, note: str, notes: list[str]) -> None:
        notes.append(note)
        self.logger(note)

    def _build_updated_sources(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        notes: list[str] = []
        recent_30m_bars = max(int(settings.LIVE_RECENT_30M_BARS), self.window_size_30m, 96)
        recent_coinbase_15m = max(int(settings.COINBASE_RECENT_15M_BARS), recent_30m_bars * 2, 200)

        coinbase_df = self.coinbase_base.copy()
        try:
            recent_coinbase = self.coinbase_client.fetch_recent_processed("BTC-USD", fifteen_minute_bars=recent_coinbase_15m)
            if recent_coinbase.empty:
                self._log_note("Coinbase live update returned no closed bars; using processed history.", notes)
            else:
                coinbase_df = _overlay_processed_frames(self.coinbase_base, recent_coinbase)
        except Exception as exc:
            self._log_note(f"Coinbase live update failed; using processed history. reason={exc}", notes)

        binance_df = self.binance_base.copy()
        try:
            recent_binance = self.binance_client.fetch_recent_processed(settings.SYMBOL, bars=recent_30m_bars)
            if recent_binance.empty:
                self._log_note("Binance live update returned no closed bars; using processed history.", notes)
            else:
                binance_df = _overlay_processed_frames(self.binance_base, recent_binance)
        except Exception as exc:
            self._log_note(f"Binance live update failed; using processed history. reason={exc}", notes)

        bybit_df = self.bybit_base.copy()
        try:
            recent_bybit = self.bybit_client.fetch_recent_processed(
                "BTCUSDT",
                bars=recent_30m_bars,
                base_df=self.bybit_base,
            )
            if recent_bybit.empty:
                self._log_note("Bybit live update returned no closed bars; using processed history.", notes)
            else:
                bybit_df = _overlay_processed_frames(self.bybit_base, recent_bybit)
        except Exception as exc:
            self._log_note(f"Bybit live update failed; using processed history. reason={exc}", notes)

        cme_df = self.cme_base.copy()
        try:
            recent_cme = self.databento_client.fetch_recent_processed(base_df=self.cme_base)
            if recent_cme.empty:
                if self.databento_client.configured:
                    self._log_note("Databento CME update returned no closed bars; using processed history.", notes)
                else:
                    self._log_note("Databento API key is not configured; using processed CME history.", notes)
            else:
                cme_df = _overlay_processed_frames(self.cme_base, recent_cme)
        except Exception as exc:
            self._log_note(f"Databento CME update failed; using processed history. reason={exc}", notes)

        merged_30m = coinbase_df.merge(binance_df, on="time", how="inner", validate="one_to_one")
        merged_30m = merged_30m.merge(bybit_df, on="time", how="inner", validate="one_to_one")
        merged_30m = merged_30m.merge(cme_df, on="time", how="inner", validate="one_to_one")
        merged_30m = _drop_helper_open_time_columns(merged_30m).sort_values("time").reset_index(drop=True)

        if merged_30m.empty:
            raise ValueError("No overlapping 30m rows remain after merging live and processed sources.")

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
