from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import bot_settings as settings
from finalPreProcess import ATR_PERIOD, compute_effective_barrier_multiplier, compute_past_only_atr
from preProcessx import load_and_fill_fed, load_fear_greed
from preprocess3 import add_technical_indicators


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_encoder_model.pt"
FINAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final.csv"
MERGED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_data.csv"
FEAR_GREED_PATH = PROJECT_ROOT / "data" / "raw" / "fear-greed" / "alternative_fear_greed_1d.csv"
FED_LIQUIDITY_PATH = PROJECT_ROOT / "data" / "raw" / "netLiq" / "fed_net_liquidity_1d.csv"
DEFAULT_STATE_FILE = PROJECT_ROOT / "runtime" / "last_processed_bar.json"
ORDER_SIDE_BUY = "BUY"
ORDER_SIDE_SELL = "SELL"
POSITION_LONG = "LONG"
POSITION_SHORT = "SHORT"
POSITION_FLAT = "FLAT"
RISK_HIGH = "high"
RISK_LOW = "low"
MAX_KLINE_LIMIT = 1500
MAX_FUNDING_LIMIT = 1000
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000


class EncoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        head_hidden_dim: int,
        max_len: int = 4096,
        output_dim: int = 3,
    ) -> None:
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
        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)

        h_mean = self.token_norm(h.mean(dim=1))
        h_last = self.token_norm(h[:, -1, :])
        h_repr = torch.cat([h_mean, h_last], dim=-1)
        h_repr = self.head_norm(h_repr)
        return self.head(h_repr)


@dataclass(frozen=True)
class RiskProfile:
    name: str
    long_sell_threshold: float
    short_buy_threshold: float
    flat_keeps_position: bool
    reverse_reopens_position: bool
    same_direction_scale_multiplier: float
    same_direction_grows_position: bool

    def derive_trade_label(self, p_buy: float, p_hold: float, p_sell: float) -> str:
        long_mask = (p_sell < self.long_sell_threshold) and (p_buy > p_hold)
        short_mask = (not long_mask) and (p_buy < self.short_buy_threshold) and (p_sell > p_hold)
        if long_mask:
            return "up"
        if short_mask:
            return "down"
        return "flat"


RISK_PROFILES = {
    RISK_HIGH: RiskProfile(
        name=RISK_HIGH,
        long_sell_threshold=0.32,
        short_buy_threshold=0.32,
        flat_keeps_position=True,
        reverse_reopens_position=True,
        same_direction_scale_multiplier=1.5,
        same_direction_grows_position=True,
    ),
    RISK_LOW: RiskProfile(
        name=RISK_LOW,
        long_sell_threshold=0.30,
        short_buy_threshold=0.30,
        flat_keeps_position=False,
        reverse_reopens_position=False,
        same_direction_scale_multiplier=1.0,
        same_direction_grows_position=False,
    ),
}


@dataclass
class SymbolRules:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal
    quantity_precision: int
    price_precision: int


@dataclass
class PositionState:
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    notional: float

    @property
    def is_open(self) -> bool:
        return self.side != POSITION_FLAT and self.quantity > 0.0


@dataclass
class AccountBalanceState:
    asset: str
    wallet_balance: float
    available_balance: float
    cross_unrealized_pnl: float
    equity: float


@dataclass
class SignalSnapshot:
    bar_time: pd.Timestamp
    close_price: float
    barrier_width: float
    probs: dict[str, float]
    trade_label: str


def round_down(value: float, quantum: Decimal) -> float:
    if quantum <= 0:
        return value
    dec_value = Decimal(str(value))
    rounded = dec_value.quantize(quantum, rounding=ROUND_DOWN)
    return float(rounded)


def round_price(value: float, quantum: Decimal) -> float:
    return round_down(value, quantum)


def now_ms() -> int:
    return int(time.time() * 1000)


class HybridFeatureNormalizer:
    def __init__(self, final_data_path: Path, config: dict) -> None:
        self.final_data_path = final_data_path
        self.config = config
        self.feature_columns = list(config["feature_columns"])
        self.target_columns = list(config["target_columns"])
        self.standard_scale_cols = list(config["standard_scale_cols"])
        self.log_return_cols = list(config["price_log_return_cols"])
        self.rolling_mean_cols = list(config["rolling_mean_cols"])
        self.atr_relative_cols = list(config["atr_relative_cols"])
        self.atr_base_col = config.get("atr_base_col")
        self.rolling_window = int(config["rolling_window"])
        self.window_size = int(config["window_size"])
        self.scaler = self._fit_scaler_from_training_data()

    def _fit_scaler_from_training_data(self) -> StandardScaler:
        df = pd.read_csv(self.final_data_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).reset_index(drop=True)

        y_soft_df = df[self.target_columns].copy()
        X_df = df[self.feature_columns].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)

        valid_rows = ~X_df.isna().any(axis=1) & ~y_soft_df.isna().any(axis=1)
        df = df.loc[valid_rows].reset_index(drop=True)
        X_df = X_df.loc[valid_rows].reset_index(drop=True)
        y_soft_df = y_soft_df.loc[valid_rows].reset_index(drop=True)

        n = len(df)
        train_end = int(n * 0.80)
        val_end = train_end + int(n * 0.10)

        train_X_raw = X_df.iloc[:train_end].copy()
        val_X_raw = X_df.iloc[train_end:val_end].copy()
        test_X_raw = X_df.iloc[val_end:].copy()

        train_y_soft = y_soft_df.iloc[:train_end].copy()
        val_y_soft = y_soft_df.iloc[train_end:val_end].copy()
        test_y_soft = y_soft_df.iloc[val_end:].copy()

        split_labels = np.array(
            ["train"] * len(train_X_raw) + ["val"] * len(val_X_raw) + ["test"] * len(test_X_raw),
            dtype=object,
        )
        all_X_raw = pd.concat([train_X_raw, val_X_raw, test_X_raw], axis=0, ignore_index=True)
        all_y_soft = pd.concat([train_y_soft, val_y_soft, test_y_soft], axis=0, ignore_index=True)
        transformed = self._apply_hybrid_steps(all_X_raw)

        valid_rows = ~transformed.isna().any(axis=1) & ~all_y_soft.isna().any(axis=1)
        transformed = transformed.loc[valid_rows].reset_index(drop=True)
        split_labels = split_labels[valid_rows.to_numpy()]
        train_mask = split_labels == "train"

        train_X_df = transformed.loc[train_mask].reset_index(drop=True)
        scaler = StandardScaler()
        scaler.fit(train_X_df[self.standard_scale_cols])
        return scaler

    def _apply_hybrid_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.atr_base_col and self.atr_relative_cols:
            atr_base = out[self.atr_base_col].replace(0, np.nan)
            for col in self.atr_relative_cols:
                out[col] = out[col] / atr_base

        for col in self.log_return_cols:
            out[col] = np.log(out[col] / out[col].shift(1))

        for col in self.rolling_mean_cols:
            trailing_mean = out[col].shift(1).rolling(window=self.rolling_window, min_periods=1).mean()
            out[col] = out[col] / trailing_mean

        return out.replace([np.inf, -np.inf], np.nan)

    def transform_live_frame(self, raw_feature_df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns if col not in raw_feature_df.columns]
        if missing:
            raise KeyError(f"Live feature frame is missing columns: {missing}")

        work_df = raw_feature_df.copy()
        transformed = self._apply_hybrid_steps(work_df[self.feature_columns])
        valid_rows = ~transformed.isna().any(axis=1)
        transformed = transformed.loc[valid_rows].copy()

        transformed.loc[:, self.standard_scale_cols] = self.scaler.transform(
            transformed[self.standard_scale_cols]
        )

        output = pd.concat(
            [
                work_df.loc[valid_rows, ["Date"]].reset_index(drop=True),
                transformed.reset_index(drop=True),
            ],
            axis=1,
        )
        return output


class ModelSignalEngine:
    def __init__(self) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        self.config = checkpoint["config"]
        self.normalizer = HybridFeatureNormalizer(FINAL_DATA_PATH, self.config)
        self.model = EncoderOnlyTransformer(
            input_dim=len(self.config["feature_columns"]),
            d_model=int(self.config["d_model"]),
            nhead=int(self.config["nhead"]),
            num_layers=int(self.config["num_layers"]),
            dim_feedforward=int(self.config["dim_feedforward"]),
            dropout=float(self.config["dropout"]),
            head_hidden_dim=int(self.config["head_hidden_dim"]),
            output_dim=int(self.config["output_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_latest(self, raw_feature_frame: pd.DataFrame) -> SignalSnapshot:
        normalized = self.normalizer.transform_live_frame(raw_feature_frame)
        if len(normalized) < self.normalizer.window_size:
            raise ValueError(
                f"Need at least {self.normalizer.window_size} valid rows for inference, got {len(normalized)}."
            )

        window = normalized[self.config["feature_columns"]].tail(self.normalizer.window_size)
        x = torch.tensor(window.to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        latest_normalized = normalized.iloc[-1]
        latest_raw = raw_feature_frame.loc[raw_feature_frame["Date"] == latest_normalized["Date"]].iloc[-1]
        bar_time = pd.to_datetime(latest_raw["Date"], errors="coerce")
        if pd.isna(bar_time):
            raise ValueError("Latest raw feature row has invalid Date.")

        return SignalSnapshot(
            bar_time=bar_time,
            close_price=float(latest_raw["close"]),
            barrier_width=float(latest_raw["barrier_width"]),
            probs={
                "p_up": float(probs[0]),
                "p_flat": float(probs[1]),
                "p_down": float(probs[2]),
            },
            trade_label="flat",
        )


class BinancePublicDataClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_server_time(self) -> int:
        payload = self._get("/fapi/v1/time")
        return int(payload["serverTime"])

    def get_exchange_info(self, symbol: str) -> SymbolRules:
        payload = self._get("/fapi/v1/exchangeInfo")
        for item in payload.get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            price_filter = next(f for f in item["filters"] if f["filterType"] == "PRICE_FILTER")
            lot_filter = next(f for f in item["filters"] if f["filterType"] == "LOT_SIZE")
            min_notional_filter = next(f for f in item["filters"] if f["filterType"] == "MIN_NOTIONAL")
            return SymbolRules(
                symbol=symbol,
                tick_size=Decimal(price_filter["tickSize"]),
                step_size=Decimal(lot_filter["stepSize"]),
                min_qty=Decimal(lot_filter["minQty"]),
                min_notional=Decimal(min_notional_filter["notional"]),
                quantity_precision=int(item["quantityPrecision"]),
                price_precision=int(item["pricePrecision"]),
            )

        raise ValueError(f"Symbol {symbol} not found in exchangeInfo.")

    def fetch_recent_klines(self, symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
        remaining = total_limit
        end_time: int | None = None
        rows: list[list] = []
        server_time = self.get_server_time()

        while remaining > 0:
            limit = min(MAX_KLINE_LIMIT, remaining)
            params: dict[str, int | str] = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_time is not None:
                params["endTime"] = end_time
            batch = self._get("/fapi/v1/klines", params=params)
            if not batch:
                break

            rows = batch + rows
            earliest_open = int(batch[0][0])
            end_time = earliest_open - 1
            if len(batch) < limit:
                break
            remaining -= len(batch)

        columns = [
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
        ]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            raise ValueError("No kline data returned from Binance.")

        df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
        numeric_cols = [c for c in columns if c not in {"ignore"}]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["close_time"] < server_time].reset_index(drop=True)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
        return df

    def fetch_recent_funding_rates(self, symbol: str, start_time: int) -> pd.DataFrame:
        rows: list[dict] = []
        cursor = start_time

        while True:
            params = {
                "symbol": symbol,
                "limit": MAX_FUNDING_LIMIT,
                "startTime": cursor,
            }
            batch = self._get("/fapi/v1/fundingRate", params=params)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < MAX_FUNDING_LIMIT:
                break
            cursor = int(batch[-1]["fundingTime"]) + 1

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No funding rate data returned from Binance.")

        df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce")
        df["lastFundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.dropna(subset=["fundingTime", "lastFundingRate"]).copy()
        df["funding_ts"] = (df["fundingTime"] // EIGHT_HOURS_MS) * EIGHT_HOURS_MS
        df = df.sort_values(["funding_ts", "fundingTime"]).drop_duplicates("funding_ts", keep="last")
        return df[["funding_ts", "lastFundingRate"]].rename(columns={"lastFundingRate": "funding_rate_8h"})


class DailyFeatureProvider:
    def __init__(self) -> None:
        self.daily_df = self._build_daily_feature_frame()

    def _build_daily_feature_frame(self) -> pd.DataFrame:
        merged_daily = pd.read_csv(MERGED_DATA_PATH)
        merged_daily["Date"] = pd.to_datetime(merged_daily["Date"], errors="coerce")
        merged_daily = merged_daily.dropna(subset=["Date"]).copy()

        fed_df = load_and_fill_fed(FED_LIQUIDITY_PATH)
        fear_greed_df = load_fear_greed(FEAR_GREED_PATH)
        mergedx_df = fed_df.merge(fear_greed_df, on="Date", how="outer")
        mergedx_df = mergedx_df.sort_values("Date")
        mergedx_df["fed_net_liquidity"] = pd.to_numeric(mergedx_df["fed_net_liquidity"], errors="coerce")
        mergedx_df["fearGreed"] = pd.to_numeric(mergedx_df["fearGreed"], errors="coerce")
        mergedx_df["fed_net_liquidity"] = mergedx_df["fed_net_liquidity"].ffill().bfill()
        mergedx_df["fearGreed"] = mergedx_df["fearGreed"].ffill().bfill()

        daily_df = merged_daily.merge(mergedx_df[["Date", "fed_net_liquidity", "fearGreed"]], on="Date", how="left")
        daily_df = daily_df.sort_values("Date").reset_index(drop=True)

        rename_map = {
            "Open": "CME_Open",
            "High": "CME_High",
            "Low": "CME_Low",
            "Close": "CME_Close",
            "Volume": "CME_Volume",
        }
        daily_df = daily_df.rename(columns=rename_map)
        full_dates = pd.date_range(
            start=daily_df["Date"].min(),
            end=max(daily_df["Date"].max(), mergedx_df["Date"].max()),
            freq="D",
        )
        daily_df = daily_df.set_index("Date").reindex(full_dates).rename_axis("Date").reset_index()
        slow_market_cols = [
            "CME_Open",
            "CME_High",
            "CME_Low",
            "CME_Close",
            "CME_Volume",
            "oi_open",
            "oi_high",
            "oi_low",
            "oi_close",
            "long_ratio",
            "short_ratio",
        ]
        macro_cols = ["fed_net_liquidity", "fearGreed"]
        daily_df[slow_market_cols] = daily_df[slow_market_cols].ffill().bfill()
        daily_df[macro_cols] = daily_df[macro_cols].ffill().bfill()
        daily_df["merge_date"] = daily_df["Date"].dt.normalize() + pd.Timedelta(days=1)
        return daily_df

    def attach_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = df.copy()
        merged["merge_date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.normalize()

        feature_cols = [
            "merge_date",
            "CME_Open",
            "CME_High",
            "CME_Low",
            "CME_Close",
            "CME_Volume",
            "oi_open",
            "oi_high",
            "oi_low",
            "oi_close",
            "long_ratio",
            "short_ratio",
            "fed_net_liquidity",
            "fearGreed",
        ]
        merged = merged.merge(self.daily_df[feature_cols], on="merge_date", how="left")

        daily_fill_cols = [c for c in feature_cols if c != "merge_date"]
        merged[daily_fill_cols] = merged[daily_fill_cols].ffill().bfill()
        merged = merged.drop(columns=["merge_date"])
        return merged


class LiveFeatureAssembler:
    def __init__(self, public_client: BinancePublicDataClient, history_bars: int) -> None:
        self.public_client = public_client
        self.history_bars = history_bars
        self.daily_provider = DailyFeatureProvider()

    def build_live_feature_frame(self, symbol: str) -> pd.DataFrame:
        klines_df = self.public_client.fetch_recent_klines(symbol=symbol, interval="30m", total_limit=self.history_bars)
        funding_df = self.public_client.fetch_recent_funding_rates(
            symbol=symbol,
            start_time=int(klines_df["open_time"].min()) - EIGHT_HOURS_MS,
        )

        merged = pd.merge_asof(
            klines_df.sort_values("open_time"),
            funding_df.sort_values("funding_ts"),
            left_on="open_time",
            right_on="funding_ts",
            direction="backward",
        )

        merged = add_technical_indicators(merged)
        merged["Date"] = merged["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        merged = self.daily_provider.attach_daily_features(merged)
        merged = self._add_live_barrier_width(merged)

        drop_cols = ["time", "open_time", "close_time", "funding_ts", "ignore"]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
        merged = merged.sort_values("Date").reset_index(drop=True)
        return merged

    def _add_live_barrier_width(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        timestamps = pd.to_datetime(out["Date"], errors="coerce", utc=True)
        for col in ["high", "low", "close"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        _, atr = compute_past_only_atr(
            high=out["high"],
            low=out["low"],
            close=out["close"],
            period=ATR_PERIOD,
        )
        effective_k, _ = compute_effective_barrier_multiplier(
            atr=atr,
            timestamps=timestamps,
        )
        out["barrier_width"] = effective_k * atr
        return out


class BinanceDemoClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, str | int | float | bool] | None = None,
    ) -> dict | list:
        payload = dict(params or {})
        payload["timestamp"] = now_ms()
        payload["recvWindow"] = settings.RECV_WINDOW_MS
        query = urlencode(payload, doseq=True)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        url = f"{self.base_url}{path}?{query}&signature={signature}"
        response = self.session.request(method=method.upper(), url=url, timeout=20)
        response.raise_for_status()
        return response.json()

    def ensure_one_way_mode(self) -> None:
        payload = self._signed_request("GET", "/fapi/v1/positionSide/dual")
        dual_side = str(payload.get("dualSidePosition", "false")).lower() == "true"
        if not dual_side:
            return
        self._signed_request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"})

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def get_open_orders(self, symbol: str) -> list[dict]:
        payload = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})
        return list(payload)

    def cancel_order(self, symbol: str, order_id: int) -> None:
        self._signed_request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def cancel_bot_orders(self, symbol: str) -> int:
        orders = self.get_open_orders(symbol)
        cancelled = 0
        for order in orders:
            client_order_id = str(order.get("clientOrderId", ""))
            if not client_order_id.startswith(settings.ORDER_CLIENT_PREFIX):
                continue
            self.cancel_order(symbol, int(order["orderId"]))
            cancelled += 1
        return cancelled

    def get_position(self, symbol: str) -> PositionState:
        payload = self._signed_request("GET", "/fapi/v3/positionRisk", {"symbol": symbol})
        if not payload:
            return PositionState(POSITION_FLAT, 0.0, 0.0, 0.0, 0.0)

        entry = next((item for item in payload if item.get("positionSide") in {"BOTH", POSITION_LONG, POSITION_SHORT}), payload[0])
        position_amt = float(entry.get("positionAmt", 0.0))
        if math.isclose(position_amt, 0.0, abs_tol=1e-12):
            return PositionState(
                side=POSITION_FLAT,
                quantity=0.0,
                entry_price=float(entry.get("entryPrice", 0.0)),
                mark_price=float(entry.get("markPrice", 0.0)),
                notional=0.0,
            )

        side = POSITION_LONG if position_amt > 0 else POSITION_SHORT
        return PositionState(
            side=side,
            quantity=abs(position_amt),
            entry_price=float(entry.get("entryPrice", 0.0)),
            mark_price=float(entry.get("markPrice", 0.0)),
            notional=abs(float(entry.get("notional", 0.0))),
        )

    def get_account_balance(self, asset: str = "USDT") -> AccountBalanceState:
        payload = self._signed_request("GET", "/fapi/v3/balance")
        entry = next((item for item in payload if item.get("asset") == asset), None)
        if entry is None:
            raise ValueError(f"{asset} bakiyesi bulunamadi.")

        wallet_balance = float(entry.get("balance", 0.0))
        available_balance = float(entry.get("availableBalance", wallet_balance))
        cross_unrealized_pnl = float(entry.get("crossUnPnl", 0.0))
        equity = wallet_balance + cross_unrealized_pnl
        return AccountBalanceState(
            asset=asset,
            wallet_balance=wallet_balance,
            available_balance=available_balance,
            cross_unrealized_pnl=cross_unrealized_pnl,
            equity=equity,
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool,
        client_order_id: str,
    ) -> dict:
        params: dict[str, str | int | float] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
            "newClientOrderId": client_order_id,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._signed_request("POST", "/fapi/v1/order", params)

    def place_protective_orders(
        self,
        symbol: str,
        position_side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> None:
        close_side = ORDER_SIDE_SELL if position_side == POSITION_LONG else ORDER_SIDE_BUY
        timestamp_suffix = str(now_ms())

        tp_params = {
            "symbol": symbol,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit_price,
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "newClientOrderId": f"{settings.ORDER_CLIENT_PREFIX}_tp_{timestamp_suffix}",
        }
        sl_params = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": stop_loss_price,
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "newClientOrderId": f"{settings.ORDER_CLIENT_PREFIX}_sl_{timestamp_suffix}",
        }

        self._signed_request("POST", "/fapi/v1/order", tp_params)
        self._signed_request("POST", "/fapi/v1/order", sl_params)


class TradeBotController:
    def __init__(
        self,
        risk_name: str,
        logger: Callable[[str], None],
        api_key: str | None = None,
        api_secret: str | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.risk_profile = RISK_PROFILES[risk_name]
        self.logger = logger
        self.public_client = BinancePublicDataClient(settings.PUBLIC_MARKET_DATA_URL)
        self.demo_client = BinanceDemoClient(
            api_key=api_key or settings.BINANCE_DEMO_API_KEY,
            api_secret=api_secret or settings.BINANCE_DEMO_API_SECRET,
            base_url=settings.BINANCE_DEMO_BASE_URL,
        )
        self.signal_engine = ModelSignalEngine()
        self.feature_assembler = LiveFeatureAssembler(
            public_client=self.public_client,
            history_bars=max(
                settings.KLINE_HISTORY_BARS,
                self.signal_engine.normalizer.rolling_window + self.signal_engine.normalizer.window_size + 64,
            ),
        )
        self.symbol_rules = self.public_client.get_exchange_info(settings.SYMBOL)
        self.stop_event = threading.Event()
        self.last_processed_bar: pd.Timestamp | None = None
        self.state_file = state_file or DEFAULT_STATE_FILE
        self.last_processed_bar = self._load_last_processed_bar()

    def validate_credentials(self) -> None:
        if not self.demo_client.api_key or not self.demo_client.api_secret:
            raise ValueError("Binance demo API key ve secret eksik. UI'dan gir veya `bot_settings.py` dosyasini doldur.")

    def start(self) -> None:
        self.validate_credentials()
        self.demo_client.ensure_one_way_mode()
        self.demo_client.set_leverage(settings.SYMBOL, settings.LEVERAGE)
        self.logger(
            f"Bot basladi | risk={self.risk_profile.name} | symbol={settings.SYMBOL} | "
            f"risk_per_trade={settings.ACCOUNT_RISK_PER_TRADE * 100:.2f}% | "
            f"sl_factor={settings.STOP_LOSS_FACTOR:.2f} | leverage={settings.LEVERAGE}x"
        )
        self.run_loop()

    def run_once(self) -> None:
        self.validate_credentials()
        self.demo_client.ensure_one_way_mode()
        self.demo_client.set_leverage(settings.SYMBOL, settings.LEVERAGE)
        self.logger(
            f"Tek seferlik calisma basladi | risk={self.risk_profile.name} | symbol={settings.SYMBOL} | "
            f"risk_per_trade={settings.ACCOUNT_RISK_PER_TRADE * 100:.2f}% | "
            f"sl_factor={settings.STOP_LOSS_FACTOR:.2f} | leverage={settings.LEVERAGE}x"
        )
        self.housekeep_orders()
        self.process_if_new_bar()

    def stop(self) -> None:
        self.stop_event.set()

    def run_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.housekeep_orders()
                self.process_if_new_bar()
            except Exception as exc:
                self.logger(f"Hata: {exc}")
            self.stop_event.wait(settings.POLL_INTERVAL_SECONDS)

    def housekeep_orders(self) -> None:
        position = self.demo_client.get_position(settings.SYMBOL)
        open_orders = self.demo_client.get_open_orders(settings.SYMBOL)
        bot_orders = [o for o in open_orders if str(o.get("clientOrderId", "")).startswith(settings.ORDER_CLIENT_PREFIX)]
        if not position.is_open and bot_orders:
            for order in bot_orders:
                self.demo_client.cancel_order(settings.SYMBOL, int(order["orderId"]))
            self.logger(f"Pozisyon yokken kalan {len(bot_orders)} bot emri iptal edildi.")

    def process_if_new_bar(self) -> None:
        raw_frame = self.feature_assembler.build_live_feature_frame(settings.SYMBOL)
        raw_frame = raw_frame.dropna(subset=["barrier_width"]).reset_index(drop=True)
        if raw_frame.empty:
            raise ValueError("Canli feature frame bos dondu.")

        snapshot = self.signal_engine.predict_latest(raw_frame)
        label = self.risk_profile.derive_trade_label(
            p_buy=snapshot.probs["p_up"],
            p_hold=snapshot.probs["p_flat"],
            p_sell=snapshot.probs["p_down"],
        )
        snapshot.trade_label = label

        if self.last_processed_bar is not None and snapshot.bar_time <= self.last_processed_bar:
            self.logger(f"Ayni mum daha once islendi, atlandi: {snapshot.bar_time:%Y-%m-%d %H:%M}")
            return

        self.last_processed_bar = snapshot.bar_time
        self._persist_last_processed_bar(snapshot.bar_time)
        self.logger(
            f"{snapshot.bar_time:%Y-%m-%d %H:%M} kapandi | close={snapshot.close_price:.2f} | "
            f"p_up={snapshot.probs['p_up']:.4f} p_flat={snapshot.probs['p_flat']:.4f} "
            f"p_down={snapshot.probs['p_down']:.4f} | signal={snapshot.trade_label}"
        )
        self.apply_trade_rules(snapshot)

    def _load_last_processed_bar(self) -> pd.Timestamp | None:
        try:
            if not self.state_file.exists():
                return None
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            value = payload.get("last_processed_bar")
            if not value:
                return None
            ts = pd.to_datetime(value, errors="coerce")
            return None if pd.isna(ts) else ts
        except Exception:
            return None

    def _persist_last_processed_bar(self, bar_time: pd.Timestamp) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"last_processed_bar": bar_time.strftime("%Y-%m-%d %H:%M:%S")}
            self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger(f"State dosyasi yazilamadi: {exc}")

    def apply_trade_rules(self, snapshot: SignalSnapshot) -> None:
        position = self.demo_client.get_position(settings.SYMBOL)
        signal = snapshot.trade_label

        if not position.is_open:
            if signal in {"up", "down"}:
                self.open_new_position(signal, snapshot)
            else:
                self.logger("Pozisyon yok ve sinyal flat; islem yapilmadi.")
            return

        if position.side == POSITION_LONG:
            same_direction = signal == "up"
            opposite_direction = signal == "down"
        else:
            same_direction = signal == "down"
            opposite_direction = signal == "up"

        if same_direction:
            self.handle_same_direction_signal(position, snapshot)
            return

        if signal == "flat":
            if self.risk_profile.flat_keeps_position:
                if not self.has_bot_protection_orders():
                    self.place_or_refresh_protection(position, snapshot)
                    self.logger("High risk flat sinyalinde koruma emirleri yeniden yazildi.")
                else:
                    self.logger("High risk flat sinyalinde pozisyon korunuyor.")
                return

            self.close_position(position, reason="Low risk flat sinyal")
            return

        if opposite_direction:
            self.close_position(position, reason="Ters sinyal")
            if self.risk_profile.reverse_reopens_position:
                self.open_new_position(signal, snapshot)
            else:
                self.logger("Low risk modunda ters sinyal sonrasi yeni pozisyon acilmadi.")

    def handle_same_direction_signal(self, position: PositionState, snapshot: SignalSnapshot) -> None:
        risk_notional, balance_state = self.compute_risk_based_notional(snapshot)
        if self.risk_profile.same_direction_grows_position:
            desired_notional = position.notional * self.risk_profile.same_direction_scale_multiplier
            target_notional = min(desired_notional, risk_notional)
            self.logger(
                f"High risk ayni yon sinyali | desired_notional={desired_notional:.2f} | "
                f"risk_cap={risk_notional:.2f} | equity={balance_state.equity:.2f}"
            )
        else:
            target_notional = min(position.notional, risk_notional)
            self.logger(
                f"Low risk ayni yon sinyali | current_notional={position.notional:.2f} | "
                f"risk_cap={risk_notional:.2f} | equity={balance_state.equity:.2f}"
            )

        position = self.rebalance_position_to_target_notional(position, snapshot, target_notional)
        self.place_or_refresh_protection(position, snapshot)

    def open_new_position(self, signal: str, snapshot: SignalSnapshot) -> None:
        side = ORDER_SIDE_BUY if signal == "up" else ORDER_SIDE_SELL
        target_notional, balance_state = self.compute_risk_based_notional(snapshot)
        quantity = self.quantity_for_notional(target_notional, snapshot.close_price)
        if quantity <= 0:
            raise ValueError("Hesaplanan quantity 0 cikti; base notional veya leverage yetersiz olabilir.")

        self.demo_client.place_market_order(
            symbol=settings.SYMBOL,
            side=side,
            quantity=quantity,
            reduce_only=False,
            client_order_id=f"{settings.ORDER_CLIENT_PREFIX}_entry_{now_ms()}",
        )
        time.sleep(1.0)
        position = self.demo_client.get_position(settings.SYMBOL)
        if not position.is_open:
            raise RuntimeError("Pozisyon acma emri gonderildi ama acik pozisyon bulunamadi.")

        self.logger(
            f"Yeni pozisyon acildi | side={position.side} | qty={position.quantity:.3f} | "
            f"entry={position.entry_price:.2f} | target_notional={target_notional:.2f} | equity={balance_state.equity:.2f}"
        )
        self.place_or_refresh_protection(position, snapshot)

    def close_position(self, position: PositionState, reason: str) -> None:
        self.demo_client.cancel_bot_orders(settings.SYMBOL)
        close_side = ORDER_SIDE_SELL if position.side == POSITION_LONG else ORDER_SIDE_BUY
        self.demo_client.place_market_order(
            symbol=settings.SYMBOL,
            side=close_side,
            quantity=self.round_quantity(position.quantity),
            reduce_only=True,
            client_order_id=f"{settings.ORDER_CLIENT_PREFIX}_close_{now_ms()}",
        )
        self.logger(f"Pozisyon kapatildi | reason={reason}")
        time.sleep(1.0)

    def place_or_refresh_protection(self, position: PositionState, snapshot: SignalSnapshot) -> None:
        self.demo_client.cancel_bot_orders(settings.SYMBOL)
        tp_price, sl_price = self.compute_tp_sl_prices(position.side, snapshot.close_price, snapshot.barrier_width)
        self.demo_client.place_protective_orders(
            symbol=settings.SYMBOL,
            position_side=position.side,
            quantity=self.round_quantity(position.quantity),
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
        )
        self.logger(
            f"TP/SL guncellendi | side={position.side} | tp={tp_price:.2f} | sl={sl_price:.2f}"
        )

    def compute_tp_sl_prices(self, position_side: str, close_price: float, barrier_width: float) -> tuple[float, float]:
        if not np.isfinite(barrier_width) or barrier_width <= 0:
            raise ValueError("Barrier width hesaplanamadi; TP/SL yerlestirilemiyor.")

        stop_distance = barrier_width * settings.STOP_LOSS_FACTOR
        if position_side == POSITION_LONG:
            tp = close_price + barrier_width
            sl = close_price - stop_distance
        else:
            tp = close_price - barrier_width
            sl = close_price + stop_distance

        tp = round_price(tp, self.symbol_rules.tick_size)
        sl = round_price(sl, self.symbol_rules.tick_size)
        return tp, sl

    def notional_to_quantity(self, notional_usdt: float, price: float) -> float:
        min_notional = float(self.symbol_rules.min_notional)
        adjusted_notional = max(notional_usdt, min_notional)
        return adjusted_notional / price

    def compute_risk_based_notional(self, snapshot: SignalSnapshot) -> tuple[float, AccountBalanceState]:
        stop_distance = snapshot.barrier_width * settings.STOP_LOSS_FACTOR
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            raise ValueError("Stop distance hesaplanamadi.")

        balance_state = self.demo_client.get_account_balance()
        risk_budget = balance_state.equity * settings.ACCOUNT_RISK_PER_TRADE
        raw_notional = risk_budget * snapshot.close_price / stop_distance
        max_notional_by_leverage = balance_state.equity * settings.LEVERAGE
        target_notional = min(raw_notional, max_notional_by_leverage)
        return target_notional, balance_state

    def rebalance_position_to_target_notional(
        self,
        position: PositionState,
        snapshot: SignalSnapshot,
        target_notional: float,
    ) -> PositionState:
        target_qty = self.quantity_for_notional(target_notional, snapshot.close_price)
        delta_qty = self.round_quantity(target_qty - position.quantity)
        step = float(self.symbol_rules.step_size)

        if abs(delta_qty) < step:
            self.logger("Hedef notional mevcut pozisyona cok yakin; boyut degistirilmedi.")
            return position

        if delta_qty > 0:
            side = ORDER_SIDE_BUY if position.side == POSITION_LONG else ORDER_SIDE_SELL
            reduce_only = False
            action = "buyutuldu"
        else:
            side = ORDER_SIDE_SELL if position.side == POSITION_LONG else ORDER_SIDE_BUY
            reduce_only = True
            action = "kucultuldu"

        self.demo_client.place_market_order(
            symbol=settings.SYMBOL,
            side=side,
            quantity=self.round_quantity(abs(delta_qty)),
            reduce_only=reduce_only,
            client_order_id=f"{settings.ORDER_CLIENT_PREFIX}_rebalance_{now_ms()}",
        )
        time.sleep(1.0)
        updated_position = self.demo_client.get_position(settings.SYMBOL)
        self.logger(
            f"Pozisyon {action} | hedef_notional={target_notional:.2f} | yeni_qty={updated_position.quantity:.3f}"
        )
        return updated_position

    def quantity_for_notional(self, notional_usdt: float, price: float) -> float:
        raw_qty = self.notional_to_quantity(notional_usdt, price)
        step = float(self.symbol_rules.step_size)
        min_qty = float(self.symbol_rules.min_qty)
        if step <= 0:
            return max(raw_qty, min_qty)
        steps = math.ceil((raw_qty / step) - 1e-12)
        qty = max(steps * step, min_qty)
        return round(qty, self.symbol_rules.quantity_precision)

    def round_quantity(self, quantity: float) -> float:
        rounded = round_down(quantity, self.symbol_rules.step_size)
        min_qty = float(self.symbol_rules.min_qty)
        if 0 < rounded < min_qty:
            rounded = min_qty
        return rounded

    def has_bot_protection_orders(self) -> bool:
        orders = self.demo_client.get_open_orders(settings.SYMBOL)
        bot_orders = [
            o
            for o in orders
            if str(o.get("clientOrderId", "")).startswith(settings.ORDER_CLIENT_PREFIX)
        ]
        return len(bot_orders) >= 2


@dataclass
class DashboardState:
    running: bool = False
    status: str = "Hazir"
    risk: str = RISK_LOW
    logs: list[str] = field(default_factory=list)
    controller: TradeBotController | None = None
    worker_thread: threading.Thread | None = None
    last_error: str | None = None


class BotDashboard:
    def __init__(self) -> None:
        self.state = DashboardState()
        self.lock = threading.Lock()

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with self.lock:
            self.state.logs.append(line)
            self.state.logs = self.state.logs[-200:]

    def start_bot(self, risk: str, api_key: str, api_secret: str) -> tuple[bool, str]:
        with self.lock:
            if self.state.worker_thread and self.state.worker_thread.is_alive():
                return False, "Bot zaten calisiyor."
            self.state.status = "Baslatiliyor..."
            self.state.risk = risk
            self.state.last_error = None
            worker = threading.Thread(
                target=self._run_bot_thread,
                args=(risk, api_key.strip(), api_secret.strip()),
                daemon=True,
            )
            self.state.worker_thread = worker
            worker.start()
        return True, "Bot baslatildi."

    def _run_bot_thread(self, risk: str, api_key: str, api_secret: str) -> None:
        try:
            resolved_api_key = api_key or settings.BINANCE_DEMO_API_KEY
            resolved_api_secret = api_secret or settings.BINANCE_DEMO_API_SECRET
            if not resolved_api_key or not resolved_api_secret:
                raise ValueError("Binance demo API key ve secret eksik. UI'dan gir veya `bot_settings.py` dosyasini doldur.")

            controller = TradeBotController(
                risk_name=risk,
                logger=self.log,
                api_key=resolved_api_key,
                api_secret=resolved_api_secret,
            )
            with self.lock:
                self.state.controller = controller
                self.state.running = True
                self.state.status = f"Calisiyor ({risk})"
            controller.start()
        except Exception as exc:
            error_text = str(exc)
            self.log(f"Baslatma hatasi: {error_text}")
            with self.lock:
                self.state.last_error = error_text
                self.state.status = f"Hata: {error_text}"
        finally:
            with self.lock:
                self.state.running = False
                self.state.controller = None
                self.state.worker_thread = None
                if not self.state.last_error:
                    self.state.status = "Durdu"

    def stop_bot(self) -> tuple[bool, str]:
        with self.lock:
            controller = self.state.controller
            if controller is None:
                return False, "Calisan bot yok."
            self.state.status = "Duruyor..."
        controller.stop()
        self.log("Durdurma istegi alindi.")
        return True, "Durdurma istegi gonderildi."

    def get_snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.state.running,
                "status": self.state.status,
                "risk": self.state.risk,
                "logs": list(self.state.logs),
                "last_error": self.state.last_error,
                "symbol": settings.SYMBOL,
                "leverage": settings.LEVERAGE,
                "risk_per_trade_pct": round(settings.ACCOUNT_RISK_PER_TRADE * 100.0, 4),
                "stop_loss_factor": settings.STOP_LOSS_FACTOR,
                "has_settings_keys": bool(settings.BINANCE_DEMO_API_KEY and settings.BINANCE_DEMO_API_SECRET),
            }

    def create_handler(self) -> type[BaseHTTPRequestHandler]:
        dashboard = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send_html(build_dashboard_html())
                    return
                if self.path == "/api/state":
                    self._send_json(dashboard.get_snapshot())
                    return
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length else b"{}"
                payload = json.loads(raw_body.decode("utf-8") or "{}")

                if self.path == "/api/start":
                    risk = str(payload.get("risk", RISK_LOW)).lower()
                    if risk not in RISK_PROFILES:
                        self._send_json({"ok": False, "message": "Gecersiz risk degeri."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    ok, message = dashboard.start_bot(
                        risk=risk,
                        api_key=str(payload.get("apiKey", "")),
                        api_secret=str(payload.get("apiSecret", "")),
                    )
                    status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
                    self._send_json({"ok": ok, "message": message}, status=status)
                    return

                if self.path == "/api/stop":
                    ok, message = dashboard.stop_bot()
                    status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
                    self._send_json({"ok": ok, "message": message}, status=status)
                    return

                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        return DashboardHandler

    def run(self, host: str = "127.0.0.1", port: int = 0, open_browser: bool = True) -> None:
        if port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
                temp_socket.bind((host, 0))
                _, port = temp_socket.getsockname()

        server = ThreadingHTTPServer((host, port), self.create_handler())
        public_host = "127.0.0.1" if host == "0.0.0.0" else host
        url = f"http://{public_host}:{port}"
        self.log(f"Dashboard hazir: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                self.log("Tarayici otomatik acilamadi; URL'yi manuel acabilirsin.")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.log("Dashboard kapatiliyor...")
        finally:
            server.shutdown()
            server.server_close()
            if self.state.controller:
                self.state.controller.stop()


def build_dashboard_html() -> str:
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC Demo Trade Bot</title>
  <style>
    :root {{
      --bg: #f4efe4;
      --panel: rgba(255,255,255,0.82);
      --panel-strong: rgba(255,255,255,0.92);
      --ink: #1b1b1b;
      --muted: #5d5a55;
      --accent: #cb5f2e;
      --accent-dark: #903f1d;
      --line: rgba(27,27,27,0.09);
      --ok: #136f4f;
      --warn: #8e5a17;
      --shadow: 0 24px 80px rgba(66, 42, 23, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI Variable", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(231, 147, 70, 0.22), transparent 28%),
        radial-gradient(circle at right 20%, rgba(23, 111, 79, 0.16), transparent 24%),
        linear-gradient(135deg, #efe6d1 0%, #f7f3ea 52%, #e5ede7 100%);
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .shell {{
      width: min(960px, 100%);
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.65);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(18px);
    }}
    .hero {{
      padding: 28px 30px 22px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(203,95,46,0.14), rgba(255,255,255,0)),
        linear-gradient(180deg, rgba(255,255,255,0.56), rgba(255,255,255,0));
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 8px rgba(203,95,46,0.12);
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(34px, 6vw, 58px);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin-top: 12px;
      max-width: 620px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      padding: 24px;
    }}
    .card {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
    }}
    .card h2 {{
      margin: 0 0 16px;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .field {{
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }}
    label {{
      font-size: 13px;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      border: 1px solid rgba(27,27,27,0.12);
      background: rgba(255,255,255,0.8);
      border-radius: 14px;
      padding: 14px 16px;
      font-size: 15px;
      color: var(--ink);
      outline: none;
    }}
    input:focus, select:focus {{
      border-color: rgba(203,95,46,0.55);
      box-shadow: 0 0 0 4px rgba(203,95,46,0.12);
    }}
    .actions {{
      display: flex;
      gap: 12px;
      margin-top: 18px;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
    }}
    button:hover {{
      transform: translateY(-1px);
    }}
    .primary {{
      flex: 1;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), #d98653);
      box-shadow: 0 18px 34px rgba(203,95,46,0.22);
    }}
    .secondary {{
      flex: 1;
      color: var(--ink);
      background: rgba(27,27,27,0.06);
    }}
    button:disabled {{
      cursor: default;
      transform: none;
      opacity: 0.55;
      box-shadow: none;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      border-radius: 999px;
      background: rgba(19,111,79,0.08);
      color: var(--ok);
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 700;
    }}
    .status-pill.warn {{
      background: rgba(142,90,23,0.08);
      color: var(--warn);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(27,27,27,0.04);
      border: 1px solid rgba(27,27,27,0.06);
    }}
    .stat .k {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .v {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .hint {{
      margin-top: 12px;
      font-size: 13px;
      line-height: 1.6;
      color: var(--muted);
    }}
    .logs {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 10px;
      max-height: 360px;
      overflow: auto;
    }}
    .logs li {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(27,27,27,0.04);
      border: 1px solid rgba(27,27,27,0.06);
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .banner {{
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(203,95,46,0.10);
      border: 1px solid rgba(203,95,46,0.18);
      color: var(--accent-dark);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 860px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow"><span class="dot"></span> BTC Demo Trade Bot</div>
      <h1>Model orada. Karar burada.</h1>
      <div class="subtitle">
        Binance demo hesabina baglan, risk profilini sec ve botu buradan yonet.
        Pozisyon boyutu stopta yaklasik %{settings.ACCOUNT_RISK_PER_TRADE * 100:.0f} hesap riski hedefler;
        TP esik, SL ise esik x {settings.STOP_LOSS_FACTOR:.2f}.
      </div>
    </section>
    <section class="grid">
      <div class="card">
        <h2>Kontrol</h2>
        <div id="status-pill" class="status-pill">Hazir</div>
        <div class="stats">
          <div class="stat"><div class="k">Symbol</div><div class="v">{settings.SYMBOL}</div></div>
          <div class="stat"><div class="k">Leverage</div><div class="v">{settings.LEVERAGE}x</div></div>
          <div class="stat"><div class="k">Risk / Trade</div><div class="v">%{settings.ACCOUNT_RISK_PER_TRADE * 100:.0f}</div></div>
          <div class="stat"><div class="k">SL Factor</div><div class="v">{settings.STOP_LOSS_FACTOR:.2f}</div></div>
        </div>
        <div class="hint">
          API anahtarlarini burada girebilirsin. Bos birakirsan bot once <code>bot_settings.py</code> icindeki degerleri dener.
        </div>
        <div class="field">
          <label for="risk">Risk seviyesi</label>
          <select id="risk">
            <option value="low">Low</option>
            <option value="high">High</option>
          </select>
        </div>
        <div class="field">
          <label for="api-key">Binance Demo API Key</label>
          <input id="api-key" type="password" placeholder="UI'dan yapistir veya bot_settings.py kullan">
        </div>
        <div class="field">
          <label for="api-secret">Binance Demo API Secret</label>
          <input id="api-secret" type="password" placeholder="UI'dan yapistir veya bot_settings.py kullan">
        </div>
        <div class="actions">
          <button id="start-btn" class="primary">Botu Baslat</button>
          <button id="stop-btn" class="secondary">Durdur</button>
        </div>
      </div>
      <div class="card">
        <h2>Canli Akis</h2>
        <ul id="logs" class="logs"></ul>
      </div>
    </section>
  </div>
  <script>
    const statusPill = document.getElementById('status-pill');
    const logsEl = document.getElementById('logs');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const riskEl = document.getElementById('risk');
    const apiKeyEl = document.getElementById('api-key');
    const apiSecretEl = document.getElementById('api-secret');

    function setStatus(text, isWarn) {{
      statusPill.textContent = text;
      statusPill.className = isWarn ? 'status-pill warn' : 'status-pill';
    }}

    async function refreshState() {{
      const res = await fetch('/api/state');
      const state = await res.json();
      setStatus(state.status, Boolean(state.last_error));
      riskEl.value = state.risk || 'low';
      startBtn.disabled = state.running;
      stopBtn.disabled = !state.running;

      logsEl.innerHTML = '';
      const logs = state.logs.length ? state.logs : ['Henuz log yok.'];
      for (const item of logs.slice().reverse()) {{
        const li = document.createElement('li');
        li.textContent = item;
        logsEl.appendChild(li);
      }}
    }}

    async function postJson(path, payload) {{
      const res = await fetch(path, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload || {{}})
      }});
      return res.json();
    }}

    startBtn.addEventListener('click', async () => {{
      startBtn.disabled = true;
      const response = await postJson('/api/start', {{
        risk: riskEl.value,
        apiKey: apiKeyEl.value,
        apiSecret: apiSecretEl.value
      }});
      if (!response.ok) {{
        alert(response.message);
      }}
      await refreshState();
    }});

    stopBtn.addEventListener('click', async () => {{
      stopBtn.disabled = true;
      const response = await postJson('/api/stop', {{}});
      if (!response.ok) {{
        alert(response.message);
      }}
      await refreshState();
    }});

    refreshState();
    setInterval(refreshState, 2000);
  </script>
</body>
</html>"""


def run_smoke_test() -> None:
    public_client = BinancePublicDataClient(settings.PUBLIC_MARKET_DATA_URL)
    signal_engine = ModelSignalEngine()
    assembler = LiveFeatureAssembler(
        public_client=public_client,
        history_bars=max(
            settings.KLINE_HISTORY_BARS,
            signal_engine.normalizer.rolling_window + signal_engine.normalizer.window_size + 64,
        ),
    )
    raw_frame = assembler.build_live_feature_frame(settings.SYMBOL)
    raw_frame = raw_frame.dropna(subset=["barrier_width"]).reset_index(drop=True)
    snapshot = signal_engine.predict_latest(raw_frame)
    risk_profile = RISK_PROFILES[RISK_LOW]
    trade_label = risk_profile.derive_trade_label(
        p_buy=snapshot.probs["p_up"],
        p_hold=snapshot.probs["p_flat"],
        p_sell=snapshot.probs["p_down"],
    )
    print(f"Latest bar: {snapshot.bar_time:%Y-%m-%d %H:%M:%S}")
    print(f"Close: {snapshot.close_price:.2f}")
    print(f"Barrier width: {snapshot.barrier_width:.4f}")
    print(
        "Probabilities: "
        f"up={snapshot.probs['p_up']:.6f}, flat={snapshot.probs['p_flat']:.6f}, down={snapshot.probs['p_down']:.6f}"
    )
    print(f"Low-risk label: {trade_label}")
    print(f"Rows used: {len(raw_frame)}")


def run_headless_bot(risk: str, api_key: str, api_secret: str) -> None:
    resolved_api_key = api_key or settings.BINANCE_DEMO_API_KEY
    resolved_api_secret = api_secret or settings.BINANCE_DEMO_API_SECRET
    if not resolved_api_key or not resolved_api_secret:
        raise ValueError("Headless mod icin Binance demo API key ve secret gerekli.")

    def logger(message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    controller = TradeBotController(
        risk_name=risk,
        logger=logger,
        api_key=resolved_api_key,
        api_secret=resolved_api_secret,
    )
    try:
        controller.start()
    except KeyboardInterrupt:
        logger("KeyboardInterrupt alindi, bot durduruluyor...")
        controller.stop()
        raise


def run_once_bot(risk: str, api_key: str, api_secret: str, state_file: str) -> None:
    resolved_api_key = api_key or settings.BINANCE_DEMO_API_KEY
    resolved_api_secret = api_secret or settings.BINANCE_DEMO_API_SECRET
    if not resolved_api_key or not resolved_api_secret:
        raise ValueError("Run-once mod icin Binance demo API key ve secret gerekli.")

    def logger(message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    controller = TradeBotController(
        risk_name=risk,
        logger=logger,
        api_key=resolved_api_key,
        api_secret=resolved_api_secret,
        state_file=Path(state_file),
    )
    controller.run_once()


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Binance demo trade bot")
    parser.add_argument("--smoke-test", action="store_true", help="Load live features and print the latest prediction")
    parser.add_argument("--headless", action="store_true", help="Run without the dashboard UI")
    parser.add_argument("--run-once", action="store_true", help="Process the latest closed candle once and exit")
    parser.add_argument("--risk", choices=[RISK_LOW, RISK_HIGH], default=RISK_LOW, help="Risk profile for headless mode")
    parser.add_argument("--api-key", default="", help="Override Binance demo API key")
    parser.add_argument("--api-secret", default="", help="Override Binance demo API secret")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=0, help="Dashboard bind port (0 picks a free port)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the dashboard in a browser")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="State file used by --run-once")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if args.run_once:
        run_once_bot(
            risk=args.risk,
            api_key=args.api_key,
            api_secret=args.api_secret,
            state_file=args.state_file,
        )
        return

    if args.headless:
        run_headless_bot(risk=args.risk, api_key=args.api_key, api_secret=args.api_secret)
        return

    BotDashboard().run(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
