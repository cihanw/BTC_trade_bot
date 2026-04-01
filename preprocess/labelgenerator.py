from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATR_PERIOD = 14
ATR_LONG_PERIOD = 50
BASE_BARRIER_MULTIPLIER = 1.5
MAX_HORIZON_BARS = 16
CONF_ALPHA = 0.5
OPPOSITE_CLASS_BETA = 0.2
DOMINANT_BASE_PROB = 0.50
DOMINANT_CONF_SPAN = 0.43
HIGH_VOL_RATIO_THRESHOLD = 2.0
HIGH_VOL_FACTOR_CAP = 1.5
ASIA_SESSION_MULTIPLIER = 1.2
ASIA_SESSION_START_HOUR = 0
ASIA_SESSION_END_HOUR = 8
INPUT_FILENAME = "merged30m.csv"
OUTPUT_FILENAME = "finalLabels.csv"
SOURCE_PREFIX = "binanceFutures"
TIME_COLUMN = "time"
REQUIRED_SOURCE_COLS = [
    f"{SOURCE_PREFIX}_open",
    f"{SOURCE_PREFIX}_high",
    f"{SOURCE_PREFIX}_low",
    f"{SOURCE_PREFIX}_close",
]


def compute_past_only_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=period).mean().shift(1)
    return true_range.rename("true_range"), atr.rename("atr_past_14")


def compute_effective_barrier_multiplier(
    atr: pd.Series,
    timestamps: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    atr_long = atr.rolling(window=ATR_LONG_PERIOD, min_periods=ATR_LONG_PERIOD).mean()
    vol_ratio = atr / atr_long

    high_vol_factor = pd.Series(1.0, index=atr.index, dtype=np.float64)
    high_vol_mask = vol_ratio > HIGH_VOL_RATIO_THRESHOLD
    high_vol_factor.loc[high_vol_mask] = np.minimum(
        (vol_ratio.loc[high_vol_mask] / HIGH_VOL_RATIO_THRESHOLD).to_numpy(dtype=np.float64),
        HIGH_VOL_FACTOR_CAP,
    )

    asia_factor = pd.Series(1.0, index=atr.index, dtype=np.float64)
    if timestamps.notna().any():
        hours = timestamps.dt.hour
        asia_mask = hours.ge(ASIA_SESSION_START_HOUR) & hours.lt(ASIA_SESSION_END_HOUR)
        asia_factor.loc[asia_mask] = ASIA_SESSION_MULTIPLIER

    effective_k = BASE_BARRIER_MULTIPLIER * high_vol_factor * asia_factor
    return effective_k.rename("effective_k"), vol_ratio.rename("atr_vol_ratio")


def resolve_same_bar_dual_hit(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    upper_barrier: float,
    lower_barrier: float,
) -> tuple[str, str]:
    up_overshoot = max(high_price - upper_barrier, 0.0)
    down_overshoot = max(lower_barrier - low_price, 0.0)

    if not np.isclose(up_overshoot, down_overshoot):
        if up_overshoot > down_overshoot:
            return "up", "dual_hit_up"
        return "down", "dual_hit_down"

    if close_price > open_price:
        return "up", "dual_hit_up"
    if close_price < open_price:
        return "down", "dual_hit_down"
    return "up", "dual_hit_up"


def build_soft_labels(
    df: pd.DataFrame,
    barrier_width: pd.Series,
    upper_barrier: pd.Series,
    lower_barrier: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    open_prices = df["open"].to_numpy(dtype=np.float64, copy=False)
    high_prices = df["high"].to_numpy(dtype=np.float64, copy=False)
    low_prices = df["low"].to_numpy(dtype=np.float64, copy=False)
    close_prices = df["close"].to_numpy(dtype=np.float64, copy=False)

    widths = barrier_width.to_numpy(dtype=np.float64, copy=False)
    upper_levels = upper_barrier.to_numpy(dtype=np.float64, copy=False)
    lower_levels = lower_barrier.to_numpy(dtype=np.float64, copy=False)

    n = len(df)
    p_up = np.full(n, np.nan, dtype=np.float64)
    p_flat = np.full(n, np.nan, dtype=np.float64)
    p_down = np.full(n, np.nan, dtype=np.float64)
    hard_labels = np.full(n, "nan", dtype=object)

    last_start = n - MAX_HORIZON_BARS
    for i in range(last_start):
        base_close = close_prices[i]
        width = widths[i]
        upper = upper_levels[i]
        lower = lower_levels[i]

        if not np.isfinite(base_close) or base_close == 0.0:
            continue
        if not np.isfinite(width) or width <= 0.0:
            continue
        if not np.isfinite(upper) or not np.isfinite(lower):
            continue

        label: str | None = None
        hit_offset: int | None = None
        hit_close = np.nan
        window_invalid = False

        for offset in range(1, MAX_HORIZON_BARS + 1):
            j = i + offset
            open_j = open_prices[j]
            high_j = high_prices[j]
            low_j = low_prices[j]
            close_j = close_prices[j]

            if not np.isfinite(open_j) or not np.isfinite(high_j) or not np.isfinite(low_j) or not np.isfinite(close_j):
                window_invalid = True
                break

            hit_up = high_j >= upper
            hit_down = low_j <= lower

            if not hit_up and not hit_down:
                continue

            if hit_up and hit_down:
                label, _ = resolve_same_bar_dual_hit(
                    open_price=open_j,
                    high_price=high_j,
                    low_price=low_j,
                    close_price=close_j,
                    upper_barrier=upper,
                    lower_barrier=lower,
                )
            elif hit_up:
                label = "up"
            else:
                label = "down"

            hit_offset = offset
            hit_close = close_j
            break

        if window_invalid:
            continue

        if label is None or hit_offset is None:
            final_close = close_prices[i + MAX_HORIZON_BARS]
            if not np.isfinite(final_close):
                continue

            r_final = np.clip((final_close - base_close) / width, -1.0, 1.0)
            flat_prob = 0.4 + 0.3 * (1.0 - abs(r_final))
            up_prob = (1.0 - flat_prob) * ((1.0 + r_final) / 2.0)
            down_prob = 1.0 - flat_prob - up_prob

            p_up[i] = up_prob
            p_flat[i] = flat_prob
            p_down[i] = down_prob
            hard_labels[i] = "flat"
            continue

        speed_score = 1.0 - (hit_offset / MAX_HORIZON_BARS)
        if label == "up":
            overshoot = (hit_close - upper) / width
        else:
            overshoot = (lower - hit_close) / width

        magnitude_score = np.tanh(overshoot)
        conf = np.clip(CONF_ALPHA * speed_score + (1.0 - CONF_ALPHA) * magnitude_score, 0.0, 1.0)
        dominant_prob = DOMINANT_BASE_PROB + DOMINANT_CONF_SPAN * conf
        opposite_prob = (1.0 - dominant_prob) * OPPOSITE_CLASS_BETA
        flat_prob = 1.0 - dominant_prob - opposite_prob
        if label == "up":
            p_up[i] = dominant_prob
            p_flat[i] = flat_prob
            p_down[i] = opposite_prob
        else:
            p_up[i] = opposite_prob
            p_flat[i] = flat_prob
            p_down[i] = dominant_prob

        hard_labels[i] = label

    labels_df = pd.DataFrame(
        {
            "p_up": p_up,
            "p_flat": p_flat,
            "p_down": p_down,
        },
        index=df.index,
    )

    prob_sum = labels_df.sum(axis=1)
    valid_sum_mask = prob_sum > 0
    labels_df.loc[valid_sum_mask, ["p_up", "p_flat", "p_down"]] = labels_df.loc[
        valid_sum_mask,
        ["p_up", "p_flat", "p_down"],
    ].div(prob_sum[valid_sum_mask], axis=0)
    return labels_df, pd.Series(hard_labels, index=df.index, name="hard_label")


def print_label_distribution(hard_labels: pd.Series) -> None:
    distribution = hard_labels.value_counts(normalize=True)
    bounds = {
        "up": (0.30, 0.40),
        "flat": (0.20, 0.40),
        "down": (0.30, 0.40),
    }

    for label, (lower_bound, upper_bound) in bounds.items():
        share = float(distribution.get(label, 0.0))
        print(f"{label.upper()} share: {share * 100:.2f}%")
        if share < lower_bound or share > upper_bound:
            print(
                f"[WARN] {label.upper()} share is outside the suggested range "
                f"({lower_bound * 100:.0f}% - {upper_bound * 100:.0f}%)."
            )

    flat_share = float(distribution.get("flat", 0.0))
    if flat_share > 0.60:
        print("[WARN] FLAT share is above 60%; effective barrier may be too wide.")
    if flat_share < 0.10:
        print("[WARN] FLAT share is below 10%; effective barrier may be too narrow.")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / INPUT_FILENAME
    output_path = project_root / "data" / "processed" / OUTPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = [TIME_COLUMN, *REQUIRED_SOURCE_COLS]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {INPUT_FILENAME}: {missing}")

    input_rows = len(df)
    working = df[[TIME_COLUMN, *REQUIRED_SOURCE_COLS]].copy()
    working = working.rename(
        columns={
            f"{SOURCE_PREFIX}_open": "open",
            f"{SOURCE_PREFIX}_high": "high",
            f"{SOURCE_PREFIX}_low": "low",
            f"{SOURCE_PREFIX}_close": "close",
        }
    )

    for col in ["open", "high", "low", "close"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    timestamps = pd.to_datetime(working[TIME_COLUMN], errors="coerce")

    _, atr = compute_past_only_atr(
        high=working["high"],
        low=working["low"],
        close=working["close"],
        period=ATR_PERIOD,
    )
    effective_k, vol_ratio = compute_effective_barrier_multiplier(atr=atr, timestamps=timestamps)
    barrier_width = (effective_k * atr).rename("barrier_width")
    upper_barrier = (working["close"] + barrier_width).rename("upper_barrier")
    lower_barrier = (working["close"] - barrier_width).rename("lower_barrier")

    labels_df, hard_labels = build_soft_labels(
        df=working,
        barrier_width=barrier_width,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
    )

    output_df = pd.DataFrame(
        {
            TIME_COLUMN: df[TIME_COLUMN],
            "p_up": labels_df["p_up"],
            "p_flat": labels_df["p_flat"],
            "p_down": labels_df["p_down"],
            "hard_label": hard_labels,
        }
    )

    valid_label_mask = output_df[["p_up", "p_flat", "p_down"]].notna().all(axis=1)
    dropped_rows = int((~valid_label_mask).sum())
    output_df = output_df.loc[valid_label_mask].reset_index(drop=True)
    hard_labels = hard_labels.loc[valid_label_mask].reset_index(drop=True)
    effective_k = effective_k.loc[valid_label_mask].reset_index(drop=True)
    vol_ratio = vol_ratio.loc[valid_label_mask].reset_index(drop=True)
    timestamps = timestamps.loc[valid_label_mask].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    high_vol_rows = int((vol_ratio > HIGH_VOL_RATIO_THRESHOLD).sum())
    asia_rows = int(
        (
            timestamps.notna()
            & timestamps.dt.hour.ge(ASIA_SESSION_START_HOUR)
            & timestamps.dt.hour.lt(ASIA_SESSION_END_HOUR)
        ).sum()
    )

    print(f"Input rows: {input_rows}")
    print(f"Output rows: {len(output_df)}")
    print(f"Dropped rows without complete labels: {dropped_rows}")
    print(f"Label source prefix: {SOURCE_PREFIX}")
    print(f"ATR period: {ATR_PERIOD}")
    print(f"ATR long-period ratio window: {ATR_LONG_PERIOD}")
    print(f"Base barrier multiplier (k): {BASE_BARRIER_MULTIPLIER:.2f}")
    print(f"Max horizon bars (Tmax): {MAX_HORIZON_BARS}")
    print(f"Confidence alpha: {CONF_ALPHA:.2f}")
    print(f"Dominant base probability: {DOMINANT_BASE_PROB:.2f}")
    print(f"Dominant confidence span: {DOMINANT_CONF_SPAN:.2f}")
    print(f"Opposite-class beta: {OPPOSITE_CLASS_BETA:.2f}")
    print(f"High-volatility rows (rho > {HIGH_VOL_RATIO_THRESHOLD:.1f}): {high_vol_rows}")
    print(f"Asia-session rows ({ASIA_SESSION_START_HOUR:02d}:00-{ASIA_SESSION_END_HOUR:02d}:00 UTC): {asia_rows}")
    print(f"Mean effective k on labeled rows: {effective_k.mean():.4f}")
    print_label_distribution(hard_labels)
    print(f"Saved: {output_path}")
    print("Added columns: p_up, p_flat, p_down, hard_label")


if __name__ == "__main__":
    main()
