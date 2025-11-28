"""
Feature Engineering Service
Extracted from improved_v6_selective_features.ipynb
"""

import numpy as np
import pandas as pd

from app.config import FEATURE_NAMES
from app.utils.helpers import _rsi_from_window


def clip_by_quantiles(s: pd.Series, mask: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Clip values by quantiles"""
    subset = s[mask & s.notna() & np.isfinite(s)]
    if len(subset) == 0:
        return s
    low = subset.quantile(lower_q)
    high = subset.quantile(upper_q)
    return s.clip(lower=low, upper=high)


def add_base_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add base return features"""
    out = df.copy()
    out["close_shift1"] = out["close"].shift(1)
    out["volume_shift1"] = out["volume"].shift(1)
    out["ret_1d"] = np.log(out["close"] / out["close_shift1"])
    out["vol_chg"] = np.log((out["volume"] + 1) / (out["volume_shift1"] + 1))
    return out


def winsorize_returns(
    df: pd.DataFrame, val_start: str, lower_q: float, upper_q: float
) -> pd.DataFrame:
    """Winsorize returns using training data quantiles"""
    out = df.copy()
    val_start_ts = pd.Timestamp(val_start)
    train_mask = out["time"] < val_start_ts
    out["ret_1d_clipped"] = clip_by_quantiles(out["ret_1d"], train_mask, lower_q, upper_q)
    out["vol_chg_clipped"] = clip_by_quantiles(out["vol_chg"], train_mask, lower_q, upper_q)
    return out


def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI series"""
    arr = close.values.astype(float)
    rsi = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period + 1:
        return pd.Series(rsi, index=close.index)
    for i in range(period, len(arr)):
        window = arr[i - period : i + 1]
        rsi[i] = _rsi_from_window(window, period)
    return pd.Series(rsi, index=close.index)


def build_features_v6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for V6 - with selective features (only +5 features)
    This is the main feature engineering function from the baseline
    """
    feat = df[["time", "close", "volume", "ret_1d_clipped", "vol_chg_clipped"]].copy()
    r = feat["ret_1d_clipped"]
    v = feat["vol_chg_clipped"]
    c = feat["close"]
    vol = feat["volume"]

    # Base features (from V4)
    # Return lags
    for lag in range(1, 11):
        if lag == 1:
            feat[f"ret_lag{lag}"] = r
        else:
            feat[f"ret_lag{lag}"] = r.shift(lag - 1)

    # Volume lags
    for lag in range(1, 6):
        if lag == 1:
            feat[f"vol_lag{lag}"] = v
        else:
            feat[f"vol_lag{lag}"] = v.shift(lag - 1)

    # Volatility, stats
    feat["vol_5"] = r.rolling(5).std()
    feat["vol_10"] = r.rolling(10).std()
    feat["vol_20"] = r.rolling(20).std()
    feat["ret_roll_min_20"] = r.rolling(20).min()
    feat["ret_roll_max_20"] = r.rolling(20).max()
    roll_mean_20 = r.rolling(20).mean()
    roll_std_20 = r.rolling(20).std()
    feat["ret_z_20"] = (r - roll_mean_20) / roll_std_20.replace(0, np.nan)
    feat["mean_ret_5"] = r.rolling(5).mean()
    feat["mean_ret_10"] = r.rolling(10).mean()
    feat["mean_ret_20"] = roll_mean_20

    # SMA, trend
    feat["sma10"] = c.rolling(10).mean()
    feat["sma20"] = c.rolling(20).mean()
    feat["price_trend_10"] = (c - feat["sma10"]) / feat["sma10"]
    feat["price_trend_20"] = (c - feat["sma20"]) / feat["sma20"]

    # RSI 14
    feat["rsi_14"] = compute_rsi_series(c, period=14)

    # Bollinger width
    std20_price = c.rolling(20).std()
    upper20 = feat["sma20"] + 2 * std20_price
    lower20 = feat["sma20"] - 2 * std20_price
    feat["bb_width_20"] = (upper20 - lower20) / feat["sma20"]

    # V6: Selective features - chỉ 5 features đơn giản
    # ROC (Rate of Change)
    feat["roc_10"] = ((c - c.shift(10)) / c.shift(10)) * 100
    feat["roc_20"] = ((c - c.shift(20)) / c.shift(20)) * 100

    # Momentum
    feat["momentum_10"] = c - c.shift(10)
    feat["momentum_20"] = c - c.shift(20)

    # Volume ratio
    volume_sma_20 = vol.rolling(20).mean()
    feat["volume_ratio"] = vol / volume_sma_20.replace(0, np.nan)

    # Calendar
    feat["dow"] = feat["time"].dt.dayofweek.astype(int)
    feat["month"] = feat["time"].dt.month.astype(int)

    # Target: next day return
    feat["y"] = feat["ret_1d_clipped"].shift(-1)

    cols = ["time"] + FEATURE_NAMES + ["y"]
    feat = feat[cols]
    feat = feat.dropna().reset_index(drop=True)
    return feat


def build_features_from_buffers(
    ret_buffer: list[float],
    vol_buffer: list[float],
    price_buffer: list[float],
    volume_buffer: list[float],
    current_date: pd.Timestamp,
) -> dict[str, float]:
    """
    Build features from buffers for iterative forecasting
    This is used during multi-step prediction when we don't have full historical data
    """
    from app.utils.helpers import last_window, rolling_mean, rolling_std

    feat_vals = {}
    current_ret = ret_buffer[-1]
    current_vol = vol_buffer[-1]
    current_price = price_buffer[-1]

    feat_vals["ret_1d_clipped"] = current_ret
    feat_vals["vol_chg_clipped"] = current_vol

    # Return lags
    for k in range(1, 11):
        if len(ret_buffer) >= k:
            feat_vals[f"ret_lag{k}"] = ret_buffer[-k]
        else:
            feat_vals[f"ret_lag{k}"] = 0.0

    # Volume lags
    for k in range(1, 6):
        if len(vol_buffer) >= k:
            feat_vals[f"vol_lag{k}"] = vol_buffer[-k]
        else:
            feat_vals[f"vol_lag{k}"] = 0.0

    ret_arr = np.array(ret_buffer, dtype=float)
    price_arr = np.array(price_buffer, dtype=float)
    volume_arr = np.array(volume_buffer, dtype=float)

    # Base features
    feat_vals["vol_5"] = rolling_std(ret_arr, 5)
    feat_vals["vol_10"] = rolling_std(ret_arr, 10)
    feat_vals["vol_20"] = rolling_std(ret_arr, 20)

    win20_ret = last_window(ret_arr, 20)
    feat_vals["ret_roll_min_20"] = float(win20_ret.min()) if len(win20_ret) > 0 else 0.0
    feat_vals["ret_roll_max_20"] = float(win20_ret.max()) if len(win20_ret) > 0 else 0.0

    mean20_ret = rolling_mean(ret_arr, 20)
    std20_ret = rolling_std(ret_arr, 20)
    feat_vals["ret_z_20"] = 0.0 if std20_ret == 0 else (current_ret - mean20_ret) / std20_ret

    feat_vals["mean_ret_5"] = rolling_mean(ret_arr, 5)
    feat_vals["mean_ret_10"] = rolling_mean(ret_arr, 10)
    feat_vals["mean_ret_20"] = mean20_ret

    sma10 = rolling_mean(price_arr, 10)
    sma20 = rolling_mean(price_arr, 20)

    feat_vals["sma10"] = sma10
    feat_vals["sma20"] = sma20
    feat_vals["price_trend_10"] = (current_price - sma10) / sma10 if sma10 != 0 else 0.0
    feat_vals["price_trend_20"] = (current_price - sma20) / sma20 if sma20 != 0 else 0.0

    if len(price_arr) >= 15:
        prices_window = price_arr[-15:]
        feat_vals["rsi_14"] = _rsi_from_window(prices_window, period=14)
    else:
        feat_vals["rsi_14"] = 50.0

    win20_price = last_window(price_arr, 20)
    if len(win20_price) > 1:
        sma20_price = float(win20_price.mean())
        std20_price = float(np.std(win20_price, ddof=1))
        bb_width = (4.0 * std20_price) / sma20_price if sma20_price != 0 else 0.0
    else:
        bb_width = 0.0
    feat_vals["bb_width_20"] = bb_width

    # V6: Selective features
    if len(price_arr) >= 10:
        feat_vals["roc_10"] = (
            ((current_price - price_arr[-10]) / price_arr[-10]) * 100
            if price_arr[-10] != 0
            else 0.0
        )
    else:
        feat_vals["roc_10"] = 0.0

    if len(price_arr) >= 20:
        feat_vals["roc_20"] = (
            ((current_price - price_arr[-20]) / price_arr[-20]) * 100
            if price_arr[-20] != 0
            else 0.0
        )
    else:
        feat_vals["roc_20"] = 0.0

    if len(price_arr) >= 10:
        feat_vals["momentum_10"] = current_price - price_arr[-10]
    else:
        feat_vals["momentum_10"] = 0.0

    if len(price_arr) >= 20:
        feat_vals["momentum_20"] = current_price - price_arr[-20]
    else:
        feat_vals["momentum_20"] = 0.0

    if len(volume_arr) >= 20:
        volume_sma_20 = rolling_mean(volume_arr, 20)
        feat_vals["volume_ratio"] = volume_arr[-1] / volume_sma_20 if volume_sma_20 != 0 else 1.0
    else:
        feat_vals["volume_ratio"] = 1.0

    feat_vals["dow"] = current_date.dayofweek
    feat_vals["month"] = current_date.month

    return feat_vals
