"""
Data loading and preprocessing utilities
"""

from pathlib import Path

import pandas as pd

from app.config import DATA_FILE, MODEL_CONFIG
from app.services.feature_engineering import add_base_returns, build_features_v6, winsorize_returns


def load_data(data_path: str | None = None) -> pd.DataFrame:
    """
    Load FPT stock data from CSV file

    Args:
        data_path: Path to CSV file. If None, uses default path from config.

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, symbol
    """
    if data_path is None:
        data_path = DATA_FILE

    if isinstance(data_path, Path):
        data_path = str(data_path)

    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def prepare_features_from_dataframe(df: pd.DataFrame) -> tuple:
    """
    Prepare features from full dataframe (for training/validation)

    Args:
        df: DataFrame with columns: time, open, high, low, close, volume, symbol

    Returns:
        Tuple of (X, y, feature_df) where:
        - X: Feature matrix (numpy array)
        - y: Target values (numpy array)
        - feature_df: DataFrame with all features
    """
    from app.config import FEATURE_NAMES

    # Add base returns
    df_feat = add_base_returns(df)

    # Winsorize returns
    df_feat = winsorize_returns(
        df_feat,
        MODEL_CONFIG["val_start"],
        MODEL_CONFIG["clip_lower_q"],
        MODEL_CONFIG["clip_upper_q"],
    )

    # Build features
    feat_df = build_features_v6(df_feat)

    # Extract X and y
    X = feat_df[FEATURE_NAMES].values
    y = feat_df["y"].values

    return X, y, feat_df


def prepare_historical_data_for_prediction(
    historical_data: list, train_df: pd.DataFrame | None = None
) -> tuple:
    """
    Prepare historical data buffers for iterative prediction

    Args:
        historical_data: List of dicts with keys: time, open, high, low, close, volume
        train_df: Optional training dataframe for winsorization parameters

    Returns:
        Tuple of (ret_buffer, vol_buffer, price_buffer, volume_buffer, last_date)
    """
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Add base returns
    df_feat = add_base_returns(df)

    # Winsorize (use training data if available, otherwise use current data)
    if train_df is not None:
        # Use training data for quantile calculation
        train_feat = add_base_returns(train_df)
        val_start_ts = pd.Timestamp(MODEL_CONFIG["val_start"])
        train_mask = train_feat["time"] < val_start_ts

        # Calculate quantiles from training data
        ret_train = train_feat["ret_1d"][train_mask & train_feat["ret_1d"].notna()]
        vol_train = train_feat["vol_chg"][train_mask & train_feat["vol_chg"].notna()]

        if len(ret_train) > 0:
            ret_low = ret_train.quantile(MODEL_CONFIG["clip_lower_q"])
            ret_high = ret_train.quantile(MODEL_CONFIG["clip_upper_q"])
            df_feat["ret_1d_clipped"] = df_feat["ret_1d"].clip(lower=ret_low, upper=ret_high)
        else:
            df_feat["ret_1d_clipped"] = df_feat["ret_1d"]

        if len(vol_train) > 0:
            vol_low = vol_train.quantile(MODEL_CONFIG["clip_lower_q"])
            vol_high = vol_train.quantile(MODEL_CONFIG["clip_upper_q"])
            df_feat["vol_chg_clipped"] = df_feat["vol_chg"].clip(lower=vol_low, upper=vol_high)
        else:
            df_feat["vol_chg_clipped"] = df_feat["vol_chg"]
    else:
        # Use current data for quantiles (fallback)
        df_feat = winsorize_returns(
            df_feat,
            df_feat["time"].max().strftime("%Y-%m-%d"),
            MODEL_CONFIG["clip_lower_q"],
            MODEL_CONFIG["clip_upper_q"],
        )

    # Extract buffers
    non_na = df_feat["ret_1d_clipped"].notna()
    ret_series = df_feat.loc[non_na, "ret_1d_clipped"].values.astype(float)
    vol_series = df_feat.loc[non_na, "vol_chg_clipped"].values.astype(float)
    close_series = df_feat.loc[non_na, "close"].values.astype(float)
    volume_series = df_feat.loc[non_na, "volume"].values.astype(float)
    time_series = df_feat.loc[non_na, "time"].values

    ret_buffer = list(ret_series[-20:])
    vol_buffer = list(vol_series[-5:])
    price_buffer = list(close_series[-20:])
    volume_buffer = list(volume_series[-20:])
    last_date = pd.Timestamp(time_series[-1])

    return ret_buffer, vol_buffer, price_buffer, volume_buffer, last_date
