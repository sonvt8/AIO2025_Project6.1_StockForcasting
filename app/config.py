"""
Configuration settings for FPT Stock Prediction API
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model artifacts directory
MODELS_DIR = BASE_DIR / "app" / "models" / "artifacts"
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "raw" / "FPT_train.csv"

# Model configuration (from V6 baseline)
MODEL_CONFIG = {
    "seed": 42,
    "val_start": "2025-01-01",
    "forecast_steps": 100,
    "clip_lower_q": 0.01,
    "clip_upper_q": 0.99,
}

# Base features (from V4)
BASE_FEATURE_NAMES = [
    "ret_1d_clipped",
    "vol_chg_clipped",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_lag5",
    "ret_lag6",
    "ret_lag7",
    "ret_lag8",
    "ret_lag9",
    "ret_lag10",
    "vol_lag1",
    "vol_lag2",
    "vol_lag3",
    "vol_lag4",
    "vol_lag5",
    "vol_5",
    "vol_10",
    "vol_20",
    "ret_roll_min_20",
    "ret_roll_max_20",
    "ret_z_20",
    "mean_ret_5",
    "mean_ret_10",
    "mean_ret_20",
    "sma10",
    "sma20",
    "price_trend_10",
    "price_trend_20",
    "rsi_14",
    "bb_width_20",
    "dow",
    "month",
]

# V6: Selective features - chỉ 5 features tốt nhất
SELECTIVE_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "roc_10",  # Rate of Change 10 days
    "roc_20",  # Rate of Change 20 days
    "momentum_10",  # Price momentum 10 days
    "momentum_20",  # Price momentum 20 days
    "volume_ratio",  # Current volume / average volume
]

FEATURE_NAMES = SELECTIVE_FEATURE_NAMES  # V6: Dùng selective features

# API Configuration
API_CONFIG = {
    "title": "FPT Stock Price Prediction API",
    "description": "Predict FPT stock prices with an ElasticNet selective-features baseline",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Model file paths (will be created after training)
MODEL_PATHS = {
    "elasticnet": MODELS_DIR / "elasticnet_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
    "calibration": MODELS_DIR / "calibration_model.pkl",
    "config": MODELS_DIR / "model_config.json",
}
