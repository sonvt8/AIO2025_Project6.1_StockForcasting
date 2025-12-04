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


def find_dataset_file() -> Path | None:
    """
    Find the dataset file in data/raw/ directory.

    Logic:
    - Only searches in data/raw/ directory
    - Looks for CSV files containing "train" in filename (case-insensitive)
    - Returns the first matching file found

    Returns:
        Path to dataset file if found, None otherwise
    """
    raw_dir = DATA_DIR / "raw"

    if not raw_dir.exists():
        print(f"[INFO] Directory {raw_dir} does not exist. No dataset file found.")
        return None

    # Search for CSV files containing "train" in filename
    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        print(f"[INFO] No CSV files found in {raw_dir}")
        return None

    # Find files with "train" in name (case-insensitive)
    train_files = [f for f in csv_files if "train" in f.name.lower()]

    if train_files:
        # Return the first matching file
        dataset_file = train_files[0]
        print(f"[INFO] Found dataset file: {dataset_file.name} (contains 'train' in filename)")
        return dataset_file
    else:
        print(
            f"[INFO] No CSV file with 'train' in filename found in {raw_dir}. "
            f"Available files: {[f.name for f in csv_files]}"
        )
        return None


# Model configuration for v2 (PatchTST)
MODEL_CONFIG = {
    "seed": 42,
    "forecast_steps": 100,
}

# PatchTST best hyperparameters (fixed from notebook)
PATCHTST_PARAMS = {
    "input_size": 100,
    "patch_len": 32,
    "stride": 4,
    "learning_rate": 0.001610814898983045,
    "max_steps": 250,
    "revin": True,
    "horizon": 100,
}

# API Configuration
API_CONFIG = {
    "title": "FPT Stock Price Prediction API",
    "description": "Predict FPT prices with PatchTST + post-processing (smooth bias)",
    "version": "2.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Model file paths (v2)
MODEL_PATHS = {
    "patchtst_ckpt": MODELS_DIR / "patchtst.pt",  # state_dict of trained model
    "best_params": MODELS_DIR / "best_params.json",
    "post_model": MODELS_DIR / "post_model.pkl",
    "smooth_config": MODELS_DIR / "smooth_config.json",
    # "revin_stats": MODELS_DIR / "revin_stats.json",  # If needed later
}

# GitHub Releases configuration (Plan A)
GITHUB_RELEASE = {
    "owner": "sonvt8",
    "repo": "AIO2025_Project6.1_StockForcasting",
    "tag": "version-2.0-patchtst",
    "assets": {
        "patchtst_ckpt": "patchtst.pt",
        "best_params": "best_params.json",
        "post_model": "post_model.pkl",
        "smooth_config": "smooth_config.json",
    },
}
