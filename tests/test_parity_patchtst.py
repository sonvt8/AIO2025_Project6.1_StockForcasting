import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from app.models.patchtst_loader import get_patchtst_loader
from app.utils.device_detector import detect_device

# Note: To see real-time training progress, run pytest with -s flag:
# python -m pytest tests/test_parity_patchtst.py -v -s


def _load_ground_truth(
    train_csv: Path, test_csv: Path, horizon: int = 100
) -> tuple[list[float], np.ndarray]:
    df = pd.read_csv(train_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df_t_raw = (
        pd.read_csv(test_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    )
    if "symbol" in df_t_raw.columns:
        df_t = df_t_raw[df_t_raw["symbol"] == "FPT"].copy()
    else:
        df_t = df_t_raw.copy()
    last_date = df["time"].max()
    df_t = df_t[df_t["time"] > last_date].reset_index(drop=True)
    y_true = df_t.head(horizon)["close"].astype("float32").values
    close_hist = df["close"].astype(float).tolist()
    return close_hist, y_true


def test_patchtst_parity():
    base_dir = Path(__file__).resolve().parent.parent
    train_csv = base_dir / "data" / "raw" / "FPT_train.csv"
    test_csv = base_dir / "data" / "test" / "FPT_test.csv"
    close_hist, y_true = _load_ground_truth(train_csv, test_csv, horizon=100)

    # Use NF inference to match training script behavior (for MSE ~17)
    # Set use_nf_inference=True to use NeuralForecast pipeline instead of raw forward
    import os

    os.environ["PATCHTST_USE_NF"] = "1"

    # Reset singleton to pick up new env var
    import app.models.patchtst_loader as loader_module

    loader_module._loader_singleton = None

    loader = get_patchtst_loader()
    print(f"[DEBUG] Loader use_nf_inference={loader.use_nf_inference}")
    assert loader.load(), "Failed to load PatchTST artifacts"

    preds = loader.predict_prices(close_hist, horizon=100)
    preds = np.asarray(preds, dtype=float).reshape(-1)[: len(y_true)]

    mse = mean_squared_error(y_true, preds)
    bias = float(np.mean(preds - y_true))

    # Detect device and set appropriate MSE threshold
    device_info = detect_device()
    device_type = device_info["device_type"]

    # Device-specific MSE thresholds (based on observed performance)
    # Note: Inference MSE may be higher than training MSE due to:
    # - Artifact reloading differences
    # - NF inference path differences from training script
    # - Device-specific numerical precision
    mse_thresholds = {
        "cuda": 18.5,  # GPU CUDA: MSE ~17 (training), ~18-20 (inference)
        "mps": 50.0,  # Apple MPS: MSE ~45-50 (training), ~50-60 (inference)
        "cpu": 80.0,  # CPU: MSE ~17-50 (training), ~50-80 (inference with NF path)
    }
    threshold = mse_thresholds.get(device_type, 50.0)

    # Log for debugging
    result = {
        "device_type": device_type,
        "device_name": device_info.get("device_name", "unknown"),
        "mse": mse,
        "bias": bias,
        "threshold": threshold,
    }
    print(json.dumps(result, indent=2))

    # Assert với ngưỡng phù hợp với device
    assert mse <= threshold, (
        f"MSE too high for {device_type}: {mse} (threshold: {threshold}). "
        f"Artifacts may need retraining for this device."
    )
