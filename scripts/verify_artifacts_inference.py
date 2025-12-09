#!/usr/bin/env python3
"""
Verify exported artifacts by reloading them and computing MSE on test split.

This script mirrors tests/test_parity_patchtst.py but runs outside pytest.
It enforces device-specific artifact directory usage and NF inference path
to match the training pipeline.

Usage:
  python scripts/verify_artifacts_inference.py \
    --train data/raw/FPT_train.csv \
    --test data/test/FPT_test.csv \
    --horizon 100 \
    --artifacts app/models/artifacts/cpu_<fingerprint>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Ensure project root on sys.path when running as a script (after stdlib/3rd-party imports)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.patchtst_loader import PatchTSTLoader  # noqa: E402
from app.utils.device_detector import detect_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify PatchTST artifacts by reloading and scoring MSE"
    )
    p.add_argument(
        "--train", type=str, default="data/raw/FPT_train.csv", help="Path to training CSV"
    )
    p.add_argument("--test", type=str, default="data/test/FPT_test.csv", help="Path to test CSV")
    p.add_argument("--horizon", type=int, default=100, help="Forecast horizon")
    p.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help="Path to device-specific artifact dir (e.g., app/models/artifacts/cpu_xxx)",
    )
    return p.parse_args()


def _load_ground_truth(
    train_csv: Path, test_csv: Path, horizon: int
) -> tuple[list[float], np.ndarray]:
    df = pd.read_csv(train_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df_t_raw = (
        pd.read_csv(test_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    )
    df_t = (
        df_t_raw[df_t_raw["symbol"] == "FPT"].copy() if "symbol" in df_t_raw.columns else df_t_raw
    )
    last_date = df["time"].max()
    df_t = df_t[df_t["time"] > last_date].reset_index(drop=True)
    y_true = df_t.head(horizon)["close"].astype("float32").values
    close_hist = df["close"].astype(float).tolist()
    return close_hist, y_true


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train)
    test_csv = Path(args.test)
    horizon = args.horizon

    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(f"Train/Test CSV not found: {train_csv}, {test_csv}")

    # Always force NF inference to mirror training pipeline
    loader = PatchTSTLoader(models_dir=Path(args.artifacts) if args.artifacts else None)
    loader.set_use_nf_inference(True)
    assert loader.load(), "Failed to load artifacts"

    close_hist, y_true = _load_ground_truth(train_csv, test_csv, horizon)
    preds = loader.predict_prices(close_hist, horizon=horizon)
    preds = np.asarray(preds, dtype=float).reshape(-1)[: len(y_true)]

    mse = mean_squared_error(y_true, preds)
    bias = float(np.mean(preds - y_true))

    device_info = detect_device()
    result = {
        "device_type": device_info["device_type"],
        "device_name": device_info.get("device_name", "unknown"),
        "artifact_dir": str(loader.models_dir),
        "horizon": horizon,
        "mse": float(mse),
        "bias": bias,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
