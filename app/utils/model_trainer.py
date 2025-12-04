"""
Model Trainer - PatchTST + Post-processing + Smooth Bias Correction (v2 canonical)
- Uses fixed hyperparameters from config (no Optuna)
- Trains on FPT_train.csv (data/raw)
- Exports artifacts for runtime inference (Plan A compatible):
  - patchtst.pt (state_dict)
  - best_params.json
  - post_model.pkl
  - smooth_config.json
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from app.config import DATA_DIR, MODEL_PATHS, PATCHTST_PARAMS

try:
    from neuralforecast import NeuralForecast  # type: ignore
    from neuralforecast.models import PatchTST  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "neuralforecast is required for training v2. Please install dependencies."
    ) from e


def load_training_series() -> tuple[pd.Series, pd.DatetimeIndex]:
    # Look for a CSV in data/raw containing 'train' per project convention
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError("data/raw directory not found")
    csvs = [p for p in raw_dir.glob("*.csv") if "train" in p.name.lower()]
    if not csvs:
        raise FileNotFoundError("No dataset found in data/raw (CSV containing 'train' in name)")
    df = pd.read_csv(csvs[0], parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("Dataset must contain 'close' column")
    return df["close"].astype("float32"), df["time"]


def train_patchtst_and_export() -> bool:
    try:
        close, dates = load_training_series()
        input_size = int(PATCHTST_PARAMS["input_size"])  # noqa
        horizon = int(PATCHTST_PARAMS["horizon"])  # noqa

        # Build model and fit via NeuralForecast (handles internals, RevIN, etc.)
        model = PatchTST(
            h=horizon,
            input_size=input_size,
            patch_len=int(PATCHTST_PARAMS["patch_len"]),
            stride=int(PATCHTST_PARAMS["stride"]),
            revin=bool(PATCHTST_PARAMS["revin"]),
            learning_rate=float(PATCHTST_PARAMS["learning_rate"]),
            max_steps=int(PATCHTST_PARAMS["max_steps"]),
            val_check_steps=10,
        )

        df_nf = pd.DataFrame(
            {
                "unique_id": "FPT",
                "ds": pd.date_range(start=dates.iloc[0], periods=len(close), freq="D"),
                "y": close.values,
            }
        )
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=df_nf, val_size=0)

        # Export state_dict (Torch weights)
        module = getattr(model, "model", model)
        state = module.state_dict()
        MODEL_PATHS["patchtst_ckpt"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, MODEL_PATHS["patchtst_ckpt"])  # patchtst.pt

        # Save fixed hparams
        with open(MODEL_PATHS["best_params"], "w", encoding="utf-8") as f:
            json.dump(PATCHTST_PARAMS, f, indent=2)

        # Build baseline forecast using NF predict (robust to internal shapes)
        forecast = nf.predict()
        pred_col = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]
        baseline = forecast[pred_col].values.astype(float)
        baseline = baseline[:horizon]

        # Train post-processing LinearRegression using folds with NF per-fold fit
        tscv = TimeSeriesSplit(n_splits=3)
        series = close.values.astype(float)
        X_post, y_post = [], []

        idxs = np.arange(len(series))
        for train_idx, val_idx in tscv.split(idxs):
            # Require enough history and at least some validation points
            if len(train_idx) < input_size or len(val_idx) == 0:
                continue
            # Build NF dataset for fold
            ds_train = pd.date_range(
                start=dates.iloc[train_idx[0]], periods=len(train_idx), freq="D"
            )
            df_train = pd.DataFrame({"unique_id": "FPT", "ds": ds_train, "y": series[train_idx]})

            h_fold = min(horizon, len(val_idx))
            model_fold = PatchTST(
                h=h_fold,
                input_size=input_size,
                patch_len=int(PATCHTST_PARAMS["patch_len"]),
                stride=int(PATCHTST_PARAMS["stride"]),
                revin=bool(PATCHTST_PARAMS["revin"]),
                learning_rate=float(PATCHTST_PARAMS["learning_rate"]),
                max_steps=int(PATCHTST_PARAMS["max_steps"]),
                val_check_steps=10,
            )
            nf_fold = NeuralForecast(models=[model_fold], freq="D")
            nf_fold.fit(df=df_train, val_size=0)
            forecast_fold = nf_fold.predict()
            pred_col_fold = [c for c in forecast_fold.columns if c not in ("unique_id", "ds")][0]
            y_pred = forecast_fold[pred_col_fold].values.astype(float)
            y_true = series[val_idx][: len(y_pred)]

            X_post.extend(y_pred.reshape(-1, 1))
            y_post.extend(y_true)

        X_post = np.array(X_post)
        y_post = np.array(y_post)
        if len(X_post) == 0:
            post_model = LinearRegression()
            post_model.coef_ = np.array([1.0])
            post_model.intercept_ = 0.0
        else:
            post_model = LinearRegression().fit(X_post, y_post)

        joblib.dump(post_model, MODEL_PATHS["post_model"])  # post_model.pkl

        # Save smooth config
        smooth_cfg = {"method": "linear", "smooth_ratio": 0.2}
        with open(MODEL_PATHS["smooth_config"], "w", encoding="utf-8") as f:
            json.dump(smooth_cfg, f, indent=2)

        print("✅ Exported artifacts to:", MODEL_PATHS["patchtst_ckpt"].parent)
        return True
    except Exception as e:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print(f"❌ Training v2 failed: {e}")
        return False


if __name__ == "__main__":
    ok = train_patchtst_and_export()
    raise SystemExit(0 if ok else 1)
