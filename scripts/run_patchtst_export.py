#!/usr/bin/env python3
"""
Run PatchTST (fixed params) end-to-end:
- Optional deterministic setup for CUDA (torch + cuBLAS)
- Env/versions snapshot (JSON)
- Load train/test CSV
- Train PatchTST baseline on (train+val)
- Train post-processing LinearRegression via TimeSeriesSplit
- Apply Smooth Linear 20% + post-processing (best method)
- Report metrics (baseline/post/smooth)
- Export artifacts: patchtst_full.pt, patchtst.pt, best_params.json,
  post_model.pkl, smooth_config.json
- Print checksums for exported files

Usage (local):
  python scripts/run_patchtst_export.py \
      --train data/raw/FPT_train.csv \
      --test  data/test/FPT_test.csv \
      --out   app/models/artifacts \
      --deterministic --workspace-config :4096:8

Usage (Colab, if CSVs missing, script will auto-download):
  python scripts/run_patchtst_export.py --out /content/artifacts \
      --deterministic --workspace-config :4096:8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# -------------------------- CLI & helpers --------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PatchTST fixed pipeline and export artifacts")
    p.add_argument(
        "--train", type=str, default="data/raw/FPT_train.csv", help="Path to training CSV"
    )
    p.add_argument("--test", type=str, default="data/test/FPT_test.csv", help="Path to test CSV")
    p.add_argument(
        "--out", type=str, default="app/models/artifacts", help="Output directory for artifacts"
    )
    p.add_argument("--horizon", type=int, default=100, help="Forecast horizon")
    p.add_argument(
        "--deterministic", action="store_true", help="Enable torch deterministic algorithms"
    )
    p.add_argument(
        "--workspace-config",
        type=str,
        default=":4096:8",
        help="CUBLAS_WORKSPACE_CONFIG value if deterministic",
    )
    p.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download CSVs from Google Drive if not found",
    )
    return p.parse_args()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------- Env snapshot --------------------------


def env_snapshot() -> dict:
    out: dict = {"python": platform.python_version(), "platform": platform.platform(), "libs": {}}
    try:
        import torch  # type: ignore

        cuda = {
            "available": torch.cuda.is_available(),
            "version": getattr(torch.version, "cuda", "N/A"),
        }
        if torch.cuda.is_available():
            cuda["device_name"] = torch.cuda.get_device_name(0)
            try:
                import torch.backends.cudnn as cudnn  # type: ignore

                cuda["cudnn"] = {
                    "version": cudnn.version(),
                    "deterministic": getattr(cudnn, "deterministic", None),
                    "benchmark": getattr(cudnn, "benchmark", None),
                }
            except Exception as e:  # pragma: no cover
                cuda["cudnn_error"] = str(e)
        out["torch"] = {"version": torch.__version__, "cuda": cuda}
    except Exception as e:
        out["torch_error"] = str(e)

    for m in [
        "neuralforecast",
        "pytorch_lightning",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "joblib",
        "optuna",
    ]:
        try:
            mod = importlib.import_module(m)
            out["libs"][m] = getattr(mod, "__version__", "N/A")
        except Exception:
            out["libs"][m] = "missing"
    return out


def configure_determinism(enable: bool, workspace_cfg: str) -> None:
    if not enable:
        return
    # cuBLAS determinism requirement with CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace_cfg)
    try:
        import torch  # type: ignore
        import torch.backends.cudnn as cudnn  # type: ignore

        torch.use_deterministic_algorithms(True)
        cudnn.benchmark = False
        cudnn.deterministic = True
        print(
            f"[DETERMINISM] Enabled. CUBLAS_WORKSPACE_CONFIG={os.getenv('CUBLAS_WORKSPACE_CONFIG')}"
        )
    except Exception as e:  # pragma: no cover
        print("[DETERMINISM] Warning:", e)


# -------------------------- Data loading --------------------------


def maybe_download(train_path: Path, test_path: Path) -> None:
    # Google Drive IDs used in your notebooks
    TRAIN_ID = "1nS9xshut38SJEX__PD_zjKFtj2CQCn7S"
    TEST_ID = "1IkzoSTHPMnOUBILN7cCPjVw9QWAuOtCs"
    try:
        import gdown  # type: ignore

        if not train_path.exists():
            train_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                f"https://drive.google.com/uc?id={TRAIN_ID}", str(train_path), quiet=False
            )
        if not test_path.exists():
            test_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={TEST_ID}", str(test_path), quiet=False)
    except Exception as e:  # pragma: no cover
        print("[DOWNLOAD] Skipped or failed:", e)


def load_data(
    train_csv: Path, test_csv: Path, download_if_missing: bool
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if download_if_missing:
        maybe_download(train_csv, test_csv)

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
    return df, df_t, df["close"].astype("float32").values


# -------------------------- Model pipeline --------------------------


def run_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int) -> dict:
    # Imports here to fail fast with a clear error if packages missing
    from neuralforecast import NeuralForecast  # type: ignore
    from neuralforecast.models import PatchTST  # type: ignore

    target_col = "close"
    y_true = test_df.head(horizon)[target_col].astype("float32").values

    # Fixed best params
    params = {
        "input_size": 100,
        "patch_len": 32,
        "stride": 4,
        "learning_rate": 0.001610814898983045,
        "max_steps": 250,
        "revin": True,
        "horizon": horizon,
    }

    # Prepare NeuralForecast datasets (daily freq)
    close_vals = train_df[target_col].astype("float32").values
    T = len(close_vals)
    train_size = int(T * 0.8)
    val_size = int(T * 0.1)

    train_nf = pd.DataFrame(
        {
            "unique_id": "FPT",
            "ds": pd.date_range(start=train_df["time"].iloc[0], periods=train_size, freq="D"),
            "y": close_vals[:train_size],
        }
    )
    val_nf = pd.DataFrame(
        {
            "unique_id": "FPT",
            "ds": pd.date_range(
                start=train_df["time"].iloc[train_size], periods=val_size, freq="D"
            ),
            "y": close_vals[train_size : train_size + val_size],
        }
    )
    train_nf_full = pd.concat([train_nf, val_nf], ignore_index=True)

    # Train PatchTST baseline
    model_patchtst = PatchTST(
        h=horizon,
        input_size=params["input_size"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        revin=params["revin"],
        learning_rate=params["learning_rate"],
        max_steps=params["max_steps"],
        val_check_steps=10,
    )
    nf = NeuralForecast(models=[model_patchtst], freq="D")
    nf.fit(df=train_nf_full, val_size=0)
    forecast = nf.predict()
    pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
    pred_baseline = forecast[pred_col].values[: len(y_true)].astype("float32")

    # Post-processing via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    X_post, y_post = [], []
    for train_idx, val_idx in tscv.split(train_nf_full):
        fold_train = train_nf_full.iloc[train_idx]
        fold_val = train_nf_full.iloc[val_idx]
        h_fold = min(horizon, len(fold_val))

        fold_model = PatchTST(
            h=h_fold,
            input_size=params["input_size"],
            patch_len=params["patch_len"],
            stride=params["stride"],
            revin=params["revin"],
            learning_rate=params["learning_rate"],
            max_steps=params["max_steps"],
            val_check_steps=10,
        )
        nf_fold = NeuralForecast(models=[fold_model], freq="D")
        nf_fold.fit(df=fold_train, val_size=0)
        f_fold = nf_fold.predict()
        c_fold = [c for c in f_fold.columns if c not in ["unique_id", "ds"]][0]
        pred_fold = f_fold[c_fold].values[: len(fold_val)]
        true_fold = fold_val["y"].values[: len(pred_fold)]
        X_post.extend(pred_fold.reshape(-1, 1))
        y_post.extend(true_fold)

    X_post = np.array(X_post)
    y_post = np.array(y_post)
    post_model = LinearRegression().fit(X_post, y_post)
    pred_post = post_model.predict(pred_baseline.reshape(-1, 1))

    # Smooth Linear 20%
    def smooth_linear_20(baseline: np.ndarray, post: np.ndarray, ratio: float = 0.2) -> np.ndarray:
        n = len(baseline)
        split = max(1, min(n - 1, int(n * ratio)))
        out = baseline.copy()
        if split > 1:
            w = np.linspace(0.0, 1.0, split)
            out[:split] = (1 - w) * baseline[:split] + w * post[:split]
            out[0] = baseline[0]
            out[split - 1] = post[split - 1]
        if split < n:
            out[split:] = post[split:]
        return out

    pred_smooth = smooth_linear_20(pred_baseline, pred_post, 0.2)

    def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        return {"mse": float(mean_squared_error(y, p)), "bias": float(np.mean(p - y))}

    res = {
        "params": params,
        "baseline": metrics(y_true, pred_baseline),
        "post": metrics(y_true, pred_post),
        "smooth20": metrics(y_true, pred_smooth),
        "objects": {
            "model_patchtst": model_patchtst,
            "post_model": post_model,
            "pred_baseline": pred_baseline,
            "pred_post": pred_post,
            "pred_smooth": pred_smooth,
        },
    }
    return res


# -------------------------- Export --------------------------


def export_artifacts(out_dir: Path, res: dict) -> dict:
    import joblib  # type: ignore
    import torch  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    model = res["objects"]["model_patchtst"]
    module = getattr(model, "model", model)
    # state_dict
    torch.save(module.state_dict(), out_dir / "patchtst.pt")
    # full module (best parity; optional)
    try:
        torch.save(module, out_dir / "patchtst_full.pt")
    except Exception as e:  # pragma: no cover
        print("[EXPORT] Warning saving full module:", e)

    # best params
    best_params = res["params"].copy()
    with (out_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # post model
    joblib.dump(res["objects"]["post_model"], out_dir / "post_model.pkl")

    # smooth config
    with (out_dir / "smooth_config.json").open("w", encoding="utf-8") as f:
        json.dump({"method": "linear", "smooth_ratio": 0.2}, f, indent=2)

    # report sizes + sha256
    report = {}
    for name in [
        "patchtst_full.pt",
        "patchtst.pt",
        "best_params.json",
        "post_model.pkl",
        "smooth_config.json",
    ]:
        p = out_dir / name
        if p.exists():
            report[name] = {"size": p.stat().st_size, "sha256": sha256(p)}
    return report


# -------------------------- Main --------------------------


def main() -> None:
    args = parse_args()

    # Determinism (must be set before torch ops)
    configure_determinism(args.deterministic, args.workspace_config)

    # Print env snapshot
    snap = env_snapshot()
    print("\n=== ENV SNAPSHOT ===")
    print(json.dumps(snap, indent=2, ensure_ascii=False))

    # Data
    train_csv = Path(args.train)
    test_csv = Path(args.test)
    df, df_test, _ = load_data(train_csv, test_csv, args.download_if_missing)

    print("\n=== DATA REPORT ===")
    train_range = f"{df['time'].min()}→{df['time'].max()}"
    print(f"Train rows={len(df)} range={train_range} | " f"Test(after train) rows={len(df_test)}")

    # Run pipeline
    print("\n=== TRAIN & PREDICT ===")
    res = run_pipeline(df, df_test, args.horizon)

    # Metrics
    print("\n=== METRICS ===")
    print(
        f"Baseline: MSE={res['baseline']['mse']:.4f}, Bias={res['baseline']['bias']:.4f}\n"
        f"Post:     MSE={res['post']['mse']:.4f},     Bias={res['post']['bias']:.4f}\n"
        f"Smooth20: MSE={res['smooth20']['mse']:.4f}, Bias={res['smooth20']['bias']:.4f}"
    )

    # Export
    out_dir = Path(args.out)
    report = export_artifacts(out_dir, res)
    print("\n=== EXPORTED ARTIFACTS ===")
    print(json.dumps(report, indent=2))

    # Session summary (compact JSON)
    summary = {
        "metrics": {
            "baseline": res["baseline"],
            "post": res["post"],
            "smooth20": res["smooth20"],
        },
        "artifacts": report,
    }
    print("\n=== SESSION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
