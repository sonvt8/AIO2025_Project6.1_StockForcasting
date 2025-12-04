"""
PatchTST Loader (v2)
Loads trained PatchTST weights and provides inference without retraining.
Artifacts expected (Plan A - auto-downloaded if missing):
- patchtst.pt (state_dict of trained model)
- best_params.json (hyperparameters)
- post_model.pkl (LinearRegression for post-processing)
- smooth_config.json ({method, smooth_ratio})
Optionally:
- revin_stats.json (if required by the model configuration)

Inference contract:
- predict_sequence(close_history: list[float], horizon: int) -> np.ndarray[float]
  Uses the last input_size points from close_history to produce horizon raw predictions
  from the neural model (before post-processing & smoothing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import torch

from app.config import GITHUB_RELEASE, MODEL_PATHS, PATCHTST_PARAMS

try:
    from neuralforecast.models import PatchTST  # type: ignore

    _NF_AVAILABLE = True
except Exception:  # pragma: no cover
    _NF_AVAILABLE = False


@dataclass
class SmoothConfig:
    method: str = "linear"
    smooth_ratio: float = 0.2


class PatchTSTLoader:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODEL_PATHS["patchtst_ckpt"].parent
        self.model: torch.nn.Module | None = None
        self.hparams: dict[str, Any] | None = None
        self.post_model = None
        self.smooth_cfg = SmoothConfig()
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def _maybe_download_artifacts(self) -> None:
        """Plan A: Auto-download artifacts from GitHub Releases if missing (public repo)."""
        owner = GITHUB_RELEASE.get("owner")
        repo = GITHUB_RELEASE.get("repo")
        tag = GITHUB_RELEASE.get("tag")
        assets = GITHUB_RELEASE.get("assets", {})

        to_download: list[tuple[str, Path]] = []
        for key, asset_name in assets.items():
            local_path = MODEL_PATHS.get(key)
            if local_path is None:
                continue
            if not local_path.exists():
                to_download.append((asset_name, local_path))

        for asset_name, local_path in to_download:
            url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset_name}"
            try:
                print(f"[INFO] Downloading artifact: {asset_name} from {url}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    local_path.write_bytes(resp.content)
                    print(f"[INFO] Saved artifact to {local_path}")
                else:
                    print(
                        f"[WARNING] Failed to download {asset_name} (HTTP {resp.status_code}). "
                        f"Please place it manually at {local_path}."
                    )
            except Exception as e:
                print(
                    f"[WARNING] Error downloading {asset_name}: {e}. "
                    f"Please place it manually at {local_path}."
                )

    def load(self) -> bool:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._maybe_download_artifacts()
        try:
            # Load hparams
            if MODEL_PATHS["best_params"].exists():
                with open(MODEL_PATHS["best_params"], encoding="utf-8") as f:
                    self.hparams = json.load(f)
            else:
                # Fallback to config defaults
                self.hparams = PATCHTST_PARAMS.copy()

            # Build model skeleton
            if not _NF_AVAILABLE:
                raise RuntimeError(
                    "neuralforecast not installed. Install 'neuralforecast' to use PatchTST."
                )

            h = int(self.hparams.get("horizon", PATCHTST_PARAMS["horizon"]))
            self.model = PatchTST(
                h=h,
                input_size=int(self.hparams.get("input_size", PATCHTST_PARAMS["input_size"])),
                patch_len=int(self.hparams.get("patch_len", PATCHTST_PARAMS["patch_len"])),
                stride=int(self.hparams.get("stride", PATCHTST_PARAMS["stride"])),
                revin=bool(self.hparams.get("revin", PATCHTST_PARAMS["revin"])),
                learning_rate=float(
                    self.hparams.get("learning_rate", PATCHTST_PARAMS["learning_rate"])
                ),
                max_steps=int(self.hparams.get("max_steps", PATCHTST_PARAMS["max_steps"])),
                val_check_steps=10,
            )

            # Load weights
            ckpt_path = MODEL_PATHS["patchtst_ckpt"]
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing model weights: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            # Accept both pure state_dict or checkpoint-like dicts
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
                # Some Lightning checkpoints prefix with 'model.'; handle common cases
                cleaned = {}
                for k, v in state.items():
                    cleaned[k.split("model.")[-1]] = v
                state = cleaned
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

            # Load post model
            if MODEL_PATHS["post_model"].exists():
                self.post_model = joblib.load(MODEL_PATHS["post_model"])
            else:
                self.post_model = None

            # Load smooth config
            if MODEL_PATHS["smooth_config"].exists():
                with open(MODEL_PATHS["smooth_config"], encoding="utf-8") as f:
                    sc = json.load(f)
                self.smooth_cfg = SmoothConfig(
                    method=sc.get("method", "linear"),
                    smooth_ratio=float(sc.get("smooth_ratio", 0.2)),
                )

            self._loaded = True
            return True
        except Exception as e:  # pragma: no cover
            print(f"[ERROR] Failed to load PatchTST artifacts: {e}")
            self._loaded = False
            return False

    # --- Core helpers ---
    def _post_process(self, baseline: np.ndarray) -> np.ndarray:
        # Ensure 1D float baseline
        baseline = np.asarray(baseline, dtype=float).reshape(-1)
        if self.post_model is None:
            return baseline
        pred = self.post_model.predict(baseline.reshape(-1, 1))
        # scikit-learn returns shape (n, 1) for some regressors; flatten to 1D
        return np.asarray(pred, dtype=float).reshape(-1)

    def _smooth_linear_20(
        self, baseline: np.ndarray, post: np.ndarray, ratio: float = 0.2
    ) -> np.ndarray:
        n = len(baseline)
        split = max(1, min(n - 1, int(n * ratio)))
        out = baseline.copy()
        # linear weights from 0..1 over first split
        if split > 1:
            w = np.linspace(0.0, 1.0, split)
            out[:split] = (1 - w) * baseline[:split] + w * post[:split]
            out[0] = baseline[0]
            out[split - 1] = post[split - 1]
        if split < n:
            out[split:] = post[split:]
        return out

    # --- Public API ---
    def predict_prices(self, close_history: list[float], horizon: int) -> np.ndarray:
        """Return final prices after post-processing + smoothing (v2 best method)."""
        baseline = self.predict_sequence(close_history, horizon)
        baseline = np.asarray(baseline, dtype=float).reshape(-1)
        post = self._post_process(baseline)
        post = np.asarray(post, dtype=float).reshape(-1)
        smooth = self._smooth_linear_20(baseline, post, self.smooth_cfg.smooth_ratio)
        smooth = np.asarray(smooth, dtype=float).reshape(-1)
        # Clamp negatives to zero
        smooth = np.maximum(smooth, 0.0)
        return smooth

    def predict_sequence(self, close_history: list[float], horizon: int) -> np.ndarray:
        """Predict baseline prices for next horizon steps using last input_size window.
        This uses the loaded PatchTST forward pass without training.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.model is None:
            raise RuntimeError("Internal model is None.")

        input_size = (
            int(self.hparams.get("input_size", PATCHTST_PARAMS["input_size"]))
            if self.hparams
            else PATCHTST_PARAMS["input_size"]
        )
        h = (
            int(self.hparams.get("horizon", PATCHTST_PARAMS["horizon"]))
            if self.hparams
            else PATCHTST_PARAMS["horizon"]
        )
        if horizon != h:
            # For safety, limit to model horizon
            horizon = min(horizon, h)

        if len(close_history) < input_size:
            raise ValueError(f"Need at least {input_size} points, got {len(close_history)}")

        window = np.array(close_history[-input_size:], dtype=np.float32)
        # Expected PatchTST input shape: [batch, channels, seq_len]
        # Our univariate series => channels=1
        x = torch.from_numpy(window[None, None, :]).float()  # shape (1, 1, input_size)

        with torch.no_grad():
            # Many NF models output y_hat scaled as same unit as input
            # Attempt direct forward; fall back to attribute access if required
            try:
                y_hat = self.model(x)
            except Exception:
                # Some NF models wrap forward in .model or .network
                if hasattr(self.model, "model"):
                    y_hat = self.model.model(x)
                else:
                    raise

        # y_hat may have shape (batch, horizon) or (batch, 1, horizon)
        y_hat_np = y_hat.detach().cpu().numpy().astype(float)
        # Squeeze all singleton dims to get 1D horizon vector
        y_hat_np = np.squeeze(y_hat_np)
        if y_hat_np.ndim > 1:
            # Fallback: flatten
            y_hat_np = y_hat_np.reshape(-1)
        y_hat_np = y_hat_np[:horizon]
        return y_hat_np


# Global accessor
_loader_singleton: PatchTSTLoader | None = None


def get_patchtst_loader() -> PatchTSTLoader:
    global _loader_singleton
    if _loader_singleton is None:
        _loader_singleton = PatchTSTLoader()
    return _loader_singleton
