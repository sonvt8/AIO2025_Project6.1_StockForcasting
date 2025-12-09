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
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import torch

from app.config import GITHUB_RELEASE, MODEL_PATHS, PATCHTST_PARAMS
from app.utils.device_detector import detect_device, get_device_artifact_dir

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
    def __init__(
        self,
        models_dir: Path | None = None,
        strict_nf: bool = True,
        use_nf_inference: bool | None = None,
        auto_train: bool | None = None,
    ) -> None:
        # Auto-detect device and use device-specific artifact directory
        base_dir = models_dir or MODEL_PATHS["patchtst_ckpt"].parent
        self.device_info = detect_device()
        self.device_artifact_dir = get_device_artifact_dir(base_dir)
        self.models_dir = self.device_artifact_dir

        self.model: torch.nn.Module | None = None
        self.hparams: dict[str, Any] | None = None
        self.post_model = None
        self.smooth_cfg = SmoothConfig()
        self._loaded = False
        # strict_nf=True: nếu NF lỗi sẽ raise để tránh âm thầm lệch kết quả
        self.strict_nf = strict_nf
        # use_nf_inference: nếu False sẽ dùng raw forward (ưu tiên ổn định với artifact đã train);
        # có thể bật qua env PATCHTST_USE_NF=1
        if use_nf_inference is None:
            env_flag = os.getenv("PATCHTST_USE_NF", "").strip()
            self.use_nf_inference = env_flag.lower() in ("1", "true", "yes")
        else:
            self.use_nf_inference = use_nf_inference
        # auto_train: tự động train/export artifacts nếu thiếu cho device hiện tại
        if auto_train is None:
            env_flag = os.getenv("PATCHTST_AUTO_TRAIN", "1").strip()
            self.auto_train = env_flag.lower() in ("1", "true", "yes")
        else:
            self.auto_train = auto_train

    @staticmethod
    def _set_deterministic_seeds(seed: int = 42) -> None:
        """Cố định seed để hạn chế trôi số giữa các lần dự đoán."""
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Không ép deterministic algorithms vì trên CPU/MPS có thể hạn chế; giữ mặc định
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self._loaded

    def _maybe_download_artifacts(self) -> None:
        """Plan A: Auto-download artifacts from GitHub Releases if missing (public repo).
        Downloads to device-specific directory.
        """
        owner = GITHUB_RELEASE.get("owner")
        repo = GITHUB_RELEASE.get("repo")
        tag = GITHUB_RELEASE.get("tag")
        assets = GITHUB_RELEASE.get("assets", {})

        # Map asset names to local filenames
        asset_to_file = {
            "patchtst_ckpt": "patchtst.pt",
            "best_params": "best_params.json",
            "post_model": "post_model.pkl",
            "smooth_config": "smooth_config.json",
        }

        to_download: list[tuple[str, Path]] = []
        for key, asset_name in assets.items():
            local_filename = asset_to_file.get(key)
            if local_filename is None:
                continue
            local_path = self.models_dir / local_filename
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

    def _check_artifacts_exist(self) -> bool:
        """Check if all required artifacts exist in device-specific directory."""
        required = ["patchtst.pt", "best_params.json", "post_model.pkl", "smooth_config.json"]
        return all((self.models_dir / f).exists() for f in required)

    def _auto_train_if_needed(self) -> None:
        """Auto-train and export artifacts if missing for current device."""
        if not self.auto_train:
            return
        if self._check_artifacts_exist():
            device_type = self.device_info["device_type"]
            print(f"[INFO] Artifacts found for device {device_type}, skipping auto-train.")
            return

        # Use sys.__stderr__ to bypass pytest output capture
        print(f"\n{'='*70}", file=sys.__stderr__, flush=True)
        print(
            f"[AUTO-TRAIN] Artifacts missing for device: {self.device_info['device_type']}",
            file=sys.__stderr__,
            flush=True,
        )
        print(
            f"[AUTO-TRAIN] Device name: {self.device_info.get('device_name', 'unknown')}",
            file=sys.__stderr__,
            flush=True,
        )
        print(
            f"[AUTO-TRAIN] Artifact directory: {self.models_dir}", file=sys.__stderr__, flush=True
        )
        print("[AUTO-TRAIN] Starting training pipeline...", file=sys.__stderr__, flush=True)
        print(
            "[AUTO-TRAIN] This may take 5-15 minutes depending on your hardware.",
            file=sys.__stderr__,
            flush=True,
        )
        print(
            "[AUTO-TRAIN] Training PatchTST with 250 steps + post-processing...",
            file=sys.__stderr__,
            flush=True,
        )
        print("[AUTO-TRAIN] Progress messages will appear below:", file=sys.__stderr__, flush=True)
        print(f"{'='*70}\n", file=sys.__stderr__, flush=True)

        try:
            import subprocess

            script_path = Path(__file__).parent.parent.parent / "scripts" / "run_patchtst_export.py"
            train_csv = Path(__file__).parent.parent.parent / "data" / "raw" / "FPT_train.csv"
            test_csv = Path(__file__).parent.parent.parent / "data" / "test" / "FPT_test.csv"

            if not script_path.exists():
                print(f"[ERROR] Cannot auto-train: script not found at {script_path}")
                return
            if not train_csv.exists():
                print(f"[ERROR] Cannot auto-train: training data not found at {train_csv}")
                return
            if not test_csv.exists():
                print(f"[WARNING] Test data not found at {test_csv}, continuing anyway...")

            cmd = [
                sys.executable,
                str(script_path),
                "--train",
                str(train_csv),
                "--test",
                str(test_csv),
                "--out",
                str(self.models_dir),
                "--deterministic",
                "--workspace-config",
                ":4096:8",
            ]
            print(f"[AUTO-TRAIN] Command: {' '.join(cmd)}\n", file=sys.__stderr__, flush=True)

            # Run with real-time output streaming so user can see progress
            # Use Popen with line buffering to show output immediately, even in pytest
            # Write to sys.__stderr__ to bypass pytest's output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                universal_newlines=True,
                bufsize=1,  # Line buffered
                encoding="utf-8",
                errors="replace",  # Handle encoding errors gracefully
            )

            # Stream output line by line to original stderr (bypasses pytest capture)
            # This ensures users see progress even when pytest captures stdout/stderr
            try:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        # Write to original stderr to bypass pytest capture
                        sys.__stderr__.write(line)
                        sys.__stderr__.flush()
                        # Also try stdout for normal runs (may be captured by pytest)
                        try:
                            sys.__stdout__.write(line)
                            sys.__stdout__.flush()
                        except Exception:
                            pass
            except Exception:
                pass

            # Wait for process to complete
            returncode = process.wait(timeout=3600)
            result = type("Result", (), {"returncode": returncode})()

            print(f"\n{'='*70}", file=sys.__stderr__, flush=True)
            if result.returncode == 0:
                device_type = self.device_info["device_type"]
                print(
                    f"[AUTO-TRAIN] ✅ Completed successfully for {device_type}!",
                    file=sys.__stderr__,
                    flush=True,
                )
                print(
                    f"[AUTO-TRAIN] Artifacts saved to: {self.models_dir}",
                    file=sys.__stderr__,
                    flush=True,
                )
            else:
                print(
                    f"[AUTO-TRAIN] ❌ Failed with exit code {result.returncode}",
                    file=sys.__stderr__,
                    flush=True,
                )
                print(
                    "[AUTO-TRAIN] Please check the output above for errors.",
                    file=sys.__stderr__,
                    flush=True,
                )
            print(f"{'='*70}\n", file=sys.__stderr__, flush=True)
        except subprocess.TimeoutExpired:
            print("\n[ERROR] Auto-train timed out after 1 hour. Please train manually.")
        except Exception as e:
            print(f"\n[ERROR] Auto-train error: {e}")
            import traceback

            traceback.print_exc()

    def load(self) -> bool:
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Auto-train if artifacts missing and enabled
        self._auto_train_if_needed()

        # Try download from GitHub if still missing (fallback)
        if not self._check_artifacts_exist():
            self._maybe_download_artifacts()

        try:
            # Load hparams (use device-specific path)
            best_params_path = self.models_dir / "best_params.json"
            if best_params_path.exists():
                with open(best_params_path, encoding="utf-8") as f:
                    self.hparams = json.load(f)
            elif MODEL_PATHS["best_params"].exists():
                # Fallback to default location
                with open(MODEL_PATHS["best_params"], encoding="utf-8") as f:
                    self.hparams = json.load(f)
            else:
                # Fallback to config defaults
                self.hparams = PATCHTST_PARAMS.copy()

            # Prefer loading full saved module if available (includes scaler/temporal norm state)
            full_path = self.models_dir / "patchtst_full.pt"
            if full_path.exists():
                try:
                    # PyTorch >=2.9: allowlist NF classes and force weights_only=False
                    globs = []
                    try:
                        from neuralforecast.models.patchtst import PatchTST_backbone  # type: ignore

                        globs.append(PatchTST_backbone)
                    except Exception:
                        pass
                    try:
                        from neuralforecast.common._modules import (  # type: ignore
                            RevIN,
                            TemporalNorm,
                        )

                        globs.extend([RevIN, TemporalNorm])
                    except Exception:
                        pass

                    try:
                        from torch.serialization import (  # type: ignore
                            add_safe_globals,
                            safe_globals,
                        )

                        try:
                            if globs:
                                add_safe_globals(globs)
                        except Exception:
                            pass
                        try:
                            if globs:
                                with safe_globals(globs):
                                    self.model = torch.load(
                                        full_path, map_location="cpu", weights_only=False
                                    )
                            else:
                                self.model = torch.load(
                                    full_path, map_location="cpu", weights_only=False
                                )
                        except TypeError:
                            # Older torch without weights_only arg
                            if globs:
                                with safe_globals(globs):
                                    self.model = torch.load(full_path, map_location="cpu")
                            else:
                                self.model = torch.load(full_path, map_location="cpu")
                    except Exception:
                        # serialization helpers not available → direct load
                        try:
                            self.model = torch.load(
                                full_path, map_location="cpu", weights_only=False
                            )
                        except TypeError:
                            self.model = torch.load(full_path, map_location="cpu")

                    self.model.eval()
                    for p in self.model.parameters():
                        p.requires_grad_(False)
                    self._use_full_module = True
                except Exception as e:
                    msg = f"[WARNING] Failed to load full module (patchtst_full.pt): {e}."
                    print(f"{msg} Falling back to state_dict.")
                    self._use_full_module = False
            else:
                self._use_full_module = False

            if not self._use_full_module:
                # Build model skeleton (NeuralForecast backbone) and load weights
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

                # Load weights (use device-specific path)
                ckpt_path = self.models_dir / "patchtst.pt"
                if not ckpt_path.exists():
                    # Fallback to default location
                    ckpt_path = MODEL_PATHS["patchtst_ckpt"]
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing model weights: {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                # Accept both pure state_dict or checkpoint-like dicts
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                cleaned = {}
                if isinstance(state, dict):
                    for k, v in state.items():
                        cleaned[k.split("model.")[-1]] = v
                state = cleaned or state
                self.model.load_state_dict(state, strict=False)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)

            # Load post model (use device-specific path)
            post_model_path = self.models_dir / "post_model.pkl"
            if post_model_path.exists():
                self.post_model = joblib.load(post_model_path)
            elif MODEL_PATHS["post_model"].exists():
                # Fallback to default location
                self.post_model = joblib.load(MODEL_PATHS["post_model"])
            else:
                self.post_model = None

            # Load smooth config (use device-specific path)
            smooth_config_path = self.models_dir / "smooth_config.json"
            if smooth_config_path.exists():
                with open(smooth_config_path, encoding="utf-8") as f:
                    sc = json.load(f)
                self.smooth_cfg = SmoothConfig(
                    method=sc.get("method", "linear"),
                    smooth_ratio=float(sc.get("smooth_ratio", 0.2)),
                )
            elif MODEL_PATHS["smooth_config"].exists():
                # Fallback to default location
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

    def _predict_sequence_via_nf(self, close_history: list[float], horizon: int) -> np.ndarray:
        """Predict baseline using NeuralForecast pipeline with zero training steps.
        This mirrors the notebook/script behavior (datatable + internal transforms).
        """
        if not _NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed. Cannot use NF-based inference.")
        # ép CPU để tránh khác biệt GPU/MPS và giữ tính ổn định
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        torch.set_grad_enabled(False)
        # Lazy imports to avoid hard dependency at import time
        import pandas as pd  # type: ignore
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.models import PatchTST as NF_PatchTST  # type: ignore

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
        horizon = min(horizon, h)
        if len(close_history) < input_size:
            raise ValueError(f"Need at least {input_size} points, got {len(close_history)}")

        # Build minimal df covering the history (daily freq)
        series = np.asarray(close_history, dtype="float32")
        df = pd.DataFrame(
            {
                "unique_id": "FPT",
                "ds": pd.date_range(
                    start=pd.Timestamp.today().normalize() - pd.Timedelta(days=len(series) - 1),
                    periods=len(series),
                    freq="D",
                ),
                "y": series,
            }
        )

        # Create a fresh NF model with same hparams, load weights, set max_steps=0
        nf_model = NF_PatchTST(
            h=h,
            input_size=input_size,
            patch_len=int(self.hparams.get("patch_len", PATCHTST_PARAMS["patch_len"]))
            if self.hparams
            else PATCHTST_PARAMS["patch_len"],
            stride=int(self.hparams.get("stride", PATCHTST_PARAMS["stride"]))
            if self.hparams
            else PATCHTST_PARAMS["stride"],
            revin=bool(self.hparams.get("revin", PATCHTST_PARAMS["revin"]))
            if self.hparams
            else PATCHTST_PARAMS["revin"],
            learning_rate=float(self.hparams.get("learning_rate", PATCHTST_PARAMS["learning_rate"]))
            if self.hparams
            else PATCHTST_PARAMS["learning_rate"],
            max_steps=0,  # critical: don't retrain, only set up data/transforms
            val_check_steps=10,
        )
        # Load full LightningModule state (use device-specific path)
        ckpt_path = self.models_dir / "patchtst.pt"
        if not ckpt_path.exists():
            ckpt_path = MODEL_PATHS["patchtst_ckpt"]
        state = torch.load(ckpt_path, map_location="cpu")
        # Accept both pure state_dict or checkpoint-like dicts
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Align keys to NF LightningModule expectation.
        if isinstance(state, dict):
            keys = list(state.keys())
            if any(k.startswith("model.") for k in keys):
                mapped = state  # already matches
            elif any(k.startswith("backbone.") for k in keys):
                mapped = {f"model.{k}": v for k, v in state.items()}
            else:
                mapped = state  # fallback
        else:
            mapped = state
        nf_model.load_state_dict(mapped, strict=False)

        nf = NeuralForecast(models=[nf_model], freq="D")
        # fit with 0 steps just to materialize datamodule/transforms
        nf.fit(df=df, val_size=0)
        forecast = nf.predict()
        col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
        y_hat = forecast[col].values.astype(float)
        return y_hat[:horizon]

    def predict_sequence(self, close_history: list[float], horizon: int) -> np.ndarray:
        """Predict baseline prices using NF pipeline for parity with notebook.
        Falls back to raw forward if NF path fails.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.model is None:
            raise RuntimeError("Internal model is None.")
        # đặt seed để hạn chế trôi số giữa các lần gọi
        self._set_deterministic_seeds(42)
        # Chọn đường suy luận: ưu tiên raw forward (ổn định) trừ khi bật NF
        if self.use_nf_inference:
            print("[DEBUG] Using NF inference path (use_nf_inference=True)")
            try:
                return self._predict_sequence_via_nf(close_history, horizon)
            except Exception as e:
                if self.strict_nf:
                    raise RuntimeError(f"NF inference failed (strict_nf=True): {e}") from e
                # nếu strict_nf=False thì rơi xuống raw forward

        # Raw forward path (đã load trọng số từ artifact)
        print("[DEBUG] Using raw forward path (use_nf_inference=False)")
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
            horizon = min(horizon, h)
        if len(close_history) < input_size:
            raise ValueError(f"Need at least {input_size} points, got {len(close_history)}")
        window = np.array(close_history[-input_size:], dtype=np.float32)
        x = torch.from_numpy(window[None, None, :]).float()
        with torch.no_grad():
            try:
                y_hat = self.model(x)
            except Exception:
                if hasattr(self.model, "model"):
                    y_hat = self.model.model(x)
                else:
                    raise
        y_hat_np = y_hat.detach().cpu().numpy().astype(float)
        y_hat_np = np.squeeze(y_hat_np)
        if y_hat_np.ndim > 1:
            y_hat_np = y_hat_np.reshape(-1)
        return y_hat_np[:horizon]


# Global accessor
_loader_singleton: PatchTSTLoader | None = None


def get_patchtst_loader() -> PatchTSTLoader:
    global _loader_singleton
    if _loader_singleton is None:
        _loader_singleton = PatchTSTLoader()
    return _loader_singleton
