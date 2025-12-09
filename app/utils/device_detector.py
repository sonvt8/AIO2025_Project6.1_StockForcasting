"""
Device detection and fingerprinting for artifact management.
Auto-detects CUDA/MPS/CPU and creates device-specific artifact paths.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch


def detect_device() -> dict[str, str]:
    """
    Detect current compute device and return device info dict.

    Returns:
        Dict with keys: device_type (cuda/mps/cpu), device_name, fingerprint
    """
    device_type = "cpu"
    device_name = "cpu"

    if torch.cuda.is_available():
        device_type = "cuda"
        try:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = getattr(torch.version, "cuda", "unknown")
            device_name = f"{device_name}_cuda{cuda_version}"
        except Exception:
            device_name = "cuda_unknown"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        device_name = "mps"

    return {
        "device_type": device_type,
        "device_name": device_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def get_device_fingerprint() -> str:
    """
    Create a unique fingerprint for the current device/environment.
    Used to cache artifacts per device type.

    Returns:
        Short hash string identifying device type (e.g., "cuda", "mps", "cpu")
    """
    device_info = detect_device()
    # Use device_type as primary fingerprint (CUDA/MPS/CPU)
    # This ensures artifacts trained on GPU are separate from CPU
    fingerprint_data = json.dumps(
        {
            "device_type": device_info["device_type"],
            "device_name": device_info["device_name"],
        },
        sort_keys=True,
    )
    return hashlib.md5(fingerprint_data.encode()).hexdigest()[:8]


def get_device_artifact_dir(base_dir: Path) -> Path:
    """
    Get device-specific artifact directory.

    Args:
        base_dir: Base artifacts directory (e.g., app/models/artifacts)

    Returns:
        Path to device-specific subdirectory (e.g., app/models/artifacts/cuda_xxx)
    """
    device_info = detect_device()
    device_type = device_info["device_type"]
    fingerprint = get_device_fingerprint()
    # Create subdirectory: artifacts/{device_type}_{fingerprint}/
    device_dir = base_dir / f"{device_type}_{fingerprint}"
    return device_dir
