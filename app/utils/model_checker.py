"""
Utility to check if models are available
"""

from app.config import MODEL_PATHS


def check_models_exist() -> tuple[bool, list[str]]:
    """
    Check if all required model files exist

    Returns:
        Tuple of (all_exist: bool, missing_files: list[str])
    """
    required_files = {
        "elasticnet": MODEL_PATHS["elasticnet"],
        "scaler": MODEL_PATHS["scaler"],
        "calibration": MODEL_PATHS["calibration"],
        "config": MODEL_PATHS["config"],
    }

    missing = []
    for _, path in required_files.items():
        if not path.exists():
            missing.append(str(path))

    return len(missing) == 0, missing


def print_model_status():
    """Print status of model files"""
    all_exist, missing = check_models_exist()

    if all_exist:
        print("✅ All model files are present")
        print(f"  - {MODEL_PATHS['elasticnet']}")
        print(f"  - {MODEL_PATHS['scaler']}")
        print(f"  - {MODEL_PATHS['calibration']}")
        print(f"  - {MODEL_PATHS['config']}")
    else:
        print("❌ Missing model files:")
        for file in missing:
            print(f"  - {file}")
        print("\nTo create models, run: python export_models.py")

    return all_exist
