"""
Model Service
Wrapper for model operations
"""

from app.models.patchtst_loader import get_patchtst_loader
from app.services.forecast_service import ForecastService


class ModelService:
    """Service for model operations (v2 - PatchTST)"""

    def __init__(self):
        self.model_loader = get_patchtst_loader()
        self.forecast_service = ForecastService()

    def ensure_models_loaded(self):
        """Ensure models are loaded"""
        if not self.model_loader.is_loaded():
            success = self.model_loader.load()
            if not success:
                raise RuntimeError("Failed to load PatchTST artifacts in app/models/artifacts/.")

    def get_model_info(self):
        """Get model information"""
        self.ensure_models_loaded()
        hparams = getattr(self.model_loader, "hparams", {}) or {}
        return {
            "status": "loaded",
            "model_type": "PatchTST",
            "features_count": None,
            "config": hparams,
        }
