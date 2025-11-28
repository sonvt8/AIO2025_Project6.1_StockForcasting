"""
Model Service
Wrapper for model operations
"""

from app.models.model_loader import get_model_loader
from app.services.forecast_service import ForecastService


class ModelService:
    """Service for model operations"""

    def __init__(self):
        self.model_loader = get_model_loader()
        self.forecast_service = ForecastService()

    def ensure_models_loaded(self):
        """Ensure models are loaded"""
        if not self.model_loader.is_loaded():
            success = self.model_loader.load_models()
            if not success:
                raise RuntimeError(
                    "Failed to load models. Ensure files exist in app/models/artifacts/."
                )

    def get_model_info(self):
        """Get model information"""
        self.ensure_models_loaded()
        return self.model_loader.get_model_info()
