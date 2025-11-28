"""
Model Loader Service
Handles loading and management of trained models
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

from app.config import MODEL_PATHS


class ModelLoader:
    """Load and manage trained models"""

    def __init__(self):
        self.elasticnet_model: ElasticNet | None = None
        self.scaler: StandardScaler | None = None
        self.calibration_model: LinearRegression | None = None
        self.model_config: dict[str, Any] | None = None
        self._loaded = False

    def load_models(self, models_dir: Path | None = None) -> bool:
        """
        Load all model artifacts

        Args:
            models_dir: Directory containing model files. If None, uses default from config.

        Returns:
            True if successful, False otherwise
        """
        if models_dir is None:
            models_dir = MODEL_PATHS["elasticnet"].parent

        # Create directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load ElasticNet model
            elasticnet_path = models_dir / "elasticnet_model.pkl"
            if elasticnet_path.exists():
                self.elasticnet_model = joblib.load(elasticnet_path)
            else:
                raise FileNotFoundError(f"ElasticNet model not found at {elasticnet_path}")

            # Load scaler
            scaler_path = models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")

            # Load calibration model
            calibration_path = models_dir / "calibration_model.pkl"
            if calibration_path.exists():
                self.calibration_model = joblib.load(calibration_path)
            else:
                # Calibration is optional, create a default one if not found
                self.calibration_model = LinearRegression()
                self.calibration_model.coef_ = np.array([1.0])
                self.calibration_model.intercept_ = 0.0

            # Load model config
            config_path = models_dir / "model_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.model_config = json.load(f)
            else:
                # Use default config from baseline
                self.model_config = {
                    "window_size": 252,
                    "window_type": "sliding",
                    "alpha": 0.0005,
                    "l1_ratio": 0.8,
                    "ensemble_weight": 0.0,  # w_naive = 0.0, w_model = 1.0
                }

            self._loaded = True
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            self._loaded = False
            return False

    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._loaded

    def get_model_info(self) -> dict[str, Any]:
        """Get model information"""
        if not self._loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_type": "ElasticNet",
            "features_count": len(self.scaler.feature_names_in_)
            if hasattr(self.scaler, "feature_names_in_")
            else 39,
            "config": self.model_config,
        }

    def predict_return(self, features: np.ndarray) -> float:
        """
        Predict next day return from features

        Args:
            features: Feature array of shape (1, n_features) or (n_features,)

        Returns:
            Predicted return (log return)
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict with ElasticNet
        raw_pred = self.elasticnet_model.predict(features_scaled)[0]

        # Apply calibration
        calibrated_pred = self.calibration_model.predict([[raw_pred]])[0]

        return float(calibrated_pred)

    def predict_price(self, features: np.ndarray, current_price: float) -> float:
        """
        Predict next day price from features

        Args:
            features: Feature array
            current_price: Current day's close price

        Returns:
            Predicted next day price
        """
        predicted_return = self.predict_return(features)
        predicted_price = current_price * np.exp(predicted_return)
        return float(predicted_price)


# Global model loader instance
_model_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
