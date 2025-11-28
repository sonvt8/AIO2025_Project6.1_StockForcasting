"""
Forecast Service
Handles multi-step forecasting logic
"""

import numpy as np
import pandas as pd

from app.config import FEATURE_NAMES
from app.models.model_loader import get_model_loader
from app.services.feature_engineering import build_features_from_buffers


class ForecastService:
    """Service for stock price forecasting"""

    def __init__(self):
        self.model_loader = get_model_loader()

    def predict_single_step(
        self,
        ret_buffer: list[float],
        vol_buffer: list[float],
        price_buffer: list[float],
        volume_buffer: list[float],
        current_date: pd.Timestamp,
    ) -> tuple[float, float]:
        """
        Predict next day's return and price

        Args:
            ret_buffer: List of recent returns (last 20)
            vol_buffer: List of recent volume changes (last 5)
            price_buffer: List of recent prices (last 20)
            volume_buffer: List of recent volumes (last 20)
            current_date: Current date

        Returns:
            Tuple of (predicted_return, predicted_price)
        """
        # Build features
        feat_vals = build_features_from_buffers(
            ret_buffer, vol_buffer, price_buffer, volume_buffer, current_date
        )

        # Convert to feature array in correct order
        feature_array = np.array([feat_vals[name] for name in FEATURE_NAMES], dtype=float)

        # Predict return
        predicted_return = self.model_loader.predict_return(feature_array)

        # Convert to price
        current_price = price_buffer[-1]
        predicted_price = current_price * np.exp(predicted_return)

        return predicted_return, predicted_price

    def predict_multi_step(
        self,
        ret_buffer: list[float],
        vol_buffer: list[float],
        price_buffer: list[float],
        volume_buffer: list[float],
        start_date: pd.Timestamp,
        n_steps: int = 100,
    ) -> dict[str, np.ndarray]:
        """
        Predict multiple steps ahead using iterative forecasting

        Args:
            ret_buffer: Initial return buffer
            vol_buffer: Initial volume change buffer
            price_buffer: Initial price buffer
            volume_buffer: Initial volume buffer
            start_date: Starting date for forecast
            n_steps: Number of steps to forecast

        Returns:
            Dictionary with keys:
            - 'returns': Array of predicted returns
            - 'prices': Array of predicted prices
            - 'dates': Array of forecast dates
        """
        # Initialize buffers (copy to avoid mutation)
        ret_buf = list(ret_buffer)
        vol_buf = list(vol_buffer)
        price_buf = list(price_buffer)
        vol_buf_full = list(volume_buffer)

        current_date = start_date
        predicted_returns = []
        predicted_prices = []
        forecast_dates = []

        # Get model config for ensemble
        model_config = self.model_loader.model_config
        ensemble_weight = model_config.get("ensemble_weight", 0.0) if model_config else 0.0

        for _ in range(n_steps):
            # Predict next step
            pred_return, pred_price = self.predict_single_step(
                ret_buf, vol_buf, price_buf, vol_buf_full, current_date
            )

            # Apply ensemble with naive model if needed
            if ensemble_weight > 0:
                naive_price = price_buf[-1]  # Naive: keep last price
                pred_price = ensemble_weight * naive_price + (1.0 - ensemble_weight) * pred_price

            predicted_returns.append(pred_return)
            predicted_prices.append(pred_price)
            forecast_dates.append(current_date)

            # Update buffers with prediction
            ret_buf.append(pred_return)
            if len(ret_buf) > 20:
                ret_buf = ret_buf[-20:]

            vol_buf.append(0.0)  # Assume no volume change for future
            if len(vol_buf) > 5:
                vol_buf = vol_buf[-5:]

            price_buf.append(pred_price)
            if len(price_buf) > 20:
                price_buf = price_buf[-20:]

            vol_buf_full.append(vol_buf_full[-1] if len(vol_buf_full) > 0 else 1.0)
            if len(vol_buf_full) > 20:
                vol_buf_full = vol_buf_full[-20:]

            # Move to next business day
            current_date = current_date + pd.offsets.BDay(1)

        return {
            "returns": np.array(predicted_returns, dtype=float),
            "prices": np.array(predicted_prices, dtype=float),
            "dates": np.array(forecast_dates),
        }

    def predict_from_historical_data(
        self, historical_data: list[dict], n_steps: int = 100, train_df: pd.DataFrame = None
    ) -> dict[str, np.ndarray]:
        """
        Predict from historical data (convenience method)

        Args:
            historical_data: List of dicts with keys: time, open, high, low, close, volume
            n_steps: Number of steps to forecast
            train_df: Optional training dataframe for proper feature engineering

        Returns:
            Dictionary with predicted returns, prices, and dates
        """
        from app.utils.data_processing import prepare_historical_data_for_prediction

        # Prepare buffers
        ret_buffer, vol_buffer, price_buffer, volume_buffer, last_date = (
            prepare_historical_data_for_prediction(historical_data, train_df)
        )

        # Forecast
        start_date = last_date + pd.offsets.BDay(1)
        return self.predict_multi_step(
            ret_buffer, vol_buffer, price_buffer, volume_buffer, start_date, n_steps
        )
