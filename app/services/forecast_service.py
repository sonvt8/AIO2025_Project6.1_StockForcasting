"""
Forecast Service (v2 - PatchTST)
- Inference-only service using trained PatchTST + post-processing + smooth bias correction
- No feature engineering, no retraining at runtime
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.patchtst_loader import get_patchtst_loader


class ForecastService:
    def __init__(self) -> None:
        self.loader = get_patchtst_loader()

    @staticmethod
    def _validate_and_sort(historical_data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(historical_data)
        if "time" not in df.columns:
            raise ValueError("historical_data must include 'time' field")
        if "close" not in df.columns:
            # allow 'price' alias
            if "price" in df.columns:
                df = df.rename(columns={"price": "close"})
            else:
                raise ValueError("historical_data must include 'close' or 'price' field")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        # basic cleaning
        df = df.dropna(subset=["time", "close"])
        df = df[df["close"] > 0]
        if len(df) == 0:
            raise ValueError("No valid historical close prices after cleaning")
        return df

    def predict_from_historical_data(
        self, historical_data: list[dict], n_steps: int
    ) -> dict[str, np.ndarray]:
        if not self.loader.is_loaded():
            raise RuntimeError("PatchTST artifacts not loaded")

        df = self._validate_and_sort(historical_data)
        last_date: pd.Timestamp = pd.Timestamp(df["time"].iloc[-1])
        close_history: list[float] = df["close"].astype(float).tolist()

        # Use model horizon; limit n_steps if needed inside loader
        prices = self.loader.predict_prices(close_history, n_steps).astype(float)

        # Build forecast dates: start from next business day
        start_date = last_date + pd.offsets.BDay(1)
        dates = pd.bdate_range(start=start_date, periods=len(prices))

        # Compute expected returns (log-returns) for UI compatibility
        returns = np.zeros_like(prices, dtype=float)
        prev_price = float(close_history[-1])
        for i, p in enumerate(prices):
            p_safe = max(p, 1e-8)
            prev_safe = max(prev_price, 1e-8)
            returns[i] = float(np.log(p_safe / prev_safe))
            prev_price = p_safe

        return {
            "returns": returns,
            "prices": prices,
            "dates": dates.values,
        }
