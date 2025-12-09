"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, ConfigDict, Field, validator


class StockDataPoint(BaseModel):
    """Single stock data point"""

    time: str = Field(..., description="Date in YYYY-MM-DD format")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")

    @validator("high")
    def high_must_be_highest(cls, v, values):
        if "low" in values and v < values["low"]:
            raise ValueError("high must be >= low")
        return v

    @validator("close")
    def close_in_range(cls, v, values):
        if "low" in values and "high" in values:
            if not (values["low"] <= v <= values["high"]):
                raise ValueError("close must be between low and high")
        return v


class SinglePredictRequest(BaseModel):
    """Request for single step prediction"""

    historical_data: list[StockDataPoint] = Field(
        ..., min_items=20, description="Historical stock data (minimum 20 days required)"
    )


class SinglePredictResponse(BaseModel):
    """Response for single step prediction"""

    predicted_price: float = Field(..., description="Predicted next day closing price")
    predicted_return: float = Field(..., description="Predicted next day log return")
    forecast_date: str = Field(..., description="Date of prediction (YYYY-MM-DD)")


class MultiPredictRequest(BaseModel):
    """Request for multi-step prediction"""

    historical_data: list[StockDataPoint] = Field(
        ..., min_items=20, description="Historical stock data (minimum 20 days required)"
    )
    n_steps: int = Field(
        default=100, ge=1, le=100, description="Number of days to forecast (1-100)"
    )


class MultiPredictResponse(BaseModel):
    """Response for multi-step prediction"""

    predictions: list[dict] = Field(
        ..., description="List of predictions with date, price, and return"
    )
    n_steps: int = Field(..., description="Number of forecasted days")


class FullPredictRequest(BaseModel):
    """Request for full 100-day prediction"""

    historical_data: list[StockDataPoint] = Field(
        ..., min_items=20, description="Historical stock data (minimum 20 days required)"
    )


class FullPredictResponse(BaseModel):
    """Response for full 100-day prediction"""

    predictions: list[dict] = Field(
        ..., description="List of 100 predictions with date, price, and return"
    )


class ModelInfoResponse(BaseModel):
    """Response for model information"""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_type: str | None = None
    features_count: int | None = None
    config: dict | None = None
    device_type: str | None = None
    device_name: str | None = None
    artifact_dir: str | None = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    message: str
    models_loaded: bool


class RealtimePredictRequest(BaseModel):
    """Request for realtime prediction (fetches data from internet)"""

    n_steps: int = Field(
        default=100, ge=1, le=100, description="Number of days to forecast (1-100)"
    )
    historical_days: int = Field(
        default=120,
        ge=20,
        le=365,
        description="Number of historical days to fetch from internet (20-365)",
    )


class RealtimePredictResponse(BaseModel):
    """Response for realtime prediction"""

    fetched_data_count: int = Field(..., description="Number of historical data points (total)")
    latest_date: str = Field(..., description="Latest date in data (YYYY-MM-DD)")
    fetched_new_data: bool = Field(
        ..., description="Whether new data was fetched from internet (False if using cached data)"
    )
    previous_last_date: str | None = Field(
        None, description="Last date in dataset before fetch (if available)"
    )
    predictions: list[dict] = Field(
        ..., description="List of predictions with date, price, and return"
    )
    n_steps: int = Field(..., description="Number of forecasted days")
    historical_data: list[dict] = Field(
        ...,
        description=(
            "Historical OHLCV data for chart display. "
            "Contains 6 years of historical data from latest_date going backwards, "
            "excluding the last day (forecast starts from next day). "
            "This includes ALL data from FPT_train.csv + newly fetched data, merged together. "
            "Approximately ~1500 trading days for better chart depth and clarity."
        ),
    )


class MetricTestRequest(BaseModel):
    """Request for metric testing against ground truth"""

    train_csv_path: str | None = Field(
        None,
        description="Path to training CSV (default: data/raw/FPT_train.csv)",
    )
    test_csv_path: str | None = Field(
        None,
        description="Path to test CSV with ground truth (default: data/test/FPT_test.csv)",
    )
    horizon: int = Field(default=100, ge=1, le=100, description="Forecast horizon (1-100)")


class MetricTestResponse(BaseModel):
    """Response for metric testing"""

    device_type: str = Field(..., description="Device type (cuda/mps/cpu)")
    device_name: str = Field(..., description="Device name")
    artifact_dir: str = Field(..., description="Artifact directory path")
    metrics: dict = Field(
        ...,
        description="Metrics: mse, bias, rmse, mae, r2, mape",
    )
    threshold: float = Field(..., description="MSE threshold for this device")
    passed: bool = Field(..., description="Whether metrics pass threshold")
    test_info: dict = Field(..., description="Test information (train/test rows, etc.)")
