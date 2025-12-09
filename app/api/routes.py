"""
API Routes
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    FullPredictRequest,
    FullPredictResponse,
    HealthResponse,
    MetricTestRequest,
    MetricTestResponse,
    ModelInfoResponse,
    MultiPredictRequest,
    MultiPredictResponse,
    RealtimePredictRequest,
    RealtimePredictResponse,
    SinglePredictRequest,
    SinglePredictResponse,
)
from app.services.data_fetcher import fetch_fpt_data_as_dict_list
from app.services.model_service import ModelService

# v2 no longer requires feature engineering utilities

router = APIRouter()

# Initialize model service (will be lazy-loaded)
_model_service = None


def get_model_service():
    """Get or create model service instance"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_service = get_model_service()
        models_loaded = model_service.model_loader.is_loaded()
        return HealthResponse(
            status="healthy", message="API is running", models_loaded=models_loaded
        )
    except Exception as e:
        return HealthResponse(status="unhealthy", message=f"Error: {str(e)}", models_loaded=False)


@router.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information including device info"""
    try:
        from app.utils.device_detector import detect_device

        model_service = get_model_service()
        model_service.ensure_models_loaded()
        info = model_service.get_model_info()

        # Add device info
        device_info = detect_device()
        info["device_type"] = device_info["device_type"]
        info["device_name"] = device_info.get("device_name", "unknown")
        info["artifact_dir"] = str(model_service.model_loader.models_dir)

        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}",
        ) from e


@router.post("/api/v1/predict/single", response_model=SinglePredictResponse)
async def predict_single(request: SinglePredictRequest):
    """
    Predict next day's stock price

    Requires minimum 20 days of historical data
    """
    try:
        model_service = get_model_service()
        model_service.ensure_models_loaded()

        # Convert to list of dicts
        historical_data = [item.dict() for item in request.historical_data]

        # Predict single step using v2 service
        results = model_service.forecast_service.predict_from_historical_data(
            historical_data, n_steps=1
        )
        forecast_date = pd.Timestamp(results["dates"][0]).strftime("%Y-%m-%d")
        pred_price = float(results["prices"][0])
        pred_return = float(results["returns"][0])
        return SinglePredictResponse(
            predicted_price=round(pred_price, 2),
            predicted_return=round(pred_return, 6),
            forecast_date=forecast_date,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}"
        ) from e


@router.post("/api/v1/predict/multi", response_model=MultiPredictResponse)
async def predict_multi(request: MultiPredictRequest):
    """
    Predict N days ahead (1-100 days)

    Requires minimum 20 days of historical data
    """
    try:
        model_service = get_model_service()
        model_service.ensure_models_loaded()

        # Convert to list of dicts
        historical_data = [item.dict() for item in request.historical_data]

        # Predict multi-step using v2 (no training data required)
        results = model_service.forecast_service.predict_from_historical_data(
            historical_data, request.n_steps
        )

        # Format predictions
        predictions = [
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "price": round(float(price), 2),
                "return": round(float(ret), 6),
            }
            for date, price, ret in zip(
                results["dates"], results["prices"], results["returns"], strict=False
            )
        ]

        return MultiPredictResponse(predictions=predictions, n_steps=len(predictions))

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}"
        ) from e


@router.post("/api/v1/predict/full", response_model=FullPredictResponse)
async def predict_full(request: FullPredictRequest):
    """
    Predict full 100 days ahead (as per baseline)

    Requires minimum 20 days of historical data
    """
    try:
        model_service = get_model_service()
        model_service.ensure_models_loaded()

        # Convert to list of dicts
        historical_data = [item.dict() for item in request.historical_data]

        # Predict 100 steps using v2 (no training data required)
        results = model_service.forecast_service.predict_from_historical_data(
            historical_data, n_steps=100
        )

        # Format predictions
        predictions = [
            {
                "id": idx + 1,
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "price": round(float(price), 2),
                "return": round(float(ret), 6),
            }
            for idx, (date, price, ret) in enumerate(
                zip(results["dates"], results["prices"], results["returns"], strict=False)
            )
        ]

        return FullPredictResponse(predictions=predictions)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}"
        ) from e


@router.post("/api/v1/predict/realtime", response_model=RealtimePredictResponse)
async def predict_realtime(request: RealtimePredictRequest):
    """
    Predict using real-time data fetched from internet

    Automatically fetches FPT stock data from internet and predicts future prices.
    This endpoint replaces the need to manually provide historical_data.

    Note: The 'historical_days' parameter is for backward compatibility only.
    When an existing dataset (FPT_train.csv) is present, the function ALWAYS returns
    ALL data from the dataset (from 2020-08-03) + newly fetched data, regardless of
    the 'historical_days' value. The parameter only affects behavior when no existing
    dataset exists.
    """
    try:
        print(
            f"[DEBUG] Received realtime prediction request: "
            f"n_steps={request.n_steps}, historical_days={request.historical_days}"
        )
        model_service = get_model_service()
        model_service.ensure_models_loaded()

        # Fetch data from internet (only missing data will be fetched)
        # NOTE: historical_days parameter is for backward compatibility only.
        # When existing dataset exists, ALL data is returned regardless of this parameter.
        try:
            print(
                f"[DEBUG] Calling fetch_fpt_data_as_dict_list with days={request.historical_days}"
            )
            historical_data, metadata = fetch_fpt_data_as_dict_list(days=request.historical_days)
            print(f"[DEBUG] Fetched {len(historical_data)} records, metadata: {metadata}")
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Data fetcher not available: {str(e)}. "
                    "Please install vnstock: pip install vnstock"
                ),
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error fetching data from internet: {str(e)}",
            ) from e

        if len(historical_data) < 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Not enough data fetched. "
                    f"Got {len(historical_data)} days, need at least 20 days."
                ),
            )

        # Get latest date from metadata or data
        latest_date = metadata.get("last_date") or historical_data[-1]["time"]

        # Validate that we have enough data and last_date is valid
        if len(historical_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No historical data available for prediction.",
            )

        # Get last date from historical data to ensure forecast starts correctly
        last_historical_date = pd.Timestamp(historical_data[-1]["time"])
        expected_forecast_start = last_historical_date + pd.offsets.BDay(1)

        # Predict multi-step with error handling
        try:
            results = model_service.forecast_service.predict_from_historical_data(
                historical_data, request.n_steps
            )
        except ValueError as e:
            # Log detailed error for debugging
            print(f"[ERROR] Prediction failed with ValueError: {str(e)}")
            print(f"[DEBUG] Historical data sample (first 3): {historical_data[:3]}")
            print(f"[DEBUG] Historical data sample (last 3): {historical_data[-3:]}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Prediction failed: {str(e)}",
            ) from e
        except Exception as e:
            # Log detailed error for debugging
            print(f"[ERROR] Prediction failed with unexpected error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {str(e)}",
            ) from e

        # Validate forecast continuity: first forecast date should be next business day
        if len(results["dates"]) > 0:
            first_forecast_date = pd.Timestamp(results["dates"][0])
            if first_forecast_date != expected_forecast_start:
                print(
                    f"[WARNING] Forecast start date mismatch: "
                    f"expected {expected_forecast_start.date()}, "
                    f"got {first_forecast_date.date()}"
                )

        # Format predictions
        predictions = [
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "price": round(float(price), 2),
                "return": round(float(ret), 6),
            }
            for date, price, ret in zip(
                results["dates"], results["prices"], results["returns"], strict=False
            )
        ]

        # Prepare historical data for chart: ALL data (FPT_train.csv + newly fetched)
        # historical_data already contains merged data from fetch_fpt_data_as_dict_list()
        # which includes ALL data from FPT_train.csv (from 2020-08-03) + newly fetched data

        # Sort by time to ensure chronological order
        historical_sorted = sorted(historical_data, key=lambda x: x["time"])

        # Debug: log full data range before processing
        if len(historical_sorted) > 0:
            first_date_full = pd.Timestamp(historical_sorted[0]["time"])
            last_date_full = pd.Timestamp(historical_sorted[-1]["time"])
            print(
                f"[DEBUG] Full historical data received: {len(historical_sorted)} records "
                f"from {first_date_full.date()} to {last_date_full.date()}"
            )

        # Last day is excluded because forecast starts from the next day
        # This ensures the chart shows historical data up to (but not including)
        # the forecast start date
        historical_for_chart = (
            historical_sorted[:-1] if len(historical_sorted) > 1 else historical_sorted
        )

        # Debug: log data range after processing
        if len(historical_for_chart) > 0:
            first_date = pd.Timestamp(historical_for_chart[0]["time"])
            last_date_chart = pd.Timestamp(historical_for_chart[-1]["time"])
            print(
                f"[DEBUG] Historical data for chart: {len(historical_for_chart)} records "
                f"from {first_date.date()} to {last_date_chart.date()} "
                f"(excluded last day: {last_date_full.date()})"
            )

        return RealtimePredictResponse(
            fetched_data_count=len(historical_data),
            latest_date=latest_date,
            fetched_new_data=metadata.get("fetched_new_data", False),
            previous_last_date=metadata.get("previous_last_date"),
            predictions=predictions,
            n_steps=len(predictions),
            historical_data=historical_for_chart,
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[ERROR] Prediction failed with ValueError: {e}")
        print(
            f"[DEBUG] Request parameters: "
            f"n_steps={request.n_steps}, historical_days={request.historical_days}"
        )
        if "historical_data" in locals():
            print(f"[DEBUG] Historical data length: {len(historical_data)}")
            if len(historical_data) > 0:
                print(f"[DEBUG] Historical data sample (first 3): {historical_data[:3]}")
                print(f"[DEBUG] Historical data sample (last 3): {historical_data[-3:]}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}"
        ) from e
    except Exception as e:
        print(f"[ERROR] Prediction failed with unexpected error: {e}")
        print(
            f"[DEBUG] Request parameters: "
            f"n_steps={request.n_steps}, historical_days={request.historical_days}"
        )
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}"
        ) from e


@router.post("/api/v1/test/metrics", response_model=MetricTestResponse)
async def test_metrics(request: MetricTestRequest):
    """
    Test model metrics against ground truth data.

    This endpoint loads train/test data, runs predictions, and compares against
    ground truth to compute MSE, Bias, RMSE, MAE, R2, and MAPE.
    Useful for validating that API predictions match expected performance.
    """
    try:
        from pathlib import Path

        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        from app.utils.device_detector import detect_device

        model_service = get_model_service()
        model_service.ensure_models_loaded()

        # Get device info
        device_info = detect_device()
        device_type = device_info["device_type"]
        device_name = device_info.get("device_name", "unknown")
        artifact_dir = str(model_service.model_loader.models_dir)

        # Load data
        base_dir = Path(__file__).parent.parent.parent
        train_csv = (
            Path(request.train_csv_path)
            if request.train_csv_path
            else base_dir / "data" / "raw" / "FPT_train.csv"
        )
        test_csv = (
            Path(request.test_csv_path)
            if request.test_csv_path
            else base_dir / "data" / "test" / "FPT_test.csv"
        )

        if not train_csv.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training CSV not found: {train_csv}",
            )
        if not test_csv.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test CSV not found: {test_csv}",
            )

        # Load and prepare data
        df_train = (
            pd.read_csv(train_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
        )
        df_test_raw = (
            pd.read_csv(test_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
        )

        if "symbol" in df_test_raw.columns:
            df_test = df_test_raw[df_test_raw["symbol"] == "FPT"].copy()
        else:
            df_test = df_test_raw.copy()

        last_train_date = df_train["time"].max()
        df_test = df_test[df_test["time"] > last_train_date].reset_index(drop=True)

        if len(df_test) < request.horizon:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough test data: {len(df_test)} < {request.horizon}",
            )

        # Get ground truth
        y_true = df_test.head(request.horizon)["close"].astype("float32").values
        close_hist = df_train["close"].astype(float).tolist()

        # Predict
        preds = model_service.model_loader.predict_prices(close_hist, request.horizon)
        preds = np.asarray(preds, dtype=float).reshape(-1)[: len(y_true)]

        # Compute metrics
        mse = float(mean_squared_error(y_true, preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))
        mape = float(np.mean(np.abs((y_true - preds) / y_true)) * 100)
        bias = float(np.mean(preds - y_true))

        # Device-specific thresholds
        mse_thresholds = {
            "cuda": 18.5,
            "mps": 50.0,
            "cpu": 50.0,
        }
        threshold = mse_thresholds.get(device_type, 50.0)
        passed = mse <= threshold

        metrics_dict = {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 4),
            "bias": round(bias, 4),
        }

        test_info = {
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "horizon": request.horizon,
            "train_range": f"{df_train['time'].min()} to {df_train['time'].max()}",
            "test_range": f"{df_test['time'].min()} to {df_test['time'].max()}",
        }

        return MetricTestResponse(
            device_type=device_type,
            device_name=device_name,
            artifact_dir=artifact_dir,
            metrics=metrics_dict,
            threshold=threshold,
            passed=passed,
            test_info=test_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metric test error: {str(e)}",
        ) from e
