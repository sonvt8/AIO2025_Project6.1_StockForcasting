"""
API Routes
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    FullPredictRequest,
    FullPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    MultiPredictRequest,
    MultiPredictResponse,
    SinglePredictRequest,
    SinglePredictResponse,
)
from app.services.model_service import ModelService
from app.utils.data_processing import load_data, prepare_historical_data_for_prediction

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
    """Get model information"""
    try:
        model_service = get_model_service()
        model_service.ensure_models_loaded()
        info = model_service.get_model_info()
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

        # Prepare buffers
        ret_buffer, vol_buffer, price_buffer, volume_buffer, last_date = (
            prepare_historical_data_for_prediction(historical_data)
        )

        # Predict single step
        forecast_date = last_date + pd.offsets.BDay(1)
        pred_return, pred_price = model_service.forecast_service.predict_single_step(
            ret_buffer, vol_buffer, price_buffer, volume_buffer, last_date
        )

        return SinglePredictResponse(
            predicted_price=round(pred_price, 2),
            predicted_return=round(pred_return, 6),
            forecast_date=forecast_date.strftime("%Y-%m-%d"),
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

        # Load training data for proper feature engineering
        try:
            train_df = load_data()
        except Exception:
            train_df = None

        # Predict multi-step
        results = model_service.forecast_service.predict_from_historical_data(
            historical_data, request.n_steps, train_df
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

        # Load training data for proper feature engineering
        try:
            train_df = load_data()
        except Exception:
            train_df = None

        # Predict 100 steps
        results = model_service.forecast_service.predict_from_historical_data(
            historical_data, n_steps=100, train_df=train_df
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
