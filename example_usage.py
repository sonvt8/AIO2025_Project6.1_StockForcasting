"""
Example usage of FPT Stock Prediction API
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import requests

# API base URL
BASE_URL = "http://localhost:8000"


def load_sample_data():
    """Load sample data from data/raw/FPT_train.csv"""
    try:
        df = pd.read_csv("data/raw/FPT_train.csv")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Get last 30 days as sample
        sample = df.tail(30)

        # Convert to API format
        historical_data = []
        for _, row in sample.iterrows():
            historical_data.append(
                {
                    "time": row["time"].strftime("%Y-%m-%d"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        return historical_data
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return dummy data
        return create_dummy_data()


def create_dummy_data():
    """Create dummy historical data for testing"""
    base_date = datetime(2025, 2, 1)
    base_price = 120.0

    historical_data = []
    for i in range(30):
        date = base_date + timedelta(days=i)
        # Skip weekends
        if date.weekday() >= 5:
            continue

        price = base_price + (i * 0.1) + (i % 3 - 1) * 0.5
        historical_data.append(
            {
                "time": date.strftime("%Y-%m-%d"),
                "open": round(price, 2),
                "high": round(price * 1.02, 2),
                "low": round(price * 0.98, 2),
                "close": round(price, 2),
                "volume": 1000000 + i * 10000,
            }
        )

    return historical_data


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("1. Testing Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")
        print(f"Models loaded: {data['models_loaded']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "=" * 60)
    print("2. Testing Model Info")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/api/v1/model/info")
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model type: {data.get('model_type', 'N/A')}")
        print(f"Features count: {data.get('features_count', 'N/A')}")
        if data.get("config"):
            print(f"Config: {json.dumps(data['config'], indent=2)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_prediction(historical_data):
    """Test single step prediction"""
    print("\n" + "=" * 60)
    print("3. Testing Single Step Prediction")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/single",
            json={"historical_data": historical_data},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        print(f"Predicted price: {data['predicted_price']}")
        print(f"Predicted return: {data['predicted_return']}")
        print(f"Forecast date: {data['forecast_date']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def test_multi_prediction(historical_data, n_steps=10):
    """Test multi-step prediction"""
    print("\n" + "=" * 60)
    print(f"4. Testing Multi-Step Prediction ({n_steps} days)")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/multi",
            json={"historical_data": historical_data, "n_steps": n_steps},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        print(f"Forecasted {data['n_steps']} days")
        print("\nFirst 5 predictions:")
        for pred in data["predictions"][:5]:
            print(f"  {pred['date']}: {pred['price']:.2f} (return: {pred['return']:.6f})")
        if len(data["predictions"]) > 5:
            print("  ...")
            print(
                f"  Last: {data['predictions'][-1]['date']}: {data['predictions'][-1]['price']:.2f}"
            )
        return True
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def test_full_prediction(historical_data):
    """Test full 100-day prediction"""
    print("\n" + "=" * 60)
    print("5. Testing Full 100-Day Prediction")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/full",
            json={"historical_data": historical_data},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        print(f"Forecasted {len(data['predictions'])} days")
        print("\nFirst 5 predictions:")
        for pred in data["predictions"][:5]:
            print(f"  ID {pred['id']}: {pred['date']} - {pred['price']:.2f}")
        print("\nLast 5 predictions:")
        for pred in data["predictions"][-5:]:
            print(f"  ID {pred['id']}: {pred['date']} - {pred['price']:.2f}")

        # Save to CSV
        df = pd.DataFrame(data["predictions"])
        df[["id", "price"]].to_csv("example_prediction.csv", index=False)
        print("\n✅ Saved predictions to example_prediction.csv")
        return True
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("FPT Stock Prediction API - Example Usage")
    print("=" * 60)
    print(f"\nAPI URL: {BASE_URL}")
    print("\nMake sure the API is running:")
    print("  uvicorn app.main:app --reload")

    # Load sample data
    print("\nLoading sample data...")
    historical_data = load_sample_data()
    print(f"Loaded {len(historical_data)} days of historical data")

    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Model Info", test_model_info()))
    results.append(("Single Prediction", test_single_prediction(historical_data)))
    results.append(("Multi-Step Prediction", test_multi_prediction(historical_data, 10)))
    results.append(("Full Prediction", test_full_prediction(historical_data)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()
