"""
Test script for API endpoints
Make sure API server is running: uvicorn app.main:app --reload
Run: python test_api.py
"""

import json
from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Models Loaded: {result['models_loaded']}")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure server is running:")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


def test_model_info():
    """Test model info endpoint"""
    print("=" * 60)
    print("TEST 2: Model Info")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/api/v1/model/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Model Type: {result.get('model_type', 'N/A')}")
            print(f"Features Count: {result.get('features_count', 'N/A')}")
            if result.get("config"):
                print(f"Config: {json.dumps(result['config'], indent=2)}")
            print("✅ Model info retrieved")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


def test_realtime_prediction():
    """Test realtime prediction endpoint"""
    print("=" * 60)
    print("TEST 3: Realtime Prediction")
    print("=" * 60)

    payload = {
        "n_steps": 30,  # Predict 30 days ahead
        "historical_days": 120,  # Use 120 days of historical data
    }

    print(f"Request payload: {json.dumps(payload, indent=2)}")
    print("\nSending request...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/realtime",
            json=payload,
            timeout=30,  # May take time to fetch data
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            print("\n" + "-" * 60)
            print("RESPONSE SUMMARY")
            print("-" * 60)
            print(f"Fetched new data: {result['fetched_new_data']}")
            print(f"Previous last date: {result.get('previous_last_date', 'N/A')}")
            print(f"Latest date: {result['latest_date']}")
            print(f"Total records: {result['fetched_data_count']}")
            print(f"Number of predictions: {result['n_steps']}")

            print("\n" + "-" * 60)
            print("FIRST 5 PREDICTIONS")
            print("-" * 60)
            for i, pred in enumerate(result["predictions"][:5], 1):
                msg = (
                    f"{i}. {pred['date']}: Price = {pred['price']:.2f}, "
                    f"Return = {pred['return']:.6f}"
                )
                print(msg)

            print("\n" + "-" * 60)
            print("LAST 5 PREDICTIONS")
            print("-" * 60)
            for i, pred in enumerate(result["predictions"][-5:], len(result["predictions"]) - 4):
                msg = (
                    f"{i}. {pred['date']}: Price = {pred['price']:.2f}, "
                    f"Return = {pred['return']:.6f}"
                )
                print(msg)

            # Calculate statistics
            prices = [p["price"] for p in result["predictions"]]
            returns = [p["return"] for p in result["predictions"]]

            print("\n" + "-" * 60)
            print("PREDICTION STATISTICS")
            print("-" * 60)
            print(f"Min price: {min(prices):.2f}")
            print(f"Max price: {max(prices):.2f}")
            print(f"Avg price: {sum(prices)/len(prices):.2f}")
            print(f"Avg return: {sum(returns)/len(returns):.6f}")

            print("\n✅ Realtime prediction successful")
        else:
            print(f"❌ Error: {response.text}")
            try:
                error_detail = response.json()
                print(f"Error detail: {json.dumps(error_detail, indent=2)}")
            except Exception:
                pass
    except requests.exceptions.Timeout:
        print("❌ Request timeout. API may be fetching data from internet...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    print()


def test_multiple_requests():
    """Test multiple requests to verify caching behavior"""
    print("=" * 60)
    print("TEST 4: Multiple Requests (Test Caching)")
    print("=" * 60)

    payload = {"n_steps": 10, "historical_days": 120}

    print("Making 3 consecutive requests...")
    for i in range(1, 4):
        print(f"\nRequest {i}:")
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/predict/realtime", json=payload, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"  Fetched new data: {result['fetched_new_data']}")
                print(f"  Latest date: {result['latest_date']}")
                # Second and third requests should not fetch new data if dataset is up to date
            else:
                print(f"  Error: {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("API TEST SUITE")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_health_check()
    test_model_info()
    test_realtime_prediction()
    test_multiple_requests()

    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\n💡 Tips:")
    print("  - If 'fetched_new_data' is False, dataset is up to date")
    print("  - If 'fetched_new_data' is True, new data was fetched from internet")
    print("  - Check 'previous_last_date' vs 'latest_date' to see what was fetched")


if __name__ == "__main__":
    main()
