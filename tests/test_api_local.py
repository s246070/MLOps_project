#!/usr/bin/env python3
"""
Local test script for the FastAPI inference app.
Tests the API endpoints locally before deployment to Cloud Run.
"""

import requests
import json

BASE_URL = "https://europe-west1-titanic-mlops-484412.cloudfunctions.net/titanic-function"


def test_health() -> bool:
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_root() -> bool:
    """Test root endpoint."""
    print("\nTesting / endpoint...")
    try:
        response = requests.get(BASE_URL)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_single_prediction() -> bool:
    """Test single prediction endpoint."""
    print("\nTesting /predict endpoint (single prediction)...")
    
    # Test case 1: First class female passenger (high survival probability)
    payload = {
        "pclass": 1,
        "sex": "female",
        "age": 25,
        "sibsp": 1,
        "parch": 0,
        "fare": 200,
        "embarked": "S"
    }
    
    try:
        print(f"  Input: {json.dumps(payload, indent=2)}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_batch_prediction() -> bool:
    """Test batch prediction endpoint."""
    print("\nTesting /batch_predict endpoint (batch predictions)...")
    
    payload = {
        "instances": [
            {
                "pclass": 1,
                "sex": "female",
                "age": 25,
                "sibsp": 1,
                "parch": 0,
                "fare": 200,
                "embarked": "S"
            },
            {
                "pclass": 3,
                "sex": "male",
                "age": 30,
                "sibsp": 0,
                "parch": 0,
                "fare": 10,
                "embarked": "S"
            },
            {
                "pclass": 2,
                "sex": "female",
                "age": 35,
                "sibsp": 1,
                "parch": 1,
                "fare": 75,
                "embarked": "C"
            }
        ]
    }
    
    try:
        print(f"  Input: {len(payload['instances'])} instances")
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"  Status: {response.status_code}")
        result = response.json()
        print(f"  Number of predictions: {len(result['predictions'])}")
        for i, pred in enumerate(result['predictions']):
            print(f"    Prediction {i+1}: {json.dumps(pred, indent=6)}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_invalid_input() -> bool:
    """Test error handling with invalid input."""
    print("\nTesting error handling with invalid input...")
    
    payload = {
        "pclass": 1,
        "sex": "invalid_value",  # Invalid sex value
        "age": 25,
        "sibsp": 1,
        "parch": 0,
        "fare": 200,
        "embarked": "S"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        # Should return 400 for invalid input
        return response.status_code == 400
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_api_docs() -> bool:
    """Test automatic API documentation."""
    print("\nTesting API documentation endpoints...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"  /docs Status: {response.status_code}")
        
        response = requests.get(f"{BASE_URL}/openapi.json")
        print(f"  /openapi.json Status: {response.status_code}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FastAPI Inference App - Local Test Suite")
    print("=" * 60)
    
    print(f"\nConnecting to: {BASE_URL}\n")
    
    results = {
        "Health Check": test_health(),
        "Root Endpoint": test_root(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Invalid Input Handling": test_invalid_input(),
        "API Documentation": test_api_docs(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
