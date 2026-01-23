import os
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080").rstrip("/")

def test_health_check():
    """
    Test the health endpoint to verify that the API is running
    and responding correctly.
    """
    response = requests.get(f"{API_URL}/health")

    assert response.status_code == 200
    assert response.text == "OK"


def test_valid_prediction():
    """
    Test a valid prediction request.
    Sends a input payload and checks that:
    - the API responds with HTTP 200
    - the response contains prediction-related fields
    """
    data = {
        "instances": [3, "male", 22.0, 1, 0, 7.25, "S"]
    }

    response = requests.post(f"{API_URL}/", json=data)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "probabilities" in result


def test_invalid_input():
    """
    Test invalid input handling.
    Sends an empty instances list and verifies that:
    - the API responds with HTTP 400
    - an error message is returned
    """
    response = requests.post(f"{API_URL}/", json={"instances": []})

    assert response.status_code == 400
    assert "error" in response.json()
