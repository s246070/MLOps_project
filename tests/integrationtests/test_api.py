import os
import requests

API_URL = os.getenv("API_URL")

def test_health_check():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.text == "OK"

def test_valid_prediction():
    data = {
        "instances": [3, "male", 22.0, 1, 0, 7.25, "s"]
    }
    response = requests.post(f"{API_URL}/", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probabilities" in result

def test_invalid_input():
    response = requests.post(f"{API_URL}/", json={"instances": []})
    assert response.status_code == 400
    assert "error" in response.json()
