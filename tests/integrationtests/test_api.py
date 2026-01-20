from fastapi.testclient import TestClient
from src.mlops_project.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ML Ops API"}

def test_another_endpoint():
    response = client.get("/another-endpoint")
    assert response.status_code == 200
    assert "result" in response.json()

def test_read_item():
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42}


def test_predict_dummy():
    response = client.post(
        "/predict",
        json={"feature1": 0.5, "feature2": 0.7, "feature3": 0.4}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
