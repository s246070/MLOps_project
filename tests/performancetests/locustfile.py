from locust import HttpUser, task, between
import random

# Predefined example passengers for prediction requests
passenger_samples = [
    [1, "male", 22.0, 1, 0, 7.25, "s"],
    [3, "female", 26.0, 0, 0, 8.05, "c"],
    [2, "male", 35.0, 0, 0, 13.0, "q"],
    [1, "female", 29.0, 1, 1, 120.0, "s"]
]

class APILoadTest(HttpUser):
    # Simulate user wait time between tasks (1 to 5 seconds)
    wait_time = between(1, 5)

    @task
    def predict(self):
        """
        Send a POST request to the prediction endpoint with a random passenger sample.
        Check that the response returns status 200 and contains 'prediction'.
        """
        sample = random.choice(passenger_samples)
        response = self.client.post("/", json={"instances": sample})

        assert response.status_code == 200, f"Prediction failed: {response.text}"
        assert "prediction" in response.json(), "Missing 'prediction' in response"

    @task
    def health_check(self):
        """
        Send a GET request to the /health endpoint to ensure service availability.
        """
        response = self.client.get("/health")
        assert response.status_code == 200, f"Health check failed: {response.text}"
        assert response.text == "OK", f"Unexpected health response: {response.text}"
