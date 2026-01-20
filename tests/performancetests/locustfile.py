from locust import HttpUser, task, between

class APIPerformanceTest(HttpUser):
    wait_time = between(1, 3)  # seconds

    @task
    def root_endpoint(self):
        self.client.get("/")

    @task
    def items_endpoint(self):
        self.client.get("/items/1")

    @task
    def predict_endpoint(self):
        self.client.post("/predict", json={
            "feature1": 0.3,
            "feature2": 0.4,
            "feature3": 0.9,
        })