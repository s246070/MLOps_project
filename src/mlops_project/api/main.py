"""
Google Cloud Function for ML Model Inference

This module implements a REST API endpoint for making predictions using a trained
logistic regression model stored in Google Cloud Storage. It's designed to be deployed
as a Google Cloud Function.

Environment Variables:
    MODEL_BUCKET: GCS bucket name containing the model (default: "mlops-project-models")
    MODEL_BLOB: Model file name in the bucket (default: "model.pkl")

The API accepts JSON requests with input features and returns predictions.
"""

import os
import pickle
import functions_framework
from google.cloud import storage

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "mlops-project-models")
MODEL_FILE = os.environ.get("MODEL_BLOB", "model.pkl")


def load_model():
    """
    Load the pickled ML model from Google Cloud Storage.
    
    Connects to the GCS bucket specified by BUCKET_NAME and retrieves
    the model file specified by MODEL_FILE, then deserializes it.
    
    Returns:
        The deserialized ML model object (typically a scikit-learn model)
        
    Raises:
        google.cloud.exceptions.NotFound: If the bucket or model file doesn't exist
        pickle.UnpicklingError: If the file is not a valid pickled object
    """
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(MODEL_FILE)
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)


# Load model at module initialization (runs once when function is first invoked)
MODEL = load_model()



def parse_instances(request):
    """
    Parse and validate input features from the HTTP request.
    
    Accepts input in two formats:
    - Raw JSON list: [1, 2, 3, 4]
    - JSON object with "instances" or "input_data" key: {"instances": [1, 2, 3, 4]}
    
    Args:
        request: Flask request object containing JSON body
        
    Returns:
        list: List of integer feature values
        
    Raises:
        ValueError: If body is not JSON, empty, or contains non-integer values
    """
    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("Body must be JSON")

    # Allow raw JSON list: [1,2,3]
    if isinstance(data, list):
        instances = data
    else:
        instances = data.get("instances") or data.get("input_data")

    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError('Send a non-empty list (raw JSON list or key "instances")')

    if not all(isinstance(x, int) for x in instances):
        raise ValueError("All values must be integers")

    return instances


@functions_framework.http
def logreg_classifier(request):
    """
    HTTP endpoint for logistic regression predictions.
    
    This is the main entry point for the Cloud Function. It processes incoming
    prediction requests, validates input, and returns model predictions.
    
    Args:
        request: Flask request object containing JSON body with input features
        
    Returns:
        dict: JSON response with either:
            - {"prediction": [predicted_class]} on success (HTTP 200)
            - {"error": error_message} on failure (HTTP 400)
            
    Example:
        POST request body: [4.5, 3.0, 1.5, 0.2]
        Response: {"prediction": [0]}
    """
    try:
        instances = parse_instances(request)

        # Convert integers to floats and reshape for model input
        # If your model expects 4 features in one sample (e.g., Iris dataset),
        # then instances should have length 4 and we reshape to (1, 4).
        X = [[float(x) for x in instances]]

        pred = MODEL.predict(X)
        return {"prediction": pred.tolist()}

    except Exception as e:
        return {"error": str(e)}, 400