#%%
import os
import pickle
import functions_framework
from google.cloud import storage

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "mlops-project-models")
MODEL_FILE = os.environ.get("MODEL_BLOB", "modelweights.pkl")


_MODEL_CACHE = None

def load_model():
    """Lazy-load model on first request to avoid startup timeout."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        client = storage.Client()
        blob = client.bucket(BUCKET_NAME).blob(MODEL_FILE)
        model_bytes = blob.download_as_bytes()
        _MODEL_CACHE = pickle.loads(model_bytes)
    return _MODEL_CACHE


def parse_instances(request):
    """Parse input JSON and return a list of numeric features.

    Accepts either a raw JSON list or an object containing "instances" or
    "input_data". Handles mixed feature types used in the Titanic model:
    - Sex: "male"/"female" mapped to 0/1
    - Embarked: "S"/"C"/"Q" mapped to 0/1/2
    - Numerics: cast to float
    """

    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("Body must be JSON")

    # allow raw JSON list: ["male", 3, ...]
    if isinstance(data, list):
        instances = data
    else:
        instances = data.get("instances") or data.get("input_data")

    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError('Send a non-empty list (raw JSON list or key "instances")')

    # Expected order from preprocessing: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    sex_map = {"male": 0, "female": 1}
    embarked_map = {"s": 0, "c": 1, "q": 2}

    converted = []
    for idx, val in enumerate(instances):
        # Sex is index 1
        if idx == 1 and isinstance(val, str):
            key = val.strip().lower()
            if key not in sex_map:
                raise ValueError("Sex must be 'male' or 'female'")
            converted.append(float(sex_map[key]))
            continue

        # Embarked is index 6
        if idx == 6 and isinstance(val, str):
            key = val.strip().lower()
            if key not in embarked_map:
                raise ValueError("Embarked must be 'S', 'C', or 'Q'")
            converted.append(float(embarked_map[key]))
            continue

        try:
            converted.append(float(val))
        except (TypeError, ValueError):
            raise ValueError(f"Feature at position {idx} must be numeric")

    return converted


@functions_framework.http
def logreg_classifier(request):
    """Cloud Function entry point for Titanic survival prediction."""
    try:
        # Lazy-load model on first invocation
        model = load_model()
        
        instances = parse_instances(request)
        expected_features = int(os.environ.get("FEATURE_COUNT", "7"))

        if len(instances) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(instances)}")

        X = [[float(x) for x in instances]]

        pred = model.predict(X)

        # If the model exposes class probabilities (e.g., scikit-learn), include them
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = None

        response = {"prediction": pred.tolist()}
        if proba is not None:
            # Map probabilities to class labels if available
            if hasattr(model, "classes_"):
                response["classes"] = model.classes_.tolist()
            response["probabilities"] = proba.tolist()

        return (response, 200)

    except Exception as e:
        return ({"error": str(e)}, 400)

@functions_framework.http
def health(request):
    return "OK"

