#%%
import os
import pickle
import functions_framework
from google.cloud import storage

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "mlops-project-models")
MODEL_FILE = os.environ.get("MODEL_BLOB", "modelweights.pkl")


def load_model():
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(MODEL_FILE)
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

MODEL = load_model()


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
    try:
        instances = parse_instances(request)
        expected_features = int(os.environ.get("FEATURE_COUNT", len(instances)))

        if len(instances) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(instances)}")

        X = [[float(x) for x in instances]]

        pred = MODEL.predict(X)

        # If the model exposes class probabilities (e.g., scikit-learn), include them
        proba = None
        if hasattr(MODEL, "predict_proba"):
            try:
                proba = MODEL.predict_proba(X)
            except Exception:
                proba = None

        response = {"prediction": pred.tolist()}
        if proba is not None:
            # Map probabilities to class labels if available
            if hasattr(MODEL, "classes_"):
                response["classes"] = MODEL.classes_.tolist()
            response["probabilities"] = proba.tolist()

        return response

    except Exception as e:
        return {"error": str(e)}, 400