
import os
import pickle
import torch
import torch.nn as nn
import functions_framework
from google.cloud import storage
from mlops_project.model import LogisticRegressionModel
from collections import OrderedDict
from mlops_project.drift_logging import append_jsonl_to_gcs, make_log_record
from prometheus_client import Counter, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import time
from mlops_project.drift_detection import run_drift_check


# this is for drift logging
LOG_BUCKET = os.environ.get("DRIFT_BUCKET", os.environ.get("MODEL_BUCKET", "mlops-project-models"))
LOG_BLOB = os.environ.get("DRIFT_LOG_BLOB", "drift/predictions_log.jsonl")

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "mlops-project-models")
MODEL_FILE = os.environ.get("MODEL_BLOB", "modelweights.pkl")
_MODEL_CACHE = None


ERROR_COUNTER = Counter("prediction_errors_total", "Number of prediction errors")
REQUEST_COUNTER = Counter("prediction_requests_total", "Number of prediction requests")
LATENCY_HIST = Histogram("prediction_latency_seconds", "Prediction latency")
REVIEW_SIZE = Summary("review_size_bytes", "Size of input review/json")


def _is_torch_state_dict(obj) -> bool:
    """Return True if the object looks like a PyTorch state_dict.

    More robust than a simple tensor check: allows numpy arrays and typical
    weight/bias key patterns such as "*.weight" and "*.bias".
    """
    if not isinstance(obj, (dict, OrderedDict)):
        return False

    def _is_tensorlike(v):
        if torch.is_tensor(v):
            return True
        # numpy arrays or similar (duck-typed) with shape and dtype
        return hasattr(v, "shape") and hasattr(v, "dtype")

    has_tensorlike = any(_is_tensorlike(v) for v in obj.values())
    has_weightish_keys = any(
        isinstance(k, str) and (
            k.endswith(".weight") or k.endswith(".bias") or "weight" in k or "bias" in k
        )
        for k in obj.keys()
    )

    return has_tensorlike and has_weightish_keys
def _strip_module_prefix(state_dict):
    # Handles keys like "module.linear.weight"
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())

def _unwrap_state_payload(obj):
    """Unwrap common payload formats that contain a state_dict.

    Examples: {"state_dict": {...}}, {"model_state_dict": {...}}, {"state": {...}}
    Returns inner dict if found, else returns obj unchanged.
    """
    if isinstance(obj, (dict, OrderedDict)):
        for key in ("state_dict", "model_state_dict", "state"):
            inner = obj.get(key)
            if isinstance(inner, (dict, OrderedDict)):
                return inner
    return obj

def _infer_input_dim(state_dict) -> int:
    # Find first 2D tensor-like -> (out_features, in_features)
    for v in state_dict.values():
        if torch.is_tensor(v):
            if v.ndim == 2:
                return v.shape[1]
            continue
        # numpy arrays or other tensor-like with shape
        if hasattr(v, "shape"):
            try:
                ndim = len(v.shape)
                if ndim == 2:
                    return int(v.shape[1])
            except Exception:
                pass
    raise ValueError("Could not infer input_dim from state_dict (no 2D weight tensor found)")

def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        client = storage.Client()
        blob = client.bucket(BUCKET_NAME).blob(MODEL_FILE)
        model_bytes = blob.download_as_bytes()
        obj = pickle.loads(model_bytes)
        obj = _unwrap_state_payload(obj)
        # If we already have a full torch module, use it
        if isinstance(obj, nn.Module):
            obj.eval()
            _MODEL_CACHE = obj
            return _MODEL_CACHE

        # Try to interpret dict/OrderedDict as a torch state_dict
        if isinstance(obj, (dict, OrderedDict)):
            try:
                sd = _strip_module_prefix(obj)
                input_dim = _infer_input_dim(sd)
                model = LogisticRegressionModel(input_dim)
                model.load_state_dict(sd, strict=False)
                model.eval()
                _MODEL_CACHE = model
            except Exception:
                # Not a recognizable torch state_dict; cache raw object
                _MODEL_CACHE = obj
        else:
            # sklearn or other pickled estimator/object
            _MODEL_CACHE = obj
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

    # --- /metrics endpoint ---
    if getattr(request, "path", "").endswith("/metrics"):
        body = generate_latest()
        headers = {"Content-Type": CONTENT_TYPE_LATEST}
        return (body, 200, headers)

    # --- Health check ---
    path = getattr(request, "path", "")
    if request.method == "GET" and path.endswith("/health"):
        return ("OK", 200)

    # --- Metrics: count requests ---
    REQUEST_COUNTER.inc()
    start_time = time.perf_counter()

    try:
        # --- Metrics: request size ---
        raw = request.get_data(cache=True) or b""
        REVIEW_SIZE.observe(len(raw))

        # ---- DIN EKSISTERENDE MODEL-KODE (beholdt 100%) ----

        # Lazy-load model on first invocation
        model = load_model()

        # Parse input instances
        instances = parse_instances(request)
        expected_features = int(os.environ.get("FEATURE_COUNT", "7"))
        if len(instances) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(instances)}")

        X = [[float(x) for x in instances]]

        # --- Torch model case ---
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                logits = model(X_tensor)
                proba_positive = torch.sigmoid(logits).reshape(-1)[0].item()
                pred = [1 if proba_positive >= 0.5 else 0]
                proba = [[1.0 - proba_positive, proba_positive]]

            # Drift logging
            record = make_log_record(
                instances,
                int(pred[0]),
                proba_positive
            )
            append_jsonl_to_gcs(LOG_BUCKET, LOG_BLOB, record)

            response = {
                "prediction": pred,
                "classes": [0, 1],
                "probabilities": proba
            }

        # --- Sklearn model ---
        else:
            if not hasattr(model, "predict"):
                raise TypeError("Loaded object is neither a torch Module nor sklearn estimator")

            pred = model.predict(X)
            proba = None

            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                except Exception:
                    proba = None

            response = {"prediction": pred.tolist()}

            if proba is not None:
                if hasattr(model, "classes_"):
                    response["classes"] = model.classes_.tolist()
                response["probabilities"] = proba.tolist()

        # --- Metrics: latency ---
        elapsed = time.perf_counter() - start_time
        LATENCY_HIST.observe(elapsed)

        # Return final response
        return (response, 200)

    except Exception as e:
        # Metrics: errors
        ERROR_COUNTER.inc()
        return ({"error": str(e)}, 400)

    # try:
    #     # Health check: respond to GET requests (e.g., /health)
    #     path = getattr(request, "path", None)
    #     if getattr(request, "method", "GET") == "GET" and (path is None or path.endswith("/health")):
    #         return ("OK", 200)

    #     # Lazy-load model on first invocation
    #     model = load_model()
        
    #     instances = parse_instances(request)
    #     expected_features = int(os.environ.get("FEATURE_COUNT", "7"))
    #     if len(instances) != expected_features:
    #         raise ValueError(f"Expected {expected_features} features, got {len(instances)}")
    #     X = [[float(x) for x in instances]]
    #     # Handle PyTorch model
    #     if isinstance(model, nn.Module):
    #         model.eval()
    #         with torch.no_grad():
    #             X_tensor = torch.tensor(X, dtype=torch.float32)
    #             logits = model(X_tensor)  # shape (1, 1)
    #             proba_positive = torch.sigmoid(logits).reshape(-1)[0].item()  # float in [0,1]
    #             pred = [1 if proba_positive >= 0.5 else 0]
    #             proba = [[1.0 - proba_positive, proba_positive]]
            
    #         # added for drift-detection
    #         record = make_log_record(instances, int(pred[0]), proba_positive if isinstance(model, nn.Module) else None)
    #         append_jsonl_to_gcs(LOG_BUCKET, LOG_BLOB, record)

    #         response = {
    #             "prediction": pred,
    #             "classes": [0, 1],
    #             "probabilities": proba
    #         }
    #     else:
    #         # Sklearn model or other estimator-like
    #         if not hasattr(model, "predict"):
    #             raise TypeError("Loaded object is neither a torch Module nor an sklearn estimator")
    #         pred = model.predict(X)
    #         proba = None
    #         if hasattr(model, "predict_proba"):
    #             try:
    #                 proba = model.predict_proba(X)
    #             except Exception:
    #                 proba = None
    #         response = {"prediction": pred.tolist()}
    #         if proba is not None:
    #             # Map probabilities to class labels if available
    #             if hasattr(model, "classes_"):
    #                 response["classes"] = model.classes_.tolist()
    #             response["probabilities"] = proba.tolist()
    #     return (response, 200)
    # except Exception as e:
    #     return ({"error": str(e)}, 400)
    


"""
The section below is an HTTP endpoint we can call to run data drift
detection. This endpoint reads how many recent predictions to ude, runs
the drift check and returns the result as an JSON response. 
"""

@functions_framework.http
def drift_check(request):
    try:
        n_latest = request.args.get("n_latest") if request.args else None
        n_latest = int(n_latest) if n_latest else 500

        result = run_drift_check(
            bucket_name=os.environ.get("DRIFT_BUCKET", BUCKET_NAME),
            reference_blob="drift/titanic_features_reference.csv",
            log_blob=os.environ.get("DRIFT_LOG_BLOB", "drift/predictions_log.jsonl"),
            n_latest=n_latest,
        )
        return (result, 200 if result.get("ok") else 400)
    except Exception as e:
        return ({"ok": False, "error": str(e)}, 400)
