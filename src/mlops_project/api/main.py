import os
import pickle
import functions_framework
from google.cloud import storage

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "mlops-project-models")
MODEL_FILE = os.environ.get("MODEL_BLOB", "model.pkl")


def load_model():
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(MODEL_FILE)
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

MODEL = load_model()



def parse_instances(request):
    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("Body must be JSON")

    # allow raw JSON list: [1,2,3]
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
    try:
        instances = parse_instances(request)

        # IMPORTANT:
        # If your model expects 4 features in one sample (iris),
        # then instances should have length 4 and we reshape to (1,4).
        X = [ [float(x) for x in instances] ]

        pred = MODEL.predict(X)
        return {"prediction": pred.tolist()}

    except Exception as e:
        return {"error": str(e)}, 400