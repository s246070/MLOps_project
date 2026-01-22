import json
from datetime import datetime, timezone
from google.cloud import storage

"""
This script logs model inputs and predictions to GCS in a JSNOL-format
so the features can later be used for monitoring and data drift detection.
"""

def append_jsonl_to_gcs(bucket_name: str, blob_name: str, record: dict) -> None:

    """
    Creating GCS client (connection to GCS) and locates bucket + blob.
    """

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    line = json.dumps(record) + "\n"

    if blob.exists():
        existing = blob.download_as_text()
        blob.upload_from_string(existing + line, content_type="application/json")
    else:
        blob.upload_from_string(line, content_type="application/json")

def make_log_record(input_features: list[float], prediction: int, proba_pos: float | None):
    """
    The record contains timestamp, input-features used by the model,
    prediction class and probability for the positive class.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": input_features,      # [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
        "prediction": prediction,
        "proba_pos": proba_pos,
    }