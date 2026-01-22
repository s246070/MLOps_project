import json
from datetime import datetime, timezone
from google.cloud import storage

def append_jsonl_to_gcs(bucket_name: str, blob_name: str, record: dict) -> None:
    """
    Appender én JSON-linje til en blob i GCS.
    OK til M27. (For high-scale ville man bruge BigQuery/PubSub.)
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
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": input_features,      # [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
        "prediction": prediction,
        "proba_pos": proba_pos,
    }