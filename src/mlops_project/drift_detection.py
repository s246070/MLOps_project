import io
import json
import pandas as pd
from google.cloud import storage
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def run_drift_check(
    bucket_name: str = "mlops-project-models",
    reference_blob: str = "drift/titanic_features_reference.csv",
    log_blob: str = "drift/predictions_log.jsonl",
    n_latest: int = 500,
) -> dict:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Load reference
    ref_bytes = bucket.blob(reference_blob).download_as_bytes()
    reference = pd.read_csv(io.BytesIO(ref_bytes))[FEATURES]

    # Load production logs
    log_text = bucket.blob(log_blob).download_as_text().strip()
    if not log_text:
        return {"ok": False, "error": "No production logs found yet"}

    records = [json.loads(line) for line in log_text.splitlines()]
    records = records[-n_latest:]

    current = pd.DataFrame([r["features"] for r in records], columns=FEATURES)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    rep = report.as_dict()
    dataset_drift = rep["metrics"][0]["result"].get("dataset_drift", None)

    return {
        "ok": True,
        "dataset_drift": dataset_drift,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }