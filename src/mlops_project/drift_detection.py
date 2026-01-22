import io
import json
import pandas as pd
from google.cloud import storage
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

"""
This script compares the reference training features with recent production features
from GCS to detect data drift using Evidently and reaturn a simple drift status.
"""

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def run_drift_check(
    bucket_name: str = "mlops-project-models",
    reference_blob: str = "drift/titanic_features_reference.csv",
    log_blob: str = "drift/predictions_log.jsonl",
    n_latest: int = 500,
) -> dict:
    # adding this to avoid confusing edges 
    if n_latest <= 0:
        return {"ok": False, "error": "n_latest must be a postive integer"}
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Load reference
    ref_bytes = bucket.blob(reference_blob).download_as_bytes()
    ref_df = pd.read_csv(io.BytesIO(ref_bytes))
    reference = ref_df[FEATURES]

    # Load production logs
    log_text = bucket.blob(log_blob).download_as_text().strip()
    if not log_text:
        return {"ok": False, "error": "No production logs found yet"}

    # JSNOL : one JSON object per line
    lines = log_text.splitlines()
    # Only keep the latest n_latest lines
    lines = lines[-n_latest:]

    records = []
    for line in lines:
        if line.strip() !="":
            records.append(json.loads(line))

        if len(records) == 0:
            return {"ok": False, "error": "Production log file exists but contains no valid records"}

    feature_rows = []
    for r in records:
        if "features" not in r:
            return{"ok": False, "error": "A log record is mising the 'features field"}
        feature_rows.append(r["features"])

    current = pd.DataFrame([r["features"] for r in records], columns=FEATURES)

    """
    Run data drift detection and return status.
    """

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("drift_report.html")

    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"].get("dataset_drift", None)

    return {
        "ok": True,
        "dataset_drift": dataset_drift,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }