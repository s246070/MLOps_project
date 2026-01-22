# M27/M28 Requirements Completion Checklist

## Overview
Analysis of the MLOps project against the specified requirements for M27 (Data Drift Detection) and M28 (Monitoring & Alerting).

---

## M27 - Data Drift Detection

### 1. ✅ COMPLETED: Check how robust your model is towards data drifting
**Status:** DONE  
**Evidence:**
- Drift detection implementation exists in [src/mlops_project/drift_detection.py](src/mlops_project/drift_detection.py)
- Uses Evidently AI library (`evidently>=0.4,<0.5` in pyproject.toml)
- `DataDriftPreset()` metric is configured to detect dataset drift across all features
- Drift detection compares reference data against production logs
- Output: `drift_report.html` shows drift analysis results (7 out of 7 features show drift)
- Reference data stored at: `drift/titanic_features_reference.csv`

**What's implemented:**
- Drift detection using Evidently's DataDriftPreset
- Comparison logic between reference and current datasets
- HTML report generation with visualizations

---

### 2. ✅ COMPLETED: Setup collection of input-output data from your deployed application
**Status:** DONE  
**Evidence:**
- Input-output logging system implemented in [src/mlops_project/drift_logging.py](src/mlops_project/drift_logging.py)
- Logs predictions + features to GCS in JSONL format
- Called from API in [src/mlops_project/api/main.py](src/mlops_project/api/main.py) lines 165-167
- Environment variables configured:
  - `DRIFT_BUCKET` - where logs are stored
  - `DRIFT_LOG_BLOB` - default: `drift/predictions_log.jsonl`
- Log record structure: `{timestamp, features, prediction, proba_pos}`
- Each API inference call automatically logs to GCS

**What's implemented:**
- JSONL logging to Google Cloud Storage
- Automatic timestamp generation (UTC)
- Feature extraction and storage
- Prediction + probability logging
- Handles append operations to existing logs

---

### 3. ✅ COMPLETED: Deploy to the cloud a drift detection API
**Status:** DONE  
**Evidence:**
- Cloud Run deployment configured in [cloudbuild.yaml](cloudbuild.yaml)
- Two Cloud Functions endpoints implemented in [src/mlops_project/api/main.py](src/mlops_project/api/main.py):
  
  **Endpoint 1: Inference API**
  - Function: `logreg_classifier`
  - Handles model inference with PyTorch models
  - Health check endpoint: GET `/health`
  - Returns predictions + probabilities
  
  **Endpoint 2: Drift Check API**
  - Function: `drift_check` (lines 201-218)
  - On-demand drift detection endpoint
  - Accepts `n_latest` parameter (default: 500)
  - Returns drift status and statistics
  - HTTP endpoint accessible via GET request

- Docker image: `eu.gcr.io/$PROJECT_ID/titanic-inference-api`
- Region: `europe-west1` (Cloud Run)
- Service account configured: `titanic-mlops-484412@appspot.gserviceaccount.com`
- Unauthenticated access allowed

**What's implemented:**
- Google Cloud Functions framework integration
- Cloud Run deployment pipeline
- Docker containerization via [dockerfiles/api.dockerfile](dockerfiles/api.dockerfile)
- CI/CD via Cloud Build
- Model loading from GCS bucket
- Automated prediction logging

---

## M28 - Monitoring & Alerting

### 4. ❌ MISSING: Instrument your API with a couple of system metrics
**Status:** NOT STARTED  
**What's missing:**
- No metrics instrumentation found in the codebase
- No Prometheus metrics library
- No OpenTelemetry integration
- No custom system metric collection

**What needs to be added:**
- Response time/latency metrics
- Request count/throughput metrics
- Error rate metrics
- Model inference duration
- GCS I/O latency
- Memory/CPU usage metrics
- HTTP status code tracking

**Recommended approach:**
- Add `prometheus-client` to dependencies
- Instrument API endpoints with decorators
- Export metrics in Prometheus format
- OR use OpenTelemetry with Google Cloud Trace/Monitoring

**Example needed:**
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
request_duration = Histogram('api_request_duration_seconds', 'Request latency', ['endpoint'])
predictions_made = Counter('predictions_total', 'Total predictions made')
inference_errors = Counter('inference_errors_total', 'Inference errors')
```

---

### 5. ❌ MISSING: Setup cloud monitoring of your instrumented application
**Status:** NOT STARTED  
**What's missing:**
- No Google Cloud Monitoring integration
- No Cloud Logging configuration
- No dashboards created
- No metric export to GCP Monitoring
- No uptime checks
- No custom metrics dashboard

**What needs to be added:**
- Google Cloud Monitoring client setup
- Custom metrics registration
- Log aggregation configuration
- Dashboard creation (Cloud Console)
- Cloud Logging agent/exporter
- Metric export from Prometheus to Cloud Monitoring

**Recommended approach:**
- Use `google-cloud-monitoring` library
- Create custom metrics in GCP Monitoring
- Setup Cloud Logging for application logs
- Create Cloud Monitoring dashboard for visualization
- Configure metric thresholds

---

### 6. ❌ MISSING: Create one or more alert systems in GCP to alert if app is not behaving correctly
**Status:** NOT STARTED  
**What's missing:**
- No GCP Alert Policies created
- No notification channels configured
- No alerting rules defined
- No Pub/Sub integration for alerts
- No email/SMS notification setup

**What needs to be added:**
- GCP Monitoring Alert Policies
- Notification channels (email, Slack, PagerDuty, etc.)
- Alert conditions:
  - High error rates
  - High latency (response time > threshold)
  - Low throughput
  - Model drift detection triggers
  - GCS bucket size/cost anomalies
  - Service downtime

**Recommended thresholds:**
- Error rate > 5%
- P95 latency > 1000ms
- Request rate < 1 req/min (potential downtime)
- Dataset drift detected
- Model inference time > 500ms

**Example alert policy structure needed:**
```yaml
- name: "API Error Rate High"
  condition: api_error_rate > 0.05
  duration: 5 minutes
  notification_channel: email, slack
  
- name: "Drift Detected"
  condition: dataset_drift == true
  notification_channel: email
  
- name: "High Latency"
  condition: p95_latency > 1000ms
  duration: 10 minutes
  notification_channel: slack
```

---

## Summary Table

| Requirement | Status | Evidence |
|-------------|--------|----------|
| M27.1: Model drift robustness check | ✅ DONE | [drift_detection.py](src/mlops_project/drift_detection.py) |
| M27.2: Input-output data collection | ✅ DONE | [drift_logging.py](src/mlops_project/drift_logging.py) |
| M27.3: Deploy drift detection API | ✅ DONE | [cloudbuild.yaml](cloudbuild.yaml), [main.py](src/mlops_project/api/main.py) |
| M28.1: System metrics instrumentation | ❌ MISSING | Need to add Prometheus/OpenTelemetry |
| M28.2: Cloud monitoring setup | ❌ MISSING | Need to configure GCP Monitoring |
| M28.3: GCP alert systems | ❌ MISSING | Need to create Alert Policies |

---

## Priority Implementation Order

1. **High Priority (M28.1):** Add metrics instrumentation to API
   - Estimated effort: 2-3 hours
   - Impact: Enables monitoring and alerting

2. **High Priority (M28.2):** Setup GCP Monitoring dashboard
   - Estimated effort: 2-3 hours
   - Impact: Visibility into system behavior

3. **High Priority (M28.3):** Create GCP Alert Policies
   - Estimated effort: 1-2 hours
   - Impact: Proactive incident notification

---

## Current Architecture Status

### ✅ Working
- Model serving via Cloud Functions
- Prediction logging to GCS
- Drift detection calculation
- Cloud Run deployment
- Docker containerization

### ⚠️ Partial
- API functionality exists but lacks instrumentation
- Drift detection works but alerts not setup
- Data collection works but monitoring not configured

### ❌ Missing
- System metrics collection
- Cloud Monitoring integration
- Alert policies and notifications
- Monitoring dashboards
- Performance baselines

