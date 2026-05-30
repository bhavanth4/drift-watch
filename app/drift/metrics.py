from prometheus_client import Counter, Histogram, Gauge

# Inference Metrics
INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total", 
    "Total number of model inference requests", 
    ["model_id", "model_version", "prediction"]
)

INFERENCE_LATENCY_MS = Histogram(
    "inference_latency_ms", 
    "Model inference latency in milliseconds", 
    ["model_id", "model_version"],
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000)
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence", 
    "Confidence scores or regression outputs for predictions", 
    ["model_id", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Drift Metrics
DRIFT_SCORE = Gauge(
    "drift_score", 
    "Drift score (PSI or KS Divergence)", 
    ["model_id", "metric_type"]
)

DRIFT_ALERTS_TOTAL = Counter(
    "drift_alerts_total", 
    "Total number of drift alerts generated", 
    ["model_id", "alert_type"]
)

LATENCY_ANOMALY_SCORE = Gauge(
    "latency_anomaly_score", 
    "Z-score of current latency vs history",
    ["model_id"]
)
