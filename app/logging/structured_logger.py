import logging
import json
import datetime
import sys

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "request_id": getattr(record, "request_id", "N/A"),
        }
        
        # Add extra fields if they exist
        if hasattr(record, "model_id"):
            log_data["model_id"] = record.model_id
        if hasattr(record, "model_version"):
            log_data["model_version"] = record.model_version
        if hasattr(record, "prediction"):
            log_data["prediction"] = record.prediction
        if hasattr(record, "confidence"):
            log_data["confidence"] = record.confidence
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "drift_score"):
            log_data["drift_score"] = record.drift_score
        
        return json.dumps(log_data)

def setup_logger(name="mlops-app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()
