import mlflow
import mlflow.sklearn
import os
import time
import socket
from urllib.parse import urlparse

def is_mlflow_reachable(uri: str) -> bool:
    """
    Performs a fast socket connection check to see if the MLflow server is active.
    Prevents long retries and DNS blocks on host machines.
    """
    try:
        parsed = urlparse(uri)
        host = parsed.hostname
        port = parsed.port or 80
        if not host:
            return False
        
        # Test connection with a fast 0.5-second timeout
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False

class MLflowTracker:
    def __init__(self, experiment_name="MLOps_Observability_Platform"):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.experiment_name = experiment_name
        self.enabled = False
        
        # Fast reachability check
        if is_mlflow_reachable(self.tracking_uri):
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment_name)
                self.enabled = True
                print(f"MLflow Tracker initialized at {self.tracking_uri} for experiment '{experiment_name}'")
            except Exception as e:
                print(f"⚠️ MLflow initialization failed: {e}. Tracking will be disabled.")
        else:
            # Fallback to localhost if running outside Docker (for local testing/pytest)
            local_uri = "http://localhost:5000"
            if is_mlflow_reachable(local_uri):
                try:
                    mlflow.set_tracking_uri(local_uri)
                    mlflow.set_experiment(experiment_name)
                    self.enabled = True
                    print(f"MLflow Tracker connected to local fallback at {local_uri}")
                except Exception:
                    pass
            
            if not self.enabled:
                print(f"📡 MLflow server at {self.tracking_uri} is unreachable. MLflow tracking disabled.")

    def _ensure_connected(self) -> bool:
        if self.enabled:
            return True
        if is_mlflow_reachable(self.tracking_uri):
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.enabled = True
                print(f"📡 MLflow Tracker lazily re-connected successfully to {self.tracking_uri}!")
                return True
            except Exception:
                pass
        return False

    def log_inference(self, model_id, model_version, prediction, confidence, latency_ms, drift_score=0.0):
        if not self._ensure_connected(): return
        try:
            with mlflow.start_run(nested=True, run_name=f"Inference_{model_id}_{model_version}"):
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("model_version", model_version)
                mlflow.log_metric("confidence", confidence if confidence is not None else 1.0)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("drift_score", drift_score)
                
                if drift_score > 0.2:
                    mlflow.set_tag("alert", "HIGH_DRIFT")
        except Exception as e:
            print(f"Failed to log inference session to MLflow: {e}")

    def log_drift_event(self, drift_score, kl_div, alert_type):
        if not self._ensure_connected(): return
        try:
            with mlflow.start_run(run_name=f"Drift_Alert_{alert_type}"):
                mlflow.log_metric("drift_score", drift_score)
                mlflow.log_metric("kl_divergence", kl_div)
                mlflow.set_tag("event_type", "DRIFT_ALERT")
                mlflow.set_tag("alert_severity", "CRITICAL" if drift_score > 0.2 else "WARNING")
        except Exception as e:
            print(f"Failed to log drift alert to MLflow: {e}")

    def log_model_switch(self, old_version, new_version):
        if not self._ensure_connected(): return
        try:
            with mlflow.start_run(run_name="Model_Deployment"):
                mlflow.log_param("action", "MODEL_SWITCH")
                mlflow.log_param("from_version", old_version)
                mlflow.log_param("to_version", new_version)
                mlflow.set_tag("lifecycle", "REDEPLOYMENT")
        except Exception as e:
            print(f"Failed to log model switch to MLflow: {e}")

    def register_model(self, sk_model, model_name, validation_metric=None):
        """
        Registers a generic scikit-learn model to the MLflow Model Registry.
        """
        if not self._ensure_connected(): return None
        try:
            with mlflow.start_run(run_name=f"Register_{model_name}") as run:
                if validation_metric is not None:
                    mlflow.log_metric("validation_score", validation_metric)
                
                # Log the scikit-learn model estimator
                mlflow.sklearn.log_model(
                    sk_model=sk_model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Transition new version to Production
                client = mlflow.tracking.MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = versions[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    print(f"Registered {model_name} version {latest_version} as Production.")
                    return latest_version
        except Exception as e:
            print(f"Failed to register model in MLflow registry: {e}")
        return None

    def get_latest_production_model_uri(self, model_name):
        if not self._ensure_connected(): return None
        try:
            client = mlflow.tracking.MlflowClient()
            for mv in client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    return f"models:/{model_name}/Production"
        except Exception as e:
            print(f"Failed to fetch model from MLflow registry: {e}")
        return None

# Global tracker instance
tracker = MLflowTracker()
