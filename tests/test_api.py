import os
import pytest
import json
import joblib
from fastapi.testclient import TestClient
from app.api.main import app
from sklearn.linear_model import LogisticRegression
import numpy as np

@pytest.fixture
def client():
    """
    Fixtured client wrapper utilizing context management ('with TestClient')
    to ensure FastAPI's lifespan initialization and shutdown (background task cancellation)
    are cleanly executed.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_requests" in data

def test_model_upload_and_predict(client, tmp_path):
    """
    Verifies model registration via binary multipart uploads, automated deployment,
    and subsequent dynamic feature-aligned predictions.
    """
    # 1. Train a dummy scikit-learn model
    X = np.array([[5.0, 1.0], [25.0, 10.0], [12.0, 3.0]])
    y = np.array(["low", "high", "low"])
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # Save dummy model in a temporary path
    model_file = tmp_path / "model.joblib"
    joblib.dump(clf, model_file)
    
    # 2. Register/Upload via API
    with open(model_file, "rb") as f:
        response = client.post(
            "/models/upload",
            data={
                "model_id": "test_fraud",
                "model_name": "Test Fraud Detector",
                "version": "v1",
                "framework": "scikit-learn",
                "task_type": "classification",
                "features": json.dumps(["amount", "distance"])
            },
            files={"file": ("model.joblib", f, "application/octet-stream")}
        )
        
    assert response.status_code == 200
    json_upload = response.json()
    assert json_upload["status"] == "success"
    assert json_upload["model_id"] == "test_fraud"
    
    # 3. Test dynamic prediction routing
    pred_response = client.post(
        "/predict/test_fraud",
        json={"features": {"amount": 15.0, "distance": 2.5}}
    )
    assert pred_response.status_code == 200
    json_resp = pred_response.json()
    assert json_resp["model_id"] == "test_fraud"
    assert json_resp["model_version"] == "v1"
    assert json_resp["prediction"] == "low"
    assert json_resp["confidence"] is not None
    assert json_resp["latency_ms"] >= 0.0
    
    # 4. Check dashboard statistics aggregation
    stats_response = client.get("/dashboard/stats")
    assert stats_response.status_code == 200
    stats_data = stats_response.json()
    assert stats_data["total_predictions"] >= 1
    assert stats_data["active_models_count"] >= 1
    
    # Find test model summary
    target_summary = next((m for m in stats_data["models"] if m["model_id"] == "test_fraud"), None)
    assert target_summary is not None
    assert target_summary["throughput"] >= 1
    assert target_summary["deployment_status"] == "ACTIVE"
    
    # 5. Active model registry endpoints
    active_resp = client.get("/models/active/test_fraud")
    assert active_resp.status_code == 200
    assert active_resp.json()["version"] == "v1"
    assert active_resp.json()["filename"] == "model_v1.joblib"

    versions_resp = client.get("/models/test_fraud/versions")
    assert versions_resp.status_code == 200
    assert len(versions_resp.json()["versions"]) >= 1

    # 6. Clean up uploaded artifacts
    import shutil
    models_dir = "models/test_fraud"
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
