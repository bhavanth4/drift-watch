import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.database import (
    SessionLocal,
    init_db,
    register_model_version,
    deploy_model,
    log_inference,
    get_active_model_meta,
)
from app.retraining.labeling import derive_label
from app.retraining.trainer import retrain_from_production_data


@pytest.fixture
def db_with_fraud_metrics(tmp_path):
    init_db()
    model_id = "fraud_detector"
    model_path = tmp_path / "model_v1.joblib"
    clf = LogisticRegression()
    X = np.array([[10, 1, 0], [25, 2, 0], [150, 25, 1], [800, 150, 1]])
    y = np.array([0, 0, 1, 1])
    clf.fit(X, y)
    joblib.dump(clf, model_path)

    db = SessionLocal()
    register_model_version(
        db,
        model_id=model_id,
        name="Fraud",
        version="v1",
        framework="scikit-learn",
        task_type="classification",
        features=["amount", "distance", "is_international"],
        file_path=str(model_path),
        filename="model_v1.joblib",
        activate=True,
    )
    deploy_model(db, model_id, "v1")

    for amount, distance, is_int in [
        (20, 2, 0),
        (35, 4, 0),
        (12, 1, 0),
        (80, 8, 0),
        (45, 5, 0),
        (60, 7, 0),
        (30, 3, 0),
        (55, 6, 0),
        (25, 2, 0),
        (40, 4, 0),
        (70, 9, 0),
        (15, 1, 0),
        (50, 5, 0),
        (65, 8, 0),
        (22, 2, 0),
    ]:
        feats = {"amount": amount, "distance": distance, "is_international": is_int}
        pred = clf.predict(np.array([[amount, distance, is_int]]))[0]
        log_inference(db, model_id, "v1", 5.0, str(pred), 0.9, feats)

    for amount, distance, is_int in [
        (1200, 200, 1),
        (1800, 350, 1),
        (950, 180, 1),
        (2100, 400, 1),
        (1500, 250, 1),
        (1100, 160, 1),
        (2000, 380, 1),
        (1300, 220, 1),
        (1700, 300, 1),
        (1600, 280, 1),
    ]:
        feats = {"amount": amount, "distance": distance, "is_international": is_int}
        pred = clf.predict(np.array([[amount, distance, is_int]]))[0]
        log_inference(db, model_id, "v1", 5.0, str(pred), 0.6, feats)

    yield db, str(model_path)
    db.close()


def test_retrain_produces_new_estimator(db_with_fraud_metrics):
    db, model_path = db_with_fraud_metrics
    model = get_active_model_meta(db, "fraud_detector")
    assert model is not None

    new_model, score, info = retrain_from_production_data(model, model_path)
    assert info["train_samples"] >= 15
    assert info["live_window_rows"] == 10
    assert score >= 0.0

    old = joblib.load(model_path)
    drift_cases = [
        {"amount": 1500, "distance": 250, "is_international": 1},
        {"amount": 2100, "distance": 400, "is_international": 1},
    ]
    old_correct = 0
    new_correct = 0
    for feats in drift_cases:
        row = np.array([[feats["amount"], feats["distance"], feats["is_international"]]])
        label = derive_label("fraud_detector", feats, "classification", "0")
        old_correct += int(old.predict(row)[0]) == label
        new_correct += int(new_model.predict(row)[0]) == label
    assert new_correct >= old_correct
    assert new_correct == len(drift_cases)
