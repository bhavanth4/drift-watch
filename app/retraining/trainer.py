"""
Fit a new scikit-learn estimator from production inference history after drift.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from app.database import InferenceMetric, SessionLocal
from app.logging.structured_logger import logger
from app.models.loader import load_model_from_file
from app.models.registry import build_model_path, version_to_filename
from app.retraining.labeling import build_labeled_dataset

MIN_SAMPLES = 15
LIVE_WINDOW = 10
METRIC_LIMIT = 200


def _records_from_metrics(
    metrics: List[InferenceMetric],
    feature_names: List[str],
    live_window: int = LIVE_WINDOW,
) -> List[Dict[str, Any]]:
    ordered = list(reversed(metrics))
    live_cutoff = max(0, len(ordered) - live_window)
    records: List[Dict[str, Any]] = []

    for idx, m in enumerate(ordered):
        try:
            features = json.loads(m.features_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not all(name in features for name in feature_names):
            continue
        records.append(
            {
                "features": features,
                "prediction": m.prediction,
                "confidence": m.confidence,
                "is_live": idx >= live_cutoff,
            }
        )
    return records


def _clone_estimator(existing: Any, task_type: str) -> Any:
    try:
        return clone(existing)
    except Exception:
        if task_type == "regression":
            return LinearRegression()
        return LogisticRegression(max_iter=1000)


def _evaluate(model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
    if len(y) < 2:
        return 0.0
    if task_type == "regression":
        return float(r2_score(y, model.predict(X)))
    return float(accuracy_score(y, model.predict(X)))


def retrain_from_production_data(model_meta: Any, file_path: str) -> Tuple[Any, float, Dict[str, Any]]:
    feature_names: List[str] = json.loads(model_meta.features)
    db = SessionLocal()
    try:
        metrics = (
            db.query(InferenceMetric)
            .filter_by(model_id=model_meta.model_id, version=model_meta.version)
            .order_by(InferenceMetric.timestamp.desc())
            .limit(METRIC_LIMIT)
            .all()
        )
    finally:
        db.close()

    if len(metrics) < MIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_SAMPLES} logged predictions to retrain; got {len(metrics)}."
        )

    records = _records_from_metrics(metrics, feature_names)
    if len(records) < MIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_SAMPLES} valid feature rows; got {len(records)}."
        )

    X_list, y_list, live_count = build_labeled_dataset(
        model_meta.model_id,
        model_meta.task_type,
        feature_names,
        records,
        live_oversample=4,
    )

    X = np.array(X_list, dtype=float)
    if model_meta.task_type == "classification":
        y = np.array(y_list)
    else:
        y = np.array([float(v) for v in y_list])

    existing = load_model_from_file(file_path)
    estimator = _clone_estimator(existing, model_meta.task_type)

    n_holdout = min(live_count or LIVE_WINDOW, max(1, len(records) // 5))
    if len(X) > n_holdout + 5:
        X_train, X_val = X[:-n_holdout], X[-n_holdout:]
        y_train, y_val = y[:-n_holdout], y[-n_holdout:]
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    logger.info(
        f"Retraining {model_meta.model_id}: samples={len(X_train)}, "
        f"holdout={len(X_val)}, live_window_rows={live_count}"
    )

    estimator.fit(X_train, y_train)
    validation_score = _evaluate(estimator, X_val, y_val, model_meta.task_type)

    return estimator, validation_score, {
        "train_samples": int(len(X_train)),
        "total_samples": int(len(X)),
        "live_window_rows": int(live_count),
        "validation_score": validation_score,
        "task_type": model_meta.task_type,
    }


def save_retrained_model(
    estimator: Any,
    model_id: str,
    version: str,
    ext: str = ".joblib",
) -> Tuple[str, str]:
    """Save artifact as model_vN.joblib and return (file_path, filename)."""
    file_path = build_model_path(model_id, version, ext)
    filename = version_to_filename(version, ext)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if ext == ".joblib":
        joblib.dump(estimator, file_path)
    else:
        import pickle

        with open(file_path, "wb") as f:
            pickle.dump(estimator, f)
    return file_path, filename
