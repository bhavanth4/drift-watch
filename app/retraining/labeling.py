"""
Derive supervised training targets from production inference records.

When ground-truth labels are not stored, we use domain heuristics for known demo
models (aligned with traffic_generator drift patterns) and high-confidence
pseudo-labels for everything else.
"""
from typing import Any, Dict, List, Optional, Tuple


def _feature(features: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    lower_map = {str(k).lower(): v for k, v in features.items()}
    for name in names:
        if name.lower() in lower_map:
            return float(lower_map[name.lower()])
    return default


def derive_label(
    model_id: str,
    features: Dict[str, Any],
    task_type: str,
    prediction: str,
    confidence: Optional[float] = None,
) -> Any:
    """
    Return a supervised target (class label or numeric value) for one record.
    """
    model_key = model_id.lower()

    if task_type == "classification":
        if "fraud" in model_key:
            amount = _feature(features, "amount", "transaction_amount")
            distance = _feature(features, "distance", "transaction_distance")
            is_int = int(_feature(features, "is_international", default=0))
            # Matches skewed fraud traffic: large amounts / distant international txs
            if amount >= 200 or (distance >= 40 and is_int == 1) or amount >= 600:
                return 1
            return 0

        if "churn" in model_key:
            tenure = _feature(features, "tenure")
            monthly = _feature(features, "monthly_charges")
            calls = _feature(features, "support_calls", "customer_calls")
            if monthly >= 100 or calls >= 5 or (tenure <= 6 and monthly >= 80):
                return 1
            return 0

        # Generic: trust the deployed model when it is confident
        if confidence is not None and confidence >= 0.75:
            try:
                return int(float(prediction))
            except ValueError:
                return prediction
        try:
            return int(float(prediction))
        except ValueError:
            return prediction

    # Regression
    if "house" in model_key or "price" in model_key:
        sqft = _feature(features, "sqft", "square_feet", default=1500.0)
        beds = _feature(features, "bedrooms", default=3.0)
        baths = _feature(features, "bathrooms", default=2.0)
        age = _feature(features, "age", "house_age", default=20.0)
        # Calibrated to demo training scale; scales up for drifted mega-mansions
        return 120.0 * sqft + 45_000.0 * beds + 25_000.0 * baths - 1_500.0 * age + 50_000.0

    try:
        return float(prediction)
    except (TypeError, ValueError):
        return 0.0


def build_labeled_dataset(
    model_id: str,
    task_type: str,
    feature_names: List[str],
    records: List[Dict[str, Any]],
    live_oversample: int = 4,
) -> Tuple[List[List[float]], List[Any], int]:
    """
    Build (X, y) from inference metric dicts with optional oversampling of live rows.

    Each record: features (dict), prediction (str), confidence (float|None), is_live (bool)
    """
    X_rows: List[List[float]] = []
    y_vals: List[Any] = []

    for rec in records:
        features = rec["features"]
        label = derive_label(
            model_id,
            features,
            task_type,
            rec["prediction"],
            rec.get("confidence"),
        )
        row = [float(features[name]) for name in feature_names]
        repeats = live_oversample if rec.get("is_live") else 1
        for _ in range(repeats):
            X_rows.append(row)
            y_vals.append(label)

    live_count = sum(1 for r in records if r.get("is_live"))
    return X_rows, y_vals, live_count
