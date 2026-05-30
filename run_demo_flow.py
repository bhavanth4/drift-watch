"""One-shot demo: baseline traffic, drift, wait for retraining."""
import time

import requests

from traffic_generator import generate_random_features

API = "http://localhost:8000"


def main():
    models = requests.get(f"{API}/models", timeout=10).json()
    active = [m for m in models if m["deployment_status"] == "ACTIVE"]
    print("Active models:", [m["model_id"] for m in active])

    headers = {"X-Request-Source": "sandbox"}
    delay = 0.2  # avoid 429 rate limit (100 req / 5s)

    for m in active:
        mid, feats = m["model_id"], m["features"]
        for _ in range(35):
            f = generate_random_features(mid, feats, inject_drift=False)
            requests.post(
                f"{API}/predict/{mid}", json={"features": f}, timeout=10, headers=headers
            )
            time.sleep(delay)
        print(f"Baseline done: {mid}")

    mid = "fraud_detector"
    feats = next(m["features"] for m in active if m["model_id"] == mid)
    for i in range(15):
        f = generate_random_features(mid, feats, inject_drift=True)
        r = requests.post(
            f"{API}/predict/{mid}", json={"features": f}, timeout=10, headers=headers
        )
        r.raise_for_status()
        data = r.json()
        print(f"Drift {i}: pred={data.get('prediction')} ver={data.get('model_version')}")
        time.sleep(delay)

    print("Waiting 35s for drift scheduler + retraining...")
    time.sleep(35)

    models2 = requests.get(f"{API}/models", timeout=10).json()
    fraud = next(m for m in models2 if m["model_id"] == mid)
    print("Fraud version after wait:", fraud["version"])

    alerts = requests.get(f"{API}/alerts", timeout=10).json()
    items = alerts if isinstance(alerts, list) else alerts.get("alerts", [])
    for a in items[-10:]:
        print(
            "Alert:",
            a.get("alert_type"),
            a.get("severity"),
            (a.get("message") or "")[:90],
            "resolved=",
            a.get("resolved"),
        )


if __name__ == "__main__":
    main()
