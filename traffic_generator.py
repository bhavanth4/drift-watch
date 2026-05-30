import requests
import time
import random
import sys
import argparse

API_BASE = "http://localhost:8000"

def generate_random_features(model_id: str, feature_names: list, inject_drift: bool = False) -> dict:
    """
    Generates realistic, schema-appropriate input features for recommended demo models.
    Supports injecting skewed out-of-distribution data when inject_drift is True.
    """
    features = {}
    
    # 1. Fraud Detector Schema
    if "fraud" in model_id.lower():
        # Baseline transaction features
        amount = random.uniform(5.0, 150.0)
        distance = random.uniform(0.1, 15.0)
        is_international = random.choice([0, 0, 0, 0, 1]) # Mostly local transactions
        
        # Inject anomalies / drift
        if inject_drift:
            amount = random.uniform(800.0, 2500.0) # Massive price spike!
            distance = random.uniform(120.0, 500.0) # Distant, weird transactions!
            is_international = 1 # Force international flag
            
        mapping = {
            "amount": amount,
            "distance": distance,
            "is_international": is_international,
            "transaction_amount": amount,
            "transaction_distance": distance
        }
        for name in feature_names:
            features[name] = mapping.get(name.lower(), random.uniform(10.0, 100.0))
            
    # 2. Customer Churn Schema
    elif "churn" in model_id.lower():
        tenure = random.randint(3, 60)
        monthly_charges = random.uniform(20.0, 85.0)
        support_calls = random.choice([0, 0, 1, 1, 2])
        
        if inject_drift:
            tenure = random.randint(1, 5) # New customers only
            monthly_charges = random.uniform(120.0, 250.0) # Sky-high monthly bills!
            support_calls = random.randint(6, 12) # Endless customer service complaints!
            
        mapping = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "customer_calls": support_calls
        }
        for name in feature_names:
            features[name] = mapping.get(name.lower(), random.uniform(5.0, 50.0))
            
    # 3. House Price Schema
    elif "house" in model_id.lower() or "price" in model_id.lower():
        sqft = random.uniform(900.0, 3200.0)
        bedrooms = random.randint(1, 4)
        bathrooms = float(random.choice([1.0, 1.5, 2.0, 2.5, 3.0]))
        age = random.uniform(2.0, 50.0)
        
        if inject_drift:
            sqft = random.uniform(5000.0, 12000.0) # Mega mansions!
            bedrooms = random.randint(6, 12) # Huge number of rooms
            bathrooms = float(random.randint(5, 8))
            age = random.uniform(90.0, 120.0) # Century-old houses
            
        mapping = {
            "sqft": sqft,
            "square_feet": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "house_age": age
        }
        for name in feature_names:
            features[name] = mapping.get(name.lower(), random.uniform(1.0, 10.0))
            
    # 4. Fallback Generic Tabular Schema
    else:
        # Standard numeric bounds
        for name in feature_names:
            val = random.uniform(0.0, 1.0) if not inject_drift else random.uniform(10.0, 50.0)
            features[name] = val
            
    return features

def main():
    parser = argparse.ArgumentParser(description="Academic MLOps Traffic and Drift Simulator")
    parser.add_argument("--drift", action="store_true", help="Start immediately with drifted features")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds to sleep between request pulses")
    args = parser.parse_args()
    
    print("=================================================================")
    print("    === MLOPS PLATFORM TRAFFIC & DRIFT SIMULATION ENGINE ===")
    print("=================================================================")
    print(f"Targeting Gateway: {API_BASE}")
    print(f"Interval: {args.interval}s | Drift Mode: {'ACTIVE' if args.drift else 'INACTIVE'}")
    print("Press Ctrl+C to shutdown traffic generation.")
    
    inject_drift = args.drift
    count = 0
    
    try:
        while True:
            # 1. Dynamically query registered models
            try:
                res = requests.get(f"{API_BASE}/models", timeout=3)
                if res.status_code != 200:
                    print("[WARNING] Gateway error when fetching registry models. Retrying...")
                    time.sleep(5)
                    continue
                models = res.json()
            except Exception as e:
                print(f"[WARNING] Failed to connect to MLOps API Gateway: {e}. Retrying in 5s...")
                time.sleep(5)
                continue
                
            active_models = [m for m in models if m["deployment_status"] == "ACTIVE"]
            if not active_models:
                print("[INFO] No ACTIVE models found in registry. Please upload and activate a model via the UI.")
                time.sleep(5)
                continue
                
            # 2. Pick a model and generate data
            model = random.choice(active_models)
            model_id = model["model_id"]
            features_list = model["features"]
            
            payload = {
                "features": generate_random_features(model_id, features_list, inject_drift=inject_drift)
            }
            
            try:
                start_time = time.time()
                pred_res = requests.post(f"{API_BASE}/predict/{model_id}", json=payload, timeout=5)
                latency = (time.time() - start_time) * 1000
                
                if pred_res.status_code == 200:
                    data = pred_res.json()
                    conf_str = f" | Confidence: {data['confidence']:.2%}" if data['confidence'] is not None else ""
                    drift_tag = "[DRIFT ACTIVE]" if inject_drift else "[STABLE]"
                    print(
                        f"[{count:04d}] {drift_tag} Model: {model_id} ({data['model_version']}) -> "
                        f"Pred: {data['prediction']}{conf_str} | Latency: {latency:.1f}ms"
                    )
                elif pred_res.status_code == 429:
                    print(f"[{count:04d}] Rate Limited (429). Slowing down traffic generation...")
                    time.sleep(3)
                else:
                    print(f"[{count:04d}] Prediction failed with status: {pred_res.status_code} | {pred_res.text}")
            except Exception as e:
                print(f"[{count:04d}] Connection error during predict request: {e}")
                
            count += 1
            time.sleep(args.interval)
            
            # Periodically print instructions on injecting drift
            if count % 20 == 0:
                print("\n[TIP] To simulate production drift and trigger automated self-healing,")
                print("   you can pass the '--drift' argument to this traffic generator script!\n")
                
    except KeyboardInterrupt:
        print("\n[INFO] Live inference traffic simulation shut down cleanly.")

if __name__ == "__main__":
    main()
