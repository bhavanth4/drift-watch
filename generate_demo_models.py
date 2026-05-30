import os
import json
import joblib
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

API_BASE = "http://localhost:8000"

def create_and_register_fraud_model():
    print("Training Fraud Detection model...")
    # amount, distance, is_international
    X = np.array([
        [10.0, 1.0, 0],
        [25.0, 2.0, 0],
        [150.0, 25.0, 1],
        [800.0, 150.0, 1],
        [5.0, 0.5, 0],
        [1200.0, 300.0, 1]
    ])
    y = np.array([0, 0, 1, 1, 0, 1]) # 0 = Legit, 1 = Fraud
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    temp_file = "temp_fraud.joblib"
    joblib.dump(clf, temp_file)
    
    print("Uploading Fraud Detection model to registry...")
    try:
        with open(temp_file, "rb") as f:
            res = requests.post(
                f"{API_BASE}/models/upload",
                data={
                    "model_id": "fraud_detector",
                    "model_name": "Credit Card Fraud Detector",
                    "version": "v1",
                    "framework": "scikit-learn",
                    "task_type": "classification",
                    "features": json.dumps(["amount", "distance", "is_international"])
                },
                files={"file": ("fraud_detector.joblib", f, "application/octet-stream")}
            )
            print("Response:", res.status_code, res.json())
    except Exception as e:
        print("Registration failed:", e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_and_register_churn_model():
    print("Training Customer Churn model...")
    # tenure, monthly_charges, support_calls
    X = np.array([
        [12, 50.0, 1],
        [2, 120.0, 5],
        [60, 30.0, 0],
        [5, 95.0, 4],
        [45, 60.0, 2],
        [1, 110.0, 7]
    ])
    y = np.array([0, 1, 0, 1, 0, 1]) # 0 = Retained, 1 = Churned
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    temp_file = "temp_churn.joblib"
    joblib.dump(clf, temp_file)
    
    print("Uploading Customer Churn model to registry...")
    try:
        with open(temp_file, "rb") as f:
            res = requests.post(
                f"{API_BASE}/models/upload",
                data={
                    "model_id": "customer_churn",
                    "model_name": "Customer Churn Predictor",
                    "version": "v1",
                    "framework": "scikit-learn",
                    "task_type": "classification",
                    "features": json.dumps(["tenure", "monthly_charges", "support_calls"])
                },
                files={"file": ("customer_churn.joblib", f, "application/octet-stream")}
            )
            print("Response:", res.status_code, res.json())
    except Exception as e:
        print("Registration failed:", e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_and_register_house_model():
    print("Training House Price Prediction model...")
    # sqft, bedrooms, bathrooms, age
    X = np.array([
        [1200, 3, 2.0, 15],
        [850, 2, 1.0, 40],
        [2400, 4, 2.5, 5],
        [3100, 4, 3.5, 2],
        [1500, 3, 2.0, 25],
        [1800, 3, 2.0, 10]
    ])
    y = np.array([280000.0, 150000.0, 490000.0, 680000.0, 320000.0, 390000.0]) # Price in USD
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    temp_file = "temp_house.joblib"
    joblib.dump(reg, temp_file)
    
    print("Uploading House Price model to registry...")
    try:
        with open(temp_file, "rb") as f:
            res = requests.post(
                f"{API_BASE}/models/upload",
                data={
                    "model_id": "house_price_predictor",
                    "model_name": "House Price Predictor",
                    "version": "v1",
                    "framework": "scikit-learn",
                    "task_type": "regression",
                    "features": json.dumps(["sqft", "bedrooms", "bathrooms", "age"])
                },
                files={"file": ("house_price_predictor.joblib", f, "application/octet-stream")}
            )
            print("Response:", res.status_code, res.json())
    except Exception as e:
        print("Registration failed:", e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    create_and_register_fraud_model()
    print("-" * 50)
    create_and_register_churn_model()
    print("-" * 50)
    create_and_register_house_model()
    print("-" * 50)
    print("All demo models generated and registered successfully!")
