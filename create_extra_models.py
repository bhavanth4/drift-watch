import os
import json
import joblib
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

API_BASE = "http://localhost:8000"

def create_and_register_loan_model():
    print("Training Loan Approval Predictor model...")
    # income (k$), credit_score, loan_amount (k$)
    X = np.array([
        [50.0, 600, 15.0],
        [120.0, 750, 300.0],
        [30.0, 580, 10.0],
        [85.0, 680, 150.0],
        [200.0, 800, 500.0],
        [40.0, 500, 50.0]
    ])
    y = np.array([1, 1, 0, 1, 1, 0]) # 1 = Approved, 0 = Rejected
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    temp_file = "temp_loan.joblib"
    joblib.dump(clf, temp_file)
    
    print("Uploading Loan Approval model to registry...")
    try:
        with open(temp_file, "rb") as f:
            res = requests.post(
                f"{API_BASE}/models/upload",
                data={
                    "model_id": "loan_predictor",
                    "model_name": "Loan Approval Predictor",
                    "version": "v1",
                    "framework": "scikit-learn",
                    "task_type": "classification",
                    "features": json.dumps(["income", "credit_score", "loan_amount"])
                },
                files={"file": ("loan_predictor.joblib", f, "application/octet-stream")}
            )
            print("Response:", res.status_code, res.json())
    except Exception as e:
        print("Registration failed:", e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_and_register_sales_model():
    print("Training Sales Predictor model...")
    # advertising_budget (k$), competitor_price, seasonality (1-4)
    X = np.array([
        [10.0, 15.0, 1],
        [50.0, 14.5, 2],
        [100.0, 16.0, 4],
        [5.0, 15.5, 1],
        [80.0, 13.0, 3],
        [20.0, 15.0, 2]
    ])
    y = np.array([25.0, 75.0, 180.0, 12.0, 110.0, 45.0]) # Sales in k units
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    temp_file = "temp_sales.joblib"
    joblib.dump(reg, temp_file)
    
    print("Uploading Sales Predictor model to registry...")
    try:
        with open(temp_file, "rb") as f:
            res = requests.post(
                f"{API_BASE}/models/upload",
                data={
                    "model_id": "sales_predictor",
                    "model_name": "Sales Volume Predictor",
                    "version": "v1",
                    "framework": "scikit-learn",
                    "task_type": "regression",
                    "features": json.dumps(["advertising_budget", "competitor_price", "seasonality"])
                },
                files={"file": ("sales_predictor.joblib", f, "application/octet-stream")}
            )
            print("Response:", res.status_code, res.json())
    except Exception as e:
        print("Registration failed:", e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    create_and_register_loan_model()
    print("-" * 50)
    create_and_register_sales_model()
    print("-" * 50)
    print("Both extra models generated and registered successfully!")
