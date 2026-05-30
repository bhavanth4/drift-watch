import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

def generate_and_save_binaries():
    output_dir = "demo_model_binaries"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating demo model binaries inside directory: '{output_dir}'...\n")

    # 1. Train and save Fraud Detector (Classification)
    print("Training 'fraud_detector'...")
    X_fraud = np.array([
        [10.0, 1.0, 0],
        [25.0, 2.0, 0],
        [150.0, 25.0, 1],
        [800.0, 150.0, 1],
        [5.0, 0.5, 0],
        [1200.0, 300.0, 1]
    ])
    y_fraud = np.array([0, 0, 1, 1, 0, 1])
    clf_fraud = LogisticRegression()
    clf_fraud.fit(X_fraud, y_fraud)
    
    # Save both standard and suffix variants
    joblib.dump(clf_fraud, os.path.join(output_dir, "fraud_detector.joblib"))
    joblib.dump(clf_fraud, os.path.join(output_dir, "fraud_detector.joblib.txt"))
    print("  -> Saved 'fraud_detector.joblib'")
    print("  -> Saved 'fraud_detector.joblib.txt'\n")

    # 2. Train and save Customer Churn (Classification)
    print("Training 'customer_churn'...")
    X_churn = np.array([
        [12, 50.0, 1],
        [2, 120.0, 5],
        [60, 30.0, 0],
        [5, 95.0, 4],
        [45, 60.0, 2],
        [1, 110.0, 7]
    ])
    y_churn = np.array([0, 1, 0, 1, 0, 1])
    clf_churn = LogisticRegression()
    clf_churn.fit(X_churn, y_churn)
    
    joblib.dump(clf_churn, os.path.join(output_dir, "customer_churn.joblib"))
    joblib.dump(clf_churn, os.path.join(output_dir, "customer_churn.joblib.txt"))
    print("  -> Saved 'customer_churn.joblib'")
    print("  -> Saved 'customer_churn.joblib.txt'\n")

    # 3. Train and save House Price Predictor (Regression)
    print("Training 'house_price_predictor'...")
    X_house = np.array([
        [1200, 3, 2.0, 15],
        [850, 2, 1.0, 40],
        [2400, 4, 2.5, 5],
        [3100, 4, 3.5, 2],
        [1500, 3, 2.0, 25],
        [1800, 3, 2.0, 10]
    ])
    y_house = np.array([280000.0, 150000.0, 490000.0, 680000.0, 320000.0, 390000.0])
    reg_house = LinearRegression()
    reg_house.fit(X_house, y_house)
    
    joblib.dump(reg_house, os.path.join(output_dir, "house_price_predictor.joblib"))
    joblib.dump(reg_house, os.path.join(output_dir, "house_price_predictor.joblib.txt"))
    print("  -> Saved 'house_price_predictor.joblib'")
    print("  -> Saved 'house_price_predictor.joblib.txt'\n")

    print("-" * 50)
    print(f"All demo binaries successfully created and 'kept aside' in standard and suffix text modes!")

if __name__ == "__main__":
    generate_and_save_binaries()
