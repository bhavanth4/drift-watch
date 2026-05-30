import os
import json
import joblib
import numpy as np

def convert_binaries_to_json():
    input_dir = "demo_model_binaries"
    print(f"Reading binary models from '{input_dir}' and converting to plain-text JSON format...\n")

    # Model schemas to enrich the JSON parameters
    models_metadata = {
        "fraud_detector": {
            "model_name": "Credit Card Fraud Detector",
            "task_type": "classification",
            "features_schema": ["amount", "distance", "is_international"]
        },
        "customer_churn": {
            "model_name": "Customer Churn Predictor",
            "task_type": "classification",
            "features_schema": ["tenure", "monthly_charges", "support_calls"]
        },
        "house_price_predictor": {
            "model_name": "House Price Predictor",
            "task_type": "regression",
            "features_schema": ["sqft", "bedrooms", "bathrooms", "age"]
        }
    }

    for model_id, meta in models_metadata.items():
        binary_file = f"{model_id}.joblib"
        binary_path = os.path.join(input_dir, binary_file)
        
        if not os.path.exists(binary_path):
            print(f"⚠️ Binary file not found: {binary_path}. Skipping.")
            continue
            
        print(f"Loading '{binary_file}'...")
        try:
            # 1. Deserialize the model binary
            model = joblib.load(binary_path)
            
            # 2. Extract weights and details
            readable_data = {
                "model_id": model_id,
                "model_name": meta["model_name"],
                "framework": "scikit-learn",
                "task_type": meta["task_type"],
                "features_schema": meta["features_schema"],
                "model_parameters": {}
            }
            
            # Check for coefficients
            if hasattr(model, "coef_"):
                coef = model.coef_
                readable_data["model_parameters"]["coefficients"] = coef.tolist()
            
            # Check for intercept/bias
            if hasattr(model, "intercept_"):
                intercept = model.intercept_
                readable_data["model_parameters"]["intercept"] = intercept.tolist() if hasattr(intercept, "tolist") else float(intercept)
            
            # Check for categories/classes
            if hasattr(model, "classes_"):
                classes = model.classes_
                readable_data["model_parameters"]["target_classes"] = classes.tolist()
                
            readable_data["model_parameters"]["serialized_size_bytes"] = os.path.getsize(binary_path)
            
            # 3. Save as beautiful, indented JSON text file
            json_file = f"{model_id}_params.json"
            json_path = os.path.join(input_dir, json_file)
            
            with open(json_path, "w") as f:
                json.dump(readable_data, f, indent=4)
                
            print(f"  -> Successfully saved readable JSON: '{json_file}'")
            
        except Exception as err:
            print(f"  ❌ Failed to convert {binary_file}: {err}")
            
    print("\n" + "=" * 50)
    print("All conversions completed! Plain-text JSON parameters are saved and kept aside.")

if __name__ == "__main__":
    convert_binaries_to_json()
