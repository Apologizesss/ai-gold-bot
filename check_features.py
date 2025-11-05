"""
Check feature count mismatch
"""

import joblib
import pandas as pd

print("=" * 70)
print("FEATURE MISMATCH CHECKER")
print("=" * 70)

# Check scaler
print("\n1. Checking scaler...")
scaler = joblib.load("results/xgboost/xgboost_scaler.pkl")
print(f"   Scaler expects: {scaler.n_features_in_} features")

if hasattr(scaler, "feature_names_in_"):
    print(f"   Feature names available: {len(scaler.feature_names_in_)}")
    print("\n   Training features:")
    for i, name in enumerate(scaler.feature_names_in_, 1):
        print(f"   {i:3d}. {name}")
else:
    print("   No feature names saved in scaler")

# Check feature importance file
print("\n2. Checking feature importance file...")
fi = pd.read_csv("results/xgboost/feature_importance.csv")
print(f"   Features in importance file: {len(fi)}")

# Check model
print("\n3. Checking model...")
try:
    model = joblib.load("results/xgboost/xgboost_model.pkl")
    print(f"   Model loaded successfully")
    if hasattr(model, "n_features_in_"):
        print(f"   Model expects: {model.n_features_in_} features")
except Exception as e:
    print(f"   Error loading model: {e}")

print("\n" + "=" * 70)
print("SOLUTION:")
print("=" * 70)
print("The model was trained with 96 features but inference produces 138.")
print("\nOptions:")
print("1. RETRAIN with current features (recommended)")
print("   Command: python train_xgboost.py")
print("\n2. Fix inference to use only the 96 training features")
print("   (requires saved feature list)")
