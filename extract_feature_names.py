"""
Extract feature names from feature_importance.csv
Quick fix to create feature_names.txt for inference
"""

import pandas as pd
from pathlib import Path

print("=" * 70)
print("EXTRACTING FEATURE NAMES FROM FEATURE_IMPORTANCE.CSV")
print("=" * 70)

# Read feature importance
fi_path = Path("results/xgboost/feature_importance.csv")
if not fi_path.exists():
    print(f"ERROR: {fi_path} not found!")
    exit(1)

fi = pd.read_csv(fi_path)
feature_names = fi["feature"].tolist()

print(f"\nFound {len(feature_names)} features")
print("\nFirst 10 features:")
for i, name in enumerate(feature_names[:10], 1):
    print(f"  {i:2d}. {name}")

# Save to feature_names.txt
output_path = Path("results/xgboost/feature_names.txt")
with open(output_path, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

print(f"\n✅ Feature names saved to: {output_path}")
print(f"   Total features: {len(feature_names)}")
print("\nYou can now run paper_trading.py")
