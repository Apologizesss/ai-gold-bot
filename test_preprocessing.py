"""
Test Data Preprocessing Pipeline
=================================
Test the complete preprocessing pipeline before model training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.data_preprocessor import DataPreprocessor


def main():
    """Test preprocessing pipeline."""

    print("=" * 80)
    print("🧪 TESTING DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # Paths
    data_file = (
        project_root
        / "results"
        / "feature_selection"
        / "XAUUSD_M15_selected_features.csv"
    )
    selected_features_file = (
        project_root / "results" / "feature_selection" / "selected_features.json"
    )

    # Check files exist
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return

    if not selected_features_file.exists():
        print(f"❌ Features file not found: {selected_features_file}")
        return

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        scaler_type="standard",  # StandardScaler
        test_size=0.2,  # 20% test
        val_size=0.1,  # 10% validation
        random_state=42,
    )

    # Test 1: Classification (binary)
    print("\n" + "=" * 80)
    print("TEST 1: BINARY CLASSIFICATION")
    print("=" * 80)

    result_cls = preprocessor.preprocess_pipeline(
        data_file=str(data_file),
        selected_features_file=str(selected_features_file),
        target_type="classification",
        n_periods=1,  # Predict next bar
        threshold=0.0,  # Any upward movement
        create_sequences=False,
    )

    print("\n✅ Binary classification preprocessing successful!")
    print(f"   Features: {len(result_cls['feature_names'])}")
    print(f"   Train samples: {len(result_cls['y_train'])}")
    print(f"   Val samples: {len(result_cls['y_val'])}")
    print(f"   Test samples: {len(result_cls['y_test'])}")

    # Test 2: Create sequences for LSTM
    print("\n" + "=" * 80)
    print("TEST 2: SEQUENCE CREATION FOR LSTM")
    print("=" * 80)

    result_seq = preprocessor.preprocess_pipeline(
        data_file=str(data_file),
        selected_features_file=str(selected_features_file),
        target_type="classification",
        n_periods=1,
        threshold=0.0,
        create_sequences=True,
        sequence_length=60,  # 60-bar lookback
    )

    print("\n✅ Sequence creation successful!")
    print(f"   Sequence shape: {result_seq['X_train_seq'].shape}")
    print(
        f"   Format: (samples, timesteps, features) = ({result_seq['X_train_seq'].shape[0]}, {result_seq['X_train_seq'].shape[1]}, {result_seq['X_train_seq'].shape[2]})"
    )

    # Test 3: Regression
    print("\n" + "=" * 80)
    print("TEST 3: REGRESSION TARGET")
    print("=" * 80)

    result_reg = preprocessor.preprocess_pipeline(
        data_file=str(data_file),
        selected_features_file=str(selected_features_file),
        target_type="regression",
        n_periods=1,
        create_sequences=False,
    )

    print("\n✅ Regression preprocessing successful!")
    print(f"   Target mean: {result_reg['y_train'].mean():.4f}%")
    print(f"   Target std: {result_reg['y_train'].std():.4f}%")

    # Test 4: Multi-class
    print("\n" + "=" * 80)
    print("TEST 4: MULTI-CLASS CLASSIFICATION")
    print("=" * 80)

    result_multi = preprocessor.preprocess_pipeline(
        data_file=str(data_file),
        selected_features_file=str(selected_features_file),
        target_type="multiclass",
        n_periods=1,
        create_sequences=False,
    )

    print("\n✅ Multi-class preprocessing successful!")
    print(f"   Classes: {sorted(result_multi['y_train'].unique())}")

    # Summary
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("\n✅ Preprocessing pipeline is ready for model training!")
    print("\nNext steps:")
    print("  1. Train XGBoost baseline model")
    print("  2. Train Random Forest model")
    print("  3. Train LSTM model (with sequences)")
    print("  4. Train CNN model")
    print("  5. Create ensemble")
    print("\nRecommended command:")
    print("  python src/models/train_xgboost.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
