"""
Data Preprocessing Module
==========================
Prepares data for model training:
1. Load selected features
2. Create target variables (classification/regression)
3. Train/validation/test split (time-series aware)
4. Feature scaling (StandardScaler, MinMaxScaler)
5. Sequence creation for LSTM/CNN
6. Handle class imbalance (SMOTE)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import json


class DataPreprocessor:
    """
    Preprocess data for ML model training.

    Features:
    - Time-series aware train/test splits
    - Multiple scaling methods
    - Target variable creation (classification/regression)
    - Sequence generation for RNN/LSTM
    - Data validation and quality checks
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize preprocessor.

        Args:
            scaler_type: 'standard', 'minmax', or 'robust'
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
        """
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.is_fitted: bool = False

    def load_data(
        self, data_file: str, selected_features_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load processed data and filter to selected features.

        Args:
            data_file: Path to processed data CSV
            selected_features_file: Path to selected features JSON (optional)

        Returns:
            DataFrame with selected features
        """
        print("\n📂 Loading Data")
        print("-" * 70)

        df = pd.read_csv(data_file)
        print(f"✅ Loaded: {Path(data_file).name}")
        print(f"   Shape: {df.shape}")

        if selected_features_file:
            with open(selected_features_file, "r") as f:
                feature_data = json.load(f)

            selected_features = feature_data["selected_features"]

            # Keep time + OHLC + selected features
            keep_cols = ["time", "open", "high", "low", "close", "tick_volume"]
            keep_cols = [col for col in keep_cols if col in df.columns]
            keep_cols.extend([col for col in selected_features if col in df.columns])

            df = df[keep_cols]
            print(f"✅ Filtered to {len(selected_features)} selected features")
            print(f"   New shape: {df.shape}")

        return df

    def create_target_classification(
        self, df: pd.DataFrame, n_periods: int = 1, threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create binary classification target.

        Args:
            df: DataFrame with 'close' column
            n_periods: Number of periods ahead to predict
            threshold: Price change threshold (% for buy signal)

        Returns:
            DataFrame with 'target' column (0=down/hold, 1=up/buy)
        """
        print(f"\n🎯 Creating Target Variable (Classification)")
        print("-" * 70)

        # Calculate future return
        df["future_return"] = (
            (df["close"].shift(-n_periods) - df["close"]) / df["close"] * 100
        )

        # Binary: 1 if price goes up by > threshold, else 0
        df["target"] = (df["future_return"] > threshold).astype(int)

        # Remove last n_periods rows (no future data)
        df = df[:-n_periods].copy()

        # Class distribution
        class_counts = df["target"].value_counts().sort_index()
        total = len(df)

        print(f"Target: Price direction {n_periods} period(s) ahead")
        print(f"Threshold: >{threshold}% return")
        print(f"\nClass Distribution:")
        print(
            f"  Class 0 (Down/Hold): {class_counts.get(0, 0):>6,} ({class_counts.get(0, 0) / total * 100:5.2f}%)"
        )
        print(
            f"  Class 1 (Up/Buy):    {class_counts.get(1, 0):>6,} ({class_counts.get(1, 0) / total * 100:5.2f}%)"
        )

        self.target_name = "target"

        return df

    def create_target_regression(
        self, df: pd.DataFrame, n_periods: int = 1
    ) -> pd.DataFrame:
        """
        Create regression target (future return %).

        Args:
            df: DataFrame with 'close' column
            n_periods: Number of periods ahead to predict

        Returns:
            DataFrame with 'target' column (future return %)
        """
        print(f"\n🎯 Creating Target Variable (Regression)")
        print("-" * 70)

        # Calculate future return
        df["target"] = (df["close"].shift(-n_periods) - df["close"]) / df["close"] * 100

        # Remove last n_periods rows
        df = df[:-n_periods].copy()

        print(f"Target: Future return % after {n_periods} period(s)")
        print(f"\nTarget Statistics:")
        print(f"  Mean:   {df['target'].mean():>8.4f}%")
        print(f"  Std:    {df['target'].std():>8.4f}%")
        print(f"  Min:    {df['target'].min():>8.4f}%")
        print(f"  Max:    {df['target'].max():>8.4f}%")
        print(f"  Median: {df['target'].median():>8.4f}%")

        self.target_name = "target"

        return df

    def create_multiclass_target(
        self,
        df: pd.DataFrame,
        n_periods: int = 1,
        thresholds: Tuple[float, float] = (-0.1, 0.1),
    ) -> pd.DataFrame:
        """
        Create multi-class target (0=down, 1=hold, 2=up).

        Args:
            df: DataFrame with 'close' column
            n_periods: Number of periods ahead
            thresholds: (down_threshold, up_threshold) in %

        Returns:
            DataFrame with 'target' column (0/1/2)
        """
        print(f"\n🎯 Creating Target Variable (Multi-class)")
        print("-" * 70)

        # Calculate future return
        df["future_return"] = (
            (df["close"].shift(-n_periods) - df["close"]) / df["close"] * 100
        )

        # Multi-class: 0=down, 1=hold, 2=up
        df["target"] = 1  # Default: hold
        df.loc[df["future_return"] < thresholds[0], "target"] = 0  # Down
        df.loc[df["future_return"] > thresholds[1], "target"] = 2  # Up

        df = df[:-n_periods].copy()

        class_counts = df["target"].value_counts().sort_index()
        total = len(df)

        print(f"Target: Price direction {n_periods} period(s) ahead")
        print(f"Thresholds: Down<{thresholds[0]}%, Hold, Up>{thresholds[1]}%")
        print(f"\nClass Distribution:")
        print(
            f"  Class 0 (Down): {class_counts.get(0, 0):>6,} ({class_counts.get(0, 0) / total * 100:5.2f}%)"
        )
        print(
            f"  Class 1 (Hold): {class_counts.get(1, 0):>6,} ({class_counts.get(1, 0) / total * 100:5.2f}%)"
        )
        print(
            f"  Class 2 (Up):   {class_counts.get(2, 0):>6,} ({class_counts.get(2, 0) / total * 100:5.2f}%)"
        )

        self.target_name = "target"

        return df

    def split_data(
        self, df: pd.DataFrame, time_column: str = "time"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets (time-series aware).

        Args:
            df: DataFrame with features and target
            time_column: Name of time column

        Returns:
            (train_df, val_df, test_df)
        """
        print(f"\n✂️  Splitting Data (Time-Series Aware)")
        print("-" * 70)

        # Sort by time
        if time_column in df.columns:
            df = df.sort_values(time_column).reset_index(drop=True)

        n = len(df)

        # Calculate split indices
        test_start = int(n * (1 - self.test_size))
        val_start = int(test_start * (1 - self.val_size))

        # Split
        train_df = df[:val_start].copy()
        val_df = df[val_start:test_start].copy()
        test_df = df[test_start:].copy()

        print(f"Total samples: {n:,}")
        print(f"\nSplit sizes:")
        print(f"  Train:      {len(train_df):>6,} ({len(train_df) / n * 100:5.2f}%)")
        print(f"  Validation: {len(val_df):>6,} ({len(val_df) / n * 100:5.2f}%)")
        print(f"  Test:       {len(test_df):>6,} ({len(test_df) / n * 100:5.2f}%)")

        if time_column in df.columns:
            print(f"\nTime ranges:")
            print(
                f"  Train: {train_df[time_column].iloc[0]} → {train_df[time_column].iloc[-1]}"
            )
            print(
                f"  Val:   {val_df[time_column].iloc[0]} → {val_df[time_column].iloc[-1]}"
            )
            print(
                f"  Test:  {test_df[time_column].iloc[0]} → {test_df[time_column].iloc[-1]}"
            )

        return train_df, val_df, test_df

    def prepare_features_and_target(
        self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target.

        Args:
            df: DataFrame with features and target
            exclude_cols: Columns to exclude from features

        Returns:
            (X, y) where X is features, y is target
        """
        if exclude_cols is None:
            exclude_cols = [
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
                "target",
                "future_return",
            ]

        # Target
        y = df["target"].copy()

        # Features (exclude metadata and target)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()

        self.feature_names = feature_cols

        return X, y

    def scale_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using fitted scaler (fit on train only).

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        print(f"\n📏 Scaling Features ({self.scaler_type})")
        print("-" * 70)

        # Fit on train only (prevent data leakage)
        self.scaler.fit(X_train)
        self.is_fitted = True

        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train), columns=X_train.columns, index=X_train.index
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        print(f"✅ Scaled {len(X_train.columns)} features")
        print(f"   Scaler: {self.scaler_type}")
        print(f"   Fitted on: {len(X_train):,} training samples")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            sequence_length: Number of time steps to look back

        Returns:
            (X_seq, y_seq) where X_seq shape is (n_samples, sequence_length, n_features)
        """
        print(f"\n🔄 Creating Sequences (lookback={sequence_length})")
        print("-" * 70)

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length : i])
            y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"✅ Created sequences")
        print(f"   Original shape: {X.shape}")
        print(f"   Sequence shape: {X_seq.shape}")
        print(f"   Lost samples: {len(X) - len(X_seq)} (first {sequence_length} rows)")

        return X_seq, y_seq

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.

        Args:
            y: Target array

        Returns:
            Dictionary of class weights
        """
        from collections import Counter

        class_counts = Counter(y)
        total = len(y)
        n_classes = len(class_counts)

        # Compute balanced weights
        weights = {
            cls: total / (n_classes * count) for cls, count in class_counts.items()
        }

        print(f"\n⚖️  Class Weights (for imbalance handling)")
        print("-" * 70)
        for cls in sorted(weights.keys()):
            print(f"  Class {cls}: {weights[cls]:.4f}")

        return weights

    def preprocess_pipeline(
        self,
        data_file: str,
        selected_features_file: str,
        target_type: str = "classification",
        n_periods: int = 1,
        threshold: float = 0.0,
        create_sequences: bool = False,
        sequence_length: int = 60,
    ) -> Dict:
        """
        Full preprocessing pipeline.

        Args:
            data_file: Path to data CSV
            selected_features_file: Path to selected features JSON
            target_type: 'classification', 'regression', or 'multiclass'
            n_periods: Periods ahead to predict
            threshold: Threshold for classification
            create_sequences: Whether to create sequences for LSTM
            sequence_length: Sequence length if creating sequences

        Returns:
            Dictionary with all preprocessed data
        """
        print("\n" + "=" * 80)
        print("🔧 DATA PREPROCESSING PIPELINE")
        print("=" * 80)

        # 1. Load data
        df = self.load_data(data_file, selected_features_file)

        # 2. Create target
        if target_type == "classification":
            df = self.create_target_classification(df, n_periods, threshold)
        elif target_type == "regression":
            df = self.create_target_regression(df, n_periods)
        elif target_type == "multiclass":
            df = self.create_multiclass_target(df, n_periods, (-0.1, 0.1))
        else:
            raise ValueError(f"Unknown target type: {target_type}")

        # 3. Split data
        train_df, val_df, test_df = self.split_data(df)

        # 4. Prepare features and target
        X_train, y_train = self.prepare_features_and_target(train_df)
        X_val, y_val = self.prepare_features_and_target(val_df)
        X_test, y_test = self.prepare_features_and_target(test_df)

        # 5. Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        result = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        }

        # 6. Create sequences if needed (for LSTM/RNN)
        if create_sequences:
            X_train_seq, y_train_seq = self.create_sequences(
                X_train_scaled.values, y_train.values, sequence_length
            )
            X_val_seq, y_val_seq = self.create_sequences(
                X_val_scaled.values, y_val.values, sequence_length
            )
            X_test_seq, y_test_seq = self.create_sequences(
                X_test_scaled.values, y_test.values, sequence_length
            )

            result["X_train_seq"] = X_train_seq
            result["X_val_seq"] = X_val_seq
            result["X_test_seq"] = X_test_seq
            result["y_train_seq"] = y_train_seq
            result["y_val_seq"] = y_val_seq
            result["y_test_seq"] = y_test_seq

        # 7. Class weights (for classification)
        if target_type in ["classification", "multiclass"]:
            result["class_weights"] = self.get_class_weights(y_train.values)

        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\n📦 Available data:")
        print(f"  • X_train: {result['X_train'].shape}")
        print(f"  • X_val:   {result['X_val'].shape}")
        print(f"  • X_test:  {result['X_test'].shape}")
        print(f"  • Features: {len(self.feature_names)}")

        if create_sequences:
            print(f"  • X_train_seq: {result['X_train_seq'].shape}")
            print(f"  • X_val_seq:   {result['X_val_seq'].shape}")
            print(f"  • X_test_seq:  {result['X_test_seq'].shape}")

        return result
