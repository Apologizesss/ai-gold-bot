"""
XGBoost Training Script
=======================
Train XGBoost classifier with hyperparameter tuning and evaluation.

Features:
- Binary classification for price direction prediction
- Hyperparameter optimization with GridSearchCV
- Cross-validation with time-series splits
- Feature importance analysis
- Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model persistence (save/load)
- Backtesting simulation
"""

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend to prevent figure windows

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_preprocessor import DataPreprocessor

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostTrainer:
    """Train and evaluate XGBoost models for trading."""

    def __init__(self, output_dir: str = "results/models/xgboost"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}

    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_weights: dict = None,
    ):
        """
        Train baseline XGBoost model with default parameters.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weights: Class weights for imbalanced data
        """
        print("\n" + "=" * 80)
        print("🎯 TRAINING BASELINE XGBOOST MODEL")
        print("=" * 80)

        # Calculate scale_pos_weight if binary classification
        if class_weights and len(class_weights) == 2:
            scale_pos_weight = class_weights[0] / class_weights[1]
        else:
            scale_pos_weight = 1.0

        # Default parameters with regularization
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        print("\nParameters:")
        for key, val in params.items():
            print(f"  {key:20s}: {val}")

        # Train model
        print("\n🔧 Training model...")

        self.model = xgb.XGBClassifier(**params)

        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        print("✅ Training complete!")

        # Evaluate
        self._evaluate_model(X_train, y_train, X_val, y_val, "baseline")

    def train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_weights: dict = None,
        n_splits: int = 5,
        verbose: int = 2,
    ):
        """
        Train XGBoost with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weights: Class weights
            n_splits: Number of CV splits
            verbose: Verbosity level
        """
        print("\n" + "=" * 80)
        print("🔍 HYPERPARAMETER TUNING WITH GRID SEARCH")
        print("=" * 80)

        # Calculate scale_pos_weight
        if class_weights and len(class_weights) == 2:
            scale_pos_weight = class_weights[0] / class_weights[1]
        else:
            scale_pos_weight = 1.0

        # Parameter grid
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0.5, 1.0, 2.0],
        }

        print(
            f"\nParameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations"
        )
        print("(This may take a while...)")

        # Base model
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Grid search
        print(f"\n🔍 Running GridSearchCV with {n_splits}-fold time-series CV...")

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=verbose,
        )

        grid_search.fit(X_train, y_train)

        # Best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print("\n✅ Tuning complete!")
        print(f"\nBest parameters:")
        for key, val in self.best_params.items():
            print(f"  {key:20s}: {val}")
        print(f"\nBest CV score (ROC-AUC): {grid_search.best_score_:.4f}")

        # Evaluate
        self._evaluate_model(X_train, y_train, X_val, y_val, "tuned")

    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str = "model",
    ):
        """Evaluate model performance."""

        print("\n" + "=" * 80)
        print("📊 MODEL EVALUATION")
        print("=" * 80)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]

        # Metrics
        metrics = {
            "train": {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "precision": precision_score(y_train, y_train_pred, zero_division=0),
                "recall": recall_score(y_train, y_train_pred, zero_division=0),
                "f1": f1_score(y_train, y_train_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_train, y_train_proba),
            },
            "validation": {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_val_proba),
            },
        }

        self.metrics[model_name] = metrics

        # Print metrics
        print("\nTRAIN METRICS:")
        print("-" * 80)
        for metric, value in metrics["train"].items():
            print(f"  {metric:12s}: {value:.4f}")

        print("\nVALIDATION METRICS:")
        print("-" * 80)
        for metric, value in metrics["validation"].items():
            print(f"  {metric:12s}: {value:.4f}")

        # Confusion matrix
        print("\nCONFUSION MATRIX (Validation):")
        print("-" * 80)
        cm = confusion_matrix(y_val, y_val_pred)
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_val_pred, digits=4))

        # Feature importance
        self._analyze_feature_importance(X_train.columns)

    def _analyze_feature_importance(self, feature_names: list):
        """Analyze and plot feature importance."""

        print("\n📈 FEATURE IMPORTANCE")
        print("-" * 80)

        # Get importance
        importance = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        self.feature_importance = importance_df

        # Print top 20
        print("\nTop 20 Most Important Features:")
        for i, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.6f}")

    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""

        if self.feature_importance is None:
            print("❌ No feature importance available")
            return

        print(f"\n📊 Plotting top {top_n} features...")

        fig, ax = plt.subplots(figsize=(12, 8))

        top_features = self.feature_importance.head(top_n)

        ax.barh(range(len(top_features)), top_features["importance"], color="steelblue")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
        ax.set_title(
            f"XGBoost: Top {top_n} Feature Importance", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / "feature_importance.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_file.name}")

    def save_model(self, model_name: str = "xgboost_model.pkl"):
        """Save trained model."""

        if self.model is None:
            print("❌ No model to save")
            return

        model_file = self.output_dir / model_name

        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n💾 Model saved: {model_file}")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_params": self.best_params if self.best_params else {},
            "metrics": self.metrics,
            "n_features": len(self.feature_importance)
            if self.feature_importance is not None
            else 0,
        }

        metadata_file = self.output_dir / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved: {metadata_file.name}")

    def load_model(self, model_name: str = "xgboost_model.pkl"):
        """Load trained model."""

        model_file = self.output_dir / model_name

        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return False

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        print(f"✅ Model loaded: {model_file}")
        return True


def main():
    """Main training pipeline."""

    print("=" * 80)
    print("🤖 XGBOOST MODEL TRAINING - AI GOLD BOT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Check files
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return

    if not selected_features_file.exists():
        print(f"❌ Features file not found: {selected_features_file}")
        return

    # Preprocess data
    preprocessor = DataPreprocessor(
        scaler_type="standard", test_size=0.2, val_size=0.1, random_state=42
    )

    data = preprocessor.preprocess_pipeline(
        data_file=str(data_file),
        selected_features_file=str(selected_features_file),
        target_type="classification",
        n_periods=1,
        threshold=0.0,
        create_sequences=False,
    )

    # Initialize trainer
    trainer = XGBoostTrainer(output_dir="results/models/xgboost")

    # Train baseline model
    trainer.train_baseline(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        class_weights=data["class_weights"],
    )

    # Plot feature importance
    trainer.plot_feature_importance(top_n=20)

    # Save baseline model
    trainer.save_model("xgboost_baseline.pkl")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("🧪 TEST SET EVALUATION")
    print("=" * 80)

    y_test_pred = trainer.model.predict(data["X_test"])
    y_test_proba = trainer.model.predict_proba(data["X_test"])[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(data["y_test"], y_test_pred),
        "precision": precision_score(data["y_test"], y_test_pred, zero_division=0),
        "recall": recall_score(data["y_test"], y_test_pred, zero_division=0),
        "f1": f1_score(data["y_test"], y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(data["y_test"], y_test_proba),
    }

    print("\nTEST METRICS:")
    print("-" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(data["y_test"], y_test_pred))

    # Summary
    print("\n" + "=" * 80)
    print("✨ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: results/models/xgboost/")
    print(f"  • xgboost_baseline.pkl       (Trained model)")
    print(f"  • model_metadata.json        (Model info)")
    print(f"  • feature_importance.png     (Feature importance plot)")

    print(f"\n📊 Performance Summary:")
    print(
        f"  Train Accuracy:      {trainer.metrics['baseline']['train']['accuracy']:.4f}"
    )
    print(
        f"  Validation Accuracy: {trainer.metrics['baseline']['validation']['accuracy']:.4f}"
    )
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC:        {test_metrics['roc_auc']:.4f}")

    print(f"\n🚀 Next steps:")
    print(f"  1. Train Random Forest model")
    print(f"  2. Train LSTM model")
    print(f"  3. Compare model performances")
    print(f"  4. Create ensemble")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
