"""
Random Forest Training Script
==============================
Train Random Forest classifier for price direction prediction.

Features:
- Binary classification with ensemble learning
- Cross-validation with time-series splits
- Feature importance analysis
- Hyperparameter tuning with RandomizedSearchCV
- Model evaluation and comparison with XGBoost
- Out-of-bag (OOB) score estimation
"""

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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


class RandomForestTrainer:
    """Train and evaluate Random Forest models for trading."""

    def __init__(self, output_dir: str = "results/models/random_forest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}
        self.oob_score = None

    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_weights: dict = None,
    ):
        """
        Train baseline Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weights: Class weights for imbalanced data
        """
        print("\n" + "=" * 80)
        print("🌲 TRAINING BASELINE RANDOM FOREST MODEL")
        print("=" * 80)

        # Convert class_weights dict to 'balanced' or None
        if class_weights:
            class_weight = "balanced"
        else:
            class_weight = None

        # Default parameters with conservative settings to reduce overfitting
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "max_samples": 0.8,
            "bootstrap": True,
            "oob_score": True,
            "class_weight": class_weight,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }

        print("\nParameters:")
        for key, val in params.items():
            print(f"  {key:20s}: {val}")

        # Train model
        print("\n🔧 Training model...")

        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)

        # Get OOB score
        self.oob_score = self.model.oob_score_

        print("✅ Training complete!")
        print(f"   OOB Score: {self.oob_score:.4f}")

        # Evaluate
        self._evaluate_model(X_train, y_train, X_val, y_val, "baseline")

    def train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_weights: dict = None,
        n_iter: int = 50,
        n_splits: int = 5,
        verbose: int = 2,
    ):
        """
        Train Random Forest with randomized hyperparameter search.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weights: Class weights
            n_iter: Number of random combinations to try
            n_splits: Number of CV splits
            verbose: Verbosity level
        """
        print("\n" + "=" * 80)
        print("🔍 HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH")
        print("=" * 80)

        # Convert class_weights
        class_weight = "balanced" if class_weights else None

        # Parameter distributions for random search
        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [10, 20, 30, 50],
            "min_samples_leaf": [5, 10, 15, 20],
            "max_features": ["sqrt", "log2", None],
            "max_samples": [0.7, 0.8, 0.9],
            "bootstrap": [True],
            "class_weight": [class_weight],
        }

        print(f"\nParameter search space: {n_iter} random combinations")
        print("(This may take several minutes...)")

        # Base model
        base_model = RandomForestClassifier(
            oob_score=True, random_state=42, n_jobs=-1, verbose=0
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Randomized search
        print(f"\n🔍 Running RandomizedSearchCV with {n_splits}-fold time-series CV...")

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=verbose,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

        # Best model
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.oob_score = self.model.oob_score_ if self.model.oob_score else None

        print("\n✅ Tuning complete!")
        print(f"\nBest parameters:")
        for key, val in self.best_params.items():
            print(f"  {key:20s}: {val}")
        print(f"\nBest CV score (ROC-AUC): {random_search.best_score_:.4f}")
        if self.oob_score:
            print(f"OOB Score: {self.oob_score:.4f}")

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

        # Add OOB score if available
        if self.oob_score:
            metrics["oob_score"] = self.oob_score

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

        if self.oob_score:
            print(f"\nOOB SCORE: {self.oob_score:.4f}")

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
        """Analyze and print feature importance."""

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

        ax.barh(
            range(len(top_features)), top_features["importance"], color="forestgreen"
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Random Forest: Top {top_n} Feature Importance",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / "feature_importance.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_file.name}")

    def compare_with_xgboost(self, xgboost_metrics: dict = None):
        """Compare Random Forest with XGBoost."""

        if xgboost_metrics is None:
            # Try to load XGBoost metrics
            xgb_metadata_file = (
                project_root / "results" / "models" / "xgboost" / "model_metadata.json"
            )
            if xgb_metadata_file.exists():
                with open(xgb_metadata_file, "r") as f:
                    metadata = json.load(f)
                    xgboost_metrics = metadata.get("metrics", {}).get("baseline", {})

        if not xgboost_metrics:
            print("⚠️  XGBoost metrics not available for comparison")
            return

        print("\n" + "=" * 80)
        print("⚔️  RANDOM FOREST vs XGBOOST COMPARISON")
        print("=" * 80)

        rf_metrics = self.metrics.get("baseline", {})

        print(
            f"\n{'Metric':<15} {'Random Forest':>15} {'XGBoost':>15} {'Difference':>15}"
        )
        print("-" * 80)

        # Validation metrics comparison
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            rf_val = rf_metrics.get("validation", {}).get(metric, 0)
            xgb_val = xgboost_metrics.get("validation", {}).get(metric, 0)
            diff = rf_val - xgb_val
            diff_sign = "+" if diff > 0 else ""

            print(
                f"{metric.upper():<15} {rf_val:>15.4f} {xgb_val:>15.4f} {diff_sign}{diff:>14.4f}"
            )

        # Winner
        rf_acc = rf_metrics.get("validation", {}).get("accuracy", 0)
        xgb_acc = xgboost_metrics.get("validation", {}).get("accuracy", 0)

        print("\n" + "=" * 80)
        if rf_acc > xgb_acc:
            print("🏆 Winner: RANDOM FOREST")
            print(
                f"   Random Forest outperforms XGBoost by {(rf_acc - xgb_acc) * 100:.2f}%"
            )
        elif xgb_acc > rf_acc:
            print("🏆 Winner: XGBOOST")
            print(
                f"   XGBoost outperforms Random Forest by {(xgb_acc - rf_acc) * 100:.2f}%"
            )
        else:
            print("🤝 Tie: Both models perform equally")

    def save_model(self, model_name: str = "random_forest_model.pkl"):
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
            "oob_score": self.oob_score,
            "n_features": len(self.feature_importance)
            if self.feature_importance is not None
            else 0,
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
        }

        metadata_file = self.output_dir / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved: {metadata_file.name}")

    def load_model(self, model_name: str = "random_forest_model.pkl"):
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
    print("🌲 RANDOM FOREST MODEL TRAINING - AI GOLD BOT")
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
    trainer = RandomForestTrainer(output_dir="results/models/random_forest")

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

    # Compare with XGBoost
    trainer.compare_with_xgboost()

    # Save baseline model
    trainer.save_model("random_forest_baseline.pkl")

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
    print(f"\nOutput directory: results/models/random_forest/")
    print(f"  • random_forest_baseline.pkl  (Trained model)")
    print(f"  • model_metadata.json         (Model info)")
    print(f"  • feature_importance.png      (Feature importance plot)")

    print(f"\n📊 Performance Summary:")
    print(
        f"  Train Accuracy:      {trainer.metrics['baseline']['train']['accuracy']:.4f}"
    )
    print(
        f"  Validation Accuracy: {trainer.metrics['baseline']['validation']['accuracy']:.4f}"
    )
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC:        {test_metrics['roc_auc']:.4f}")
    if trainer.oob_score:
        print(f"  OOB Score:           {trainer.oob_score:.4f}")

    print(f"\n🚀 Next steps:")
    print(f"  1. Train LSTM model with sequences")
    print(f"  2. Create ensemble (RF + XGBoost)")
    print(f"  3. Backtest strategies")
    print(f"  4. Deploy best model")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
