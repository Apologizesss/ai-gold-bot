"""
Ensemble Model - Voting Classifier
===================================
Combine multiple trained models for better predictions.

Features:
- Soft voting ensemble (probability averaging)
- Hard voting ensemble (majority voting)
- Weighted voting with custom weights
- Model performance comparison
- Ensemble evaluation and metrics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class EnsembleModel:
    """
    Ensemble multiple trained models for improved predictions.

    Combines XGBoost, Random Forest, and potentially other models
    using voting strategies.
    """

    def __init__(self, output_dir: str = "results/models/ensemble"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.ensemble = None
        self.metrics = {}
        self.voting_type = "soft"

    def load_model(self, model_name: str, model_path: str) -> bool:
        """
        Load a trained model.

        Args:
            model_name: Name identifier (e.g., 'xgboost', 'random_forest')
            model_path: Path to pickled model file

        Returns:
            True if successful, False otherwise
        """
        model_file = Path(model_path)

        if not model_file.exists():
            print(f"[Error] Model not found: {model_file}")
            return False

        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            self.models[model_name] = model
            print(f"[OK] Loaded: {model_name} from {model_file.name}")
            return True

        except Exception as e:
            print(f"[Error] Error loading {model_name}: {e}")
            return False

    def create_voting_ensemble(
        self,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ):
        """
        Create voting ensemble from loaded models.

        Args:
            voting: 'soft' (probability average) or 'hard' (majority vote)
            weights: Optional weights for each model
        """
        print("\n" + "=" * 80)
        print(f"[Target] CREATING ENSEMBLE MODEL (voting={voting})")
        print("=" * 80)

        if len(self.models) < 2:
            print("[Error] Need at least 2 models for ensemble")
            return

        # Prepare estimators list for VotingClassifier
        estimators = [(name, model) for name, model in self.models.items()]

        print(f"\nModels in ensemble:")
        for i, (name, model) in enumerate(estimators):
            weight = weights[i] if weights else 1.0
            print(f"  {i + 1}. {name:20s} (weight: {weight})")

        # Create ensemble
        self.voting_type = voting
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1,
        )

        print(f"\n[OK] Ensemble created with {len(estimators)} models")
        print(f"   Voting type: {voting}")
        if weights:
            print(f"   Weights: {weights}")

    def fit_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the ensemble model.

        Note: Individual models should already be trained.
        This just fits the VotingClassifier wrapper.
        """
        if self.ensemble is None:
            print("[Error] Create ensemble first using create_voting_ensemble()")
            return

        print("\n[Feature Engineering] Fitting ensemble...")
        self.ensemble.fit(X_train, y_train)
        print("[OK] Ensemble fitted!")

    def evaluate_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """
        Evaluate ensemble on all datasets.

        Returns:
            Dictionary with metrics for all sets
        """
        print("\n" + "=" * 80)
        print("[Stats] ENSEMBLE EVALUATION")
        print("=" * 80)

        results = {}

        # Evaluate on each dataset
        for name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            # Predictions
            y_pred = self.ensemble.predict(X)

            if self.voting_type == "soft":
                y_proba = self.ensemble.predict_proba(X)[:, 1]
            else:
                # For hard voting, use individual model probabilities
                y_proba = np.mean(
                    [
                        model.predict_proba(X)[:, 1]
                        for model in self.ensemble.estimators_
                    ],
                    axis=0,
                )

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_proba),
            }

            results[name] = {
                "metrics": metrics,
                "predictions": y_pred,
                "probabilities": y_proba,
            }

            # Print metrics
            print(f"\n{name.upper()} METRICS:")
            print("-" * 80)
            for metric, value in metrics.items():
                print(f"  {metric:12s}: {value:.4f}")

            # Confusion matrix
            if name == "test":
                print("\nCONFUSION MATRIX (Test):")
                print("-" * 80)
                cm = confusion_matrix(y, y_pred)
                print(cm)
                print("\nClassification Report:")
                print(classification_report(y, y_pred, digits=4))

        self.metrics = results
        return results

    def compare_all_models(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare all individual models with ensemble on test set.

        Returns:
            DataFrame with comparison metrics
        """
        print("\n" + "=" * 80)
        print("⚔️  MODEL COMPARISON (Test Set)")
        print("=" * 80)

        comparison_data = []

        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name.upper(),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            comparison_data.append(metrics)

        # Add ensemble
        if self.ensemble and self.metrics:
            ensemble_metrics = {
                "model": "ENSEMBLE",
                **self.metrics["test"]["metrics"],
            }
            comparison_data.append(ensemble_metrics)

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("accuracy", ascending=False)

        # Print table
        print("\n" + comparison_df.to_string(index=False))

        # Find best model
        best_model = comparison_df.iloc[0]["model"]
        best_acc = comparison_df.iloc[0]["accuracy"]

        print("\n" + "=" * 80)
        print(f"🏆 BEST MODEL: {best_model}")
        print(f"   Test Accuracy: {best_acc:.4f}")
        print("=" * 80)

        return comparison_df

    def save_ensemble(self, filename: str = "ensemble_model.pkl"):
        """Save ensemble model."""
        if self.ensemble is None:
            print("[Error] No ensemble to save")
            return

        model_file = self.output_dir / filename

        with open(model_file, "wb") as f:
            pickle.dump(self.ensemble, f)

        print(f"\n[Save] Ensemble saved: {model_file}")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "voting_type": self.voting_type,
            "models": list(self.models.keys()),
            "n_models": len(self.models),
            "metrics": {
                k: v["metrics"] for k, v in self.metrics.items() if "metrics" in v
            },
        }

        metadata_file = self.output_dir / "ensemble_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Metadata saved: {metadata_file.name}")

    def load_ensemble(self, filename: str = "ensemble_model.pkl") -> bool:
        """Load ensemble model."""
        model_file = self.output_dir / filename

        if not model_file.exists():
            print(f"[Error] Ensemble not found: {model_file}")
            return False

        with open(model_file, "rb") as f:
            self.ensemble = pickle.load(f)

        print(f"[OK] Ensemble loaded: {model_file}")
        return True


def main():
    """Main ensemble creation and evaluation pipeline."""

    print("=" * 80)
    print("[Target] ENSEMBLE MODEL CREATION - AI GOLD BOT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Import preprocessor
    from src.models.data_preprocessor import DataPreprocessor

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
    xgboost_model = (
        project_root / "results" / "models" / "xgboost" / "xgboost_baseline.pkl"
    )
    rf_model = (
        project_root
        / "results"
        / "models"
        / "random_forest"
        / "random_forest_baseline.pkl"
    )

    # Check files
    if not data_file.exists():
        print(f"[Error] Data file not found: {data_file}")
        return

    if not xgboost_model.exists():
        print(f"[Error] XGBoost model not found: {xgboost_model}")
        return

    if not rf_model.exists():
        print(f"[Error] Random Forest model not found: {rf_model}")
        return

    # Preprocess data
    print("📂 Loading and preprocessing data...")
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

    # Initialize ensemble
    ensemble = EnsembleModel(output_dir="results/models/ensemble")

    # Load models
    print("\n" + "=" * 80)
    print("📦 LOADING TRAINED MODELS")
    print("=" * 80 + "\n")

    ensemble.load_model("xgboost", str(xgboost_model))
    ensemble.load_model("random_forest", str(rf_model))

    # Create soft voting ensemble
    ensemble.create_voting_ensemble(voting="soft", weights=None)

    # Fit ensemble (just wrapper, models already trained)
    ensemble.fit_ensemble(data["X_train"], data["y_train"])

    # Evaluate ensemble
    results = ensemble.evaluate_ensemble(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    # Compare all models
    comparison_df = ensemble.compare_all_models(data["X_test"], data["y_test"])

    # Save comparison
    comparison_file = ensemble.output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n[OK] Comparison saved: {comparison_file.name}")

    # Save ensemble
    ensemble.save_ensemble("ensemble_soft_voting.pkl")

    # Summary
    print("\n" + "=" * 80)
    print("✨ ENSEMBLE CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: results/models/ensemble/")
    print(f"  • ensemble_soft_voting.pkl    (Ensemble model)")
    print(f"  • ensemble_metadata.json      (Model info)")
    print(f"  • model_comparison.csv        (Performance comparison)")

    print(f"\n[Stats] Final Performance Summary:")
    print(f"  Ensemble Test Accuracy: {results['test']['metrics']['accuracy']:.4f}")
    print(f"  Ensemble Test ROC-AUC:  {results['test']['metrics']['roc_auc']:.4f}")

    print(f"\n[Launch] Next steps:")
    print(f"  1. Try weighted voting ensemble")
    print(f"  2. Add LSTM to ensemble")
    print(f"  3. Backtest strategies")
    print(f"  4. Deploy to paper trading")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
