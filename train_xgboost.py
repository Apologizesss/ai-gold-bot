"""
XGBoost Trainer for Trading Predictions
========================================
Alternative to LSTM - often works better on tabular data with limited samples
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: XGBoost not installed!")
    print("Install with: pip install xgboost")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import json
import joblib


class XGBoostTrainer:
    """Train XGBoost model for trading predictions"""

    def __init__(self, data_path, test_size=0.2):
        print("=" * 70)
        print("XGBOOST TRADING MODEL TRAINER")
        print("=" * 70)
        print()
        print(f"Data: {data_path}")
        print(f"Test size: {test_size * 100:.0f}%")
        print()

        self.data_path = data_path
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.model = None

    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Get numeric columns only (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if "target" not in df.columns:
            print("ERROR: No target column found!")
            return None, None, None

        # Remove target from features
        feature_cols = [col for col in numeric_cols if col != "target"]

        print(f"  Found {len(feature_cols)} features")

        # Extract features and target
        X = df[feature_cols].values
        y = df["target"].values

        # Handle NaN
        if np.isnan(X).any():
            print("  Warning: Found NaN, filling with 0")
            X = np.nan_to_num(X, 0)

        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Target distribution:")
        for val, count in zip(unique, counts):
            label = "DOWN" if val == 0 else "UP"
            pct = count / len(y) * 100
            print(f"    {label}: {count:,}/{len(y):,} ({pct:.1f}%)")

        return X, y, feature_cols

    def split_data(self, X, y):
        """Split into train/test sets"""
        print()
        print("Splitting data...")

        # Use stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        print(f"  Train: {len(X_train):,} samples ({np.mean(y_train) * 100:.1f}% UP)")
        print(f"  Test:  {len(X_test):,} samples ({np.mean(y_test) * 100:.1f}% UP)")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print()
        print("Training XGBoost model...")
        print()

        # Calculate scale_pos_weight for imbalanced data
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"  Class balance: {n_neg} DOWN, {n_pos} UP")
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        print()

        # Create model with good defaults
        self.model = XGBClassifier(
            n_estimators=200,  # Number of trees
            max_depth=6,  # Tree depth
            learning_rate=0.1,  # Step size
            subsample=0.8,  # Fraction of samples per tree
            colsample_bytree=0.8,  # Fraction of features per tree
            gamma=1,  # Minimum loss reduction for split
            min_child_weight=3,  # Minimum sum of weights in child
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=20,  # Stop if no improvement
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbosity=1,
        )

        # Train with early stopping
        eval_set = [(X_train, y_train), (X_test, y_test)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
        )

        print()
        print(f"  Training completed after {self.model.best_iteration} iterations")
        print()

        return self.model

    def evaluate_model(self, X_test, y_test, feature_names):
        """Evaluate model performance"""
        print("=" * 70)
        print("EVALUATION")
        print("=" * 70)
        print()

        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0

        print(f"Test Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        print(f"Test AUC-ROC:   {auc:.4f}")
        print()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"  TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
        print(f"  FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

        # Prediction distribution
        print("Prediction Distribution:")
        unique, counts = np.unique(y_pred, return_counts=True)
        for val, count in zip(unique, counts):
            label = "DOWN" if val == 0 else "UP"
            pct = count / len(y_pred) * 100
            print(f"  {label}: {count}/{len(y_pred)} ({pct:.1f}%)")
        print()

        # Probability statistics
        print("Prediction Probabilities:")
        print(f"  Mean: {np.mean(y_pred_proba):.4f}")
        print(f"  Std:  {np.std(y_pred_proba):.4f}")
        print(f"  Min:  {np.min(y_pred_proba):.4f}")
        print(f"  Max:  {np.max(y_pred_proba):.4f}")
        print()

        # Feature importance
        print("Top 20 Most Important Features:")
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        for i, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']:30s} : {row['importance']:.4f}")
        print()

        # Save feature importance
        results_dir = Path("results/xgboost")
        results_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "feature_importance": importance_df.to_dict("records"),
        }

    def plot_results(self, X_test, y_test, metrics, output_dir):
        """Plot evaluation results"""
        print("Plotting results...")

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Feature importance (top 15)
        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.model.get_booster().feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(15)
        )

        axes[0, 0].barh(range(len(importance_df)), importance_df["importance"])
        axes[0, 0].set_yticks(range(len(importance_df)))
        axes[0, 0].set_yticklabels(importance_df["feature"], fontsize=8)
        axes[0, 0].set_xlabel("Importance")
        axes[0, 0].set_title("Top 15 Feature Importance")
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = metrics["auc"]
            axes[0, 1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random")
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        except:
            axes[0, 1].text(0.5, 0.5, "ROC Curve\nN/A", ha="center", va="center")

        # 3. Prediction probability distribution
        axes[1, 0].hist(
            y_pred_proba[y_test == 0], bins=30, alpha=0.5, label="DOWN (actual)"
        )
        axes[1, 0].hist(
            y_pred_proba[y_test == 1], bins=30, alpha=0.5, label="UP (actual)"
        )
        axes[1, 0].axvline(0.5, color="red", linestyle="--", label="Threshold")
        axes[1, 0].set_xlabel("Predicted Probability (UP)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Prediction Probability Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Confusion matrix heatmap
        cm = np.array(metrics["confusion_matrix"])
        im = axes[1, 1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[1, 1].set_title("Confusion Matrix")
        plt.colorbar(im, ax=axes[1, 1])

        tick_marks = np.arange(2)
        axes[1, 1].set_xticks(tick_marks)
        axes[1, 1].set_yticks(tick_marks)
        axes[1, 1].set_xticklabels(["DOWN", "UP"])
        axes[1, 1].set_yticklabels(["DOWN", "UP"])

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        axes[1, 1].set_ylabel("True Label")
        axes[1, 1].set_xlabel("Predicted Label")

        plt.tight_layout()
        plot_path = output_dir / "xgboost_results.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {plot_path}")

    def save_model(self, feature_names, metrics):
        """Save model and results"""
        print()
        print("Saving model and results...")

        results_dir = Path("results/xgboost")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = results_dir / "xgboost_model.json"
        self.model.save_model(model_path)
        print(f"  Model saved: {model_path}")

        # Save as pickle (for easier loading)
        pickle_path = results_dir / "xgboost_model.pkl"
        joblib.dump(self.model, pickle_path)
        print(f"  Model (pkl) saved: {pickle_path}")

        # Save scaler
        scaler_path = results_dir / "xgboost_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"  Scaler saved: {scaler_path}")

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_path": str(self.data_path),
            "n_features": len(feature_names),
            "model_params": self.model.get_params(),
            "best_iteration": int(self.model.best_iteration),
            "metrics": metrics,
        }

        results_path = results_dir / "xgboost_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {results_path}")

        return results_dir

    def train(self):
        """Main training pipeline"""
        print()
        print("=" * 70)
        print("TRAINING PROCESS")
        print("=" * 70)
        print()

        # Load data
        X, y, feature_names = self.load_and_prepare_data()
        if X is None:
            return None

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale features (XGBoost doesn't require it, but can help)
        print()
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.train_model(X_train_scaled, y_train, X_test_scaled, y_test)

        # Evaluate
        metrics = self.evaluate_model(X_test_scaled, y_test, feature_names)

        # Save
        results_dir = self.save_model(feature_names, metrics)

        # Plot
        self.plot_results(X_test_scaled, y_test, metrics, results_dir)

        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print(f"Final Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        print()
        print(f"Model saved to: {results_dir}")
        print()

        return self.model, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost model for trading")
    parser.add_argument(
        "--data-path", "-d", required=True, help="Path to CSV with features and target"
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)",
    )

    args = parser.parse_args()

    trainer = XGBoostTrainer(data_path=args.data_path, test_size=args.test_size)

    model, metrics = trainer.train()


if __name__ == "__main__":
    main()
