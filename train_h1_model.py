"""
Train LSTM Model on H1 Data
============================
Simple trainer for H1 timeframe data with advanced features
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import json


class H1LSTMTrainer:
    def __init__(
        self, data_path, sequence_length=60, features=67, epochs=150, batch_size=32
    ):
        print("=" * 70)
        print("H1 LSTM TRAINER")
        print("=" * 70)
        print()
        print(f"Data: {data_path}")
        print(f"Sequence length: {sequence_length}")
        print(f"Features: {features}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print()

        self.data_path = data_path
        self.sequence_length = sequence_length
        self.n_features = features
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def load_and_prepare_data(self):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df):,} rows")

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from features if present
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        print(f"  Found {len(numeric_cols)} numeric features")

        # Extract features and target
        X = df[numeric_cols].values
        y = df["target"].values

        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")

        # Check for NaN
        if np.isnan(X).any():
            print("  Warning: Found NaN values, filling with 0")
            X = np.nan_to_num(X, 0)

        # Store feature count
        self.n_features = X.shape[1]
        print(f"  Using {self.n_features} features")

        return X, y

    def create_sequences(self, X, y):
        print(f"Creating sequences (length={self.sequence_length})...")

        X_seq = []
        y_seq = []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"  Created {len(X_seq):,} sequences")
        print(f"  Sequence shape: {X_seq.shape}")

        return X_seq, y_seq

    def split_data(self, X, y):
        print("Splitting data...")

        # Split: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )

        print(f"  Train: {len(X_train):,} samples")
        print(f"  Val:   {len(X_val):,} samples")
        print(f"  Test:  {len(X_test):,} samples")

        # Check class balance
        print(
            f"  Train UP: {np.sum(y_train)}/{len(y_train)} ({np.mean(y_train) * 100:.1f}%)"
        )
        print(f"  Val UP:   {np.sum(y_val)}/{len(y_val)} ({np.mean(y_val) * 100:.1f}%)")
        print(
            f"  Test UP:  {np.sum(y_test)}/{len(y_test)} ({np.mean(y_test) * 100:.1f}%)"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        print("Building LSTM model...")

        model = keras.Sequential(
            [
                layers.Input(shape=(self.sequence_length, self.n_features)),
                # First LSTM layer with return sequences
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                # Second LSTM layer with return sequences
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                # Third LSTM layer (no return sequences)
                layers.LSTM(32),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                # Dense layers
                layers.Dense(16, activation="relu"),
                layers.Dropout(0.2),
                # Output
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        print("  Model built successfully")
        model.summary()

        return model

    def train(self):
        print()
        print("=" * 70)
        print("TRAINING PROCESS")
        print("=" * 70)
        print()

        # Load and prepare data
        X, y = self.load_and_prepare_data()

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_seq, y_seq)

        # Build model
        model = self.build_model()

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
        )

        # Train
        print()
        print("Starting training...")
        print()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        print()
        print("=" * 70)
        print("EVALUATION")
        print("=" * 70)
        print()

        # Evaluate on test set
        print("Evaluating on test set...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall:    {test_rec:.4f}")
        print(f"Test F1-Score:  {test_f1:.4f}")
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

        # Save results
        print("Saving results...")

        # Create results directory
        results_dir = Path("results/h1_training")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = results_dir / "h1_lstm_model.keras"
        model.save(model_path)
        print(f"  Model saved: {model_path}")

        # Save scaler
        import joblib

        scaler_path = results_dir / "h1_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"  Scaler saved: {scaler_path}")

        # Save metrics
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_path": str(self.data_path),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "test_accuracy": float(test_acc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "final_epoch": len(history.history["loss"]),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
        }

        results_path = results_dir / "h1_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {results_path}")

        # Plot training history
        print("  Plotting training history...")
        self.plot_history(history, results_dir)

        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Model saved to: {model_path}")
        print()

        return model, history, results

    def plot_history(self, history, output_dir):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"], label="Train")
        axes[0, 0].plot(history.history["val_accuracy"], label="Val")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history.history["loss"], label="Train")
        axes[0, 1].plot(history.history["val_loss"], label="Val")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(history.history["precision"], label="Train")
        axes[1, 0].plot(history.history["val_precision"], label="Val")
        axes[1, 0].set_title("Model Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(history.history["recall"], label="Train")
        axes[1, 1].plot(history.history["val_recall"], label="Val")
        axes[1, 1].set_title("Model Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "h1_training_history.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()


def main():
    data_path = "data/raw/XAUUSD_H1_20251101_172515_advanced_features_with_target.csv"

    trainer = H1LSTMTrainer(
        data_path=data_path, sequence_length=60, features=67, epochs=150, batch_size=32
    )

    model, history, results = trainer.train()


if __name__ == "__main__":
    main()
