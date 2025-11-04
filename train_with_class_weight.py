"""
Improved LSTM Training with Class Weight Support
Fixes class imbalance problem by using class weights
"""

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend to prevent figure windows

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class ImprovedLSTMTrainer:
    """LSTM Trainer with class weight support"""

    def __init__(
        self,
        data_path="data/processed/XAUUSD_M15_features_complete.csv",
        sequence_length=60,
        batch_size=64,
        epochs=100,
        learning_rate=0.001,
        use_features=70,
        use_class_weight=True,
        architecture="stacked",
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_features = use_features
        self.use_class_weight = use_class_weight
        self.architecture = architecture

        self.model = None
        self.scaler = StandardScaler()
        self.class_weights = None

        print("=" * 80)
        print("🎯 IMPROVED LSTM TRAINER WITH CLASS WEIGHT")
        print("=" * 80)
        print(f"Data: {self.data_path}")
        print(f"Sequence: {sequence_length}")
        print(f"Batch: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Features: {use_features}")
        print(f"Architecture: {architecture}")
        print(f"Class Weight: {'✅ Enabled' if use_class_weight else '❌ Disabled'}")
        print("=" * 80)

    def load_data(self):
        """Load and prepare data"""
        print("\n📂 Loading data...")

        if not self.data_path.exists():
            print(f"❌ ERROR: File not found: {self.data_path}")
            return False

        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} rows")

        # Find target column
        target_cols = ["target", "Target", "label", "Label"]
        self.target_col = None
        for col in target_cols:
            if col in self.df.columns:
                self.target_col = col
                break

        if self.target_col is None:
            print("❌ ERROR: No target column found!")
            return False

        # Check class distribution
        print(f"\n📊 Class Distribution:")
        value_counts = self.df[self.target_col].value_counts()
        for value, count in value_counts.items():
            label = "UP" if value == 1 else "DOWN"
            pct = count / len(self.df) * 100
            print(f"   {label}: {count:,} ({pct:.2f}%)")

        # Calculate imbalance ratio
        if len(value_counts) == 2:
            ratio = value_counts.max() / value_counts.min()
            print(f"\n⚖️  Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 1.2:
                print(f"⚠️  Class imbalance detected! Using class weights...")

        # Select features - only numeric columns
        exclude_cols = [
            self.target_col,
            "timestamp",
            "Timestamp",
            "datetime",
            "DateTime",
            "time",
            "Time",
            "symbol",
            "Symbol",
            "timeframe",
            "Timeframe",
        ]

        # Get only numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Use top features if specified
        if self.use_features and self.use_features < len(feature_cols):
            # Try to load feature importance
            importance_file = Path("results/feature_selection/feature_importance.csv")
            if importance_file.exists():
                importance_df = pd.read_csv(importance_file)
                if "feature" in importance_df.columns:
                    top_features = (
                        importance_df["feature"].head(self.use_features).tolist()
                    )
                    feature_cols = [f for f in top_features if f in feature_cols]
                    print(
                        f"\n✨ Using top {len(feature_cols)} features from importance ranking"
                    )
                else:
                    feature_cols = feature_cols[: self.use_features]
                    print(f"\n✨ Using first {len(feature_cols)} features")
            else:
                feature_cols = feature_cols[: self.use_features]
                print(f"\n✨ Using first {len(feature_cols)} features")

        self.feature_cols = feature_cols
        print(f"📊 Features: {len(self.feature_cols)}")

        return True

    def prepare_sequences(self):
        """Create sequences for LSTM"""
        print("\n🔄 Creating sequences...")

        # Convert to numeric and handle any non-numeric values
        X = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce").values
        y = pd.to_numeric(self.df[self.target_col], errors="coerce").values

        # Remove NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        print(f"✅ Clean samples: {len(X):,}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq = []
        y_seq = []

        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"✅ Created {len(X_seq):,} sequences")
        print(f"📐 Shape: {X_seq.shape}")

        # Split data (80/10/10)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train):,} ({len(X_train) / len(X_seq) * 100:.1f}%)")
        print(f"   Val:   {len(X_val):,} ({len(X_val) / len(X_seq) * 100:.1f}%)")
        print(f"   Test:  {len(X_test):,} ({len(X_test) / len(X_seq) * 100:.1f}%)")

        # Compute class weights
        if self.use_class_weight:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            self.class_weights = dict(zip(classes, weights))

            print(f"\n⚖️  Class Weights:")
            for cls, weight in self.class_weights.items():
                label = "UP" if cls == 1 else "DOWN"
                print(f"   {label} ({cls}): {weight:.4f}")

        return True

    def build_model(self):
        """Build LSTM model"""
        print(f"\n🏗️  Building {self.architecture.upper()} LSTM model...")

        input_shape = (self.sequence_length, len(self.feature_cols))

        model = keras.Sequential()

        if self.architecture == "simple":
            model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=False))
            model.add(layers.Dropout(0.3))

        elif self.architecture == "stacked":
            model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.LSTM(64, return_sequences=False))
            model.add(layers.Dropout(0.3))

        elif self.architecture == "attention":
            model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.LSTM(64, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.Attention())
            model.add(layers.Flatten())

        elif self.architecture == "bidirectional":
            model.add(
                layers.Bidirectional(
                    layers.LSTM(64, return_sequences=True), input_shape=input_shape
                )
            )
            model.add(layers.Dropout(0.3))
            model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=False)))
            model.add(layers.Dropout(0.3))

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Output layers
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation="sigmoid"))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        self.model = model

        print("\n📋 Model Summary:")
        self.model.summary()

        return True

    def train(self):
        """Train model"""
        print("\n🚀 Starting training...")

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )

        # Train
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop, reduce_lr],
            class_weight=self.class_weights if self.use_class_weight else None,
            verbose=1,
        )

        print("\n✅ Training completed!")
        return True

    def evaluate(self):
        """Evaluate model"""
        print("\n📊 Evaluating model...")

        # Predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)

        print("\n" + "=" * 80)
        print("📈 TEST RESULTS")
        print("=" * 80)
        print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["DOWN", "UP"]))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"              DOWN    UP")
        print(f"Actual DOWN   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       UP     {cm[1][0]:4d}  {cm[1][1]:4d}")

        # Save results
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="binary"
        )

        results = {
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "architecture": self.architecture,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "use_class_weight": self.use_class_weight,
            "class_weights": (
                {int(k): float(v) for k, v in self.class_weights.items()}
                if self.class_weights
                else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        results_file = Path("results/lstm/lstm_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")

        # Save model
        model_file = Path("models/lstm_best_model.h5")
        self.model.save(model_file)
        print(f"✅ Model saved to: {model_file}")

        return accuracy

    def run(self):
        """Run full pipeline"""
        print("\n" + "=" * 80)
        print("🚀 STARTING IMPROVED LSTM TRAINING PIPELINE")
        print("=" * 80)

        if not self.load_data():
            return None

        if not self.prepare_sequences():
            return None

        if not self.build_model():
            return None

        if not self.train():
            return None

        accuracy = self.evaluate()

        print("\n" + "=" * 80)
        print("✅ TRAINING PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 80)

        return accuracy


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Train LSTM with class weights")
    parser.add_argument(
        "--data-path",
        default="data/processed/XAUUSD_M15_features_complete.csv",
        help="Path to data",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=60, help="Sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--features", type=int, default=70, help="Number of features")
    parser.add_argument(
        "--architecture",
        choices=["simple", "stacked", "attention", "bidirectional"],
        default="stacked",
        help="Model architecture",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class weights",
    )

    args = parser.parse_args()

    trainer = ImprovedLSTMTrainer(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_features=args.features,
        use_class_weight=not args.no_class_weight,
        architecture=args.architecture,
    )

    accuracy = trainer.run()

    if accuracy is not None:
        print(f"\n🎉 Training successful! Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"\n❌ Training failed!")


if __name__ == "__main__":
    main()
