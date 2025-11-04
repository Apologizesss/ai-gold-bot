"""
LSTM Training with Focal Loss
==============================
Uses Focal Loss to handle class imbalance better than class weights.
Focal Loss focuses more on hard-to-classify examples.
"""

import matplotlib

matplotlib.use("Agg")  # Silent mode

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for binary classification

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (gamma >= 0)
    """

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )

        # Modulating factor
        p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
        modulating_factor = tf.pow(1 - p_t, self.gamma)

        # Alpha weighting
        alpha_factor = tf.where(y_true == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_factor * modulating_factor * cross_entropy

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class FocalLossTrainer:
    """LSTM Trainer with Focal Loss"""

    def __init__(
        self,
        data_path,
        sequence_length=60,
        batch_size=64,
        epochs=100,
        learning_rate=0.001,
        use_features=140,
        architecture="stacked",
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_features = use_features
        self.architecture = architecture
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        print("=" * 80)
        print("🎯 LSTM TRAINER WITH FOCAL LOSS")
        print("=" * 80)
        print(f"Data: {data_path}")
        print(f"Sequence: {sequence_length}")
        print(f"Batch: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Features: {use_features}")
        print(f"Architecture: {architecture}")
        print(f"Focal Loss - Alpha: {focal_alpha}, Gamma: {focal_gamma}")
        print("=" * 80)

    def load_data(self):
        """Load and prepare data"""
        print("\n" + "=" * 80)
        print("🚀 STARTING FOCAL LOSS TRAINING PIPELINE")
        print("=" * 80)

        print("\n📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} rows")

        # Check target
        self.target_col = "target"
        if self.target_col not in self.df.columns:
            print(f"❌ Target column '{self.target_col}' not found!")
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
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"✅ Created {len(X_seq):,} sequences")
        print(f"📐 Shape: {X_seq.shape}")

        # Split data (time-series aware - no shuffle)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        print(f"\n📊 Data Split:")
        print(
            f"   Train: {len(self.X_train):,} ({len(self.X_train) / len(X_seq) * 100:.1f}%)"
        )
        print(
            f"   Val:   {len(self.X_val):,} ({len(self.X_val) / len(X_seq) * 100:.1f}%)"
        )
        print(
            f"   Test:  {len(self.X_test):,} ({len(self.X_test) / len(X_seq) * 100:.1f}%)"
        )

        return True

    def build_model(self):
        """Build LSTM model"""
        print(f"\n🏗️  Building {self.architecture.upper()} LSTM model...")

        n_features = self.X_train.shape[2]

        model = keras.Sequential()

        if self.architecture == "simple":
            # Simple LSTM
            model.add(
                layers.LSTM(
                    64, input_shape=(self.sequence_length, n_features), dropout=0.2
                )
            )
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dropout(0.2))

        elif self.architecture == "stacked":
            # Stacked LSTM
            model.add(
                layers.LSTM(
                    128,
                    return_sequences=True,
                    input_shape=(self.sequence_length, n_features),
                    dropout=0.2,
                )
            )
            model.add(layers.Dropout(0.3))
            model.add(layers.LSTM(64, dropout=0.2))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dropout(0.2))

        elif self.architecture == "attention":
            # LSTM with attention mechanism
            lstm_out = layers.LSTM(
                128,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features),
                dropout=0.2,
            )(model.input if hasattr(model, "input") else None)

            # Manual attention implementation
            inputs = layers.Input(shape=(self.sequence_length, n_features))
            lstm_out = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)

            # Attention layer
            attention = layers.Dense(1, activation="tanh")(lstm_out)
            attention = layers.Flatten()(attention)
            attention = layers.Activation("softmax")(attention)
            attention = layers.RepeatVector(128)(attention)
            attention = layers.Permute([2, 1])(attention)

            # Apply attention
            merged = layers.Multiply()([lstm_out, attention])
            merged = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(merged)

            model = keras.Model(inputs=inputs, outputs=merged)
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dropout(0.2))

        else:
            # Bidirectional LSTM
            model.add(
                layers.Bidirectional(
                    layers.LSTM(
                        64,
                        return_sequences=True,
                        input_shape=(self.sequence_length, n_features),
                    )
                )
            )
            model.add(layers.Dropout(0.3))
            model.add(layers.Bidirectional(layers.LSTM(32)))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dropout(0.2))

        # Output layer
        model.add(layers.Dense(1, activation="sigmoid"))

        # Compile with Focal Loss
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=focal_loss,
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model

        print("\n📋 Model Summary:")
        self.model.summary()

        return True

    def train(self):
        """Train model"""
        print("\n🚀 Starting training with Focal Loss...")

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
            "loss_function": "focal_loss",
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        results_file = Path("results/lstm/lstm_focal_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")

        # Save model
        model_file = Path("models/lstm_focal_model.h5")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_file)
        print(f"✅ Model saved to: {model_file}")

        return accuracy

    def run(self):
        """Run full pipeline"""
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
        print("✅ FOCAL LOSS TRAINING PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 80)

        return accuracy


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Train LSTM with Focal Loss")
    parser.add_argument(
        "--data-path",
        default="data/processed/XAUUSD_M15_features_with_target.csv",
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
    parser.add_argument("--features", type=int, default=140, help="Number of features")
    parser.add_argument(
        "--architecture",
        choices=["simple", "stacked", "attention", "bidirectional"],
        default="stacked",
        help="Model architecture",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha parameter (0-1)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (>= 0)",
    )

    args = parser.parse_args()

    trainer = FocalLossTrainer(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_features=args.features,
        architecture=args.architecture,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    accuracy = trainer.run()

    if accuracy is not None:
        print(f"\n🎉 Training successful! Accuracy: {accuracy * 100:.2f}%")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()
