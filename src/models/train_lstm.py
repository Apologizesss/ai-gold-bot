"""
LSTM Model Training Pipeline for Gold Price Prediction
Handles sequence creation, model architecture, training, and evaluation
"""

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend to prevent figure windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMTrainingPipeline:
    """Complete pipeline for LSTM model training on gold price data"""

    def __init__(
        self,
        data_path="data/processed/XAUUSD_M15_features_complete.csv",
        sequence_length=60,
        prediction_horizon=1,
        test_size=0.2,
        val_size=0.1,
        batch_size=32,
        epochs=50,
        learning_rate=0.001,
        use_top_features=70,
    ):
        """
        Initialize LSTM training pipeline

        Args:
            data_path: Path to processed feature data
            sequence_length: Number of timesteps to look back
            prediction_horizon: Steps ahead to predict (1 = next candle)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            use_top_features: Number of top features to use (None = use all)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_top_features = use_top_features

        # Initialize storage
        self.df = None
        self.feature_columns = None
        self.scaler = None
        self.model = None
        self.history = None
        self.results = {}

        # Create output directories
        self.model_dir = Path("models")
        self.results_dir = Path("results/lstm")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🧠 LSTM Training Pipeline Initialized")
        print("=" * 80)
        print(f"📂 Data: {self.data_path}")
        print(f"📊 Sequence Length: {self.sequence_length}")
        print(f"🎯 Prediction Horizon: {self.prediction_horizon}")
        print(f"🔢 Batch Size: {self.batch_size}")
        print(f"🔁 Epochs: {self.epochs}")
        print(f"📈 Learning Rate: {self.learning_rate}")
        print(
            f"✨ Top Features: {self.use_top_features if self.use_top_features else 'All'}"
        )
        print("=" * 80)

    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("\n📥 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")

        # Convert time column if exists
        if "time" in self.df.columns:
            self.df["time"] = pd.to_datetime(self.df["time"])
            self.df = self.df.sort_values("time").reset_index(drop=True)

        # Define feature selection based on importance
        if self.use_top_features:
            print(f"\n🎯 Selecting top {self.use_top_features} features...")
            # Top features based on Phase C analysis
            top_features = [
                "DONCH_lower",
                "WMA_20",
                "BB_upper",
                "EMA_50",
                "SMA_50",
                "EMA_20",
                "days_to_month_end",
                "EMA_200",
                "KELT_upper",
                "BB_lower",
                "hour_cos",
                "ATR_14",
                "SMA_100",
                "body_size_pct",
                "VOL_SMA_20",
                "SMA_20",
                "EMA_5",
                "BB_pct_b",
                "hour_sin",
                "VWAP",
                "week_of_year",
                "kurt_20",
                "OBV_SMA",
                "RSI_28",
                "EMA_10",
                "HL_range_pct",
                "volatility_50",
                "is_liquid_hours",
                "volatility_10",
                "session_london",
                "EMA_100",
                "WMA_10",
                "SMA_10",
                "WMA_50",
                "ROC_14",
                "DONCH_upper",
                "BB_width",
                "CCI_20",
                "KELT_lower",
                "skew_20",
                "volatility_20",
                "upper_shadow_pct",
                "day_of_week",
                "OBV",
                "session_newyork",
                "month",
                "MACD_signal",
                "RSI_14",
                "MACD",
                "session_tokyo",
                "is_tokyo_session",
                "lower_shadow_pct",
                "month_sin",
                "ADX_14",
                "STOCH_k",
                "overlap_london_newyork",
                "volume",
                "WILLIAMS_R",
                "is_london_session",
                "returns",
                "is_newyork_session",
                "day_of_month",
                "KELT_width",
                "high",
                "MFI_14",
                "close",
                "STOCH_d",
                "MACD_hist",
                "day_sin",
                "day_cos",
                "overlap_tokyo_london",
                "open",
                "low",
                "hour",
                "DONCH_width",
            ]
            # Filter to available columns
            self.feature_columns = [f for f in top_features if f in self.df.columns][
                : self.use_top_features
            ]
        else:
            # Use all numeric columns except target-related
            exclude_cols = [
                "time",
                "target",
                "target_binary",
                "target_3class",
                "future_return",
            ]
            self.feature_columns = [
                col
                for col in self.df.columns
                if col not in exclude_cols
                and self.df[col].dtype in ["float64", "int64"]
            ]

        print(f"✅ Selected {len(self.feature_columns)} features")

        # Handle missing values
        print("\n🔧 Handling missing values...")
        self.df[self.feature_columns] = (
            self.df[self.feature_columns]
            .fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0)
        )

        # Create target variable (price direction: 1=up, 0=down)
        if "target_binary" not in self.df.columns:
            print("\n🎯 Creating target variable...")
            self.df["future_close"] = self.df["close"].shift(-self.prediction_horizon)
            self.df["target_binary"] = (
                self.df["future_close"] > self.df["close"]
            ).astype(int)
            self.df = self.df[: -self.prediction_horizon]  # Remove rows without target

        # Check class balance
        target_counts = self.df["target_binary"].value_counts()
        print(f"\n📊 Target Distribution:")
        print(
            f"   Down (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0) / len(self.df) * 100:.2f}%)"
        )
        print(
            f"   Up (1):   {target_counts.get(1, 0):,} ({target_counts.get(1, 0) / len(self.df) * 100:.2f}%)"
        )

        return self

    def create_sequences(self):
        """Create sequences for LSTM input"""
        print("\n🔄 Creating sequences...")

        X = self.df[self.feature_columns].values
        y = self.df["target_binary"].values

        # Scale features
        print("📏 Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq = []
        y_seq = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.sequence_length : i])
            y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"✅ Created {len(X_seq):,} sequences")
        print(f"   Shape: {X_seq.shape} (samples, timesteps, features)")

        # Time-series split (no shuffle)
        test_idx = int(len(X_seq) * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))

        self.X_train = X_seq[:val_idx]
        self.y_train = y_seq[:val_idx]
        self.X_val = X_seq[val_idx:test_idx]
        self.y_val = y_seq[val_idx:test_idx]
        self.X_test = X_seq[test_idx:]
        self.y_test = y_seq[test_idx:]

        print(f"\n📂 Dataset splits:")
        print(
            f"   Train: {len(self.X_train):,} sequences ({len(self.X_train) / len(X_seq) * 100:.1f}%)"
        )
        print(
            f"   Val:   {len(self.X_val):,} sequences ({len(self.X_val) / len(X_seq) * 100:.1f}%)"
        )
        print(
            f"   Test:  {len(self.X_test):,} sequences ({len(self.X_test) / len(X_seq) * 100:.1f}%)"
        )

        return self

    def build_model(self, architecture="bidirectional"):
        """
        Build LSTM model architecture

        Args:
            architecture: 'simple', 'stacked', 'bidirectional', or 'attention'
        """
        print(f"\n🏗️ Building {architecture} LSTM model...")

        input_shape = (self.sequence_length, len(self.feature_columns))

        if architecture == "simple":
            # Simple LSTM
            model = models.Sequential(
                [
                    layers.LSTM(64, input_shape=input_shape, return_sequences=False),
                    layers.Dropout(0.3),
                    layers.Dense(
                        32, activation="relu", kernel_regularizer=regularizers.l2(0.01)
                    ),
                    layers.Dropout(0.3),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )

        elif architecture == "stacked":
            # Stacked LSTM
            model = models.Sequential(
                [
                    layers.LSTM(128, input_shape=input_shape, return_sequences=True),
                    layers.Dropout(0.3),
                    layers.LSTM(64, return_sequences=False),
                    layers.Dropout(0.3),
                    layers.Dense(
                        32, activation="relu", kernel_regularizer=regularizers.l2(0.01)
                    ),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )

        elif architecture == "bidirectional":
            # Bidirectional LSTM (recommended)
            model = models.Sequential(
                [
                    layers.Bidirectional(
                        layers.LSTM(64, return_sequences=True), input_shape=input_shape
                    ),
                    layers.Dropout(0.3),
                    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
                    layers.Dropout(0.3),
                    layers.Dense(
                        32, activation="relu", kernel_regularizer=regularizers.l2(0.01)
                    ),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )

        elif architecture == "attention":
            # LSTM with attention mechanism
            inputs = layers.Input(shape=input_shape)
            lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
            lstm_out = layers.Dropout(0.3)(lstm_out)

            # Attention layer
            attention = layers.Dense(1, activation="tanh")(lstm_out)
            attention = layers.Flatten()(attention)
            attention = layers.Activation("softmax")(attention)
            attention = layers.RepeatVector(64)(attention)
            attention = layers.Permute([2, 1])(attention)

            # Apply attention
            sent_representation = layers.multiply([lstm_out, attention])
            sent_representation = layers.Lambda(
                lambda xin: tf.keras.backend.sum(xin, axis=1)
            )(sent_representation)

            # Dense layers
            dense = layers.Dense(
                32, activation="relu", kernel_regularizer=regularizers.l2(0.01)
            )(sent_representation)
            dense = layers.Dropout(0.2)(dense)
            outputs = layers.Dense(1, activation="sigmoid")(dense)

            model = models.Model(inputs=inputs, outputs=outputs)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        self.model = model

        # Print model summary
        print("\n📋 Model Architecture:")
        model.summary()

        # Count parameters
        total_params = model.count_params()
        print(f"\n🔢 Total parameters: {total_params:,}")

        return self

    def train_model(self):
        """Train the LSTM model"""
        print("\n🚀 Starting training...")

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )

        model_checkpoint = callbacks.ModelCheckpoint(
            str(self.model_dir / "lstm_best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(self.y_train), y=self.y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"\n⚖️ Class weights: {class_weight_dict}")

        # Train model
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1,
        )

        print("\n✅ Training completed!")

        return self

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")

        # Predictions
        y_train_pred = (
            (self.model.predict(self.X_train, verbose=0) > 0.5).astype(int).flatten()
        )
        y_val_pred = (
            (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int).flatten()
        )
        y_test_pred = (
            (self.model.predict(self.X_test, verbose=0) > 0.5).astype(int).flatten()
        )

        # Prediction probabilities
        y_test_proba = self.model.predict(self.X_test, verbose=0).flatten()

        # Calculate metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        val_acc = accuracy_score(self.y_val, y_val_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)

        print("\n" + "=" * 80)
        print("🎯 PERFORMANCE METRICS")
        print("=" * 80)
        print(f"Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"Val Accuracy:   {val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")

        # Detailed test metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_test_pred, average="binary"
        )

        print(f"\n📈 Test Set Metrics:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        # Classification report
        print("\n📋 Classification Report (Test Set):")
        print(
            classification_report(
                self.y_test, y_test_pred, target_names=["Down (0)", "Up (1)"]
            )
        )

        # Store results
        self.results = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "predictions": y_test_pred.tolist(),
            "probabilities": y_test_proba.tolist(),
            "true_labels": self.y_test.tolist(),
        }

        return self

    def plot_results(self):
        """Plot training history and results"""
        print("\n📊 Generating plots...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Training history - Loss
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(self.history.history["loss"], label="Train Loss", linewidth=2)
        plt.plot(self.history.history["val_loss"], label="Val Loss", linewidth=2)
        plt.title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Training history - Accuracy
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(self.history.history["accuracy"], label="Train Acc", linewidth=2)
        plt.plot(self.history.history["val_accuracy"], label="Val Acc", linewidth=2)
        plt.title("Model Accuracy Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(
            self.y_test, (self.model.predict(self.X_test, verbose=0) > 0.5).astype(int)
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"],
        )
        plt.title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 4. Prediction distribution
        ax4 = plt.subplot(2, 3, 4)
        y_test_proba = self.model.predict(self.X_test, verbose=0).flatten()
        plt.hist(
            y_test_proba[self.y_test == 0],
            bins=50,
            alpha=0.6,
            label="Down (0)",
            color="red",
        )
        plt.hist(
            y_test_proba[self.y_test == 1],
            bins=50,
            alpha=0.6,
            label="Up (1)",
            color="green",
        )
        plt.title("Prediction Probability Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Precision-Recall over time
        ax5 = plt.subplot(2, 3, 5)
        if "precision" in self.history.history:
            plt.plot(
                self.history.history["precision"], label="Train Precision", linewidth=2
            )
            plt.plot(
                self.history.history["val_precision"],
                label="Val Precision",
                linewidth=2,
            )
            plt.plot(self.history.history["recall"], label="Train Recall", linewidth=2)
            plt.plot(
                self.history.history["val_recall"], label="Val Recall", linewidth=2
            )
            plt.title("Precision & Recall Over Epochs", fontsize=12, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 6. Performance comparison
        ax6 = plt.subplot(2, 3, 6)
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        scores = [
            self.results["test_accuracy"],
            self.results["test_precision"],
            self.results["test_recall"],
            self.results["test_f1"],
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = plt.bar(metrics, scores, color=colors, alpha=0.7, edgecolor="black")
        plt.title("Test Set Performance Metrics", fontsize=12, fontweight="bold")
        plt.ylabel("Score")
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / "lstm_training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot: {plot_path}")

        # plt.show()

        return self

    def save_model_and_results(self):
        """Save trained model, scaler, and results"""
        print("\n💾 Saving model and results...")

        # Save model (already saved best weights during training)
        model_path = self.model_dir / "lstm_final_model.h5"
        self.model.save(str(model_path))
        print(f"✅ Saved model: {model_path}")

        # Save scaler
        import joblib

        scaler_path = self.model_dir / "lstm_scaler.pkl"
        joblib.dump(self.scaler, str(scaler_path))
        print(f"✅ Saved scaler: {scaler_path}")

        # Save feature columns
        features_path = self.model_dir / "lstm_features.json"
        with open(features_path, "w") as f:
            json.dump({"features": self.feature_columns}, f, indent=2)
        print(f"✅ Saved features: {features_path}")

        # Save configuration
        config = {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "num_features": len(self.feature_columns),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "training_date": datetime.now().isoformat(),
        }
        config_path = self.model_dir / "lstm_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved config: {config_path}")

        # Save detailed results
        results_path = self.results_dir / "lstm_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved results: {results_path}")

        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_path = self.results_dir / "lstm_training_history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"✅ Saved history: {history_path}")

        return self

    def run_full_pipeline(self, architecture="bidirectional"):
        """Run the complete training pipeline"""
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print("🚀 STARTING FULL LSTM TRAINING PIPELINE")
        print("=" * 80)

        try:
            # Execute pipeline
            self.load_and_prepare_data()
            self.create_sequences()
            self.build_model(architecture=architecture)
            self.train_model()
            self.evaluate_model()
            self.plot_results()
            self.save_model_and_results()

            # Calculate total time
            elapsed = datetime.now() - start_time

            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"⏱️ Total time: {elapsed}")
            print(f"🎯 Test Accuracy: {self.results['test_accuracy']:.4f}")
            print(f"📊 Test F1-Score: {self.results['test_f1']:.4f}")
            print("=" * 80)

            return self

        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ PIPELINE FAILED")
            print("=" * 80)
            print(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


def main():
    """Main execution function"""

    # Configuration
    config = {
        "data_path": "data/processed/XAUUSD_M15_features_complete.csv",
        "sequence_length": 60,  # Look back 60 candles (15 hours for M15)
        "prediction_horizon": 1,  # Predict next candle
        "test_size": 0.2,  # 20% for testing
        "val_size": 0.1,  # 10% of training for validation
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "use_top_features": 70,  # Use top 70 features
    }

    # Initialize and run pipeline
    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture="bidirectional")

    print("\n🎉 LSTM Model training complete!")
    print(f"📂 Models saved in: {pipeline.model_dir}")
    print(f"📊 Results saved in: {pipeline.results_dir}")


if __name__ == "__main__":
    main()
