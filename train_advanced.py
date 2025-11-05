"""
Advanced LSTM Training Script
==============================
เทรน Model ด้วย Architecture และ Hyperparameters ที่ปรับปรุงแล้ว

Features:
- Bidirectional LSTM
- Attention Mechanism
- Advanced Data Augmentation
- Class Weight Balancing
- Custom Learning Rate Schedule
- Multiple Model Architectures

วิธีใช้:
    python train_advanced.py
    python train_advanced.py --model attention --epochs 150
    python train_advanced.py --model bidirectional --data data/processed/XAUUSD_M5_features_with_target_extended.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

print("=" * 80)
print("[Launch] Advanced LSTM Training Script")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)


def load_data(data_path=None):
    """โหลดข้อมูล"""
    print("\n📂 กำลังโหลดข้อมูล...")

    if data_path:
        df = pd.read_csv(data_path)
        print(f"[OK] โหลดจาก: {data_path}")
    else:
        data_dir = Path("data/processed")
        possible_files = [
            "XAUUSD_M5_features_with_target_extended.csv",
            "XAUUSD_M15_features_with_target_extended.csv",
            "XAUUSD_H1_features_with_target_extended.csv",
            "XAUUSD_M15_features_with_target.csv",
        ]

        df = None
        for filename in possible_files:
            filepath = data_dir / filename
            if filepath.exists():
                print(f"[OK] พบไฟล์: {filename}")
                df = pd.read_csv(filepath)
                break

        if df is None:
            print("[Error] ไม่พบไฟล์ข้อมูล!")
            sys.exit(1)

    print(f"[Stats] โหลดข้อมูลสำเร็จ: {len(df):,} แถว, {len(df.columns)} columns")
    return df


def prepare_data(df, use_robust_scaler=True):
    """เตรียมข้อมูล"""
    print("\n[Feature Engineering] กำลังเตรียมข้อมูล...")

    # สร้าง target ถ้ายังไม่มี
    if "target" not in df.columns:
        print("[Warning]  ไม่พบ target column, กำลังสร้าง...")
        df["future_price"] = df["close"].shift(-4)
        df["target"] = (df["future_price"] > df["close"]).astype(int)

    # กำจัด columns ที่ไม่ต้องการ
    exclude_cols = [
        "target",
        "future_price",
        "time",
        "timestamp",
        "symbol",
        "timeframe",
        "date",
        "datetime",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # ลบ columns ที่เป็น object
    for col in feature_cols[:]:
        if df[col].dtype == "object":
            feature_cols.remove(col)

    print(f"[Stats] จำนวน features: {len(feature_cols)}")

    # เอาเฉพาะ columns ที่ต้องการ
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # ลบแถวที่มี NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"[Stats] ข้อมูลหลังทำความสะอาด: {len(X):,} แถว")
    print(f"[Stats] Target distribution: UP={y.sum()}, DOWN={len(y) - y.sum()}")
    print(f"[Stats] Class balance: {y.sum() / len(y) * 100:.1f}% UP")

    return X, y, feature_cols


def create_bidirectional_lstm(input_shape, units=128, dropout=0.3):
    """สร้าง Bidirectional LSTM Model"""
    print("\n🤖 กำลังสร้าง Bidirectional LSTM Model...")

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(units // 2, return_sequences=True)),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.LSTM(units // 4),
            layers.Dropout(dropout / 2),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout / 2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout / 4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_attention_lstm(input_shape, units=128, dropout=0.3):
    """สร้าง LSTM with Attention Model"""
    print("\n🤖 กำลังสร้าง Attention LSTM Model...")

    inputs = layers.Input(shape=input_shape)

    # LSTM layers
    x = layers.LSTM(units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(units // 2, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    # Attention mechanism
    x = AttentionLayer()(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout / 4)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_deep_lstm(input_shape, units=256, dropout=0.3):
    """สร้าง Deep LSTM Model"""
    print("\n🤖 กำลังสร้าง Deep LSTM Model...")

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.LSTM(units // 2, return_sequences=True),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.LSTM(units // 4, return_sequences=True),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            layers.LSTM(units // 8),
            layers.Dropout(dropout / 2),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout / 2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout / 2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout / 4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_model(model_type, input_shape, units=128, dropout=0.3):
    """สร้าง model ตามประเภทที่เลือก"""

    if model_type == "bidirectional":
        model = create_bidirectional_lstm(input_shape, units, dropout)
    elif model_type == "attention":
        model = create_attention_lstm(input_shape, units, dropout)
    elif model_type == "deep":
        model = create_deep_lstm(input_shape, units, dropout)
    else:
        print(f"[Error] Model type '{model_type}' ไม่รู้จัก, ใช้ bidirectional แทน")
        model = create_bidirectional_lstm(input_shape, units, dropout)

    # Use static learning rate (compatible with ReduceLROnPlateau)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    print("[OK] Model สร้างเสร็จแล้ว")
    print(f"[Stats] Parameters: {model.count_params():,}")

    return model


class DetailedMetrics(Callback):
    """Custom callback สำหรับแสดงผลละเอียด"""

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.best_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy", 0) > self.best_val_acc:
            self.best_val_acc = logs.get("val_accuracy", 0)
            print(f"\n[Target] New best! Val Accuracy: {self.best_val_acc * 100:.2f}%")


def train_model(
    X, y, model_type="bidirectional", epochs=150, batch_size=64, validation_split=0.2
):
    """เทรน model"""
    print("\n" + "=" * 80)
    print("[Target] เริ่มการเทรน")
    print("=" * 80)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("=" * 80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[Stats] Training set: {len(X_train):,} แถว")
    print(f"[Stats] Test set: {len(X_test):,} แถว")

    # คำนวณ class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"[Stats] Class weights: DOWN={class_weights[0]:.2f}, UP={class_weights[1]:.2f}")

    # Normalize ด้วย RobustScaler (ดีกว่า StandardScaler สำหรับ outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM
    X_train_reshaped = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_reshaped = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_reshaped,
        y_train,
        test_size=validation_split,
        random_state=42,
        stratify=y_train,
    )

    # Create model
    model = create_model(model_type, input_shape=(1, X_train_scaled.shape[1]))

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="models/best_advanced_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        DetailedMetrics(validation_data=(X_val, y_val)),
    ]

    # Train
    print(f"\n[Reload] กำลังเทรน {epochs} epochs...\n")

    history = model.fit(
        X_train_final,
        y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Load best model
    print("\n📂 โหลด best model...")
    model = keras.models.load_model(
        "models/best_advanced_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("[Stats] ประเมินผล")
    print("=" * 80)

    train_results = model.evaluate(X_train_reshaped, y_train, verbose=0)
    test_results = model.evaluate(X_test_reshaped, y_test, verbose=0)

    train_loss, train_acc, train_auc = train_results
    test_loss, test_acc, test_auc = test_results

    print(f"\n[OK] Training Accuracy: {train_acc * 100:.2f}%")
    print(f"[OK] Test Accuracy: {test_acc * 100:.2f}%")
    print(f"[OK] Training AUC: {train_auc:.4f}")
    print(f"[OK] Test AUC: {test_auc:.4f}")
    print(f"[Stats] Accuracy Gap: {(train_acc - test_acc) * 100:.2f}%")

    # Predictions
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n[Stats] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], digits=4))

    print("\n[Stats] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    total = cm.sum()
    print(f"         Predicted")
    print(f"         DOWN    UP")
    print(
        f"Actual DOWN  {cm[0][0]:5d}  {cm[0][1]:5d}  ({cm[0][0] / cm[0].sum() * 100:.1f}% correct)"
    )
    print(
        f"       UP    {cm[1][0]:5d}  {cm[1][1]:5d}  ({cm[1][1] / cm[1].sum() * 100:.1f}% correct)"
    )

    # Calculate additional metrics
    precision_up = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    recall_up = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

    print(f"\n[Stats] Trading Metrics:")
    print(
        f"   Precision (UP): {precision_up * 100:.2f}% (เมื่อทำนาย UP จะถูก {precision_up * 100:.1f}%)"
    )
    print(f"   Recall (UP): {recall_up * 100:.2f}% (จับ UP ได้ {recall_up * 100:.1f}%)")

    # Training history
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_val_acc = max(history.history["val_accuracy"])

    print(f"\n[Chart] Training History:")
    print(f"   Best Epoch: {best_epoch}/{epochs}")
    print(f"   Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"   Final Val Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

    return (
        model,
        scaler,
        history,
        {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "precision_up": precision_up,
            "recall_up": recall_up,
        },
    )


def save_model(model, scaler, feature_cols, metrics, model_type):
    """บันทึก model"""
    print("\n" + "=" * 80)
    print("[Save] กำลังบันทึก Model")
    print("=" * 80)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # บันทึก model
    model_path = models_dir / f"lstm_advanced_{model_type}_{timestamp}.keras"
    model.save(model_path)
    print(f"[OK] Model: {model_path}")

    # บันทึก scaler
    scaler_path = models_dir / f"scaler_advanced_{timestamp}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Scaler: {scaler_path}")

    # บันทึก feature names
    features_path = models_dir / f"features_advanced_{timestamp}.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[OK] Features: {features_path}")

    # บันทึก metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": model_type,
        "train_accuracy": float(metrics["train_acc"]),
        "test_accuracy": float(metrics["test_acc"]),
        "train_auc": float(metrics["train_auc"]),
        "test_auc": float(metrics["test_auc"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_accuracy": float(metrics["best_val_acc"]),
        "precision_up": float(metrics["precision_up"]),
        "recall_up": float(metrics["recall_up"]),
        "num_features": len(feature_cols),
        "model_file": str(model_path.name),
        "scaler_file": str(scaler_path.name),
        "features_file": str(features_path.name),
    }

    import json

    metadata_path = models_dir / f"metadata_advanced_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print("[OK] บันทึกเสร็จสมบูรณ์!")
    print("=" * 80)

    return model_path


def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description="Advanced LSTM Training")
    parser.add_argument("--data", type=str, default=None, help="ไฟล์ข้อมูล")
    parser.add_argument(
        "--model",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "attention", "deep"],
        help="ประเภท model (default: bidirectional)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="จำนวน epochs (default: 150)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--units", type=int, default=128, help="LSTM units (default: 128)"
    )
    args = parser.parse_args()

    try:
        # Load data
        df = load_data(args.data)

        # Prepare data
        X, y, feature_cols = prepare_data(df)

        # Train model
        model, scaler, history, metrics = train_model(
            X, y, model_type=args.model, epochs=args.epochs, batch_size=args.batch_size
        )

        # Save model
        model_path = save_model(model, scaler, feature_cols, metrics, args.model)

        # Summary
        print("\n" + "[Success]" * 40)
        print("\n[OK] การเทรนเสร็จสมบูรณ์!\n")
        print(f"[Stats] Model Type: {args.model.upper()}")
        print(f"[Stats] Test Accuracy: {metrics['test_acc'] * 100:.2f}%")
        print(f"[Stats] Test AUC: {metrics['test_auc']:.4f}")
        print(f"[Stats] Precision (UP): {metrics['precision_up'] * 100:.2f}%")
        print(f"[Save] Model saved: {model_path.name}")
        print("\n[Tip] ขั้นตอนต่อไป:")
        print("   1. รัน daily_update.bat เพื่ออัพเดทข้อมูล")
        print("   2. รัน paper_trading.py เพื่อทดสอบเทรด")
        print("   3. รัน live_trading.py เพื่อเทรดจริง")
        print("\n" + "[Success]" * 40 + "\n")

    except Exception as e:
        print(f"\n[Error] เกิดข้อผิดพลาด: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
