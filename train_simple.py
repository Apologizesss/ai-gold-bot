"""
Simple LSTM Training Script
============================
วิธีใช้ง่ายๆ สำหรับเทรน LSTM model

วิธีใช้:
    python train_simple.py

หรือ:
    python train_simple.py --epochs 100 --batch_size 32
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import sys
import argparse

# Deep Learning
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 80)
print("[Launch] Simple LSTM Training Script")
print("=" * 80)


def load_data():
    """โหลดข้อมูลที่มีอยู่"""
    print("\n📂 กำลังโหลดข้อมูล...")

    data_dir = Path("data/processed")

    # ลองหาไฟล์ที่มี
    possible_files = [
        "XAUUSD_M5_features_with_target_extended.csv",
        "XAUUSD_M15_features_with_target_extended.csv",
        "XAUUSD_H1_features_with_target_extended.csv",
        "XAUUSD_H4_features_with_target_extended.csv",
        "XAUUSD_M15_features_with_target.csv",
        "XAUUSD_H1_features_complete.csv",
        "XAUUSD_M5_features_complete.csv",
    ]

    df = None
    for filename in possible_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"[OK] พบไฟล์: {filename}")
            df = pd.read_csv(filepath)
            break

    if df is None:
        # ลองใช้ไฟล์ที่สร้างจาก daily_update
        recent_file = Path("data/processed_data_20251105.csv")
        if recent_file.exists():
            print(f"[OK] พบไฟล์: {recent_file.name}")
            df = pd.read_csv(recent_file)
        else:
            print("[Error] ไม่พบไฟล์ข้อมูล!")
            print("\n[Tip] แนะนำ: รัน daily_update.bat เพื่อดึงข้อมูลก่อน")
            sys.exit(1)

    print(f"[Stats] โหลดข้อมูลสำเร็จ: {len(df)} แถว, {len(df.columns)} columns")
    return df


def prepare_data(df):
    """เตรียมข้อมูลสำหรับเทรน"""
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

    # ลบ columns ที่เป็น object หรือ string
    for col in feature_cols[:]:
        if df[col].dtype == "object":
            feature_cols.remove(col)
            print(f"   [Warning]  ข้ามcolumn: {col} (ประเภทข้อมูลไม่เหมาะสม)")

    print(f"[Stats] จำนวน features: {len(feature_cols)}")

    # เอาเฉพาะ columns ที่ต้องการ
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # ลบแถวที่มี NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"[Stats] ข้อมูลหลังทำความสะอาด: {len(X)} แถว")
    print(f"[Stats] Target distribution: UP={y.sum()}, DOWN={len(y) - y.sum()}")

    if len(X) < 100:
        print("[Error] ข้อมูลไม่เพียงพอ! (ต้องการอย่างน้อย 100 แถว)")
        sys.exit(1)

    return X, y, feature_cols


def create_model(input_shape, units=64, dropout=0.3):
    """สร้าง LSTM model"""
    print("\n🤖 กำลังสร้าง Model...")

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units // 2),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout / 2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("[OK] Model สร้างเสร็จแล้ว")
    print(f"[Stats] Parameters: {model.count_params():,}")

    return model


def train_model(X, y, epochs=50, batch_size=32, validation_split=0.2):
    """เทรน model"""
    print("\n" + "=" * 80)
    print("[Target] เริ่มการเทรน")
    print("=" * 80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[Stats] Training set: {len(X_train)} แถว")
    print(f"[Stats] Test set: {len(X_test)} แถว")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_reshaped = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # Create model
    model = create_model(input_shape=(1, X_train_scaled.shape[1]))

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1, min_lr=1e-7
        ),
    ]

    # Train
    print(f"\n[Reload] กำลังเทรน {epochs} epochs...\n")

    history = model.fit(
        X_train_reshaped,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("[Stats] ประเมินผล")
    print("=" * 80)

    train_loss, train_acc = model.evaluate(X_train_reshaped, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)

    print(f"\n[OK] Training Accuracy: {train_acc * 100:.2f}%")
    print(f"[OK] Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n[Stats] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    print("\n[Stats] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"         Predicted")
    print(f"         DOWN  UP")
    print(f"Actual DOWN  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       UP    {cm[1][0]:4d}  {cm[1][1]:4d}")

    return (
        model,
        scaler,
        history,
        {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
        },
    )


def save_model(model, scaler, feature_cols, metrics):
    """บันทึก model"""
    print("\n" + "=" * 80)
    print("[Save] กำลังบันทึก Model")
    print("=" * 80)

    # สร้างโฟลเดอร์
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # บันทึก model
    model_path = models_dir / f"lstm_simple_{timestamp}.keras"
    model.save(model_path)
    print(f"[OK] Model: {model_path}")

    # บันทึก scaler
    scaler_path = models_dir / f"scaler_simple_{timestamp}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Scaler: {scaler_path}")

    # บันทึก feature names
    features_path = models_dir / f"features_simple_{timestamp}.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[OK] Features: {features_path}")

    # บันทึก metadata
    metadata = {
        "timestamp": timestamp,
        "train_accuracy": float(metrics["train_acc"]),
        "test_accuracy": float(metrics["test_acc"]),
        "train_loss": float(metrics["train_loss"]),
        "test_loss": float(metrics["test_loss"]),
        "num_features": len(feature_cols),
        "model_file": str(model_path.name),
        "scaler_file": str(scaler_path.name),
        "features_file": str(features_path.name),
    }

    import json

    metadata_path = models_dir / f"metadata_simple_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print("[OK] บันทึกเสร็จสมบูรณ์!")
    print("=" * 80)

    return model_path


def main():
    """ฟังก์ชันหลัก"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple LSTM Training")
    parser.add_argument("--data", type=str, default=None, help="ไฟล์ข้อมูล (optional)")
    parser.add_argument(
        "--epochs", type=int, default=100, help="จำนวน epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--units", type=int, default=64, help="LSTM units (default: 64)"
    )
    args = parser.parse_args()

    try:
        # Load data
        if args.data:
            print(f"📂 กำลังโหลดข้อมูลจาก: {args.data}")
            df = pd.read_csv(args.data)
            print(f"[Stats] โหลดข้อมูลสำเร็จ: {len(df)} แถว, {len(df.columns)} columns")
        else:
            df = load_data()

        # Prepare data
        X, y, feature_cols = prepare_data(df)

        # Train model
        model, scaler, history, metrics = train_model(
            X, y, epochs=args.epochs, batch_size=args.batch_size
        )

        # Save model
        model_path = save_model(model, scaler, feature_cols, metrics)

        # Summary
        print("\n" + "[Success]" * 40)
        print("\n[OK] การเทรนเสร็จสมบูรณ์!\n")
        print(f"[Stats] Test Accuracy: {metrics['test_acc'] * 100:.2f}%")
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
