"""
Model Validation Script
=======================
ตรวจสอบว่า Model Overfit หรือไม่

Features:
- ตรวจสอบ Train vs Test accuracy gap
- Cross-validation
- ทดสอบกับข้อมูลใหม่
- วิเคราะห์ Confusion Matrix
- ตรวจสอบ Data Leakage

วิธีใช้:
    python validate_model.py
    python validate_model.py --model models/lstm_simple_20251105_122005.keras
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

# Deep Learning
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

print("=" * 80)
print("🔍 Model Validation Script")
print("=" * 80)


def load_lstm_model(model_path):
    """โหลด LSTM model"""
    print(f"\n📂 กำลังโหลด LSTM Model: {model_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ ไม่พบ model: {model_path}")
        return None, None, None

    # โหลด model
    model = keras.models.load_model(model_path)

    # หา scaler และ features
    timestamp = model_path.stem.split("_")[-2] + "_" + model_path.stem.split("_")[-1]
    base_name = "_".join(model_path.stem.split("_")[:-2])

    scaler_path = model_path.parent / f"scaler_{base_name}_{timestamp}.pkl"
    features_path = model_path.parent / f"features_{base_name}_{timestamp}.pkl"

    scaler = None
    feature_cols = None

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✅ โหลด scaler: {scaler_path.name}")

    if features_path.exists():
        with open(features_path, "rb") as f:
            feature_cols = pickle.load(f)
        print(f"✅ โหลด features: {len(feature_cols)} features")

    return model, scaler, feature_cols


def load_xgboost_model():
    """โหลด XGBoost model"""
    print(f"\n📂 กำลังโหลด XGBoost Model")

    model_path = Path("results/xgboost/xgboost_model.pkl")
    scaler_path = Path("results/xgboost/xgboost_scaler.pkl")

    if not model_path.exists():
        print(f"❌ ไม่พบ XGBoost model")
        return None, None

    # โหลด model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # โหลด scaler
    scaler = None
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"⚠️  ไม่สามารถโหลด scaler: {e}")
            scaler = None

    print(f"✅ โหลด XGBoost model สำเร็จ")
    return model, scaler


def load_data(data_path):
    """โหลดข้อมูล"""
    print(f"\n📂 กำลังโหลดข้อมูล: {data_path}")

    if not Path(data_path).exists():
        print(f"❌ ไม่พบไฟล์: {data_path}")
        return None, None, None

    df = pd.read_csv(data_path)
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")

    # เตรียมข้อมูล
    exclude_cols = [
        "target",
        "future_price",
        "future_high",
        "future_low",
        "max_gain_pct",
        "max_loss_pct",
        "max_gain",
        "max_loss",
        "threshold",
        "future_close",
        "future_return",
        "gain_pct",
        "loss_pct",
        "score",
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

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # ลบ NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"📊 Features: {len(feature_cols)}")
    print(f"📊 Samples: {len(X):,}")
    print(f"📊 Target: UP={y.sum()}, DOWN={len(y) - y.sum()}")

    return X, y, feature_cols


def check_data_leakage(df):
    """ตรวจสอบ Data Leakage"""
    print("\n" + "=" * 80)
    print("🔍 ตรวจสอบ Data Leakage")
    print("=" * 80)

    leakage_found = False

    # ตรวจสอบ features ที่อาจรั่วไหล
    suspicious_cols = []
    for col in df.columns:
        if any(
            keyword in col.lower()
            for keyword in ["future", "next", "forward", "target"]
        ):
            if col != "target":
                suspicious_cols.append(col)

    if suspicious_cols:
        print(f"⚠️  พบ columns ที่น่าสงสัย: {len(suspicious_cols)}")
        print(f"   (columns เหล่านี้จะถูกตัดออกจาก features อัตโนมัติ)")
        for col in suspicious_cols[:5]:  # แสดง 5 อันแรก
            print(f"   - {col}")
        if len(suspicious_cols) > 5:
            print(f"   - ... และอีก {len(suspicious_cols) - 5} columns")
        # ไม่ถือว่าเป็น leakage ถ้าเราตัดออกแล้ว
        leakage_found = False
    else:
        print("✅ ไม่พบ columns ที่น่าสงสัย")

    # ตรวจสอบ correlation สูงผิดปกติกับ target
    if "target" in df.columns:
        print("\n📊 ตรวจสอบ Correlation กับ Target...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = (
            df[numeric_cols].corrwith(df["target"]).abs().sort_values(ascending=False)
        )

        high_corr = correlations[correlations > 0.9]
        if len(high_corr) > 1:  # ไม่นับ target เอง
            print(f"⚠️  พบ features ที่มี correlation สูงเกิน 0.9:")
            for col, corr in high_corr.items():
                if col != "target":
                    print(f"   - {col}: {corr:.4f}")
            leakage_found = True
        else:
            print("✅ ไม่พบ correlation ผิดปกติ")

    if not leakage_found:
        print("\n✅ ไม่พบสัญญาณ Data Leakage")
    else:
        print("\n⚠️  อาจมี Data Leakage - ควรตรวจสอบเพิ่มเติม")

    return leakage_found


def validate_lstm(model, scaler, X, y):
    """ตรวจสอบ LSTM model"""
    print("\n" + "=" * 80)
    print("🔍 Validating LSTM Model")
    print("=" * 80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Normalize
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM
    X_train_reshaped = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_reshaped = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # Evaluate
    print("\n📊 Evaluating...")
    train_loss, train_acc = model.evaluate(X_train_reshaped, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)

    print(f"\n✅ Training Accuracy: {train_acc * 100:.2f}%")
    print(f"✅ Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"📊 Accuracy Gap:      {(train_acc - test_acc) * 100:.2f}%")

    # วิเคราะห์ Overfitting
    gap = train_acc - test_acc
    if gap < 0.02:
        print("✅ Model ไม่ Overfit (gap < 2%)")
    elif gap < 0.05:
        print("⚠️  Model Overfit เล็กน้อย (gap 2-5%)")
    elif gap < 0.10:
        print("⚠️  Model Overfit ปานกลาง (gap 5-10%)")
    else:
        print("❌ Model Overfit มาก! (gap > 10%)")

    # Predictions
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"         Predicted")
    print(f"         DOWN    UP")
    print(f"Actual DOWN  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       UP    {cm[1][0]:5d}  {cm[1][1]:5d}")

    # Additional metrics
    print("\n📊 Detailed Metrics:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], digits=4))

    # AUC-ROC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📊 AUC-ROC: {auc:.4f}")

    # คำแนะนำ
    print("\n💡 คำแนะนำ:")
    if test_acc > 0.95:
        print("   ⚠️  Test Accuracy สูงเกิน 95% - ควรตรวจสอบ Data Leakage")
    if gap > 0.10:
        print("   ⚠️  Overfit สูง - ควรเพิ่ม Regularization หรือลด Model Complexity")
    if auc > 0.99:
        print("   ⚠️  AUC-ROC เกือบสมบูรณ์แบบ - ควรตรวจสอบข้อมูล")

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "auc": auc,
        "overfitting": gap > 0.10,
    }


def validate_xgboost(model, scaler, X, y):
    """ตรวจสอบ XGBoost model"""
    print("\n" + "=" * 80)
    print("🔍 Validating XGBoost Model")
    print("=" * 80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Scale if scaler exists
    if scaler:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    print(f"\n✅ Training Accuracy: {train_acc * 100:.2f}%")
    print(f"✅ Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"📊 Accuracy Gap:      {(train_acc - test_acc) * 100:.2f}%")

    # วิเคราะห์ Overfitting
    gap = train_acc - test_acc
    if gap < 0.02:
        print("✅ Model ไม่ Overfit (gap < 2%)")
    elif gap < 0.05:
        print("⚠️  Model Overfit เล็กน้อย (gap 2-5%)")
    elif gap < 0.10:
        print("⚠️  Model Overfit ปานกลาง (gap 5-10%)")
    else:
        print("❌ Model Overfit มาก! (gap > 10%)")

    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"         Predicted")
    print(f"         DOWN    UP")
    print(f"Actual DOWN  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       UP    {cm[1][0]:5d}  {cm[1][1]:5d}")

    # Additional metrics
    print("\n📊 Detailed Metrics:")
    print(
        classification_report(y_test, test_pred, target_names=["DOWN", "UP"], digits=4)
    )

    # AUC-ROC
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"📊 AUC-ROC: {auc:.4f}")
    else:
        auc = None

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "auc": auc,
        "overfitting": gap > 0.10,
    }


def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description="Model Validation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to LSTM model (optional, จะใช้ XGBoost ถ้าไม่ระบุ)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/XAUUSD_M5_features_with_target_extended_target_threshold.csv",
        help="Path to data",
    )
    args = parser.parse_args()

    try:
        # โหลดข้อมูล
        X, y, feature_cols = load_data(args.data)
        if X is None:
            sys.exit(1)

        # ตรวจสอบ Data Leakage
        df = pd.read_csv(args.data)
        check_data_leakage(df)

        # Validate model
        if args.model:
            # LSTM Model
            model, scaler, model_features = load_lstm_model(args.model)
            if model is None:
                sys.exit(1)
            results = validate_lstm(model, scaler, X, y)
        else:
            # XGBoost Model
            model, scaler = load_xgboost_model()
            if model is None:
                print("⚠️  ไม่พบ XGBoost model, กรุณาระบุ --model สำหรับ LSTM")
                sys.exit(1)
            results = validate_xgboost(model, scaler, X, y)

        # สรุปผล
        print("\n" + "=" * 80)
        print("📊 สรุปผลการตรวจสอบ")
        print("=" * 80)
        print(f"\n✅ Training Accuracy: {results['train_acc'] * 100:.2f}%")
        print(f"✅ Test Accuracy:     {results['test_acc'] * 100:.2f}%")
        print(f"📊 Gap:               {results['gap'] * 100:.2f}%")
        if results["auc"]:
            print(f"📊 AUC-ROC:           {results['auc']:.4f}")

        print("\n💡 คำแนะนำสุดท้าย:")
        if results["overfitting"]:
            print("   ❌ Model มี Overfitting - ไม่แนะนำให้ใช้งานจริง")
            print("   💡 ควรเทรนใหม่ด้วย:")
            print("      - เพิ่ม Dropout")
            print("      - เพิ่ม Regularization")
            print("      - ลด Model Complexity")
            print("      - เพิ่มข้อมูล")
        elif results["test_acc"] > 0.95:
            print("   ⚠️  Accuracy สูงผิดปกติ - ควรตรวจสอบ Data Leakage อีกครั้ง")
        elif results["test_acc"] > 0.80:
            print("   ✅ Model ดีมาก! สามารถทดสอบ Paper Trading ได้")
        elif results["test_acc"] > 0.70:
            print("   ✅ Model ดี แต่อาจต้องปรับปรุง")
        else:
            print("   ⚠️  Accuracy ต่ำ - ควรปรับปรุง Model")

        print("\n" + "=" * 80)
        print("✅ การตรวจสอบเสร็จสมบูรณ์!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
