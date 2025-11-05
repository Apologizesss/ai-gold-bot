"""
Improve Target Definition Script
=================================
ปรับปรุง Target ให้ง่ายและแม่นยำขึ้น

Strategies:
1. Price Change Threshold - ราคาต้องขึ้น/ลงเกินเทรสโฮลด์
2. Support/Resistance Bounce - เด้งจาก S/R
3. Trend Following - ตามเทรนด์
4. Volatility-based - ขึ้นอยู่กับความผันผวน

วิธีใช้:
    python improve_target.py
    python improve_target.py --strategy threshold --threshold 0.5
    python improve_target.py --strategy trend --periods 10
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


def load_data(filepath):
    """โหลดข้อมูล"""
    print(f"\n📂 กำลังโหลดข้อมูลจาก: {filepath}")

    if not Path(filepath).exists():
        print(f"❌ ไม่พบไฟล์: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")

    # ตรวจสอบ columns ที่จำเป็น
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ ขาด columns: {missing_cols}")
        sys.exit(1)

    return df


def strategy_threshold(df, threshold_pct=0.3, lookahead=4):
    """
    Strategy 1: Price Change Threshold
    ราคาต้องขึ้นเกิน threshold_pct % ใน lookahead แท่งข้างหน้า

    ข้อดี: ง่าย, ชัดเจน, กรองสัญญาณที่ไม่แน่ชัดออก
    """
    print(f"\n🎯 Strategy: Price Change Threshold")
    print(f"   Threshold: {threshold_pct}%")
    print(f"   Lookahead: {lookahead} candles")

    # คำนวณราคาสูงสุดใน lookahead แท่งข้างหน้า
    df["future_high"] = df["high"].shift(-1).rolling(window=lookahead).max()
    df["future_low"] = df["low"].shift(-1).rolling(window=lookahead).min()

    # คำนวณ % change
    df["max_gain_pct"] = ((df["future_high"] - df["close"]) / df["close"]) * 100
    df["max_loss_pct"] = ((df["close"] - df["future_low"]) / df["close"]) * 100

    # สร้าง target
    # UP (1) = ราคาขึ้นเกิน threshold และไม่ลงเกิน threshold
    # DOWN (0) = ราคาลงเกิน threshold หรือขึ้นไม่ถึง threshold
    df["target"] = 0
    df.loc[
        (df["max_gain_pct"] >= threshold_pct) & (df["max_loss_pct"] < threshold_pct),
        "target",
    ] = 1

    return df


def strategy_volatility_adjusted(df, atr_multiplier=1.5, lookahead=4):
    """
    Strategy 2: Volatility-Adjusted Target
    ใช้ ATR เป็นตัวกำหนด threshold (ปรับตามความผันผวน)

    ข้อดี: ปรับตามสภาพตลาด, realistic
    """
    print(f"\n🎯 Strategy: Volatility-Adjusted (ATR-based)")
    print(f"   ATR Multiplier: {atr_multiplier}x")
    print(f"   Lookahead: {lookahead} candles")

    # คำนวณ ATR ถ้ายังไม่มี
    if "ATR" not in df.columns:
        print("   กำลังคำนวณ ATR...")
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14).mean()

    # คำนวณราคาในอนาคต
    df["future_high"] = df["high"].shift(-1).rolling(window=lookahead).max()
    df["future_low"] = df["low"].shift(-1).rolling(window=lookahead).min()

    # ใช้ ATR เป็น threshold
    df["threshold"] = df["ATR"] * atr_multiplier
    df["max_gain"] = df["future_high"] - df["close"]
    df["max_loss"] = df["close"] - df["future_low"]

    # Target: ขึ้นเกิน threshold และไม่ลงเกิน threshold
    df["target"] = 0
    df.loc[
        (df["max_gain"] >= df["threshold"]) & (df["max_loss"] < df["threshold"]),
        "target",
    ] = 1

    return df


def strategy_trend_following(df, trend_periods=20, lookahead=4):
    """
    Strategy 3: Trend Following
    เทรดตามทิศทางเทรนด์หลัก

    ข้อดี: ตามเทรนด์, win rate อาจสูงกว่า
    """
    print(f"\n🎯 Strategy: Trend Following")
    print(f"   Trend Periods: {trend_periods}")
    print(f"   Lookahead: {lookahead} candles")

    # คำนวณ Moving Average เป็นเทรนด์
    if f"SMA_{trend_periods}" not in df.columns:
        df[f"SMA_{trend_periods}"] = df["close"].rolling(window=trend_periods).mean()

    # ทิศทางเทรนด์
    df["trend"] = np.where(df["close"] > df[f"SMA_{trend_periods}"], 1, 0)

    # ราคาในอนาคต
    df["future_close"] = df["close"].shift(-lookahead)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # Target: เทรนด์ขึ้น และราคาขึ้นในอนาคต
    df["target"] = 0
    df.loc[(df["trend"] == 1) & (df["future_return"] > 0), "target"] = 1

    return df


def strategy_support_resistance(df, lookback=20, lookahead=4, threshold_pct=0.2):
    """
    Strategy 4: Support/Resistance Bounce
    เทรดเมื่อราคาเด้งจาก S/R

    ข้อดี: สัญญาณคุณภาพสูง, เหมาะสำหรับ swing trading
    """
    print(f"\n🎯 Strategy: Support/Resistance Bounce")
    print(f"   Lookback: {lookback}")
    print(f"   Threshold: {threshold_pct}%")

    # หา Support (low ต่ำสุดใน lookback)
    df["support"] = df["low"].rolling(window=lookback).min()

    # หา Resistance (high สูงสุดใน lookback)
    df["resistance"] = df["high"].rolling(window=lookback).max()

    # ระยะห่างจาก S/R (เป็น %)
    df["dist_to_support"] = ((df["close"] - df["support"]) / df["close"]) * 100
    df["dist_to_resistance"] = ((df["resistance"] - df["close"]) / df["close"]) * 100

    # ราคาในอนาคต
    df["future_high"] = df["high"].shift(-1).rolling(window=lookahead).max()
    df["future_return"] = ((df["future_high"] - df["close"]) / df["close"]) * 100

    # Target: ใกล้ support และเด้งขึ้น
    df["target"] = 0
    df.loc[
        (df["dist_to_support"] <= threshold_pct)  # ใกล้ support
        & (df["future_return"] > threshold_pct),  # เด้งขึ้น
        "target",
    ] = 1

    return df


def strategy_combined(df, threshold_pct=0.3, atr_multiplier=1.0, lookahead=4):
    """
    Strategy 5: Combined (Best of All)
    รวมหลายๆ strategy เข้าด้วยกัน

    ข้อดี: สัญญาณคุณภาพสูงสุด (แต่อาจน้อย)
    """
    print(f"\n🎯 Strategy: Combined")
    print(f"   Combining multiple strategies...")

    # คำนวณ ATR
    if "ATR" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14).mean()

    # คำนวณ SMA
    if "SMA_20" not in df.columns:
        df["SMA_20"] = df["close"].rolling(window=20).mean()

    # ราคาในอนาคต
    df["future_high"] = df["high"].shift(-1).rolling(window=lookahead).max()
    df["future_low"] = df["low"].shift(-1).rolling(window=lookahead).min()

    # เงื่อนไขที่ 1: Price threshold
    df["gain_pct"] = ((df["future_high"] - df["close"]) / df["close"]) * 100
    df["loss_pct"] = ((df["close"] - df["future_low"]) / df["close"]) * 100
    condition1 = (df["gain_pct"] >= threshold_pct) & (df["loss_pct"] < threshold_pct)

    # เงื่อนไขที่ 2: ATR-based
    df["threshold"] = df["ATR"] * atr_multiplier
    df["max_gain"] = df["future_high"] - df["close"]
    condition2 = df["max_gain"] >= df["threshold"]

    # เงื่อนไขที่ 3: Trend
    df["trend"] = df["close"] > df["SMA_20"]
    condition3 = df["trend"] == True

    # Target: ต้องผ่านอย่างน้อย 2 จาก 3 เงื่อนไข
    df["score"] = (
        condition1.astype(int) + condition2.astype(int) + condition3.astype(int)
    )
    df["target"] = (df["score"] >= 2).astype(int)

    return df


def analyze_target(df, strategy_name):
    """วิเคราะห์ target ที่ได้"""
    print("\n" + "=" * 80)
    print("📊 วิเคราะห์ Target")
    print("=" * 80)

    # ลบแถวที่เป็น NaN
    df_clean = df.dropna(subset=["target"])

    total = len(df_clean)
    up_count = df_clean["target"].sum()
    down_count = total - up_count

    up_pct = (up_count / total) * 100
    down_pct = (down_count / total) * 100

    print(f"\n📊 Target Distribution:")
    print(f"   Total samples: {total:,}")
    print(f"   UP (1):   {up_count:,} ({up_pct:.2f}%)")
    print(f"   DOWN (0): {down_count:,} ({down_pct:.2f}%)")

    # คำนวณ class imbalance ratio
    if down_count > 0:
        ratio = up_count / down_count
        print(f"   UP/DOWN Ratio: {ratio:.2f}")

        if 0.4 <= ratio <= 0.6:
            print("   ✅ Class balance ดีมาก!")
        elif 0.3 <= ratio <= 0.7:
            print("   ⚠️  Class imbalance เล็กน้อย")
        else:
            print("   ❌ Class imbalance สูง! (อาจต้องใช้ class weights)")

    # ตรวจสอบคุณภาพ
    print(f"\n💡 คำแนะนำ:")
    if up_pct < 30 or up_pct > 70:
        print("   • Target มี imbalance สูง - แนะนำใช้ class weights")
    if up_pct < 20 or up_pct > 80:
        print("   • Target imbalance มากเกินไป - ควรปรับ strategy")
    if 40 <= up_pct <= 60:
        print("   • Target สมดุลดีมาก - เหมาะสำหรับเทรน")

    return df_clean


def save_improved_data(df, original_filepath, strategy_name):
    """บันทึกข้อมูลที่ปรับปรุงแล้ว"""
    print("\n" + "=" * 80)
    print("💾 กำลังบันทึกข้อมูล")
    print("=" * 80)

    # สร้างชื่อไฟล์ใหม่
    original_path = Path(original_filepath)
    new_filename = original_path.stem + f"_target_{strategy_name}.csv"
    new_filepath = original_path.parent / new_filename

    # บันทึก
    df.to_csv(new_filepath, index=False)
    print(f"✅ บันทึกไฟล์: {new_filepath}")

    return new_filepath


def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description="Improve Target Definition")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/XAUUSD_M5_features_with_target_extended.csv",
        help="ไฟล์ข้อมูล",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="threshold",
        choices=["threshold", "volatility", "trend", "support", "combined"],
        help="Strategy (default: threshold)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Price threshold % (default: 0.3)"
    )
    parser.add_argument(
        "--lookahead", type=int, default=4, help="จำนวนแท่งข้างหน้า (default: 4)"
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=1.5,
        help="ATR multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--trend-periods", type=int, default=20, help="Trend periods (default: 20)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🎯 Improve Target Definition")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Strategy: {args.strategy}")
    print("=" * 80)

    try:
        # Load data
        df = load_data(args.data)

        # Apply strategy
        if args.strategy == "threshold":
            df = strategy_threshold(df, args.threshold, args.lookahead)
        elif args.strategy == "volatility":
            df = strategy_volatility_adjusted(df, args.atr_multiplier, args.lookahead)
        elif args.strategy == "trend":
            df = strategy_trend_following(df, args.trend_periods, args.lookahead)
        elif args.strategy == "support":
            df = strategy_support_resistance(
                df, lookahead=args.lookahead, threshold_pct=args.threshold
            )
        elif args.strategy == "combined":
            df = strategy_combined(
                df, args.threshold, args.atr_multiplier, args.lookahead
            )

        # Analyze
        df_clean = analyze_target(df, args.strategy)

        # Save
        new_filepath = save_improved_data(df_clean, args.data, args.strategy)

        # Summary
        print("\n" + "=" * 80)
        print("✅ เสร็จสมบูรณ์!")
        print("=" * 80)
        print(f"\n💡 ขั้นตอนต่อไป:")
        print(f"   python train_simple.py --data {new_filepath}")
        print(f"   หรือ")
        print(
            f"   python train_advanced.py --model bidirectional --data {new_filepath}"
        )
        print()

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
