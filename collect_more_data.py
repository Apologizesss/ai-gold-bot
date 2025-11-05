"""
Improved Data Collection Script
================================
ดึงข้อมูลย้อนหลังมากขึ้น และหลาย timeframes

วิธีใช้:
    python collect_more_data.py
    python collect_more_data.py --days 180 --timeframes M5 M15 H1 H4
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys

# Import collector
from src.data_collection.mt5_collector import MT5Collector
from src.features.feature_pipeline import FeaturePipeline


def collect_data_for_timeframe(symbol, timeframe, days):
    """ดึงข้อมูลสำหรับ timeframe หนึ่ง"""
    print("\n" + "=" * 80)
    print(f"📥 กำลังดึงข้อมูล: {symbol} - {timeframe}")
    print("=" * 80)

    collector = MT5Collector(symbol=symbol, timeframe=timeframe)

    # เชื่อมต่อ MT5
    if not collector.initialize():
        print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
        return None

    if not collector.check_symbol():
        print(f"❌ ไม่สามารถเข้าถึง {symbol} ได้")
        mt5.shutdown()
        return None

    # ดึงข้อมูล
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(
        f"📊 ช่วงเวลา: {start_date.strftime('%Y-%m-%d')} ถึง {end_date.strftime('%Y-%m-%d')}"
    )

    df = collector.collect_historical_data(date_from=start_date, date_to=end_date)

    mt5.shutdown()

    if df is None or len(df) == 0:
        print("❌ ไม่สามารถดึงข้อมูลได้")
        return None

    print(f"✅ ดึงข้อมูลสำเร็จ: {len(df)} แท่งเทียน")

    # เปลี่ยนชื่อ column
    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif "time" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["time"]

    return df


def add_features_and_target(df):
    """เพิ่ม features และ target"""
    print("\n🔧 กำลังสร้าง features...")

    # เพิ่ม features
    pipeline = FeaturePipeline()
    df_features = pipeline.add_features(df)

    print(f"✅ สร้าง features สำเร็จ: {len(df_features.columns)} columns")

    # สร้าง target
    print("🎯 กำลังสร้าง target...")
    df_features["future_price"] = df_features["close"].shift(-4)
    df_features["target"] = (df_features["future_price"] > df_features["close"]).astype(
        int
    )

    # ลบแถวที่มี NaN ใน columns สำคัญ
    important_cols = ["open", "high", "low", "close", "target"]
    df_clean = df_features.dropna(subset=important_cols)

    print(f"✅ ข้อมูลหลังทำความสะอาด: {len(df_clean)} แถว")
    print(
        f"📊 Target distribution: UP={df_clean['target'].sum()}, DOWN={len(df_clean) - df_clean['target'].sum()}"
    )

    return df_clean


def save_data(df, symbol, timeframe):
    """บันทึกข้อมูล"""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{timeframe}_features_with_target_extended.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"💾 บันทึกไฟล์: {filepath}")

    return filepath


def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description="Collect more historical data")
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--days", type=int, default=180, help="จำนวนวันย้อนหลัง (default: 180)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["M5", "M15", "H1", "H4"],
        help="Timeframes ที่ต้องการ (default: M5 M15 H1 H4)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 IMPROVED DATA COLLECTION")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print("=" * 80)

    results = {}

    for timeframe in args.timeframes:
        try:
            # ดึงข้อมูล
            df = collect_data_for_timeframe(args.symbol, timeframe, args.days)

            if df is None:
                print(f"⚠️  ข้าม {timeframe} - ไม่สามารถดึงข้อมูลได้")
                continue

            # เพิ่ม features และ target
            df_processed = add_features_and_target(df)

            if len(df_processed) < 100:
                print(f"⚠️  ข้าม {timeframe} - ข้อมูลไม่เพียงพอ ({len(df_processed)} แถว)")
                continue

            # บันทึกข้อมูล
            filepath = save_data(df_processed, args.symbol, timeframe)

            results[timeframe] = {
                "rows": len(df_processed),
                "features": len(df_processed.columns),
                "file": str(filepath),
            }

            print(f"✅ {timeframe} เสร็จสมบูรณ์!")

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับ {timeframe}: {e}")
            continue

    # สรุปผล
    print("\n" + "=" * 80)
    print("📊 สรุปผลการดึงข้อมูล")
    print("=" * 80)

    if not results:
        print("❌ ไม่มีข้อมูลที่ดึงสำเร็จ")
        sys.exit(1)

    for timeframe, info in results.items():
        print(f"\n✅ {timeframe}:")
        print(f"   แถว: {info['rows']:,}")
        print(f"   Features: {info['features']}")
        print(f"   ไฟล์: {info['file']}")

    print("\n" + "=" * 80)
    print("✅ เสร็จสมบูรณ์!")
    print("=" * 80)
    print("\n💡 ขั้นตอนต่อไป:")
    print("   1. เทรน model ด้วยข้อมูลใหม่")
    print("      python train_simple.py")
    print("   2. หรือเลือก timeframe เฉพาะ:")
    for timeframe in results.keys():
        print(
            f"      python train_simple.py --data data/processed/{args.symbol}_{timeframe}_features_with_target_extended.csv"
        )
    print()


if __name__ == "__main__":
    main()
