"""
Clean Data Leakage Script
==========================
ลบ columns ที่มีข้อมูลอนาคตออก (Data Leakage)

วิธีใช้:
    python clean_data_leakage.py
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("🧹 Clean Data Leakage Script")
print("=" * 80)

# โหลดข้อมูล
input_file = (
    "data/processed/XAUUSD_M5_features_with_target_extended_target_volatility.csv"
)
print(f"\n📂 โหลดข้อมูล: {input_file}")

df = pd.read_csv(input_file)
print(f"✅ โหลดสำเร็จ: {len(df):,} แถว, {len(df.columns)} columns")

# Columns ที่ต้องลบ (มีข้อมูลอนาคต)
future_cols = [
    "future_price",
    "future_high",
    "future_low",
    "future_close",
    "future_return",
    "max_gain",
    "max_loss",
    "max_gain_pct",
    "max_loss_pct",
    "gain_pct",
    "loss_pct",
    "threshold",
    "score",
]

# หา columns ที่มีจริงในข้อมูล
cols_to_drop = [col for col in future_cols if col in df.columns]

print(f"\n🗑️  กำลังลบ {len(cols_to_drop)} columns ที่มีข้อมูลอนาคต:")
for col in cols_to_drop:
    print(f"   - {col}")

# ลบ columns
df_clean = df.drop(columns=cols_to_drop)

print(f"\n✅ ข้อมูลหลังทำความสะอาด:")
print(f"   แถว: {len(df_clean):,}")
print(f"   Columns: {len(df_clean.columns)}")
print(f"   Target UP: {df_clean['target'].sum()}")
print(f"   Target DOWN: {len(df_clean) - df_clean['target'].sum()}")

# บันทึก
output_file = "data/processed/XAUUSD_M5_clean.csv"
df_clean.to_csv(output_file, index=False)

print(f"\n💾 บันทึกไฟล์: {output_file}")

print("\n" + "=" * 80)
print("✅ เสร็จสมบูรณ์!")
print("=" * 80)
print("\n💡 ขั้นตอนต่อไป:")
print(f"   python train_xgboost.py --data-path {output_file}")
print()
