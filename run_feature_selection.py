"""
Run Feature Selection Pipeline
===============================
Execute feature selection to reduce from 143 to optimal feature set.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.feature_selector import FeatureSelector


def main():
    """Run feature selection pipeline."""

    print("=" * 80)
    print("🎯 FEATURE SELECTION - AI GOLD BOT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Paths
    data_file = project_root / "data" / "processed" / "XAUUSD_M15_features_complete.csv"
    importance_file = (
        project_root / "results" / "feature_analysis" / "feature_importance_ranking.csv"
    )
    output_dir = project_root / "results" / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return

    if not importance_file.exists():
        print(f"❌ Importance file not found: {importance_file}")
        return

    print(f"📂 Loading data: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"   Shape: {df.shape}")
    print(f"   Features: {len(df.columns)}\n")

    # Initialize selector
    selector = FeatureSelector(
        importance_threshold=0.80,  # 80% cumulative importance
        correlation_threshold=0.95,  # Remove if |r| >= 0.95
        variance_threshold=0.01,  # Remove if variance < 0.01
    )

    # Run selection
    selected_features, removed_features = selector.select_features(
        df=df,
        importance_file=str(importance_file),
        target_features=None,  # Use threshold instead
    )

    # Print categorized features
    selector.print_selected_features()

    # Save results
    output_file = output_dir / "selected_features.json"
    selector.save_selected_features(str(output_file))

    # Save reduced dataset
    print("\n💾 Saving Reduced Dataset")
    print("-" * 80)

    # Keep only selected features (plus original price data for reference)
    keep_cols = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
    ] + selected_features
    keep_cols = [col for col in keep_cols if col in df.columns]

    df_reduced = df[keep_cols].copy()

    output_data_file = output_dir / "XAUUSD_M15_selected_features.csv"
    df_reduced.to_csv(output_data_file, index=False)

    file_size = output_data_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Saved: {output_data_file.name}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Shape: {df_reduced.shape}")

    # Generate report
    report_file = output_dir / "feature_selection_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE SELECTION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Importance threshold: {selector.importance_threshold * 100}%\n")
        f.write(f"Correlation threshold: {selector.correlation_threshold}\n")
        f.write(f"Variance threshold: {selector.variance_threshold}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original features: {len(df.columns)}\n")
        f.write(f"Selected features: {len(selected_features)}\n")
        f.write(f"Removed features: {len(removed_features)}\n")
        f.write(f"Reduction: {len(df.columns) - len(selected_features)} features ")
        f.write(f"(-{(1 - len(selected_features) / len(df.columns)) * 100:.1f}%)\n\n")

        f.write("SELECTED FEATURES\n")
        f.write("-" * 80 + "\n")
        categories = selector.get_feature_categories(selected_features)
        for cat, feats in categories.items():
            f.write(f"\n{cat} ({len(feats)}):\n")
            for feat in sorted(feats):
                f.write(f"  + {feat}\n")

        f.write("\n\nREMOVED FEATURES\n")
        f.write("-" * 80 + "\n")
        for feat, reason in sorted(removed_features.items()):
            f.write(f"  - {feat:40s} | {reason}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Saved report: {report_file.name}\n")

    # Summary
    print("=" * 80)
    print("✨ FEATURE SELECTION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files in: results/feature_selection/")
    print(f"  • selected_features.json           (Feature list + metadata)")
    print(f"  • XAUUSD_M15_selected_features.csv (Reduced dataset)")
    print(f"  • feature_selection_report.txt     (Detailed report)")
    print(f"\n📊 Next step: Data preprocessing & model training!")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
