"""
Feature Selector Module
=======================
Intelligent feature selection based on:
1. Feature importance (XGBoost ranking)
2. Correlation analysis (remove duplicates)
3. Constant/low-variance removal
4. Domain knowledge priorities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json


class FeatureSelector:
    """
    Select optimal feature set for model training.

    Strategies:
    - Remove perfect duplicates (|r| = 1.0)
    - Remove highly correlated features (keep higher importance)
    - Keep top N features by importance
    - Remove constant/near-constant features
    """

    def __init__(
        self,
        importance_threshold: float = 0.80,  # Keep features for 80% cumulative importance
        correlation_threshold: float = 0.95,  # Remove if |r| >= 0.95
        variance_threshold: float = 0.01,  # Remove if variance < threshold
    ):
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold

        self.selected_features: List[str] = []
        self.removed_features: Dict[str, str] = {}  # feature: reason
        self.feature_importance: Optional[pd.DataFrame] = None

    def load_importance_ranking(self, importance_file: str) -> pd.DataFrame:
        """Load feature importance from CSV."""
        df = pd.read_csv(importance_file)
        self.feature_importance = df
        print(f"✅ Loaded importance ranking: {len(df)} features")
        return df

    def remove_duplicates(self) -> List[str]:
        """
        Remove perfect duplicate features based on known correlations.

        Returns:
            List of features to remove
        """
        print("\n🔍 Step 1: Removing Perfect Duplicates (r = 1.000)")
        print("-" * 70)

        # Known perfect duplicates from correlation analysis
        duplicates_to_remove = [
            # Time features (keep the simpler one)
            "zscore_20",  # Duplicate of BB_pct_b
            "time_of_day_normalized",  # Duplicate of minutes_since_midnight
            "days_since_month_start",  # Duplicate of day_of_month
            "is_low_liquidity",  # Inverse of is_liquid_hours
            # Moving averages (Bollinger/Keltner middle bands ARE the MAs)
            "BB_middle",  # IS SMA_20
            "KELT_middle",  # IS EMA_20
            # Support/Resistance (duplicates of Donchian)
            "resistance",  # Same as DONCH_upper
            "support",  # Same as DONCH_lower
            # Day of week duplicates
            "is_first_day_of_week",  # Same as is_monday
            "is_last_day_of_week",  # Same as is_friday
            # Session overlaps
            "is_peak_hours",  # Same as overlap_london_newyork
            "is_first_hour_london",  # Same as overlap_tokyo_london
            # Time encodings (keep sin, remove some cos for redundancy)
            "month_cos",  # Month already captured by month + month_sin
            # Returns (log_returns ~= returns for small changes)
            "log_returns",  # Keep returns (simpler)
        ]

        for feat in duplicates_to_remove:
            self.removed_features[feat] = "Perfect duplicate (r=1.000)"
            print(f"   ❌ Remove: {feat:40s} (perfect duplicate)")

        print(f"\n✅ Removed {len(duplicates_to_remove)} perfect duplicates")
        return duplicates_to_remove

    def remove_redundant_moving_averages(self) -> List[str]:
        """
        Remove redundant moving averages.
        Keep EMA series + selected SMA/WMA.
        """
        print("\n🔍 Step 2: Removing Redundant Moving Averages")
        print("-" * 70)

        # Keep: EMA_5, 10, 20, 50, 100, 200 (full spectrum)
        # Keep: SMA_20, 50, 100 (common references)
        # Keep: WMA_20 (ranked #2 in importance!)
        # Remove: SMA_5, SMA_10, WMA_10, WMA_50 (redundant with EMA)

        redundant_mas = [
            "SMA_5",  # Too similar to EMA_5
            "SMA_10",  # Too similar to EMA_10
            "WMA_10",  # Redundant, EMA_10 sufficient
            "WMA_50",  # Redundant, EMA_50 + SMA_50 sufficient
        ]

        for feat in redundant_mas:
            self.removed_features[feat] = "Redundant moving average"
            print(f"   ❌ Remove: {feat:40s} (redundant MA)")

        print(f"\n✅ Removed {len(redundant_mas)} redundant moving averages")
        return redundant_mas

    def remove_constant_features(self, df: pd.DataFrame) -> List[str]:
        """Remove features with very low variance (near-constant)."""
        print("\n🔍 Step 3: Removing Constant/Low-Variance Features")
        print("-" * 70)

        constant_features = []

        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                variance = df[col].var()
                n_unique = df[col].nunique()

                # Remove if variance too low or only 1 unique value
                if variance < self.variance_threshold or n_unique <= 1:
                    constant_features.append(col)
                    self.removed_features[col] = (
                        f"Low variance ({variance:.6f}) or constant"
                    )
                    print(
                        f"   ❌ Remove: {col:40s} (var={variance:.6f}, unique={n_unique})"
                    )

        print(f"\n✅ Removed {len(constant_features)} constant features")
        return constant_features

    def select_by_importance(
        self, importance_df: pd.DataFrame, threshold: float = 0.80
    ) -> List[str]:
        """
        Select features by cumulative importance threshold.

        Args:
            importance_df: DataFrame with columns ['feature', 'importance', 'cumulative_pct']
            threshold: Cumulative importance threshold (0.80 = 80%)

        Returns:
            List of selected features
        """
        print(f"\n🔍 Step 4: Selecting by Importance (>= {threshold * 100}%)")
        print("-" * 70)

        # Get features up to threshold
        mask = importance_df["cumulative_pct"] <= (threshold * 100)
        top_features = importance_df[mask]["feature"].tolist()

        # Also keep a few more critical features even if just past threshold
        buffer_features = importance_df[
            (importance_df["cumulative_pct"] > threshold * 100)
            & (importance_df["cumulative_pct"] <= (threshold * 100 + 5))  # +5% buffer
        ]["feature"].tolist()[:5]  # Max 5 extra

        top_features.extend(buffer_features)

        print(
            f"   ✅ Selected {len(top_features)} features for {threshold * 100}% importance"
        )

        return top_features

    def remove_high_correlation_pairs(
        self, df: pd.DataFrame, features: List[str], importance_df: pd.DataFrame
    ) -> List[str]:
        """
        Remove one feature from highly correlated pairs.
        Keep the one with higher importance.
        """
        print(
            f"\n🔍 Step 5: Removing High Correlation (|r| >= {self.correlation_threshold})"
        )
        print("-" * 70)

        # Calculate correlation matrix for selected features
        corr_matrix = df[features].corr().abs()

        # Create importance lookup
        importance_dict = dict(
            zip(importance_df["feature"], importance_df["importance"])
        )

        to_remove = set()
        checked_pairs = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]

                if feat1 in to_remove or feat2 in to_remove:
                    continue

                pair_key = tuple(sorted([feat1, feat2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                corr_val = corr_matrix.iloc[i, j]

                if corr_val >= self.correlation_threshold:
                    # Keep the more important one
                    imp1 = importance_dict.get(feat1, 0)
                    imp2 = importance_dict.get(feat2, 0)

                    if imp1 >= imp2:
                        to_remove.add(feat2)
                        self.removed_features[feat2] = (
                            f"High corr with {feat1} (r={corr_val:.3f})"
                        )
                        print(
                            f"   ❌ Remove: {feat2:35s} (r={corr_val:.3f} with {feat1})"
                        )
                    else:
                        to_remove.add(feat1)
                        self.removed_features[feat1] = (
                            f"High corr with {feat2} (r={corr_val:.3f})"
                        )
                        print(
                            f"   ❌ Remove: {feat1:35s} (r={corr_val:.3f} with {feat2})"
                        )

        print(f"\n✅ Removed {len(to_remove)} features due to high correlation")

        return list(to_remove)

    def select_features(
        self,
        df: pd.DataFrame,
        importance_file: str,
        target_features: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Main feature selection pipeline.

        Args:
            df: DataFrame with all features
            importance_file: Path to feature importance ranking CSV
            target_features: Target number of features (optional)

        Returns:
            (selected_features, removed_features_dict)
        """
        print("\n" + "=" * 80)
        print("🎯 FEATURE SELECTION PIPELINE")
        print("=" * 80)
        print(f"Starting with {len(df.columns)} features")

        # Load importance
        importance_df = self.load_importance_ranking(importance_file)

        # Step 1: Remove perfect duplicates
        dup_removed = self.remove_duplicates()

        # Step 2: Remove redundant MAs
        ma_removed = self.remove_redundant_moving_averages()

        # Step 3: Remove constant features
        const_removed = self.remove_constant_features(df)

        # Get remaining features
        all_removed = set(dup_removed + ma_removed + const_removed)
        remaining_features = [col for col in df.columns if col not in all_removed]

        # Filter importance to remaining features
        importance_filtered = importance_df[
            importance_df["feature"].isin(remaining_features)
        ].copy()

        # Recalculate cumulative percentage
        importance_filtered["cumulative_pct"] = (
            importance_filtered["importance"].cumsum()
            / importance_filtered["importance"].sum()
            * 100
        )

        # Step 4: Select by importance
        if target_features:
            # Select exact number
            selected = importance_filtered.head(target_features)["feature"].tolist()
            print(f"\n   🎯 Selected top {target_features} features by importance")
        else:
            # Select by threshold
            selected = self.select_by_importance(
                importance_filtered, self.importance_threshold
            )

        # Step 5: Remove high correlation within selected features
        if len(selected) > 0:
            high_corr_removed = self.remove_high_correlation_pairs(
                df, selected, importance_filtered
            )
            selected = [f for f in selected if f not in high_corr_removed]

        self.selected_features = selected

        # Summary
        print("\n" + "=" * 80)
        print("📊 FEATURE SELECTION SUMMARY")
        print("=" * 80)
        print(f"Original features:        {len(df.columns)}")
        print(f"Perfect duplicates:       -{len(dup_removed)}")
        print(f"Redundant MAs:            -{len(ma_removed)}")
        print(f"Constant features:        -{len(const_removed)}")
        print(
            f"High correlation:         -{len([k for k, v in self.removed_features.items() if 'High corr' in v])}"
        )
        print(f"{'=' * 40}")
        print(f"✅ Final selected:         {len(selected)} features")
        print(
            f"   Reduction:              {len(df.columns) - len(selected)} features (-{(1 - len(selected) / len(df.columns)) * 100:.1f}%)"
        )

        return selected, self.removed_features

    def get_feature_categories(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize selected features."""
        categories = {
            "Moving Averages": [],
            "Momentum": [],
            "Volatility": [],
            "Volume": [],
            "Time": [],
            "Price Action": [],
            "Statistical": [],
            "Candlestick": [],
            "Other": [],
        }

        for feat in features:
            if any(x in feat for x in ["SMA", "EMA", "WMA"]):
                categories["Moving Averages"].append(feat)
            elif any(
                x in feat
                for x in ["RSI", "MACD", "STOCH", "WILLR", "CCI", "MOM", "ROC"]
            ):
                categories["Momentum"].append(feat)
            elif any(
                x in feat
                for x in ["BB_", "ATR", "KELT_", "DONCH_", "volatility", "range"]
            ):
                categories["Volatility"].append(feat)
            elif any(x in feat for x in ["VOL_", "OBV", "MFI", "VWAP"]):
                categories["Volume"].append(feat)
            elif any(
                x in feat
                for x in ["hour", "day", "week", "month", "session", "is_", "overlap"]
            ):
                categories["Time"].append(feat)
            elif any(x in feat for x in ["body", "wick", "HL_", "OC_"]):
                categories["Price Action"].append(feat)
            elif any(x in feat for x in ["skew", "kurt", "zscore"]):
                categories["Statistical"].append(feat)
            elif "PATTERN" in feat:
                categories["Candlestick"].append(feat)
            else:
                categories["Other"].append(feat)

        return {k: v for k, v in categories.items() if v}  # Remove empty categories

    def save_selected_features(self, output_file: str):
        """Save selected features to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        data = {
            "selected_features": self.selected_features,
            "removed_features": self.removed_features,
            "n_selected": len(self.selected_features),
            "n_removed": len(self.removed_features),
            "parameters": {
                "importance_threshold": self.importance_threshold,
                "correlation_threshold": self.correlation_threshold,
                "variance_threshold": self.variance_threshold,
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Saved feature selection to: {output_file}")

    def print_selected_features(self):
        """Print categorized selected features."""
        categories = self.get_feature_categories(self.selected_features)

        print("\n" + "=" * 80)
        print("📋 SELECTED FEATURES BY CATEGORY")
        print("=" * 80)

        for cat, feats in categories.items():
            print(f"\n{cat} ({len(feats)}):")
            for feat in sorted(feats):
                print(f"   ✓ {feat}")

        print("\n" + "=" * 80)
