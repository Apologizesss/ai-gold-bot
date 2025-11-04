"""
Comprehensive Training Script - Test All Approaches
===================================================
Trains models with all 4 approaches to find the best one:
1. Balanced Dataset (Undersample)
2. Focal Loss
3. Attention Architecture
4. Bidirectional LSTM

Runs sequentially and compares all results.
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import sys


class ComprehensiveTrainer:
    """Train and compare all approaches"""

    def __init__(self, epochs=100, batch_size=64, features=140):
        self.epochs = epochs
        self.batch_size = batch_size
        self.features = features
        self.results = []

        print("=" * 80)
        print("🚀 COMPREHENSIVE TRAINING - ALL APPROACHES")
        print("=" * 80)
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Features: {features}")
        print("=" * 80)
        print()

    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def train_approach(self, name, command, description):
        """Train with a specific approach"""
        print("\n" + "=" * 80)
        print(f"🎯 APPROACH: {name}")
        print("=" * 80)
        print(f"Description: {description}")
        print(f"Command: {command}")
        print("=" * 80)

        start_time = time.time()

        try:
            self.log(f"Starting {name}...")

            # Run training
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log(
                    f"✅ {name} completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)"
                )
                return True, elapsed
            else:
                self.log(f"❌ {name} failed with return code {result.returncode}")
                print(
                    "STDOUT:",
                    result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                )
                print(
                    "STDERR:",
                    result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
                )
                return False, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self.log(f"⏰ {name} timeout after {elapsed:.1f}s")
            return False, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"❌ {name} error: {str(e)}")
            return False, elapsed

    def load_results(self, result_file, approach_name):
        """Load results from JSON file"""
        try:
            if Path(result_file).exists():
                with open(result_file, "r") as f:
                    data = json.load(f)
                    data["approach"] = approach_name
                    return data
            else:
                self.log(f"⚠️  Results file not found: {result_file}")
                return None
        except Exception as e:
            self.log(f"❌ Error loading results: {str(e)}")
            return None

    def run_all(self):
        """Run all training approaches"""

        self.log("Starting comprehensive training...")
        print()

        # Approach 1: Balanced Dataset (Undersample)
        print("\n" + "#" * 80)
        print("# APPROACH 1: BALANCED DATASET (UNDERSAMPLE)")
        print("#" * 80)

        success, elapsed = self.train_approach(
            name="Balanced Undersample",
            command=f"python train_with_class_weight.py "
            f"--data-path data/processed/XAUUSD_M15_features_with_target_balanced_undersample.csv "
            f"--epochs {self.epochs} "
            f"--batch-size {self.batch_size} "
            f"--architecture stacked "
            f"--features {self.features} "
            f"--no-class-weight",
            description="Train on balanced dataset (50/50 split) using undersample",
        )

        if success:
            results = self.load_results(
                "results/lstm/lstm_results.json", "Balanced_Undersample"
            )
            if results:
                results["training_time"] = elapsed
                results["approach_details"] = "Balanced dataset with undersample"
                self.results.append(results)
                # Backup results
                backup_file = Path(
                    "results/comparison/approach1_balanced_undersample.json"
                )
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_file, "w") as f:
                    json.dump(results, f, indent=2)

        time.sleep(2)

        # Approach 2: Focal Loss
        print("\n" + "#" * 80)
        print("# APPROACH 2: FOCAL LOSS")
        print("#" * 80)

        success, elapsed = self.train_approach(
            name="Focal Loss",
            command=f"python train_with_focal_loss.py "
            f"--data-path data/processed/XAUUSD_M15_features_with_target.csv "
            f"--epochs {self.epochs} "
            f"--batch-size {self.batch_size} "
            f"--architecture stacked "
            f"--features {self.features} "
            f"--focal-alpha 0.25 "
            f"--focal-gamma 2.0",
            description="Train with Focal Loss to handle imbalance",
        )

        if success:
            results = self.load_results(
                "results/lstm/lstm_focal_results.json", "Focal_Loss"
            )
            if results:
                results["training_time"] = elapsed
                results["approach_details"] = "Focal Loss (alpha=0.25, gamma=2.0)"
                self.results.append(results)
                # Backup results
                backup_file = Path("results/comparison/approach2_focal_loss.json")
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_file, "w") as f:
                    json.dump(results, f, indent=2)

        time.sleep(2)

        # Approach 3: Attention Architecture
        print("\n" + "#" * 80)
        print("# APPROACH 3: ATTENTION ARCHITECTURE")
        print("#" * 80)

        success, elapsed = self.train_approach(
            name="Attention LSTM",
            command=f"python train_with_class_weight.py "
            f"--data-path data/processed/XAUUSD_M15_features_with_target.csv "
            f"--epochs {self.epochs} "
            f"--batch-size {self.batch_size} "
            f"--architecture attention "
            f"--features {self.features} "
            f"--no-class-weight",
            description="Train with Attention mechanism for better feature focus",
        )

        if success:
            results = self.load_results(
                "results/lstm/lstm_results.json", "Attention_LSTM"
            )
            if results:
                results["training_time"] = elapsed
                results["approach_details"] = "LSTM with Attention mechanism"
                self.results.append(results)
                # Backup results
                backup_file = Path("results/comparison/approach3_attention.json")
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_file, "w") as f:
                    json.dump(results, f, indent=2)

        time.sleep(2)

        # Approach 4: Bidirectional LSTM
        print("\n" + "#" * 80)
        print("# APPROACH 4: BIDIRECTIONAL LSTM")
        print("#" * 80)

        success, elapsed = self.train_approach(
            name="Bidirectional LSTM",
            command=f"python train_with_class_weight.py "
            f"--data-path data/processed/XAUUSD_M15_features_with_target.csv "
            f"--epochs {self.epochs} "
            f"--batch-size {self.batch_size} "
            f"--architecture bidirectional "
            f"--features {self.features} "
            f"--no-class-weight",
            description="Train with Bidirectional LSTM to learn from both directions",
        )

        if success:
            results = self.load_results(
                "results/lstm/lstm_results.json", "Bidirectional_LSTM"
            )
            if results:
                results["training_time"] = elapsed
                results["approach_details"] = "Bidirectional LSTM"
                self.results.append(results)
                # Backup results
                backup_file = Path("results/comparison/approach4_bidirectional.json")
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_file, "w") as f:
                    json.dump(results, f, indent=2)

        # Generate comparison report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive comparison report"""

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE RESULTS COMPARISON")
        print("=" * 80)

        if not self.results:
            print("\n❌ No results to compare!")
            return

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Sort by accuracy
        df = df.sort_values("test_accuracy", ascending=False)

        # Display results
        print("\n" + "=" * 80)
        print("🏆 RANKING BY ACCURACY")
        print("=" * 80)
        print()

        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            medal = ["🥇", "🥈", "🥉", "4️⃣"][rank - 1] if rank <= 4 else f"{rank}."

            print(f"{medal} {row['approach']}")
            print(
                f"   Accuracy:  {row['test_accuracy']:.4f} ({row['test_accuracy'] * 100:.2f}%)"
            )
            print(f"   Precision: {row['test_precision']:.4f}")
            print(f"   Recall:    {row['test_recall']:.4f}")
            print(f"   F1-Score:  {row['test_f1']:.4f}")
            print(
                f"   Time:      {row['training_time']:.1f}s ({row['training_time'] / 60:.1f} min)"
            )
            print(f"   Details:   {row.get('approach_details', 'N/A')}")
            print()

        # Statistical summary
        print("=" * 80)
        print("📈 STATISTICAL SUMMARY")
        print("=" * 80)
        print()
        print(
            f"Best Accuracy:    {df['test_accuracy'].max():.4f} ({df['test_accuracy'].max() * 100:.2f}%)"
        )
        print(
            f"Worst Accuracy:   {df['test_accuracy'].min():.4f} ({df['test_accuracy'].min() * 100:.2f}%)"
        )
        print(
            f"Mean Accuracy:    {df['test_accuracy'].mean():.4f} ({df['test_accuracy'].mean() * 100:.2f}%)"
        )
        print(f"Std Deviation:    {df['test_accuracy'].std():.4f}")
        print()
        print(f"Best F1-Score:    {df['test_f1'].max():.4f}")
        print(f"Best Precision:   {df['test_precision'].max():.4f}")
        print(f"Best Recall:      {df['test_recall'].max():.4f}")
        print()

        # Check if any approach reached target
        target = 0.60
        reached_target = df[df["test_accuracy"] >= target]

        if len(reached_target) > 0:
            print("=" * 80)
            print(
                f"✅ {len(reached_target)} APPROACH(ES) REACHED {target * 100:.0f}% ACCURACY!"
            )
            print("=" * 80)
            print()
            for idx, row in reached_target.iterrows():
                print(f"   ✅ {row['approach']}: {row['test_accuracy'] * 100:.2f}%")
        else:
            gap = target - df["test_accuracy"].max()
            print("=" * 80)
            print(f"📊 NONE REACHED {target * 100:.0f}% TARGET")
            print("=" * 80)
            print(f"   Best: {df['test_accuracy'].max() * 100:.2f}%")
            print(f"   Gap:  {gap * 100:.2f}%")
            print()

        # Recommendations
        print("=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        print()

        best = df.iloc[0]
        print(f"✅ Best approach: {best['approach']}")
        print(f"   Accuracy: {best['test_accuracy'] * 100:.2f}%")
        print()

        if best["test_accuracy"] >= 0.60:
            print("🎉 EXCELLENT! This approach shows promise!")
            print()
            print("Next steps:")
            print("  1. Train longer with this approach (200+ epochs)")
            print("  2. Fine-tune hyperparameters")
            print("  3. Ensemble with other good approaches")
            print("  4. Test on different timeframes")
        elif best["test_accuracy"] >= 0.55:
            print("👍 GOOD! This approach is working!")
            print()
            print("Next steps:")
            print("  1. Increase training epochs (150-200)")
            print("  2. Try different learning rates")
            print("  3. Combine best approaches (ensemble)")
            print("  4. Feature engineering")
        else:
            print("⚠️  NEEDS IMPROVEMENT")
            print()
            print("Next steps:")
            print("  1. Collect more data (2+ years)")
            print("  2. Better feature engineering")
            print("  3. Try different timeframes (M5, M30, H1)")
            print("  4. Consider other algorithms (XGBoost, RandomForest)")
            print("  5. Check data quality and preprocessing")

        # Balance check
        print()
        print("=" * 80)
        print("⚖️  PREDICTION BALANCE CHECK")
        print("=" * 80)
        print()

        for idx, row in df.iterrows():
            precision = row["test_precision"]
            recall = row["test_recall"]
            diff = abs(precision - recall)

            if diff < 0.15:
                status = "✅ Balanced"
            elif diff < 0.30:
                status = "⚠️  Slightly imbalanced"
            else:
                status = "❌ Imbalanced"

            print(
                f"{row['approach']:30s} | Prec: {precision:.3f} | Rec: {recall:.3f} | Diff: {diff:.3f} | {status}"
            )

        # Save comparison
        comparison_file = Path("results/comparison/all_approaches_comparison.csv")
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(comparison_file, index=False)

        print()
        print("=" * 80)
        print(f"📁 RESULTS SAVED")
        print("=" * 80)
        print(f"   Comparison CSV: {comparison_file}")
        print(f"   Individual results: results/comparison/approach*.json")
        print("=" * 80)
        print()

        # Final summary
        print("=" * 80)
        print("✅ COMPREHENSIVE TRAINING COMPLETE!")
        print("=" * 80)
        print()
        print(f"Trained {len(self.results)} approaches")
        print(f"Best accuracy: {df['test_accuracy'].max() * 100:.2f}%")
        print(f"Best approach: {df.iloc[0]['approach']}")
        print()
        print("Check results/comparison/ for detailed analysis")
        print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and compare all approaches")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--features", type=int, default=140, help="Number of features (default: 140)"
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("🚀 COMPREHENSIVE MODEL TRAINING")
    print("=" * 80)
    print()
    print("This will train 4 different approaches sequentially:")
    print("  1. Balanced Dataset (Undersample)")
    print("  2. Focal Loss")
    print("  3. Attention Architecture")
    print("  4. Bidirectional LSTM")
    print()
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Features: {args.features}")
    print()
    print("This will take approximately 40-60 minutes total.")
    print()

    try:
        response = input("Continue? (y/n): ")
        if response.lower() != "y":
            print("Cancelled by user")
            return
    except:
        print("Auto-continuing...")

    # Create trainer
    trainer = ComprehensiveTrainer(
        epochs=args.epochs, batch_size=args.batch_size, features=args.features
    )

    # Run all approaches
    try:
        trainer.run_all()
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("Partial results may be available in results/comparison/")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
