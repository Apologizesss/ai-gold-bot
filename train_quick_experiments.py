"""
Quick Batch Training Script - Test 6 Most Promising LSTM Configurations
Runs experiments to find the most accurate model in reasonable time (~30-60 minutes)
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import sys


class QuickExperimentRunner:
    """Run quick LSTM training experiments and compare results"""

    def __init__(self):
        self.results = []
        self.experiments_dir = Path("results/experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🚀 QUICK BATCH LSTM TRAINING - 6 BEST CONFIGURATIONS")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Expected time: 30-60 minutes")
        print("=" * 80)
        print()

    def run_experiment(self, name, description, command):
        """Run a single training experiment"""
        print("\n" + "=" * 80)
        print(f"🧪 EXPERIMENT: {name}")
        print("=" * 80)
        print(f"📝 {description}")
        print(f"💻 {command}")
        print("-" * 80)

        start_time = time.time()

        try:
            # Set environment for UTF-8 encoding
            import os

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Run training
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # Print output in real-time (last 3 lines only)
            last_lines = []
            for line in process.stdout:
                last_lines.append(line.strip())
                if len(last_lines) > 3:
                    last_lines.pop(0)
                # Print progress indicators
                if "Epoch" in line or "accuracy" in line or "loss" in line:
                    print(f"\r{line.strip()[:80]}", end="", flush=True)

            process.wait()
            print()  # New line after progress

            elapsed = time.time() - start_time

            # Load results
            try:
                with open("results/lstm/lstm_results.json", "r") as f:
                    metrics = json.load(f)

                # Store experiment results
                experiment_result = {
                    "name": name,
                    "description": description,
                    "command": command,
                    "train_accuracy": metrics["train_accuracy"],
                    "val_accuracy": metrics["val_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "test_f1": metrics["test_f1"],
                    "training_time": elapsed,
                    "status": "SUCCESS",
                }

                # Save individual experiment results
                exp_file = self.experiments_dir / f"{name.replace(' ', '_')}.json"
                with open(exp_file, "w") as f:
                    json.dump(experiment_result, f, indent=2)

                self.results.append(experiment_result)

                # Print results
                print("\n✅ COMPLETED!")
                print(f"⏱️  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
                print(
                    f"📊 Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
                )
                print(f"📈 Test F1: {metrics['test_f1']:.4f}")
                print(
                    f"🎯 Precision: {metrics['test_precision']:.4f} | Recall: {metrics['test_recall']:.4f}"
                )

                # Quick evaluation
                test_acc = metrics["test_accuracy"]
                if test_acc >= 0.60:
                    print("   ⭐ EXCELLENT! (≥60%)")
                elif test_acc >= 0.55:
                    print("   ✅ GOOD! (≥55%)")
                elif test_acc >= 0.52:
                    print("   ✓ TRADEABLE (≥52%)")
                else:
                    print("   ⚠️  Below target (<52%)")

                # Rename model files to preserve them
                import shutil

                model_src = Path("models/lstm_best_model.h5")
                model_dst = Path(f"models/{name.replace(' ', '_')}_best.h5")
                if model_src.exists():
                    shutil.copy(str(model_src), str(model_dst))
                    print(f"💾 Model saved: {model_dst.name}")

            except Exception as e:
                print(f"\n❌ ERROR loading results: {str(e)}")
                experiment_result = {
                    "name": name,
                    "description": description,
                    "command": command,
                    "status": "FAILED",
                    "error": str(e),
                    "training_time": elapsed,
                }
                self.results.append(experiment_result)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
            experiment_result = {
                "name": name,
                "description": description,
                "command": command,
                "status": "FAILED",
                "error": str(e),
                "training_time": elapsed,
            }
            self.results.append(experiment_result)

        print("=" * 80)
        return experiment_result

    def run_quick_experiments(self):
        """Run 6 most promising experiments"""

        experiments = [
            {
                "name": "EXP1_More_Features",
                "description": "Use 100 features (vs 70 baseline)",
                "command": "./venv/Scripts/python.exe train_models.py --features 100 --epochs 100",
            },
            {
                "name": "EXP2_Long_Sequence",
                "description": "Longer lookback window (120 timesteps vs 60)",
                "command": "./venv/Scripts/python.exe train_models.py --sequence-length 120 --epochs 100",
            },
            {
                "name": "EXP3_Stacked_LSTM",
                "description": "Deeper model with stacked LSTM layers",
                "command": "./venv/Scripts/python.exe train_models.py --mode stacked --epochs 100",
            },
            {
                "name": "EXP4_M5_Timeframe",
                "description": "5-minute candles (more data, 17K+ samples)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe M5 --sequence-length 120 --epochs 100",
            },
            {
                "name": "EXP5_H1_Timeframe",
                "description": "1-hour candles (less noise, cleaner signals)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe H1 --sequence-length 60 --epochs 100",
            },
            {
                "name": "EXP6_Best_Combo",
                "description": "Best combination: 100 features + seq=120 + stacked",
                "command": "./venv/Scripts/python.exe train_models.py --mode stacked --features 100 --sequence-length 120 --epochs 150",
            },
        ]

        print(f"\n📋 EXPERIMENTS TO RUN: {len(experiments)}")
        print("=" * 80)
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']}")
            print(f"   {exp['description']}")
        print("=" * 80)
        print("\n⏱️  Estimated total time: 30-60 minutes")
        print("💡 Each experiment runs independently and saves its own model")
        print()

        # Run all experiments
        for i, exp in enumerate(experiments, 1):
            print(f"\n\n{'#' * 80}")
            print(
                f"# PROGRESS: {i}/{len(experiments)} ({i / len(experiments) * 100:.1f}%)"
            )
            print(f"{'#' * 80}\n")

            self.run_experiment(
                name=exp["name"], description=exp["description"], command=exp["command"]
            )

            # Show running stats
            completed = sum(1 for r in self.results if r["status"] == "SUCCESS")
            failed = sum(1 for r in self.results if r["status"] == "FAILED")

            if self.results:
                total_time = sum(r.get("training_time", 0) for r in self.results)
                avg_time = total_time / len(self.results)
                remaining = (len(experiments) - i) * avg_time

                print(f"\n📊 Stats: {completed} ✅ | {failed} ❌")
                print(
                    f"⏱️  Elapsed: {total_time / 60:.1f} min | Remaining: ~{remaining / 60:.1f} min"
                )

                # Show current best
                if completed > 0:
                    successful = [r for r in self.results if r["status"] == "SUCCESS"]
                    best = max(successful, key=lambda x: x["test_accuracy"])
                    print(
                        f"🏆 Current Best: {best['name']} ({best['test_accuracy'] * 100:.2f}%)"
                    )

        self.generate_report()

    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n\n" + "=" * 80)
        print("📊 GENERATING FINAL REPORT")
        print("=" * 80)

        # Filter successful experiments
        successful = [r for r in self.results if r["status"] == "SUCCESS"]

        if not successful:
            print("\n❌ No successful experiments to report!")
            return

        # Create DataFrame
        df = pd.DataFrame(successful)
        df = df.sort_values("test_accuracy", ascending=False)

        # Save detailed results
        csv_path = self.experiments_dir / "quick_experiments_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved: {csv_path}")

        # Print ranking
        print("\n" + "=" * 80)
        print("🏆 FINAL RANKING - ALL EXPERIMENTS")
        print("=" * 80)

        for i, (idx, row) in enumerate(df.iterrows(), 1):
            print(f"\n{i}. {row['name']}")
            print(
                f"   📊 Accuracy: {row['test_accuracy']:.4f} ({row['test_accuracy'] * 100:.2f}%)"
            )
            print(f"   📈 F1-Score: {row['test_f1']:.4f}")
            print(
                f"   🎯 Precision: {row['test_precision']:.4f} | Recall: {row['test_recall']:.4f}"
            )
            print(
                f"   ⏱️  Time: {row['training_time']:.1f}s ({row['training_time'] / 60:.1f} min)"
            )
            print(f"   💡 {row['description']}")

            # Evaluation
            test_acc = row["test_accuracy"]
            if test_acc >= 0.60:
                status = "⭐ EXCELLENT"
            elif test_acc >= 0.55:
                status = "✅ GOOD"
            elif test_acc >= 0.52:
                status = "✓ TRADEABLE"
            else:
                status = "⚠️  BELOW TARGET"
            print(f"   {status}")

        # Best model details
        best = df.iloc[0]
        print("\n" + "=" * 80)
        print("🥇 BEST MODEL")
        print("=" * 80)
        print(f"Name: {best['name']}")
        print(
            f"Accuracy: {best['test_accuracy']:.4f} ({best['test_accuracy'] * 100:.2f}%)"
        )
        print(f"F1-Score: {best['test_f1']:.4f}")
        print(f"Precision: {best['test_precision']:.4f}")
        print(f"Recall: {best['test_recall']:.4f}")
        print(
            f"Training Time: {best['training_time']:.1f}s ({best['training_time'] / 60:.1f} min)"
        )
        print(f"\nCommand to reproduce:")
        print(f"  {best['command']}")
        print(f"\nModel file: models/{best['name'].replace(' ', '_')}_best.h5")

        # Comparison with baseline
        print("\n" + "=" * 80)
        print("📈 IMPROVEMENT OVER BASELINE")
        print("=" * 80)
        baseline_acc = 0.5055  # From first training
        print(f"Baseline (original): {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")
        print(
            f"Best model: {best['test_accuracy']:.4f} ({best['test_accuracy'] * 100:.2f}%)"
        )
        improvement = (best["test_accuracy"] - baseline_acc) * 100
        print(f"Improvement: {improvement:+.2f}%")

        if improvement > 5:
            print("🚀 SIGNIFICANT IMPROVEMENT!")
        elif improvement > 2:
            print("✅ Good improvement")
        elif improvement > 0:
            print("✓ Slight improvement")
        else:
            print("⚠️  No improvement over baseline")

        # Statistics
        print("\n" + "=" * 80)
        print("📊 STATISTICS")
        print("=" * 80)
        print(f"Total Experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")
        print(
            f"Total Time: {df['training_time'].sum() / 60:.1f} min ({df['training_time'].sum() / 3600:.2f} hours)"
        )
        print(
            f"\nAccuracy Range: {df['test_accuracy'].min():.4f} - {df['test_accuracy'].max():.4f}"
        )
        print(f"Average Accuracy: {df['test_accuracy'].mean():.4f}")

        # Target achievement
        above_52 = len(df[df["test_accuracy"] >= 0.52])
        above_55 = len(df[df["test_accuracy"] >= 0.55])
        above_60 = len(df[df["test_accuracy"] >= 0.60])

        print(f"\n🎯 Target Achievement:")
        print(
            f"  ≥52% (Tradeable): {above_52}/{len(successful)} ({above_52 / len(successful) * 100:.0f}%)"
        )
        print(
            f"  ≥55% (Good): {above_55}/{len(successful)} ({above_55 / len(successful) * 100:.0f}%)"
        )
        print(
            f"  ≥60% (Excellent): {above_60}/{len(successful)} ({above_60 / len(successful) * 100:.0f}%)"
        )

        # Recommendations
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        print(f"\n1. USE THIS MODEL FOR PRODUCTION:")
        print(f"   {best['name']}")
        print(f"   File: models/{best['name'].replace(' ', '_')}_best.h5")

        if len(df) >= 3:
            print(f"\n2. ENSEMBLE RECOMMENDATION:")
            print(f"   Combine top 3 models for even better results:")
            weights = [0.5, 0.3, 0.2]
            for i in range(min(3, len(df))):
                model = df.iloc[i]
                print(
                    f"   • {model['name']} (weight: {weights[i]}) - {model['test_accuracy'] * 100:.2f}%"
                )

            avg_top3 = df.head(3)["test_accuracy"].mean()
            print(f"   Expected ensemble accuracy: ~{avg_top3 * 100:.2f}%")

        print(f"\n3. NEXT STEPS:")
        print(
            f"   • Compare with XGBoost: ./venv/Scripts/python.exe src/models/compare_models.py"
        )
        print(
            f"   • Run backtesting: ./venv/Scripts/python.exe src/backtesting/run_backtest.py"
        )
        print(f"   • Create ensemble: Combine LSTM + XGBoost")

        # Save summary report
        report_path = self.experiments_dir / "quick_summary_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("QUICK LSTM EXPERIMENTS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("RANKING:\n")
            f.write("-" * 80 + "\n")
            for i, (idx, row) in enumerate(df.iterrows(), 1):
                f.write(f"\n{i}. {row['name']}\n")
                f.write(f"   Accuracy: {row['test_accuracy'] * 100:.2f}%\n")
                f.write(f"   F1-Score: {row['test_f1']:.4f}\n")
                f.write(f"   Description: {row['description']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"BEST MODEL: {best['name']}\n")
            f.write(f"Accuracy: {best['test_accuracy'] * 100:.2f}%\n")
            f.write(f"Command: {best['command']}\n")
            f.write(f"Improvement over baseline: {improvement:+.2f}%\n")

        print(f"\n✅ Saved summary: {report_path}")

        print("\n" + "=" * 80)
        print("✅ ALL EXPERIMENTS COMPLETED!")
        print("=" * 80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {df['training_time'].sum() / 60:.1f} minutes")
        print("=" * 80)


def main():
    """Main execution"""
    runner = QuickExperimentRunner()
    runner.run_quick_experiments()


if __name__ == "__main__":
    main()
