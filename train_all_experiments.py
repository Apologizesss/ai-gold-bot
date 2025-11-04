"""
Batch Training Script - Test Multiple LSTM Configurations
Runs all experiments to find the most accurate model
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time


class ExperimentRunner:
    """Run multiple LSTM training experiments and compare results"""

    def __init__(self):
        self.results = []
        self.experiments_dir = Path("results/experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🚀 BATCH LSTM TRAINING - FINDING BEST CONFIGURATION")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

    def run_experiment(self, name, description, command):
        """Run a single training experiment"""
        print("\n" + "=" * 80)
        print(f"🧪 EXPERIMENT: {name}")
        print("=" * 80)
        print(f"📝 Description: {description}")
        print(f"💻 Command: {command}")
        print("-" * 80)

        start_time = time.time()

        try:
            # Run training
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env={"PYTHONIOENCODING": "utf-8"},
            )

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
                print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
                print(
                    f"⏱️  Training Time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)"
                )
                print(
                    f"📊 Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
                )
                print(f"📈 Test F1-Score: {metrics['test_f1']:.4f}")

                # Rename model files to preserve them
                model_src = Path("models/lstm_best_model.h5")
                model_dst = Path(f"models/lstm_{name.replace(' ', '_')}_best.h5")
                if model_src.exists():
                    import shutil

                    shutil.copy(str(model_src), str(model_dst))
                    print(f"💾 Saved model: {model_dst}")

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

    def run_all_experiments(self):
        """Run all planned experiments"""

        experiments = [
            {
                "name": "Baseline_70features",
                "description": "Original configuration (70 features, seq=60)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 60 --epochs 100",
            },
            {
                "name": "More_Features_100",
                "description": "Use 100 features instead of 70",
                "command": "./venv/Scripts/python.exe train_models.py --features 100 --sequence-length 60 --epochs 100",
            },
            {
                "name": "All_Features",
                "description": "Use all available features",
                "command": "./venv/Scripts/python.exe train_models.py --features 0 --sequence-length 60 --epochs 100",
            },
            {
                "name": "Long_Sequence_120",
                "description": "Longer lookback window (120 timesteps)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 120 --epochs 100",
            },
            {
                "name": "Long_Sequence_180",
                "description": "Very long lookback window (180 timesteps)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 180 --epochs 100",
            },
            {
                "name": "Stacked_LSTM",
                "description": "Deeper model with stacked LSTM layers",
                "command": "./venv/Scripts/python.exe train_models.py --mode stacked --features 70 --epochs 100",
            },
            {
                "name": "Attention_LSTM",
                "description": "LSTM with attention mechanism",
                "command": "./venv/Scripts/python.exe train_models.py --mode attention --features 70 --epochs 100",
            },
            {
                "name": "Simple_LSTM",
                "description": "Simplest LSTM architecture",
                "command": "./venv/Scripts/python.exe train_models.py --mode simple --features 70 --epochs 100",
            },
            {
                "name": "M5_Timeframe",
                "description": "Train on 5-minute candles (more data)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe M5 --sequence-length 120 --epochs 100",
            },
            {
                "name": "H1_Timeframe",
                "description": "Train on 1-hour candles (less noise)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe H1 --sequence-length 60 --epochs 100",
            },
            {
                "name": "H4_Timeframe",
                "description": "Train on 4-hour candles (long-term)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe H4 --sequence-length 30 --epochs 100",
            },
            {
                "name": "Extended_Training_200",
                "description": "Train for 200 epochs with patience",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --epochs 200",
            },
            {
                "name": "Large_Batch_128",
                "description": "Larger batch size for faster convergence",
                "command": "./venv/Scripts/python.exe train_models.py --batch-size 128 --epochs 100",
            },
            {
                "name": "Small_Batch_32",
                "description": "Smaller batch size for more stable training",
                "command": "./venv/Scripts/python.exe train_models.py --batch-size 32 --epochs 100",
            },
            {
                "name": "Lower_LR_0.0005",
                "description": "Lower learning rate for careful training",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.0005 --epochs 150",
            },
            {
                "name": "Higher_LR_0.002",
                "description": "Higher learning rate for faster learning",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.002 --epochs 100",
            },
        ]

        print(f"\n📋 PLANNED EXPERIMENTS: {len(experiments)}")
        print("-" * 80)
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']}: {exp['description']}")
        print("-" * 80)
        print()

        input("Press ENTER to start all experiments (this will take 2-4 hours)...")

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

            # Show progress
            completed = sum(1 for r in self.results if r["status"] == "SUCCESS")
            failed = sum(1 for r in self.results if r["status"] == "FAILED")
            print(f"\n📊 Current Stats: {completed} succeeded, {failed} failed")

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
        csv_path = self.experiments_dir / "all_experiments_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved detailed results: {csv_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("🏆 TOP 5 BEST MODELS BY TEST ACCURACY")
        print("=" * 80)

        top5 = df.head(5)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n{i}. {row['name']}")
            print(
                f"   Test Accuracy:  {row['test_accuracy']:.4f} ({row['test_accuracy'] * 100:.2f}%)"
            )
            print(f"   Test F1-Score:  {row['test_f1']:.4f}")
            print(f"   Test Precision: {row['test_precision']:.4f}")
            print(f"   Test Recall:    {row['test_recall']:.4f}")
            print(
                f"   Training Time:  {row['training_time']:.1f}s ({row['training_time'] / 60:.1f} min)"
            )
            print(f"   Description:    {row['description']}")

        # Best by different metrics
        print("\n" + "=" * 80)
        print("🎯 BEST BY DIFFERENT METRICS")
        print("=" * 80)

        best_acc = df.loc[df["test_accuracy"].idxmax()]
        print(
            f"\n✅ Best Accuracy: {best_acc['name']} ({best_acc['test_accuracy'] * 100:.2f}%)"
        )

        best_f1 = df.loc[df["test_f1"].idxmax()]
        print(f"📈 Best F1-Score: {best_f1['name']} ({best_f1['test_f1']:.4f})")

        best_balanced = df.loc[
            (df["test_precision"] - df["test_recall"]).abs().idxmin()
        ]
        print(
            f"⚖️  Most Balanced: {best_balanced['name']} (Prec={best_balanced['test_precision']:.4f}, Rec={best_balanced['test_recall']:.4f})"
        )

        fastest = df.loc[df["training_time"].idxmin()]
        print(f"⚡ Fastest: {fastest['name']} ({fastest['training_time']:.1f}s)")

        # Statistics
        print("\n" + "=" * 80)
        print("📈 OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Experiments:    {len(self.results)}")
        print(f"Successful:           {len(successful)}")
        print(f"Failed:               {len(self.results) - len(successful)}")
        print(
            f"Total Training Time:  {df['training_time'].sum():.1f}s ({df['training_time'].sum() / 3600:.2f} hours)"
        )
        print(
            f"\nAccuracy Range:       {df['test_accuracy'].min():.4f} - {df['test_accuracy'].max():.4f}"
        )
        print(f"Average Accuracy:     {df['test_accuracy'].mean():.4f}")
        print(f"Std Dev Accuracy:     {df['test_accuracy'].std():.4f}")

        # Target achievement
        print("\n" + "=" * 80)
        print("🎯 TARGET ACHIEVEMENT")
        print("=" * 80)
        above_52 = len(df[df["test_accuracy"] >= 0.52])
        above_55 = len(df[df["test_accuracy"] >= 0.55])
        above_60 = len(df[df["test_accuracy"] >= 0.60])

        print(
            f"Above 52% (Tradeable):  {above_52}/{len(successful)} ({above_52 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 55% (Good):       {above_55}/{len(successful)} ({above_55 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 60% (Excellent):  {above_60}/{len(successful)} ({above_60 / len(successful) * 100:.1f}%)"
        )

        # Recommendations
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)

        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['name']}")
        print(f"   Command: {best_model['command']}")
        print(f"   Accuracy: {best_model['test_accuracy'] * 100:.2f}%")
        print(f"   Use this model for: Production trading")

        if len(df) >= 3:
            print(f"\n🎯 ENSEMBLE RECOMMENDATION:")
            print(f"   Combine top 3 models for best results:")
            for i in range(min(3, len(df))):
                model = df.iloc[i]
                weight = [0.5, 0.3, 0.2][i]
                print(f"   - {model['name']} (weight: {weight})")

        # Save summary report
        report_path = self.experiments_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LSTM EXPERIMENTS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("TOP 5 MODELS:\n")
            f.write("-" * 80 + "\n")
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                f.write(f"\n{i}. {row['name']}\n")
                f.write(f"   Accuracy: {row['test_accuracy'] * 100:.2f}%\n")
                f.write(f"   F1-Score: {row['test_f1']:.4f}\n")
                f.write(f"   Command: {row['command']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Best Model: {best_model['name']}\n")
            f.write(f"Accuracy: {best_model['test_accuracy'] * 100:.2f}%\n")
            f.write(f"Command: {best_model['command']}\n")

        print(f"\n✅ Saved summary report: {report_path}")

        print("\n" + "=" * 80)
        print("✅ ALL EXPERIMENTS COMPLETED!")
        print("=" * 80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main execution"""
    runner = ExperimentRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
