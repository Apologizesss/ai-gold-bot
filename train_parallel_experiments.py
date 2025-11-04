"""
Parallel Training Script - Run Multiple LSTM Experiments Simultaneously
Uses multiprocessing to train multiple models at once for faster results
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import os


class ParallelExperimentRunner:
    """Run multiple LSTM training experiments in parallel"""

    def __init__(self, max_workers=None):
        """
        Initialize parallel experiment runner

        Args:
            max_workers: Maximum number of parallel processes (default: CPU count - 1)
        """
        self.results = []
        self.experiments_dir = Path("results/experiments_parallel")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect optimal worker count
        cpu_count = mp.cpu_count()
        if max_workers is None:
            # Leave 1-2 cores free for system
            self.max_workers = max(1, cpu_count - 1)
        else:
            self.max_workers = min(max_workers, cpu_count)

        print("=" * 80)
        print("🚀 PARALLEL LSTM TRAINING - FINDING BEST CONFIGURATION")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CPU Cores Available: {cpu_count}")
        print(f"Parallel Workers: {self.max_workers}")
        print(f"Expected Speedup: {self.max_workers}x faster than sequential")
        print("=" * 80)
        print()

    def run_single_experiment(self, experiment):
        """Run a single training experiment (called in parallel)"""
        name = experiment["name"]
        description = experiment["description"]
        command = experiment["command"]

        print(f"\n🧪 [{name}] Starting experiment...")
        print(f"📝 [{name}] {description}")

        start_time = time.time()

        try:
            # Create unique output directory for this experiment
            exp_output_dir = self.experiments_dir / name.replace(" ", "_")
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Modify command to save results to unique location
            results_file = exp_output_dir / "lstm_results.json"
            model_file = exp_output_dir / "lstm_model.h5"

            # Add output path arguments to command
            modified_command = f"{command} --output-dir {exp_output_dir}"

            # Run training
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "TF_CPP_MIN_LOG_LEVEL": "2",  # Reduce TensorFlow logging
                },
            )

            elapsed = time.time() - start_time

            # Load results from standard location (since command doesn't support custom output)
            try:
                # Copy results to experiment directory
                default_results = Path("results/lstm/lstm_results.json")
                if default_results.exists():
                    shutil.copy(default_results, results_file)

                # Copy model to experiment directory
                default_model = Path("models/lstm_best_model.h5")
                if default_model.exists():
                    shutil.copy(default_model, model_file)

                # Load metrics
                with open(results_file, "r") as f:
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
                    "timestamp": datetime.now().isoformat(),
                }

                # Save individual experiment results
                exp_file = exp_output_dir / "experiment_results.json"
                with open(exp_file, "w") as f:
                    json.dump(experiment_result, f, indent=2)

                # Print results
                print(f"\n✅ [{name}] COMPLETED!")
                print(f"⏱️  [{name}] Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
                print(
                    f"📊 [{name}] Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
                )
                print(f"📈 [{name}] F1-Score: {metrics['test_f1']:.4f}")

                return experiment_result

            except Exception as e:
                print(f"\n⚠️  [{name}] WARNING: Could not load results - {str(e)}")
                experiment_result = {
                    "name": name,
                    "description": description,
                    "command": command,
                    "status": "COMPLETED_NO_RESULTS",
                    "error": str(e),
                    "training_time": elapsed,
                    "timestamp": datetime.now().isoformat(),
                }
                return experiment_result

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ [{name}] FAILED: {str(e)}")
            experiment_result = {
                "name": name,
                "description": description,
                "command": command,
                "status": "FAILED",
                "error": str(e),
                "training_time": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
            return experiment_result

    def run_all_experiments(self):
        """Run all planned experiments in parallel"""

        experiments = [
            # Quick baseline tests (fast)
            {
                "name": "Quick_Baseline",
                "description": "Quick baseline test (50 epochs)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 60 --epochs 50",
            },
            {
                "name": "Quick_Stacked",
                "description": "Quick stacked LSTM test",
                "command": "./venv/Scripts/python.exe train_models.py --mode stacked --features 70 --epochs 50",
            },
            {
                "name": "Quick_Attention",
                "description": "Quick attention LSTM test",
                "command": "./venv/Scripts/python.exe train_models.py --mode attention --features 70 --epochs 50",
            },
            {
                "name": "Quick_Simple",
                "description": "Quick simple LSTM test",
                "command": "./venv/Scripts/python.exe train_models.py --mode simple --features 70 --epochs 50",
            },
            # Feature variations
            {
                "name": "Features_50",
                "description": "Use top 50 features",
                "command": "./venv/Scripts/python.exe train_models.py --features 50 --sequence-length 60 --epochs 100",
            },
            {
                "name": "Features_100",
                "description": "Use top 100 features",
                "command": "./venv/Scripts/python.exe train_models.py --features 100 --sequence-length 60 --epochs 100",
            },
            {
                "name": "Features_All",
                "description": "Use all available features",
                "command": "./venv/Scripts/python.exe train_models.py --features 0 --sequence-length 60 --epochs 100",
            },
            # Sequence length variations
            {
                "name": "Seq_30",
                "description": "Short lookback (30 steps)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 30 --epochs 100",
            },
            {
                "name": "Seq_90",
                "description": "Medium lookback (90 steps)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 90 --epochs 100",
            },
            {
                "name": "Seq_120",
                "description": "Long lookback (120 steps)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --sequence-length 120 --epochs 100",
            },
            # Timeframe variations
            {
                "name": "Timeframe_M5",
                "description": "5-minute timeframe (more data)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe M5 --sequence-length 120 --epochs 100",
            },
            {
                "name": "Timeframe_H1",
                "description": "1-hour timeframe (less noise)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe H1 --sequence-length 60 --epochs 100",
            },
            {
                "name": "Timeframe_H4",
                "description": "4-hour timeframe (long-term)",
                "command": "./venv/Scripts/python.exe train_models.py --timeframe H4 --sequence-length 30 --epochs 100",
            },
            # Batch size variations
            {
                "name": "Batch_32",
                "description": "Small batch for stability",
                "command": "./venv/Scripts/python.exe train_models.py --batch-size 32 --epochs 100",
            },
            {
                "name": "Batch_128",
                "description": "Large batch for speed",
                "command": "./venv/Scripts/python.exe train_models.py --batch-size 128 --epochs 100",
            },
            {
                "name": "Batch_256",
                "description": "Very large batch",
                "command": "./venv/Scripts/python.exe train_models.py --batch-size 256 --epochs 100",
            },
            # Learning rate variations
            {
                "name": "LR_0.0001",
                "description": "Very low learning rate",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.0001 --epochs 150",
            },
            {
                "name": "LR_0.0005",
                "description": "Low learning rate",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.0005 --epochs 150",
            },
            {
                "name": "LR_0.002",
                "description": "High learning rate",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.002 --epochs 100",
            },
            {
                "name": "LR_0.005",
                "description": "Very high learning rate",
                "command": "./venv/Scripts/python.exe train_models.py --learning-rate 0.005 --epochs 100",
            },
            # Extended training
            {
                "name": "Extended_200",
                "description": "Extended training (200 epochs)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --epochs 200",
            },
            {
                "name": "Extended_300",
                "description": "Very extended training (300 epochs)",
                "command": "./venv/Scripts/python.exe train_models.py --features 70 --epochs 300",
            },
        ]

        print(f"\n📋 PLANNED EXPERIMENTS: {len(experiments)}")
        print(f"🔄 Running {self.max_workers} experiments in parallel")
        print(
            f"⏱️  Estimated time: {len(experiments) / self.max_workers * 5:.1f} - {len(experiments) / self.max_workers * 10:.1f} minutes"
        )
        print("-" * 80)
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']}: {exp['description']}")
        print("-" * 80)
        print()

        # Option to customize experiments
        print("\n💡 TIP: You can edit this script to add/remove experiments")
        print(f"📂 Results will be saved to: {self.experiments_dir}")
        print()

        response = input("Press ENTER to start parallel training (or 'q' to quit): ")
        if response.lower() == "q":
            print("Cancelled.")
            return

        # Run experiments in parallel
        print("\n" + "=" * 80)
        print("🚀 STARTING PARALLEL TRAINING")
        print("=" * 80)

        start_time = time.time()
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(self.run_single_experiment, exp): exp
                for exp in experiments
            }

            # Process completed experiments as they finish
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    self.results.append(result)

                    if result["status"] == "SUCCESS":
                        completed += 1
                    else:
                        failed += 1

                    # Show progress
                    total = completed + failed
                    progress = total / len(experiments) * 100
                    print(
                        f"\n📊 Progress: {total}/{len(experiments)} ({progress:.1f}%) | ✅ {completed} | ❌ {failed}"
                    )

                except Exception as e:
                    print(f"\n❌ Exception in {exp['name']}: {str(e)}")
                    failed += 1
                    self.results.append(
                        {
                            "name": exp["name"],
                            "description": exp["description"],
                            "command": exp["command"],
                            "status": "EXCEPTION",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("✅ ALL PARALLEL EXPERIMENTS COMPLETED!")
        print("=" * 80)
        print(f"Total Time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print(f"Successful: {completed}/{len(experiments)}")
        print(f"Failed: {failed}/{len(experiments)}")
        print(f"Average Time per Experiment: {total_time / len(experiments):.1f}s")
        print("=" * 80)

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
            print(f"Total experiments run: {len(self.results)}")
            print("Check individual experiment directories for error logs.")

            # Save all results anyway
            all_results_file = self.experiments_dir / "all_results.json"
            with open(all_results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n✅ Saved all results to: {all_results_file}")
            return

        # Create DataFrame
        df = pd.DataFrame(successful)
        df = df.sort_values("test_accuracy", ascending=False)

        # Save detailed results
        csv_path = self.experiments_dir / "successful_experiments.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved successful experiments: {csv_path}")

        # Save all results (including failed)
        all_results_file = self.experiments_dir / "all_results.json"
        with open(all_results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved all results: {all_results_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("🏆 TOP 10 BEST MODELS BY TEST ACCURACY")
        print("=" * 80)

        top10 = df.head(10)
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
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

        best_precision = df.loc[df["test_precision"].idxmax()]
        print(
            f"🎯 Best Precision: {best_precision['name']} ({best_precision['test_precision']:.4f})"
        )

        best_recall = df.loc[df["test_recall"].idxmax()]
        print(
            f"🔍 Best Recall: {best_recall['name']} ({best_recall['test_recall']:.4f})"
        )

        best_balanced = df.loc[
            (df["test_precision"] - df["test_recall"]).abs().idxmin()
        ]
        print(
            f"⚖️  Most Balanced: {best_balanced['name']} (Prec={best_balanced['test_precision']:.4f}, Rec={best_balanced['test_recall']:.4f})"
        )

        fastest = df.loc[df["training_time"].idxmin()]
        print(
            f"⚡ Fastest Training: {fastest['name']} ({fastest['training_time']:.1f}s)"
        )

        # Statistics
        print("\n" + "=" * 80)
        print("📈 OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Experiments:    {len(self.results)}")
        print(f"Successful:           {len(successful)}")
        print(f"Failed:               {len(self.results) - len(successful)}")
        print(f"Success Rate:         {len(successful) / len(self.results) * 100:.1f}%")
        print(
            f"Total Training Time:  {df['training_time'].sum():.1f}s ({df['training_time'].sum() / 3600:.2f} hours)"
        )
        print(
            f"Avg Time per Model:   {df['training_time'].mean():.1f}s ({df['training_time'].mean() / 60:.1f} min)"
        )
        print(
            f"\nAccuracy Range:       {df['test_accuracy'].min():.4f} - {df['test_accuracy'].max():.4f}"
        )
        print(
            f"Average Accuracy:     {df['test_accuracy'].mean():.4f} ({df['test_accuracy'].mean() * 100:.2f}%)"
        )
        print(
            f"Median Accuracy:      {df['test_accuracy'].median():.4f} ({df['test_accuracy'].median() * 100:.2f}%)"
        )
        print(f"Std Dev Accuracy:     {df['test_accuracy'].std():.4f}")

        # Target achievement
        print("\n" + "=" * 80)
        print("🎯 TARGET ACHIEVEMENT")
        print("=" * 80)
        above_50 = len(df[df["test_accuracy"] >= 0.50])
        above_52 = len(df[df["test_accuracy"] >= 0.52])
        above_55 = len(df[df["test_accuracy"] >= 0.55])
        above_60 = len(df[df["test_accuracy"] >= 0.60])
        above_65 = len(df[df["test_accuracy"] >= 0.65])

        print(
            f"Above 50% (Baseline):   {above_50}/{len(successful)} ({above_50 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 52% (Tradeable):  {above_52}/{len(successful)} ({above_52 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 55% (Good):       {above_55}/{len(successful)} ({above_55 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 60% (Excellent):  {above_60}/{len(successful)} ({above_60 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 65% (Outstanding):{above_65}/{len(successful)} ({above_65 / len(successful) * 100:.1f}%)"
        )

        # Recommendations
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)

        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['name']}")
        print(f"   Description: {best_model['description']}")
        print(f"   Command: {best_model['command']}")
        print(f"   Test Accuracy: {best_model['test_accuracy'] * 100:.2f}%")
        print(f"   F1-Score: {best_model['test_f1']:.4f}")
        print(f"   Training Time: {best_model['training_time']:.1f}s")
        print(
            f"   💾 Model saved at: {self.experiments_dir / best_model['name'].replace(' ', '_') / 'lstm_model.h5'}"
        )

        if len(df) >= 3:
            print(f"\n🎯 ENSEMBLE RECOMMENDATION:")
            print(f"   Combine top 3-5 models for robust predictions:")
            for i in range(min(5, len(df))):
                model = df.iloc[i]
                weight = [0.40, 0.25, 0.20, 0.10, 0.05][i]
                print(
                    f"   {i + 1}. {model['name']} - Accuracy: {model['test_accuracy']:.4f} (weight: {weight})"
                )

        # Save summary report
        report_path = self.experiments_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PARALLEL LSTM EXPERIMENTS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parallel Workers: {self.max_workers}\n")
            f.write(f"Total Experiments: {len(self.results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(self.results) - len(successful)}\n\n")

            f.write("TOP 10 MODELS:\n")
            f.write("-" * 80 + "\n")
            for i, (idx, row) in enumerate(top10.iterrows(), 1):
                f.write(f"\n{i}. {row['name']}\n")
                f.write(f"   Accuracy: {row['test_accuracy'] * 100:.2f}%\n")
                f.write(f"   F1-Score: {row['test_f1']:.4f}\n")
                f.write(f"   Precision: {row['test_precision']:.4f}\n")
                f.write(f"   Recall: {row['test_recall']:.4f}\n")
                f.write(f"   Training Time: {row['training_time']:.1f}s\n")
                f.write(f"   Command: {row['command']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST MODEL:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Name: {best_model['name']}\n")
            f.write(f"Accuracy: {best_model['test_accuracy'] * 100:.2f}%\n")
            f.write(f"F1-Score: {best_model['test_f1']:.4f}\n")
            f.write(f"Command: {best_model['command']}\n")
            f.write(
                f"\nModel Location: {self.experiments_dir / best_model['name'].replace(' ', '_') / 'lstm_model.h5'}\n"
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Accuracy: {df['test_accuracy'].mean() * 100:.2f}%\n")
            f.write(f"Best Accuracy: {df['test_accuracy'].max() * 100:.2f}%\n")
            f.write(f"Worst Accuracy: {df['test_accuracy'].min() * 100:.2f}%\n")
            f.write(
                f"Total Training Time: {df['training_time'].sum() / 60:.1f} minutes\n"
            )

        print(f"\n✅ Saved summary report: {report_path}")

        print("\n" + "=" * 80)
        print("✅ PARALLEL TRAINING COMPLETE!")
        print("=" * 80)
        print(f"📂 All results saved to: {self.experiments_dir}")
        print(f"📊 CSV results: {csv_path}")
        print(f"📄 Summary report: {report_path}")
        print("=" * 80)


def main():
    """Main execution"""
    import sys

    # Parse command line arguments
    max_workers = None
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
            print(f"Using {max_workers} parallel workers (from command line)")
        except ValueError:
            print(f"Invalid worker count: {sys.argv[1]}")
            print("Usage: python train_parallel_experiments.py [max_workers]")
            sys.exit(1)

    runner = ParallelExperimentRunner(max_workers=max_workers)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
