"""
Config-Based Parallel Training Script
Load experiments from JSON config and run them in parallel
"""

import json
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import sys


class ConfigBasedRunner:
    """Run experiments based on JSON configuration"""

    def __init__(self, config_path="config/parallel_experiments.json"):
        self.config_path = Path(config_path)
        self.results = []
        self.experiments_dir = Path("results/experiments_parallel")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load experiment configuration from JSON"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            print("=" * 80)
            print("📋 CONFIG-BASED PARALLEL TRAINING")
            print("=" * 80)
            print(f"Config: {self.config_path}")
            print(f"Description: {self.config.get('description', 'N/A')}")
            print(f"Version: {self.config.get('version', '1.0')}")
            print("=" * 80)

        except FileNotFoundError:
            print(f"❌ ERROR: Config file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in config file: {e}")
            sys.exit(1)

    def get_enabled_experiments(self):
        """Get list of enabled experiments from config"""
        experiments = self.config.get("experiments", [])
        enabled = [exp for exp in experiments if exp.get("enabled", True)]
        return enabled

    def get_experiment_set(self, set_name):
        """Get predefined set of experiments"""
        sets = self.config.get("experiment_sets", {})
        if set_name not in sets:
            print(f"❌ ERROR: Experiment set '{set_name}' not found")
            return []

        exp_set = sets[set_name]
        exp_names = exp_set.get("experiments", [])

        if exp_names == "auto":
            return self.get_enabled_experiments()

        # Get experiments by name
        all_experiments = self.config.get("experiments", [])
        selected = [exp for exp in all_experiments if exp["name"] in exp_names]
        return selected

    def run_single_experiment(self, experiment):
        """Run a single training experiment"""
        name = experiment["name"]
        description = experiment["description"]
        command = experiment["command"]

        print(f"\n🧪 [{name}] Starting...")
        print(f"📝 [{name}] {description}")

        start_time = time.time()

        try:
            # Create unique output directory
            exp_output_dir = self.experiments_dir / name.replace(" ", "_")
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            results_file = exp_output_dir / "lstm_results.json"
            model_file = exp_output_dir / "lstm_model.h5"

            # Run training
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "TF_CPP_MIN_LOG_LEVEL": "2",
                },
            )

            elapsed = time.time() - start_time

            # Copy results from default location
            try:
                default_results = Path("results/lstm/lstm_results.json")
                if default_results.exists():
                    shutil.copy(default_results, results_file)

                default_model = Path("models/lstm_best_model.h5")
                if default_model.exists():
                    shutil.copy(default_model, model_file)

                # Load metrics
                with open(results_file, "r") as f:
                    metrics = json.load(f)

                experiment_result = {
                    "name": name,
                    "description": description,
                    "command": command,
                    "priority": experiment.get("priority", "medium"),
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

                # Save individual results
                exp_file = exp_output_dir / "experiment_results.json"
                with open(exp_file, "w") as f:
                    json.dump(experiment_result, f, indent=2)

                print(f"✅ [{name}] COMPLETED!")
                print(f"⏱️  [{name}] Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
                print(
                    f"📊 [{name}] Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
                )
                print(f"📈 [{name}] F1-Score: {metrics['test_f1']:.4f}")

                return experiment_result

            except Exception as e:
                print(f"⚠️  [{name}] WARNING: Could not load results - {str(e)}")
                return {
                    "name": name,
                    "description": description,
                    "command": command,
                    "status": "COMPLETED_NO_RESULTS",
                    "error": str(e),
                    "training_time": elapsed,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [{name}] FAILED: {str(e)}")
            return {
                "name": name,
                "description": description,
                "command": command,
                "status": "FAILED",
                "error": str(e),
                "training_time": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

    def run_experiments(self, experiments, max_workers=None):
        """Run experiments in parallel"""
        if not experiments:
            print("❌ No experiments to run!")
            return

        # Determine worker count
        cpu_count = mp.cpu_count()
        if max_workers is None:
            max_workers = self.config.get("max_workers")
        if max_workers is None:
            max_workers = max(1, cpu_count - 1)
        else:
            max_workers = min(max_workers, cpu_count)

        print(f"\n📊 EXPERIMENT SUMMARY")
        print(f"Total Experiments: {len(experiments)}")
        print(f"CPU Cores: {cpu_count}")
        print(f"Parallel Workers: {max_workers}")
        print(f"Expected Speedup: {max_workers}x")
        print(
            f"Estimated Time: {len(experiments) / max_workers * 5:.0f}-{len(experiments) / max_workers * 10:.0f} minutes"
        )
        print("-" * 80)

        # Group by priority
        by_priority = {}
        for exp in experiments:
            priority = exp.get("priority", "medium")
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(exp)

        for priority, exps in by_priority.items():
            print(f"{priority.upper()}: {len(exps)} experiments")

        print("-" * 80)
        for i, exp in enumerate(experiments, 1):
            priority_marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                exp.get("priority", "medium"), "⚪"
            )
            print(f"{i}. {priority_marker} {exp['name']}: {exp['description']}")
        print("-" * 80)

        response = input("\nPress ENTER to start training (or 'q' to quit): ")
        if response.lower() == "q":
            print("Cancelled.")
            return

        # Run parallel training
        print("\n" + "=" * 80)
        print("🚀 STARTING PARALLEL TRAINING")
        print("=" * 80)

        start_time = time.time()
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_exp = {
                executor.submit(self.run_single_experiment, exp): exp
                for exp in experiments
            }

            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    self.results.append(result)

                    if result["status"] == "SUCCESS":
                        completed += 1
                    else:
                        failed += 1

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
                            "status": "EXCEPTION",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("✅ PARALLEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total Time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print(f"Successful: {completed}/{len(experiments)}")
        print(f"Failed: {failed}/{len(experiments)}")
        print(f"Average: {total_time / len(experiments):.1f}s per experiment")
        print("=" * 80)

        self.generate_report()

    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 80)
        print("📊 GENERATING REPORT")
        print("=" * 80)

        successful = [r for r in self.results if r.get("status") == "SUCCESS"]

        if not successful:
            print("❌ No successful experiments!")
            # Save all results
            all_file = self.experiments_dir / "all_results.json"
            with open(all_file, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Saved: {all_file}")
            return

        # Create DataFrame
        df = pd.DataFrame(successful)
        df = df.sort_values("test_accuracy", ascending=False)

        # Save results
        csv_path = self.experiments_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

        # Save all results
        all_file = self.experiments_dir / "all_results.json"
        with open(all_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved: {all_file}")

        # Print top 10
        print("\n🏆 TOP 10 MODELS")
        print("-" * 80)
        print(f"{'Rank':<6} {'Name':<25} {'Accuracy':<12} {'F1':<10} {'Time':<10}")
        print("-" * 80)

        for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
            print(
                f"{i:<6} {row['name'][:24]:<25} {row['test_accuracy'] * 100:>6.2f}%     {row['test_f1']:>6.4f}    {row['training_time'] / 60:>5.1f} min"
            )

        # Best model
        best = df.iloc[0]
        print("\n🏆 BEST MODEL")
        print("-" * 80)
        print(f"Name:        {best['name']}")
        print(f"Accuracy:    {best['test_accuracy'] * 100:.2f}%")
        print(f"F1-Score:    {best['test_f1']:.4f}")
        print(f"Precision:   {best['test_precision']:.4f}")
        print(f"Recall:      {best['test_recall']:.4f}")
        print(f"Time:        {best['training_time'] / 60:.1f} min")
        print(f"Command:     {best['command']}")

        # Statistics
        print("\n📈 STATISTICS")
        print("-" * 80)
        print(f"Experiments:     {len(self.results)}")
        print(f"Successful:      {len(successful)}")
        print(f"Failed:          {len(self.results) - len(successful)}")
        print(f"Avg Accuracy:    {df['test_accuracy'].mean() * 100:.2f}%")
        print(f"Best Accuracy:   {df['test_accuracy'].max() * 100:.2f}%")
        print(f"Avg F1-Score:    {df['test_f1'].mean():.4f}")

        # Targets
        above_60 = len(df[df["test_accuracy"] >= 0.60])
        above_55 = len(df[df["test_accuracy"] >= 0.55])
        above_52 = len(df[df["test_accuracy"] >= 0.52])

        print(f"\n🎯 TARGETS")
        print(
            f"Above 60%:       {above_60}/{len(successful)} ({above_60 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 55%:       {above_55}/{len(successful)} ({above_55 / len(successful) * 100:.1f}%)"
        )
        print(
            f"Above 52%:       {above_52}/{len(successful)} ({above_52 / len(successful) * 100:.1f}%)"
        )

        # Save report
        report_path = self.experiments_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write("CONFIG-BASED PARALLEL TRAINING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {self.config_path}\n\n")
            f.write(f"Best Model: {best['name']}\n")
            f.write(f"Accuracy: {best['test_accuracy'] * 100:.2f}%\n")
            f.write(f"Command: {best['command']}\n")

        print(f"\n✅ Saved: {report_path}")
        print("=" * 80)


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments from config file")
    parser.add_argument(
        "--config", default="config/parallel_experiments.json", help="Config file path"
    )
    parser.add_argument("--set", help="Run predefined experiment set")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--list-sets", action="store_true", help="List available experiment sets"
    )

    args = parser.parse_args()

    runner = ConfigBasedRunner(config_path=args.config)

    # List sets if requested
    if args.list_sets:
        print("\n📋 AVAILABLE EXPERIMENT SETS:")
        print("-" * 80)
        sets = runner.config.get("experiment_sets", {})
        for name, info in sets.items():
            print(f"\n{name}:")
            print(f"  {info.get('description', 'N/A')}")
            exps = info.get("experiments", [])
            if exps == "auto":
                print(f"  Experiments: All enabled")
            else:
                print(f"  Experiments: {len(exps)}")
                for exp_name in exps:
                    print(f"    - {exp_name}")
        print("-" * 80)
        return

    # Get experiments to run
    if args.set:
        experiments = runner.get_experiment_set(args.set)
        print(f"\n🎯 Running experiment set: {args.set}")
    else:
        experiments = runner.get_enabled_experiments()
        print(f"\n🎯 Running all enabled experiments")

    if not experiments:
        print("❌ No experiments to run!")
        return

    # Run experiments
    runner.run_experiments(experiments, max_workers=args.workers)


if __name__ == "__main__":
    main()
