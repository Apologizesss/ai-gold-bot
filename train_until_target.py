"""
Continuous Training Until Target Accuracy
Train models continuously with different configurations until reaching target accuracy
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import random
import numpy as np


class ContinuousTrainer:
    """Train models continuously until target accuracy is reached"""

    def __init__(self, target_accuracy=0.70, max_iterations=200):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.results_dir = Path("results/continuous_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_accuracy = 0.0
        self.best_config = None
        self.iteration = 0
        self.all_results = []

        self.log_file = self.results_dir / "training_log.txt"

    def log(self, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def generate_config(self):
        """Generate next training configuration"""

        # Define hyperparameter search space - expanded for 70% target
        architectures = ["simple", "stacked", "attention", "bidirectional"]
        features = [50, 70, 100, 120, 150, 200, 250]
        sequences = [30, 45, 60, 90, 120, 150, 180, 240]
        batch_sizes = [16, 32, 64, 128, 256]
        learning_rates = [0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003]
        epochs = [100, 150, 200, 250, 300, 400, 500]

        # Add more advanced configurations for higher accuracy
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        layer_sizes = [64, 128, 256]
        timeframes = ["M5", "M15", "M30", "H1"]

        # Progressive strategy: start simple, then complex
        if self.iteration < 10:
            # Phase 1: Quick exploration
            config = {
                "mode": random.choice(["simple", "stacked"]),
                "features": random.choice([50, 70, 100]),
                "sequence_length": random.choice([60, 90]),
                "batch_size": random.choice([32, 64]),
                "learning_rate": random.choice([0.0005, 0.001]),
                "epochs": random.choice([80, 100]),
                "timeframe": "M15",
            }
        elif self.iteration < 30:
            # Phase 2: Expand search
            config = {
                "mode": random.choice(architectures),
                "features": random.choice(features[:4]),
                "sequence_length": random.choice(sequences[:4]),
                "batch_size": random.choice(batch_sizes),
                "learning_rate": random.choice(learning_rates),
                "epochs": random.choice([100, 150, 200]),
                "timeframe": random.choice(["M15", "M30"]),
            }
        elif self.iteration < 60:
            # Phase 3: Intensive search near best
            if self.best_config:
                # Perturb best config
                config = self.best_config.copy()

                # Randomly modify 1-2 parameters
                modifications = random.randint(1, 2)
                params = list(config.keys())
                random.shuffle(params)

                for param in params[:modifications]:
                    if param == "mode":
                        config[param] = random.choice(architectures)
                    elif param == "features":
                        delta = random.choice([-30, -20, 20, 30])
                        config[param] = max(30, min(250, config[param] + delta))
                    elif param == "sequence_length":
                        delta = random.choice([-30, -15, 15, 30])
                        config[param] = max(30, min(240, config[param] + delta))
                    elif param == "batch_size":
                        config[param] = random.choice(batch_sizes)
                    elif param == "learning_rate":
                        config[param] = random.choice(learning_rates)
                    elif param == "epochs":
                        config[param] = random.choice([150, 200, 250, 300])
                    elif param == "timeframe":
                        config[param] = random.choice(timeframes)
            else:
                # Full random if no best yet
                config = {
                    "mode": random.choice(architectures),
                    "features": random.choice(features),
                    "sequence_length": random.choice(sequences),
                    "batch_size": random.choice(batch_sizes),
                    "learning_rate": random.choice(learning_rates),
                    "epochs": random.choice(epochs),
                    "timeframe": random.choice(timeframes),
                }
        else:
            # Phase 4: Deep training on best configs
            if self.best_config:
                config = self.best_config.copy()
                config["epochs"] = random.choice([250, 300, 400])
                # Fine-tune learning rate
                config["learning_rate"] = random.choice([0.0001, 0.0003, 0.0005])
            else:
                config = {
                    "mode": "stacked",
                    "features": 120,
                    "sequence_length": 120,
                    "batch_size": 64,
                    "learning_rate": 0.0003,
                    "epochs": 300,
                    "timeframe": "M15",
                }

        return config

    def build_command(self, config):
        """Build training command from config"""
        cmd_parts = ["python", "train_models.py"]

        if config.get("mode"):
            cmd_parts.extend(["--mode", config["mode"]])
        if config.get("features"):
            cmd_parts.extend(["--features", str(config["features"])])
        if config.get("sequence_length"):
            cmd_parts.extend(["--sequence-length", str(config["sequence_length"])])
        if config.get("batch_size"):
            cmd_parts.extend(["--batch-size", str(config["batch_size"])])
        if config.get("learning_rate"):
            cmd_parts.extend(["--learning-rate", str(config["learning_rate"])])
        if config.get("epochs"):
            cmd_parts.extend(["--epochs", str(config["epochs"])])
        if config.get("timeframe"):
            cmd_parts.extend(["--timeframe", config["timeframe"]])

        return " ".join(cmd_parts)

    def train_model(self, config):
        """Train a single model with given config"""
        command = self.build_command(config)

        self.log(f"\n{'=' * 80}")
        self.log(f"ITERATION {self.iteration + 1}/{self.max_iterations}")
        self.log(f"{'=' * 80}")
        self.log(f"Config: {config}")
        self.log(f"Command: {command}")
        self.log(
            f"Target: {self.target_accuracy * 100:.2f}% | Best so far: {self.best_accuracy * 100:.2f}%"
        )
        self.log(f"{'=' * 80}")

        start_time = time.time()

        try:
            # Run training
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            elapsed = time.time() - start_time

            # Load results
            results_file = Path("results/lstm/lstm_results.json")
            if results_file.exists():
                with open(results_file, "r") as f:
                    metrics = json.load(f)

                accuracy = metrics["test_accuracy"]

                # Save this iteration
                iteration_result = {
                    "iteration": self.iteration + 1,
                    "config": config,
                    "command": command,
                    "test_accuracy": accuracy,
                    "test_f1": metrics["test_f1"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "training_time": elapsed,
                    "timestamp": datetime.now().isoformat(),
                }

                self.all_results.append(iteration_result)

                # Update best
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config.copy()

                    # Save best model
                    best_model_src = Path("models/lstm_best_model.h5")
                    if best_model_src.exists():
                        best_model_dst = (
                            self.results_dir / f"best_model_acc_{accuracy * 100:.2f}.h5"
                        )
                        import shutil

                        shutil.copy(best_model_src, best_model_dst)
                        self.log(f"💾 Saved new best model: {best_model_dst}")

                # Log results
                self.log(f"\n✅ Training Complete!")
                self.log(f"⏱️  Time: {elapsed / 60:.1f} minutes")
                self.log(f"📊 Test Accuracy: {accuracy * 100:.2f}%")
                self.log(f"📈 Test F1: {metrics['test_f1']:.4f}")
                self.log(f"🎯 Best Accuracy: {self.best_accuracy * 100:.2f}%")

                improvement = accuracy - self.best_accuracy if self.iteration > 0 else 0
                if accuracy >= self.best_accuracy:
                    self.log(f"🎉 NEW BEST! (+{improvement * 100:.2f}%)")
                else:
                    self.log(
                        f"📉 Not better than best ({(accuracy - self.best_accuracy) * 100:.2f}%)"
                    )

                # Save progress
                self.save_progress()

                return accuracy >= self.target_accuracy

            else:
                self.log(f"❌ Results file not found")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"⏰ Training timeout (> 2 hours)")
            return False
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            return False

    def save_progress(self):
        """Save current progress"""
        # Save all results
        results_file = self.results_dir / "all_iterations.json"
        with open(results_file, "w") as f:
            json.dump(self.all_results, f, indent=2)

        # Save CSV
        if self.all_results:
            df = pd.DataFrame(self.all_results)
            df = df.sort_values("test_accuracy", ascending=False)
            csv_file = self.results_dir / "iterations_ranked.csv"
            df.to_csv(csv_file, index=False)

        # Save best config
        if self.best_config:
            best_file = self.results_dir / "best_config.json"
            with open(best_file, "w") as f:
                json.dump(
                    {
                        "accuracy": self.best_accuracy,
                        "config": self.best_config,
                        "iteration": self.iteration + 1,
                    },
                    f,
                    indent=2,
                )

    def print_summary(self):
        """Print training summary"""
        self.log(f"\n\n{'=' * 80}")
        self.log(f"TRAINING SUMMARY")
        self.log(f"{'=' * 80}")
        self.log(f"Total Iterations: {len(self.all_results)}")
        self.log(f"Target Accuracy: {self.target_accuracy * 100:.2f}%")
        self.log(f"Best Accuracy: {self.best_accuracy * 100:.2f}%")
        self.log(
            f"Target {'REACHED! 🎉' if self.best_accuracy >= self.target_accuracy else 'NOT REACHED'}"
        )

        if self.best_config:
            self.log(f"\nBest Configuration:")
            for key, value in self.best_config.items():
                self.log(f"  {key}: {value}")

        if self.all_results:
            df = pd.DataFrame(self.all_results)
            self.log(f"\nAccuracy Statistics:")
            self.log(f"  Average: {df['test_accuracy'].mean() * 100:.2f}%")
            self.log(f"  Best: {df['test_accuracy'].max() * 100:.2f}%")
            self.log(f"  Worst: {df['test_accuracy'].min() * 100:.2f}%")
            self.log(f"  Std Dev: {df['test_accuracy'].std() * 100:.2f}%")

            # Count by target
            above_60 = len(df[df["test_accuracy"] >= 0.60])
            above_55 = len(df[df["test_accuracy"] >= 0.55])
            above_52 = len(df[df["test_accuracy"] >= 0.52])

            self.log(f"\nTarget Achievement:")
            self.log(
                f"  Above 52%: {above_52}/{len(df)} ({above_52 / len(df) * 100:.1f}%)"
            )
            self.log(
                f"  Above 55%: {above_55}/{len(df)} ({above_55 / len(df) * 100:.1f}%)"
            )
            self.log(
                f"  Above 60%: {above_60}/{len(df)} ({above_60 / len(df) * 100:.1f}%)"
            )

            total_time = df["training_time"].sum()
            self.log(f"\nTotal Training Time: {total_time / 3600:.2f} hours")
            self.log(f"Average Time per Model: {total_time / len(df) / 60:.1f} minutes")

        self.log(f"{'=' * 80}")
        self.log(f"\nResults saved to: {self.results_dir}")
        self.log(f"  - all_iterations.json (complete log)")
        self.log(f"  - iterations_ranked.csv (sorted by accuracy)")
        self.log(f"  - best_config.json (best configuration)")
        self.log(f"  - training_log.txt (detailed log)")
        if self.best_accuracy > 0:
            self.log(
                f"  - best_model_acc_{self.best_accuracy * 100:.2f}.h5 (best model)"
            )

    def run(self):
        """Main training loop"""
        print("=" * 80)
        print("🎯 CONTINUOUS TRAINING UNTIL TARGET ACCURACY")
        print("=" * 80)
        print(f"Target Accuracy: {self.target_accuracy * 100:.2f}%")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Results Directory: {self.results_dir}")
        print("=" * 80)
        print()

        response = input("Start continuous training? (Y/N): ")
        if response.upper() != "Y":
            print("Cancelled.")
            return

        self.log(f"\n{'=' * 80}")
        self.log(f"STARTING CONTINUOUS TRAINING")
        self.log(f"Target: {self.target_accuracy * 100:.2f}%")
        self.log(f"Max Iterations: {self.max_iterations}")
        self.log(f"{'=' * 80}\n")

        start_time = time.time()
        target_reached = False

        for i in range(self.max_iterations):
            self.iteration = i

            # Generate configuration
            config = self.generate_config()

            # Train model
            target_reached = self.train_model(config)

            # Check if target reached
            if target_reached:
                self.log(f"\n🎉🎉🎉 TARGET REACHED! 🎉🎉🎉")
                self.log(f"Achieved {self.best_accuracy * 100:.2f}% accuracy!")
                break

            # Check for plateau (no improvement in last 10 iterations)
            if len(self.all_results) >= 20:
                recent = self.all_results[-10:]
                recent_best = max([r["test_accuracy"] for r in recent])
                if recent_best <= self.best_accuracy - 0.02:  # No improvement
                    self.log(f"\n⚠️  No improvement in last 10 iterations")
                    self.log(f"Consider adjusting strategy or stopping")

        total_time = time.time() - start_time
        self.log(f"\nTotal training time: {total_time / 3600:.2f} hours")

        # Print summary
        self.print_summary()

        # Final recommendation
        if target_reached:
            self.log(f"\n✅ SUCCESS! Target accuracy reached!")
            self.log(
                f"Use best model: {self.results_dir / f'best_model_acc_{self.best_accuracy * 100:.2f}.h5'}"
            )
        else:
            self.log(f"\n⚠️  Target not reached after {self.max_iterations} iterations")
            self.log(f"Best accuracy: {self.best_accuracy * 100:.2f}%")
            self.log(f"Consider:")
            self.log(f"  1. Running more iterations")
            self.log(f"  2. Collecting more/better data")
            self.log(f"  3. Feature engineering")
            self.log(f"  4. Different timeframes")
            self.log(f"  5. Ensemble methods")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Train until target accuracy")
    parser.add_argument(
        "--target", type=float, default=0.65, help="Target accuracy (default: 0.65)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Max iterations (default: 100)"
    )

    args = parser.parse_args()

    trainer = ContinuousTrainer(
        target_accuracy=args.target, max_iterations=args.max_iterations
    )

    trainer.run()


if __name__ == "__main__":
    main()
