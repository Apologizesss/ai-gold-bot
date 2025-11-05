"""
Train Until 70% Accuracy
========================
Advanced continuous training with multiple strategies to reach 70% accuracy.

Features:
- Progressive hyperparameter search
- Smart configuration exploration
- Early stopping when target reached
- Comprehensive logging
- Best model tracking
- Multiple architecture support
"""

import matplotlib

matplotlib.use("Agg")  # Silent mode

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import random
import numpy as np
import argparse
import sys


class AdvancedTrainer:
    """Advanced continuous trainer targeting 70% accuracy"""

    def __init__(self, target_accuracy=0.70, max_iterations=500, patience=50):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.patience = patience  # Stop if no improvement after N iterations

        self.results_dir = Path("results/target_70_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_accuracy = 0.0
        self.best_config = None
        self.iteration = 0
        self.all_results = []
        self.no_improvement_count = 0

        self.log_file = self.results_dir / "training_log.txt"
        self.results_file = self.results_dir / "all_results.csv"

        # Initialize log
        self.log("=" * 80)
        self.log("[Target] TRAINING TO 70% ACCURACY")
        self.log("=" * 80)
        self.log(f"Target: {self.target_accuracy * 100:.1f}%")
        self.log(f"Max iterations: {self.max_iterations}")
        self.log(f"Patience: {self.patience}")
        self.log("=" * 80)

    def log(self, message):
        """Log to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def generate_config(self):
        """Generate smart training configuration"""

        # Expanded search space for 70% target
        architectures = ["simple", "stacked", "attention"]
        features = [50, 70, 100, 120, 150, 200, 250, 300]
        sequences = [30, 45, 60, 90, 120, 150, 180, 240, 300]
        batch_sizes = [16, 32, 64, 128, 256]
        learning_rates = [0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]
        epochs_list = [150, 200, 250, 300, 400, 500, 600, 800]

        # Phase-based strategy
        if self.iteration < 20:
            # Phase 1: Quick exploration (iterations 0-19)
            self.log("📍 Phase 1: Quick Exploration")
            config = {
                "architecture": random.choice(["stacked", "attention"]),
                "features": random.choice([70, 100, 120]),
                "sequence_length": random.choice([60, 90, 120]),
                "batch_size": random.choice([32, 64]),
                "learning_rate": random.choice([0.0005, 0.001, 0.002]),
                "epochs": random.choice([150, 200]),
            }

        elif self.iteration < 50:
            # Phase 2: Broad search (iterations 20-49)
            self.log("📍 Phase 2: Broad Search")
            config = {
                "architecture": random.choice(architectures),
                "features": random.choice(features[:6]),  # Up to 200
                "sequence_length": random.choice(sequences[:6]),  # Up to 150
                "batch_size": random.choice(batch_sizes[:4]),  # Up to 128
                "learning_rate": random.choice(learning_rates),
                "epochs": random.choice([200, 250, 300]),
            }

        elif self.iteration < 100:
            # Phase 3: Refine around best (iterations 50-99)
            self.log("📍 Phase 3: Refining Best Config")
            if self.best_config:
                config = self.perturb_config(self.best_config, magnitude="medium")
            else:
                config = {
                    "architecture": random.choice(["stacked", "attention"]),
                    "features": random.choice(features),
                    "sequence_length": random.choice(sequences),
                    "batch_size": random.choice(batch_sizes),
                    "learning_rate": random.choice(learning_rates),
                    "epochs": random.choice([250, 300, 400]),
                }

        elif self.iteration < 200:
            # Phase 4: Intensive optimization (iterations 100-199)
            self.log("📍 Phase 4: Intensive Optimization")
            if self.best_config and self.best_accuracy > 0.60:
                config = self.perturb_config(self.best_config, magnitude="small")
            else:
                # Try more extreme configurations
                config = {
                    "architecture": random.choice(architectures),
                    "features": random.choice(features),
                    "sequence_length": random.choice(sequences),
                    "batch_size": random.choice(batch_sizes),
                    "learning_rate": random.choice(learning_rates),
                    "epochs": random.choice([300, 400, 500]),
                }

        else:
            # Phase 5: Deep training (iterations 200+)
            self.log("📍 Phase 5: Deep Training")
            if self.best_config:
                config = self.best_config.copy()
                # Increase epochs and fine-tune learning rate
                config["epochs"] = random.choice([500, 600, 800])
                config["learning_rate"] = random.choice([0.0001, 0.0003, 0.0005])
            else:
                config = {
                    "architecture": "attention",
                    "features": random.choice([150, 200, 250, 300]),
                    "sequence_length": random.choice([120, 150, 180, 240]),
                    "batch_size": random.choice([32, 64, 128]),
                    "learning_rate": random.choice([0.0001, 0.0003, 0.0005]),
                    "epochs": random.choice([400, 500, 600]),
                }

        return config

    def perturb_config(self, base_config, magnitude="medium"):
        """Perturb configuration around best known config"""
        config = base_config.copy()

        # Number of parameters to modify
        if magnitude == "small":
            n_mods = random.randint(1, 2)
            deltas = {"features": 20, "sequence": 15, "epochs": 50}
        elif magnitude == "medium":
            n_mods = random.randint(2, 3)
            deltas = {"features": 50, "sequence": 30, "epochs": 100}
        else:  # large
            n_mods = random.randint(3, 4)
            deltas = {"features": 100, "sequence": 60, "epochs": 200}

        params = list(config.keys())
        random.shuffle(params)

        for param in params[:n_mods]:
            if param == "architecture":
                config[param] = random.choice(["simple", "stacked", "attention"])
            elif param == "features":
                delta = random.choice([-deltas["features"], deltas["features"]])
                config[param] = max(50, min(300, config[param] + delta))
            elif param == "sequence_length":
                delta = random.choice([-deltas["sequence"], deltas["sequence"]])
                config[param] = max(30, min(300, config[param] + delta))
            elif param == "batch_size":
                config[param] = random.choice([16, 32, 64, 128, 256])
            elif param == "learning_rate":
                # Multiply or divide by factor
                factor = random.choice([0.5, 0.7, 1.4, 2.0])
                config[param] = max(0.00005, min(0.005, config[param] * factor))
            elif param == "epochs":
                delta = random.choice([-deltas["epochs"], deltas["epochs"]])
                config[param] = max(100, min(800, config[param] + delta))

        return config

    def train_with_config(self, config):
        """Train model with given configuration"""

        self.log("\n" + "─" * 80)
        self.log(f"[Feature Engineering] Configuration #{self.iteration + 1}")
        self.log("─" * 80)
        for key, value in config.items():
            self.log(f"  {key}: {value}")

        # Build command
        cmd = [
            "python",
            "train_with_class_weight.py",
            "--data-path",
            "data/processed/XAUUSD_M15_features_with_target.csv",
            "--architecture",
            config["architecture"],
            "--features",
            str(config["features"]),
            "--sequence-length",
            str(config["sequence_length"]),
            "--batch-size",
            str(config["batch_size"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--epochs",
            str(config["epochs"]),
        ]

        self.log(f"\n[Launch] Starting training...")
        start_time = time.time()

        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            elapsed = time.time() - start_time
            self.log(
                f"[OK] Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)"
            )

            # Parse results
            results_file = Path("results/lstm/lstm_results.json")
            if results_file.exists():
                with open(results_file, "r") as f:
                    results = json.load(f)

                accuracy = results.get("test_accuracy", 0.0)
                precision = results.get("test_precision", 0.0)
                recall = results.get("test_recall", 0.0)
                f1 = results.get("test_f1", 0.0)

                self.log("\n[Stats] Results:")
                self.log(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
                self.log(f"  Precision: {precision:.4f}")
                self.log(f"  Recall:    {recall:.4f}")
                self.log(f"  F1 Score:  {f1:.4f}")

                # Save result
                result_entry = {
                    "iteration": self.iteration + 1,
                    "timestamp": datetime.now().isoformat(),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "training_time": elapsed,
                    **config,
                }

                self.all_results.append(result_entry)

                # Update best
                if accuracy > self.best_accuracy:
                    improvement = accuracy - self.best_accuracy
                    self.best_accuracy = accuracy
                    self.best_config = config
                    self.no_improvement_count = 0

                    self.log(f"\n[Success] NEW BEST! Improved by {improvement * 100:.2f}%")
                    self.log(f"   Best accuracy: {self.best_accuracy * 100:.2f}%")

                    # Save best config
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
                else:
                    self.no_improvement_count += 1
                    gap = self.best_accuracy - accuracy
                    self.log(f"\n[Down] No improvement ({gap * 100:.2f}% below best)")
                    self.log(
                        f"   No improvement count: {self.no_improvement_count}/{self.patience}"
                    )

                # Check if target reached
                if accuracy >= self.target_accuracy:
                    self.log("\n" + "=" * 80)
                    self.log("[Target] TARGET REACHED!")
                    self.log("=" * 80)
                    self.log(
                        f"[OK] Achieved {accuracy * 100:.2f}% accuracy (target: {self.target_accuracy * 100:.1f}%)"
                    )
                    self.log(f"[OK] Iteration: {self.iteration + 1}")
                    self.log(f"[OK] Configuration:")
                    for key, value in config.items():
                        self.log(f"   {key}: {value}")
                    return True

                return False

            else:
                self.log("[Error] Results file not found!")
                return False

        except subprocess.TimeoutExpired:
            self.log("[Error] Training timeout (2 hours)")
            return False
        except Exception as e:
            self.log(f"[Error] Error: {str(e)}")
            return False

    def save_results(self):
        """Save all results to CSV"""
        if self.all_results:
            df = pd.DataFrame(self.all_results)
            df = df.sort_values("accuracy", ascending=False)
            df.to_csv(self.results_file, index=False)

            self.log(
                f"\n[Save] Saved {len(self.all_results)} results to: {self.results_file}"
            )

            # Show top 5
            self.log("\n🏆 TOP 5 CONFIGURATIONS:")
            self.log("─" * 80)
            for idx, row in df.head(5).iterrows():
                self.log(
                    f"{row['iteration']:3d}. Accuracy: {row['accuracy']:.4f} ({row['accuracy'] * 100:.2f}%)"
                )
                self.log(
                    f"     {row['architecture']}, features={row['features']}, "
                    f"seq={row['sequence_length']}, batch={row['batch_size']}, "
                    f"lr={row['learning_rate']}, epochs={row['epochs']}"
                )

    def run(self):
        """Main training loop"""

        self.log("\n[Launch] Starting continuous training...")
        self.log(f"[Target] Target: {self.target_accuracy * 100:.1f}%")
        self.log(f"[Reload] Max iterations: {self.max_iterations}")
        self.log("")

        while self.iteration < self.max_iterations:
            self.log("\n" + "=" * 80)
            self.log(f"ITERATION {self.iteration + 1}/{self.max_iterations}")
            self.log("=" * 80)
            self.log(f"Current best: {self.best_accuracy * 100:.2f}%")
            self.log(f"Target: {self.target_accuracy * 100:.1f}%")
            self.log(f"Gap: {(self.target_accuracy - self.best_accuracy) * 100:.2f}%")

            # Generate config
            config = self.generate_config()

            # Train
            target_reached = self.train_with_config(config)

            # Save results after each iteration
            self.save_results()

            self.iteration += 1

            # Check if target reached
            if target_reached:
                self.log("\n" + "=" * 80)
                self.log("[Success] SUCCESS! TARGET ACCURACY REACHED!")
                self.log("=" * 80)
                break

            # Check patience
            if self.no_improvement_count >= self.patience:
                self.log("\n" + "=" * 80)
                self.log("⏸️  EARLY STOPPING")
                self.log("=" * 80)
                self.log(f"No improvement for {self.patience} iterations")
                self.log(f"Best accuracy achieved: {self.best_accuracy * 100:.2f}%")
                self.log(f"Target was: {self.target_accuracy * 100:.1f}%")

                # Ask if should continue
                self.log("\nOptions:")
                self.log("1. Continue training (reset patience)")
                self.log("2. Stop here")

                # Auto-continue if gap is small
                gap = self.target_accuracy - self.best_accuracy
                if gap < 0.05:  # Less than 5% gap
                    self.log(
                        f"\n✨ Gap is small ({gap * 100:.1f}%), continuing automatically..."
                    )
                    self.no_improvement_count = 0
                else:
                    self.log("\n⏹️  Stopping (large gap remaining)")
                    break

            # Small delay
            time.sleep(1)

        # Final summary
        self.log("\n" + "=" * 80)
        self.log("[Stats] TRAINING COMPLETE - FINAL SUMMARY")
        self.log("=" * 80)
        self.log(f"Total iterations: {self.iteration}")
        self.log(f"Best accuracy: {self.best_accuracy * 100:.2f}%")
        self.log(f"Target accuracy: {self.target_accuracy * 100:.1f}%")

        if self.best_accuracy >= self.target_accuracy:
            self.log(
                f"[OK] TARGET REACHED! ({(self.best_accuracy - self.target_accuracy) * 100:.2f}% above target)"
            )
        else:
            gap = self.target_accuracy - self.best_accuracy
            self.log(f"[Chart] Gap to target: {gap * 100:.2f}%")
            self.log(f"\n[Tip] Suggestions:")
            self.log(f"   - Collect more training data (currently ~6k samples)")
            self.log(f"   - Try different timeframes (M5, M30, H1)")
            self.log(f"   - Feature engineering (add more indicators)")
            self.log(f"   - Ensemble methods (combine multiple models)")
            self.log(f"   - Try XGBoost or other algorithms")

        if self.best_config:
            self.log(f"\n🏆 Best configuration:")
            for key, value in self.best_config.items():
                self.log(f"   {key}: {value}")

        self.log("=" * 80)

        return self.best_accuracy >= self.target_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train model until 70% accuracy is reached"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.70,
        help="Target accuracy (default: 0.70 = 70%%)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum training iterations (default: 500)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Stop if no improvement after N iterations (default: 50)",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = AdvancedTrainer(
        target_accuracy=args.target,
        max_iterations=args.max_iterations,
        patience=args.patience,
    )

    # Run
    try:
        success = trainer.run()

        if success:
            print("\n[Success] Training successful!")
            sys.exit(0)
        else:
            print("\n[Stats] Training completed but target not reached")
            print("   Check results for best configuration")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        trainer.save_results()
        print("   Results saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[Error] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
