"""
Model Comparison Script - Compare LSTM vs XGBoost Performance
Generates comprehensive comparison reports and visualizations
"""

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend to prevent figure windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ModelComparison:
    """Compare performance of different ML models"""

    def __init__(self):
        """Initialize comparison tool"""
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.comparison_dir = Path("results/comparison")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.results = {}
        self.predictions = {}

        print("=" * 80)
        print("📊 MODEL COMPARISON TOOL")
        print("=" * 80)

    def load_model_results(self, model_name, results_path):
        """Load results for a specific model"""
        print(f"\n📥 Loading {model_name} results...")

        try:
            with open(results_path, "r") as f:
                results = json.load(f)

            self.results[model_name] = results
            print(f"✅ Loaded {model_name} results")

            # Display key metrics
            if "test_accuracy" in results:
                print(f"   Accuracy: {results['test_accuracy']:.4f}")
            if "test_precision" in results:
                print(f"   Precision: {results['test_precision']:.4f}")
            if "test_recall" in results:
                print(f"   Recall: {results['test_recall']:.4f}")
            if "test_f1" in results:
                print(f"   F1-Score: {results['test_f1']:.4f}")

            return True

        except FileNotFoundError:
            print(f"⚠️  Results file not found: {results_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")
            return False

    def load_all_models(self):
        """Load results from all available models"""
        print("\n🔍 Searching for trained models...")

        # Define model configurations
        model_configs = {
            "LSTM": self.results_dir / "lstm" / "lstm_results.json",
            "XGBoost": self.results_dir / "feature_analysis" / "xgboost_results.json",
            "Random Forest": self.results_dir / "random_forest" / "rf_results.json",
        }

        loaded_count = 0
        for model_name, results_path in model_configs.items():
            if self.load_model_results(model_name, results_path):
                loaded_count += 1

        print(f"\n✅ Loaded {loaded_count} model(s)")
        return self

    def compare_metrics(self):
        """Compare key metrics across models"""
        print("\n📊 Comparing metrics...")

        if not self.results:
            print("⚠️  No models loaded for comparison")
            return self

        # Extract metrics
        metrics_data = []
        metric_names = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ]

        for model_name, results in self.results.items():
            row = {"Model": model_name}
            for metric in metric_names:
                row[metric] = results.get(metric, np.nan)
            metrics_data.append(row)

        # Create DataFrame
        self.metrics_df = pd.DataFrame(metrics_data)

        # Display comparison table
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE COMPARISON")
        print("=" * 80)
        print(self.metrics_df.to_string(index=False))
        print("=" * 80)

        # Find best model for each metric
        print("\n🏆 BEST PERFORMERS:")
        for metric in metric_names:
            if metric in self.metrics_df.columns:
                best_idx = self.metrics_df[metric].idxmax()
                best_model = self.metrics_df.loc[best_idx, "Model"]
                best_score = self.metrics_df.loc[best_idx, metric]
                print(f"   {metric}: {best_model} ({best_score:.4f})")

        return self

    def plot_metric_comparison(self):
        """Plot bar charts comparing metrics"""
        print("\n📊 Creating metric comparison plots...")

        if not hasattr(self, "metrics_df"):
            print("⚠️  No metrics data available")
            return self

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Model Performance Comparison", fontsize=16, fontweight="bold", y=0.995
        )

        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        titles = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 2, idx % 2]

            if metric in self.metrics_df.columns:
                data = self.metrics_df[["Model", metric]].dropna()
                bars = ax.bar(
                    data["Model"],
                    data[metric],
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.4f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )

                ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
                ax.set_ylabel("Score", fontsize=10)
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis="y")
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = self.comparison_dir / "metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {plot_path}")

        # plt.show()  # Disabled to prevent figure windows

        return self

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\n📊 Creating confusion matrix comparison...")

        models_with_cm = {
            name: res
            for name, res in self.results.items()
            if "true_labels" in res and "predictions" in res
        }

        if not models_with_cm:
            print("⚠️  No confusion matrix data available")
            return self

        n_models = len(models_with_cm)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        fig.suptitle(
            "Confusion Matrix Comparison", fontsize=16, fontweight="bold", y=1.02
        )

        for idx, (model_name, results) in enumerate(models_with_cm.items()):
            y_true = np.array(results["true_labels"])
            y_pred = np.array(results["predictions"])

            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Down (0)", "Up (1)"],
                yticklabels=["Down (0)", "Up (1)"],
                ax=axes[idx],
                cbar_kws={"shrink": 0.8},
            )
            axes[idx].set_title(model_name, fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("True Label", fontsize=10)
            axes[idx].set_xlabel("Predicted Label", fontsize=10)

            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            axes[idx].text(
                0.5,
                -0.15,
                f"Accuracy: {accuracy:.4f}",
                ha="center",
                transform=axes[idx].transAxes,
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save plot
        plot_path = self.comparison_dir / "confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {plot_path}")

        # plt.show()  # Disabled to prevent figure windows

        return self

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("\n📊 Creating ROC curve comparison...")

        models_with_proba = {
            name: res
            for name, res in self.results.items()
            if "true_labels" in res and "probabilities" in res
        }

        if not models_with_proba:
            print("⚠️  No probability data available for ROC curves")
            return self

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, len(models_with_proba)))

        for idx, (model_name, results) in enumerate(models_with_proba.items()):
            y_true = np.array(results["true_labels"])
            y_proba = np.array(results["probabilities"])

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            # Plot
            ax.plot(
                fpr,
                tpr,
                color=colors[idx],
                linewidth=2.5,
                label=f"{model_name} (AUC = {auc:.4f})",
            )

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random (AUC = 0.5000)")

        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        plt.tight_layout()

        # Save plot
        plot_path = self.comparison_dir / "roc_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {plot_path}")

        # plt.show()  # Disabled to prevent figure windows

        return self

    def plot_radar_chart(self):
        """Create radar chart comparing all metrics"""
        print("\n📊 Creating radar chart...")

        if not hasattr(self, "metrics_df"):
            print("⚠️  No metrics data available")
            return self

        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.metrics_df.columns]
        available_labels = [
            labels[i] for i, m in enumerate(metrics) if m in self.metrics_df.columns
        ]

        if not available_metrics:
            print("⚠️  No metrics available for radar chart")
            return self

        # Number of variables
        num_vars = len(available_metrics)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.metrics_df)))

        for idx, row in self.metrics_df.iterrows():
            values = row[available_metrics].tolist()
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2.5,
                label=row["Model"],
                color=colors[idx],
            )
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_labels, fontsize=11, fontweight="bold")

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9
        )

        plt.title(
            "Model Performance Radar Chart",
            fontsize=14,
            fontweight="bold",
            pad=20,
            y=1.08,
        )

        plt.tight_layout()

        # Save plot
        plot_path = self.comparison_dir / "radar_chart.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {plot_path}")

        # plt.show()  # Disabled to prevent figure windows

        return self

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n📝 Generating comparison report...")

        report_path = self.comparison_dir / "comparison_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Compared: {len(self.results)}\n")
            f.write("=" * 80 + "\n\n")

            # Metrics comparison
            if hasattr(self, "metrics_df"):
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(self.metrics_df.to_string(index=False) + "\n\n")

                # Best performers
                f.write("BEST PERFORMERS\n")
                f.write("-" * 80 + "\n")
                for metric in [
                    "test_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                ]:
                    if metric in self.metrics_df.columns:
                        best_idx = self.metrics_df[metric].idxmax()
                        best_model = self.metrics_df.loc[best_idx, "Model"]
                        best_score = self.metrics_df.loc[best_idx, metric]
                        f.write(f"{metric:20s}: {best_model:15s} ({best_score:.4f})\n")
                f.write("\n")

            # Individual model details
            f.write("DETAILED MODEL RESULTS\n")
            f.write("-" * 80 + "\n\n")

            for model_name, results in self.results.items():
                f.write(f"### {model_name} ###\n")
                f.write("-" * 40 + "\n")

                # Write all available metrics
                for key, value in results.items():
                    if key not in ["predictions", "probabilities", "true_labels"]:
                        if isinstance(value, float):
                            f.write(f"  {key:25s}: {value:.6f}\n")
                        else:
                            f.write(f"  {key:25s}: {value}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"✅ Saved report: {report_path}")

        return self

    def save_comparison_data(self):
        """Save comparison data to CSV"""
        print("\n💾 Saving comparison data...")

        if hasattr(self, "metrics_df"):
            csv_path = self.comparison_dir / "metrics_comparison.csv"
            self.metrics_df.to_csv(csv_path, index=False)
            print(f"✅ Saved: {csv_path}")

        return self

    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        print("\n" + "=" * 80)
        print("🚀 STARTING MODEL COMPARISON")
        print("=" * 80)

        try:
            self.load_all_models()

            if not self.results:
                print("\n⚠️  No models found to compare!")
                print("   Please train models first using:")
                print("   - python src/models/train_lstm.py")
                print("   - python src/models/train_xgboost.py")
                return self

            self.compare_metrics()
            self.plot_metric_comparison()
            self.plot_confusion_matrices()
            self.plot_roc_curves()
            self.plot_radar_chart()
            self.generate_comparison_report()
            self.save_comparison_data()

            print("\n" + "=" * 80)
            print("✅ COMPARISON COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📂 Results saved in: {self.comparison_dir}")
            print("=" * 80)

            return self

        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ COMPARISON FAILED")
            print("=" * 80)
            print(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


def main():
    """Main execution function"""
    comparator = ModelComparison()
    comparator.run_full_comparison()

    print("\n🎉 Model comparison complete!")
    print(f"📊 Check results in: {comparator.comparison_dir}")


if __name__ == "__main__":
    main()
