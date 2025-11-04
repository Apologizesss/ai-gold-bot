# -*- coding: utf-8 -*-
"""
Main Training Launcher Script
Trains LSTM models with different configurations and compares with XGBoost baseline
"""

import argparse
import sys
import io
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.train_lstm import LSTMTrainingPipeline


def train_lstm_simple():
    """Train simple LSTM model"""
    print("\n🚀 Training Simple LSTM Model...")

    config = {
        "data_path": "data/processed/XAUUSD_M15_features_complete.csv",
        "sequence_length": 60,
        "prediction_horizon": 1,
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "use_top_features": 70,
    }

    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture="simple")

    return pipeline


def train_lstm_stacked():
    """Train stacked LSTM model"""
    print("\n🚀 Training Stacked LSTM Model...")

    config = {
        "data_path": "data/processed/XAUUSD_M15_features_complete.csv",
        "sequence_length": 60,
        "prediction_horizon": 1,
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "use_top_features": 70,
    }

    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture="stacked")

    return pipeline


def train_lstm_bidirectional():
    """Train bidirectional LSTM model (RECOMMENDED)"""
    print("\n🚀 Training Bidirectional LSTM Model (RECOMMENDED)...")

    config = {
        "data_path": "data/processed/XAUUSD_M15_features_complete.csv",
        "sequence_length": 60,
        "prediction_horizon": 1,
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "use_top_features": 70,
    }

    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture="bidirectional")

    return pipeline


def train_lstm_attention():
    """Train LSTM with attention mechanism"""
    print("\n🚀 Training LSTM with Attention...")

    config = {
        "data_path": "data/processed/XAUUSD_M15_features_complete.csv",
        "sequence_length": 60,
        "prediction_horizon": 1,
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "use_top_features": 70,
    }

    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture="attention")

    return pipeline


def train_lstm_custom(
    sequence_length=60,
    batch_size=64,
    epochs=100,
    learning_rate=0.001,
    top_features=70,
    architecture="bidirectional",
    timeframe="M15",
):
    """Train LSTM with custom configuration"""
    print(f"\n🚀 Training Custom LSTM Model ({architecture})...")
    print(f"   Timeframe: {timeframe}")
    print(f"   Sequence Length: {sequence_length}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Top Features: {top_features}")

    config = {
        "data_path": f"data/processed/XAUUSD_{timeframe}_features_complete.csv",
        "sequence_length": sequence_length,
        "prediction_horizon": 1,
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "use_top_features": top_features,
    }

    pipeline = LSTMTrainingPipeline(**config)
    pipeline.run_full_pipeline(architecture=architecture)

    return pipeline


def train_all_architectures():
    """Train all LSTM architectures and compare"""
    print("\n" + "=" * 80)
    print("🚀 TRAINING ALL LSTM ARCHITECTURES")
    print("=" * 80)

    architectures = ["simple", "stacked", "bidirectional", "attention"]
    results = {}

    for arch in architectures:
        try:
            print(f"\n{'=' * 80}")
            print(f"Training {arch.upper()} architecture...")
            print(f"{'=' * 80}")

            pipeline = train_lstm_custom(architecture=arch)
            results[arch] = {
                "test_accuracy": pipeline.results.get("test_accuracy", 0),
                "test_f1": pipeline.results.get("test_f1", 0),
                "test_precision": pipeline.results.get("test_precision", 0),
                "test_recall": pipeline.results.get("test_recall", 0),
            }

            print(f"\n✅ {arch.upper()} completed!")

        except Exception as e:
            print(f"\n❌ {arch.upper()} failed: {str(e)}")
            results[arch] = None

    # Print summary
    print("\n" + "=" * 80)
    print("📊 ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 80)

    for arch, metrics in results.items():
        if metrics:
            print(f"\n{arch.upper()}:")
            print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"  Precision: {metrics['test_precision']:.4f}")
            print(f"  Recall:    {metrics['test_recall']:.4f}")
            print(f"  F1-Score:  {metrics['test_f1']:.4f}")
        else:
            print(f"\n{arch.upper()}: FAILED")

    # Find best architecture
    best_arch = max(
        [(arch, m) for arch, m in results.items() if m],
        key=lambda x: x[1]["test_f1"],
    )

    print("\n" + "=" * 80)
    print(f"🏆 BEST ARCHITECTURE: {best_arch[0].upper()}")
    print(f"   F1-Score: {best_arch[1]['test_f1']:.4f}")
    print("=" * 80)

    return results


def main():
    """Main execution with command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train LSTM models for Gold price prediction"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="bidirectional",
        choices=["simple", "stacked", "bidirectional", "attention", "all", "custom"],
        help="Training mode (default: bidirectional)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="M15",
        choices=["M5", "M15", "M30", "H1", "H4", "D1"],
        help="Timeframe to use (default: M15)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Number of timesteps to look back (default: 60)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--features",
        type=int,
        default=70,
        help="Number of top features to use (default: 70, 0=all)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🧠 LSTM MODEL TRAINING LAUNCHER")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        if args.mode == "simple":
            pipeline = train_lstm_simple()

        elif args.mode == "stacked":
            pipeline = train_lstm_stacked()

        elif args.mode == "bidirectional":
            pipeline = train_lstm_bidirectional()

        elif args.mode == "attention":
            pipeline = train_lstm_attention()

        elif args.mode == "all":
            results = train_all_architectures()

        elif args.mode == "custom":
            pipeline = train_lstm_custom(
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                top_features=args.features if args.features > 0 else None,
                architecture="bidirectional",
                timeframe=args.timeframe,
            )

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Models saved in: models/")
        print(f"📊 Results saved in: results/lstm/")
        print("=" * 80)

        print("\n💡 Next Steps:")
        print("   1. Compare with XGBoost: python src/models/compare_models.py")
        print("   2. Run backtesting: python src/backtesting/run_backtest.py")
        print("   3. Analyze features: python analyze_features.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Quick start examples (comment out main() to use these)
    # train_lstm_bidirectional()  # Recommended default
    # train_all_architectures()   # Compare all architectures
    # train_lstm_custom(sequence_length=120, epochs=50)  # Custom config

    # Command-line mode
    main()
