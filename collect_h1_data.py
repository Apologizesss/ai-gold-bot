"""
Collect H1 (1 Hour) Data for XAUUSD
===================================
Collects 2 years of hourly data for better trend detection
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_collection.mt5_collector import MT5Collector


def main():
    print("=" * 70)
    print("COLLECTING H1 DATA FOR XAUUSD (5 YEARS)")
    print("=" * 70)
    print()
    print("Timeframe: H1 (1 Hour)")
    print("Period: 1825 days (5 years)")
    print("Symbol: XAUUSD")
    print()

    # Initialize collector
    collector = MT5Collector(symbol="XAUUSD", timeframe="H1", output_dir="data/raw")

    # Connect to MT5
    print("Connecting to MT5...")
    if not collector.initialize():
        print("ERROR: Failed to initialize MT5")
        print()
        print("Troubleshooting:")
        print("  1. Make sure MT5 is running")
        print("  2. Make sure you're logged in")
        print("  3. Enable Algo Trading (should show green)")
        return

    print("SUCCESS: Connected to MT5")

    try:
        # Check symbol
        print("Checking XAUUSD symbol...")
        if not collector.check_symbol():
            print("ERROR: XAUUSD symbol not available")
            return

        print("SUCCESS: Symbol available")

        # Collect data
        print()
        print("Collecting H1 data (5 years - this may take 60-90 seconds)...")
        df = collector.collect_historical_data(days=1825)

        if df is None:
            print("ERROR: Failed to collect data")
            return

        print(f"SUCCESS: Collected {len(df):,} bars")

        # Validate data
        print()
        print("Validating data...")
        is_valid, issues = collector.validate_data(df)

        if not is_valid:
            print("WARNING: Data validation issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("SUCCESS: Data validation passed")

        # Get summary
        summary = collector.get_data_summary(df)
        print()
        print("Data Summary:")
        print(f"   Total bars: {summary['total_bars']:,}")
        print(f"   Date range: {summary['date_range']}")
        print(
            f"   Price range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}"
        )

        # Save data
        print()
        print("Saving data...")
        filename = f"XAUUSD_H1_5Y_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = collector.save_data(df, filename)

        print(f"SUCCESS: Data saved to: {filepath}")
        print()
        print("=" * 70)
        print("H1 DATA COLLECTION COMPLETE (5 YEARS)!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(
            "  1. Create features: python create_advanced_features.py --input <filepath>"
        )
        print("  2. Add target: python add_target_column.py --data-path <filepath>")
        print("  3. Train models")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        collector.shutdown()
        print("Disconnected from MT5")


if __name__ == "__main__":
    main()
