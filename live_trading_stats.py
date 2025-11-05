"""
Live Trading Statistics Viewer
==============================
View performance statistics from live trading logs
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def load_state():
    """Load current trading state"""
    state_file = Path("logs/live_trading/live_trading_state.json")

    if not state_file.exists():
        print("No trading state found. Have you started live trading yet?")
        return None

    with open(state_file, "r") as f:
        return json.load(f)


def calculate_statistics(state):
    """Calculate trading statistics"""

    closed_trades = state.get("closed_trades", [])

    if not closed_trades:
        print("\nNo closed trades yet!")
        return None

    # Basic stats
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in closed_trades if t.get("pnl", 0) < 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

    # P&L stats
    total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    total_wins = sum(t.get("pnl", 0) for t in winning_trades)
    total_losses = abs(sum(t.get("pnl", 0) for t in losing_trades))

    avg_win = total_wins / num_wins if num_wins > 0 else 0
    avg_loss = total_losses / num_losses if num_losses > 0 else 0

    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Find largest trades
    all_pnls = [t.get("pnl", 0) for t in closed_trades]
    largest_win = max(all_pnls) if all_pnls else 0
    largest_loss = min(all_pnls) if all_pnls else 0

    # Account stats
    current_balance = state.get("account_balance", 0)
    initial_balance = state.get("initial_balance", current_balance)

    return {
        "total_trades": total_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "current_balance": current_balance,
        "initial_balance": initial_balance,
        "closed_trades": closed_trades,
    }


def display_summary(stats):
    """Display trading summary"""

    print("\n" + "=" * 70)
    print("LIVE TRADING PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Overall Performance
    print("\n" + "-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    print(f"Total Trades:        {stats['total_trades']}")
    print(f"Winning Trades:      {stats['num_wins']} ({stats['win_rate']:.1f}%)")
    print(f"Losing Trades:       {stats['num_losses']}")
    print()

    # P&L
    print("-" * 70)
    print("PROFIT & LOSS")
    print("-" * 70)

    pnl_color = "+" if stats["total_pnl"] > 0 else ""
    print(f"Net P&L:             {pnl_color}${stats['total_pnl']:.2f}")
    print(f"Total Wins:          +${stats['total_wins']:.2f}")
    print(f"Total Losses:        -${stats['total_losses']:.2f}")
    print(f"Average Win:         +${stats['avg_win']:.2f}")
    print(f"Average Loss:        -${stats['avg_loss']:.2f}")
    print(f"Largest Win:         +${stats['largest_win']:.2f}")
    print(f"Largest Loss:        ${stats['largest_loss']:.2f}")
    print()

    # Performance Metrics
    print("-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)

    print(f"Win Rate:            {stats['win_rate']:.1f}%")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")

    # Evaluate performance
    if stats["win_rate"] >= 55 and stats["profit_factor"] >= 1.5:
        status = "[OK] GOOD - Keep trading"
    elif stats["win_rate"] >= 50 and stats["profit_factor"] >= 1.2:
        status = "[Warning] ACCEPTABLE - Monitor closely"
    else:
        status = "[Error] POOR - Consider stopping"

    print(f"Status:              {status}")
    print()

    # Account
    print("-" * 70)
    print("ACCOUNT")
    print("-" * 70)

    print(f"Initial Balance:     ${stats['initial_balance']:.2f}")
    print(f"Current Balance:     ${stats['current_balance']:.2f}")

    roi = (
        (
            (stats["current_balance"] - stats["initial_balance"])
            / stats["initial_balance"]
            * 100
        )
        if stats["initial_balance"] > 0
        else 0
    )
    roi_color = "+" if roi > 0 else ""
    print(f"Return on Investment: {roi_color}{roi:.2f}%")
    print()


def display_recent_trades(trades, n=10):
    """Display recent trades"""

    print("-" * 70)
    print(f"LAST {n} TRADES")
    print("-" * 70)
    print()

    recent = trades[-n:] if len(trades) > n else trades

    for i, trade in enumerate(reversed(recent), 1):
        direction = trade.get("direction", "UNKNOWN")
        entry = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl = trade.get("pnl", 0)

        result = "WIN" if pnl > 0 else "LOSS"
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"${pnl:.2f}"

        entry_time = trade.get("entry_time", "Unknown")
        exit_time = trade.get("exit_time", "Unknown")

        print(
            f"{i:2d}. {direction:5s} | Entry: {entry:.2f} | Exit: {exit_price:.2f} | {pnl_str:>10s} | {result}"
        )
        print(f"    In: {entry_time}  Out: {exit_time}")
        print()


def export_to_csv(trades, filename="trading_history.csv"):
    """Export trades to CSV"""

    if not trades:
        print("No trades to export")
        return

    df = pd.DataFrame(trades)

    # Select relevant columns
    columns = [
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "pnl",
        "stop_loss",
        "take_profit",
    ]

    available_columns = [col for col in columns if col in df.columns]
    df_export = df[available_columns]

    output_path = Path("logs/live_trading") / filename
    df_export.to_csv(output_path, index=False)

    print(f"\n[OK] Exported {len(trades)} trades to: {output_path}")


def main():
    """Main function"""

    print("=" * 70)
    print("LIVE TRADING STATISTICS VIEWER")
    print("=" * 70)

    # Load state
    state = load_state()
    if not state:
        return

    # Calculate statistics
    stats = calculate_statistics(state)
    if not stats:
        return

    # Display summary
    display_summary(stats)

    # Display recent trades
    display_recent_trades(stats["closed_trades"], n=10)

    # Export option
    print("-" * 70)
    print("EXPORT OPTIONS")
    print("-" * 70)

    try:
        export = input("\nExport to CSV? (y/n): ").strip().lower()
        if export == "y":
            filename = input("Filename (default: trading_history.csv): ").strip()
            if not filename:
                filename = "trading_history.csv"
            export_to_csv(stats["closed_trades"], filename)
    except KeyboardInterrupt:
        print("\n\nExport cancelled")

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
