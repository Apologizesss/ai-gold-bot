"""
Compare Performance: Paper Trading vs Live Trading
===================================================
Compare performance metrics between paper and live trading
"""

import json
from pathlib import Path
from datetime import datetime


def load_paper_trading_state():
    """Load paper trading state"""
    state_file = Path("logs/paper_trading/paper_trading_state.json")

    if not state_file.exists():
        return None

    with open(state_file, "r") as f:
        return json.load(f)


def load_live_trading_state():
    """Load live trading state"""
    state_file = Path("logs/live_trading/live_trading_state.json")

    if not state_file.exists():
        return None

    with open(state_file, "r") as f:
        return json.load(f)


def calculate_metrics(state, name="Trading"):
    """Calculate trading metrics from state"""

    closed_trades = state.get("closed_trades", [])

    if not closed_trades:
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

    # Account stats
    initial_balance = state.get("initial_capital", state.get("initial_balance", 10000))
    current_balance = state.get(
        "balance", state.get("account_balance", initial_balance)
    )

    roi = (
        ((current_balance - initial_balance) / initial_balance * 100)
        if initial_balance > 0
        else 0
    )

    return {
        "name": name,
        "total_trades": total_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "initial_balance": initial_balance,
        "current_balance": current_balance,
        "roi": roi,
    }


def compare_metrics(paper_metrics, live_metrics):
    """Compare two sets of metrics"""

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: PAPER TRADING vs LIVE TRADING")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Overall comparison
    print("\n" + "-" * 80)
    print("OVERVIEW")
    print("-" * 80)
    print(
        f"{'Metric':<25s} | {'Paper Trading':>15s} | {'Live Trading':>15s} | {'Difference':>15s}"
    )
    print("-" * 80)

    # Total trades
    print(
        f"{'Total Trades':<25s} | {paper_metrics['total_trades']:>15d} | {live_metrics['total_trades']:>15d} | {live_metrics['total_trades'] - paper_metrics['total_trades']:>15d}"
    )

    # Win rate
    paper_wr = paper_metrics["win_rate"]
    live_wr = live_metrics["win_rate"]
    diff_wr = live_wr - paper_wr
    print(
        f"{'Win Rate':<25s} | {paper_wr:>14.1f}% | {live_wr:>14.1f}% | {diff_wr:>+14.1f}%"
    )

    # Profit factor
    paper_pf = paper_metrics["profit_factor"]
    live_pf = live_metrics["profit_factor"]
    diff_pf = live_pf - paper_pf
    print(
        f"{'Profit Factor':<25s} | {paper_pf:>15.2f} | {live_pf:>15.2f} | {diff_pf:>+15.2f}"
    )

    # Average win
    paper_aw = paper_metrics["avg_win"]
    live_aw = live_metrics["avg_win"]
    diff_aw = live_aw - paper_aw
    print(
        f"{'Average Win':<25s} | ${paper_aw:>14.2f} | ${live_aw:>14.2f} | ${diff_aw:>+14.2f}"
    )

    # Average loss
    paper_al = paper_metrics["avg_loss"]
    live_al = live_metrics["avg_loss"]
    diff_al = live_al - paper_al
    print(
        f"{'Average Loss':<25s} | ${paper_al:>14.2f} | ${live_al:>14.2f} | ${diff_al:>+14.2f}"
    )

    # ROI
    paper_roi = paper_metrics["roi"]
    live_roi = live_metrics["roi"]
    diff_roi = live_roi - paper_roi
    print(
        f"{'ROI':<25s} | {paper_roi:>14.2f}% | {live_roi:>14.2f}% | {diff_roi:>+14.2f}%"
    )

    print()

    # Analysis
    print("-" * 80)
    print("ANALYSIS")
    print("-" * 80)

    # Win rate comparison
    if abs(diff_wr) <= 5:
        wr_status = "[OK] Similar performance"
    elif diff_wr > 5:
        wr_status = "[OK] Live trading performing BETTER"
    else:
        wr_status = "[Warning] Live trading performing WORSE"
    print(f"Win Rate:        {wr_status}")

    # Profit factor comparison
    if abs(diff_pf) <= 0.3:
        pf_status = "[OK] Similar performance"
    elif diff_pf > 0.3:
        pf_status = "[OK] Live trading performing BETTER"
    else:
        pf_status = "[Warning] Live trading performing WORSE"
    print(f"Profit Factor:   {pf_status}")

    # ROI comparison
    if abs(diff_roi) <= 2:
        roi_status = "[OK] Similar performance"
    elif diff_roi > 2:
        roi_status = "[OK] Live trading performing BETTER"
    else:
        roi_status = "[Warning] Live trading performing WORSE"
    print(f"ROI:             {roi_status}")

    # Overall assessment
    print()
    print("-" * 80)
    print("OVERALL ASSESSMENT")
    print("-" * 80)

    warnings = []

    if live_wr < paper_wr - 5:
        warnings.append("Live win rate significantly lower than paper")

    if live_pf < paper_pf - 0.3:
        warnings.append("Live profit factor significantly lower than paper")

    if live_roi < paper_roi - 5:
        warnings.append("Live ROI significantly lower than paper")

    if live_wr < 50:
        warnings.append("Live win rate below 50% (losing more than winning)")

    if live_pf < 1.2:
        warnings.append("Live profit factor below 1.2 (barely profitable)")

    if warnings:
        print("\n[Warning] ISSUES DETECTED:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print("\nRECOMMENDATION: Review trading strategy or stop live trading")
    else:
        print("\n[OK] Live trading performance is acceptable")
        print("Continue monitoring and compare regularly")

    # Slippage analysis
    print()
    print("-" * 80)
    print("SLIPPAGE & COSTS ANALYSIS")
    print("-" * 80)

    slippage = paper_roi - live_roi
    if abs(slippage) <= 1:
        slippage_status = "[OK] Minimal slippage/costs"
    elif abs(slippage) <= 3:
        slippage_status = "[Note] Moderate slippage/costs (expected)"
    else:
        slippage_status = "[Warning] High slippage/costs (investigate)"

    print(f"Estimated Slippage: {slippage:.2f}%")
    print(f"Status:             {slippage_status}")
    print()
    print("Note: Slippage includes spreads, commissions, and execution differences")

    print()


def display_individual_metrics(metrics):
    """Display metrics for a single system"""

    print(f"\n{metrics['name'].upper()}")
    print("-" * 40)
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Wins/Losses:       {metrics['num_wins']}/{metrics['num_losses']}")
    print(f"Win Rate:          {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"Average Win:       ${metrics['avg_win']:.2f}")
    print(f"Average Loss:      ${metrics['avg_loss']:.2f}")
    print(f"Net P&L:           ${metrics['total_pnl']:.2f}")
    print(f"ROI:               {metrics['roi']:.2f}%")
    print(
        f"Balance:           ${metrics['initial_balance']:.2f} → ${metrics['current_balance']:.2f}"
    )


def main():
    """Main function"""

    print("=" * 80)
    print("PERFORMANCE COMPARISON TOOL")
    print("=" * 80)

    # Load paper trading
    paper_state = load_paper_trading_state()
    if not paper_state:
        print("\n[Error] Paper trading state not found!")
        print("Location: logs/paper_trading/paper_trading_state.json")
        print("\nHave you run paper trading yet?")
        return

    # Load live trading
    live_state = load_live_trading_state()
    if not live_state:
        print("\n[Error] Live trading state not found!")
        print("Location: logs/live_trading/live_trading_state.json")
        print("\nHave you run live trading yet?")
        return

    # Calculate metrics
    paper_metrics = calculate_metrics(paper_state, "Paper Trading")
    live_metrics = calculate_metrics(live_state, "Live Trading")

    if not paper_metrics:
        print("\n[Error] No paper trading data available")
        return

    if not live_metrics:
        print("\n[Error] No live trading data available")
        return

    # Display individual metrics
    display_individual_metrics(paper_metrics)
    display_individual_metrics(live_metrics)

    # Compare
    compare_metrics(paper_metrics, live_metrics)

    print("\n" + "=" * 80)
    print("END OF COMPARISON")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
