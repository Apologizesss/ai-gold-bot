"""
Backtesting Framework for Trading Model
========================================
Comprehensive backtesting with realistic trading costs, risk management,
and detailed performance metrics.

Features:
- Realistic spread, slippage, and commission
- Position sizing and risk management
- Multiple entry/exit strategies
- Walk-forward validation support
- Detailed performance metrics (Sharpe, drawdown, profit factor)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import joblib
from typing import Dict, List, Tuple, Optional


class TradingBacktest:
    """
    Comprehensive backtesting framework for trading strategies
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        data_path: str,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        spread_pips: float = 3.0,  # XAUUSD typical spread
        slippage_pips: float = 1.0,  # Realistic slippage
        commission_pct: float = 0.0,  # Commission % (if any)
        pip_value: float = 0.01,  # XAUUSD: $0.01 per pip for 0.01 lot
    ):
        """
        Initialize backtesting framework

        Args:
            model_path: Path to trained model (.pkl)
            scaler_path: Path to feature scaler (.pkl)
            data_path: Path to test data CSV
            initial_capital: Starting capital ($)
            risk_per_trade: Risk per trade (fraction of capital, e.g., 0.01 = 1%)
            spread_pips: Bid-ask spread in pips
            slippage_pips: Expected slippage in pips
            commission_pct: Commission as % of trade value
            pip_value: Value of 1 pip in $ (depends on lot size)
        """
        print("=" * 70)
        print("BACKTESTING FRAMEWORK")
        print("=" * 70)
        print()

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path

        # Capital and risk
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

        # Trading costs
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_pct = commission_pct
        self.pip_value = pip_value

        # Load model and scaler
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print()

        # Trading state
        self.equity_curve = []
        self.trades = []
        self.current_position = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare test data"""
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df):,} rows")

        # Ensure we have required columns
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                print(f"ERROR: Missing required column: {col}")
                return None

        # Handle timestamp/time column
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"])
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        else:
            print("ERROR: No timestamp/time column found!")
            return None

        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print()

        return df

    def calculate_position_size(self, capital: float, stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk management

        Args:
            capital: Current account equity
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Position size in lots
        """
        # Amount to risk in dollars
        risk_amount = capital * self.risk_per_trade

        # Stop loss in dollars
        stop_loss_dollars = stop_loss_pips * self.pip_value

        # Position size (simplified for XAUUSD)
        if stop_loss_dollars > 0:
            position_size = risk_amount / stop_loss_dollars
        else:
            position_size = 0.01  # Minimum

        # Limit to reasonable range (0.01 to 1.0 lots)
        position_size = max(0.01, min(1.0, position_size))

        return position_size

    def calculate_trading_costs(
        self, entry_price: float, position_size: float, direction: str
    ) -> float:
        """
        Calculate total trading costs (spread + slippage + commission)

        Args:
            entry_price: Entry price
            position_size: Position size in lots
            direction: 'long' or 'short'

        Returns:
            Total cost in dollars
        """
        # Spread cost (always paid on entry)
        spread_cost = self.spread_pips * self.pip_value * position_size

        # Slippage cost (random, but we use average)
        slippage_cost = self.slippage_pips * self.pip_value * position_size

        # Commission (if any)
        trade_value = entry_price * position_size
        commission = trade_value * self.commission_pct

        total_cost = spread_cost + slippage_cost + commission

        return total_cost

    def generate_signal(
        self,
        features: np.ndarray,
        current_price: float,
        support: float,
        resistance: float,
        confidence_threshold: float = 0.65,
    ) -> Dict:
        """
        Generate trading signal based on model prediction

        Args:
            features: Feature array for prediction
            current_price: Current market price
            support: Support level
            resistance: Resistance level
            confidence_threshold: Minimum probability to trade

        Returns:
            Signal dict with direction, confidence, stop_loss, take_profit
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get prediction
        prob_up = self.model.predict_proba(features_scaled)[0][1]

        # Calculate distances to S/R levels
        dist_to_support = abs(current_price - support) / current_price * 100
        dist_to_resistance = abs(current_price - resistance) / current_price * 100

        # Generate signal
        signal = {
            "direction": None,
            "probability": prob_up,
            "confidence": "NONE",
            "entry_price": current_price,
            "stop_loss": None,
            "take_profit": None,
            "stop_loss_pips": 0,
            "take_profit_pips": 0,
        }

        # Strategy: Support/Resistance bounce
        # LONG signal: Near support + high probability UP
        if dist_to_support < 0.5 and prob_up > confidence_threshold:
            signal["direction"] = "long"
            signal["confidence"] = "HIGH" if prob_up > 0.75 else "MEDIUM"

            # Stop loss: Below support (ATR-based, simplified to 50 pips)
            signal["stop_loss_pips"] = 50
            signal["stop_loss"] = current_price - (signal["stop_loss_pips"] * 0.01)

            # Take profit: 2x risk (100 pips)
            signal["take_profit_pips"] = 100
            signal["take_profit"] = current_price + (signal["take_profit_pips"] * 0.01)

        # SHORT signal: Near resistance + high probability DOWN
        elif dist_to_resistance < 0.5 and prob_up < (1 - confidence_threshold):
            signal["direction"] = "short"
            signal["confidence"] = "HIGH" if prob_up < 0.25 else "MEDIUM"

            # Stop loss: Above resistance
            signal["stop_loss_pips"] = 50
            signal["stop_loss"] = current_price + (signal["stop_loss_pips"] * 0.01)

            # Take profit: 2x risk
            signal["take_profit_pips"] = 100
            signal["take_profit"] = current_price - (signal["take_profit_pips"] * 0.01)

        return signal

    def simulate_trade(
        self,
        entry_bar: pd.Series,
        exit_bars: pd.DataFrame,
        signal: Dict,
        current_capital: float,
    ) -> Optional[Dict]:
        """
        Simulate a single trade with realistic execution

        Args:
            entry_bar: Bar where trade is entered
            exit_bars: Subsequent bars for exit simulation
            signal: Trading signal dict
            current_capital: Current account equity

        Returns:
            Trade result dict or None if no valid exit
        """
        direction = signal["direction"]
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        take_profit = signal["take_profit"]
        stop_loss_pips = signal["stop_loss_pips"]

        # Calculate position size
        position_size = self.calculate_position_size(current_capital, stop_loss_pips)

        # Calculate entry costs
        entry_costs = self.calculate_trading_costs(
            entry_price, position_size, direction
        )

        # Track trade
        trade = {
            "entry_time": entry_bar["time"],
            "entry_price": entry_price,
            "direction": direction,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "probability": signal["probability"],
            "confidence": signal["confidence"],
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pips": 0,
            "profit_loss": 0,
            "costs": entry_costs,
            "net_profit_loss": -entry_costs,  # Start with costs
            "bars_held": 0,
        }

        # Simulate bars until exit
        for i, (idx, bar) in enumerate(exit_bars.iterrows()):
            trade["bars_held"] = i + 1

            # Check for stop loss hit
            if direction == "long":
                if bar["low"] <= stop_loss:
                    trade["exit_time"] = bar["time"]
                    trade["exit_price"] = stop_loss
                    trade["exit_reason"] = "STOP_LOSS"
                    trade["pips"] = (stop_loss - entry_price) / 0.01
                    break
                # Check for take profit hit
                elif bar["high"] >= take_profit:
                    trade["exit_time"] = bar["time"]
                    trade["exit_price"] = take_profit
                    trade["exit_reason"] = "TAKE_PROFIT"
                    trade["pips"] = (take_profit - entry_price) / 0.01
                    break

            elif direction == "short":
                if bar["high"] >= stop_loss:
                    trade["exit_time"] = bar["time"]
                    trade["exit_price"] = stop_loss
                    trade["exit_reason"] = "STOP_LOSS"
                    trade["pips"] = (entry_price - stop_loss) / 0.01
                    break
                elif bar["low"] <= take_profit:
                    trade["exit_time"] = bar["time"]
                    trade["exit_price"] = take_profit
                    trade["exit_reason"] = "TAKE_PROFIT"
                    trade["pips"] = (entry_price - take_profit) / 0.01
                    break

            # Max hold time: 24 bars (24 hours for H1)
            if i >= 24:
                trade["exit_time"] = bar["time"]
                trade["exit_price"] = bar["close"]
                trade["exit_reason"] = "TIME_EXIT"
                if direction == "long":
                    trade["pips"] = (bar["close"] - entry_price) / 0.01
                else:
                    trade["pips"] = (entry_price - bar["close"]) / 0.01
                break

        # If no exit found, return None
        if trade["exit_price"] is None:
            return None

        # Calculate profit/loss
        trade["profit_loss"] = trade["pips"] * self.pip_value * position_size
        trade["net_profit_loss"] = trade["profit_loss"] - entry_costs

        return trade

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        confidence_threshold: float = 0.65,
    ) -> Dict:
        """
        Run full backtest on data

        Args:
            start_date: Start date (YYYY-MM-DD) or None for all data
            end_date: End date (YYYY-MM-DD) or None for all data
            confidence_threshold: Minimum probability to trade

        Returns:
            Backtest results dict
        """
        print("=" * 70)
        print("RUNNING BACKTEST")
        print("=" * 70)
        print()
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade * 100:.1f}%")
        print(f"Spread: {self.spread_pips} pips")
        print(f"Slippage: {self.slippage_pips} pips")
        print(f"Confidence Threshold: {confidence_threshold:.2f}")
        print()

        # Load data
        df = self.load_data()
        if df is None:
            return None

        # Filter by date if specified
        if start_date:
            df = df[df["time"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["time"] <= pd.to_datetime(end_date)]

        print(f"Backtesting on {len(df):,} bars")
        print()

        # Get feature columns (exclude metadata and non-numeric)
        exclude_cols = [
            "time",
            "timestamp",
            "target",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "timeframe",
            "tick_volume",
            "real_volume",
            "spread",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Get S/R columns if they exist
        has_sr = "support" in df.columns and "resistance" in df.columns

        # Initialize tracking
        current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [{"time": df.iloc[0]["time"], "equity": current_capital}]

        # Iterate through bars
        print("Simulating trades...")
        for i in range(len(df) - 25):  # Leave room for exit simulation
            bar = df.iloc[i]

            # Skip if we don't have enough data
            if pd.isna(bar[feature_cols]).any():
                continue

            # Get features
            features = bar[feature_cols].values

            # Get S/R levels
            support = bar["support"] if has_sr else bar["close"] * 0.995
            resistance = bar["resistance"] if has_sr else bar["close"] * 1.005

            # Generate signal
            signal = self.generate_signal(
                features, bar["close"], support, resistance, confidence_threshold
            )

            # If signal, simulate trade
            if signal["direction"] is not None:
                # Get next bars for exit simulation
                exit_bars = df.iloc[i + 1 : i + 26]

                # Simulate trade
                trade = self.simulate_trade(bar, exit_bars, signal, current_capital)

                if trade is not None:
                    # Update capital
                    current_capital += trade["net_profit_loss"]

                    # Record trade
                    self.trades.append(trade)

                    # Update equity curve
                    self.equity_curve.append(
                        {"time": trade["exit_time"], "equity": current_capital}
                    )

        print(f"Completed: {len(self.trades)} trades executed")
        print()

        # Calculate metrics
        metrics = self.calculate_metrics(current_capital)

        return metrics

    def calculate_metrics(self, final_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        print("=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print()

        if len(self.trades) == 0:
            print("No trades executed!")
            return {}

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["net_profit_loss"] > 0])
        losing_trades = len(trades_df[trades_df["net_profit_loss"] < 0])
        win_rate = winning_trades / total_trades * 100

        total_profit = trades_df[trades_df["net_profit_loss"] > 0][
            "net_profit_loss"
        ].sum()
        total_loss = abs(
            trades_df[trades_df["net_profit_loss"] < 0]["net_profit_loss"].sum()
        )
        net_profit = final_capital - self.initial_capital
        return_pct = (net_profit / self.initial_capital) * 100

        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Average metrics
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        avg_trade = net_profit / total_trades

        # Equity curve metrics
        equity_df = pd.DataFrame(self.equity_curve)
        peak = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified - assuming daily returns)
        equity_df["returns"] = equity_df["equity"].pct_change()
        sharpe = (
            equity_df["returns"].mean() / equity_df["returns"].std() * np.sqrt(252)
            if equity_df["returns"].std() > 0
            else 0
        )

        # Exit reason breakdown
        exit_reasons = trades_df["exit_reason"].value_counts()

        # Print results
        print("OVERALL PERFORMANCE")
        print("-" * 70)
        print(f"Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"Final Capital:      ${final_capital:,.2f}")
        print(f"Net Profit/Loss:    ${net_profit:,.2f} ({return_pct:+.2f}%)")
        print(f"Total Trades:       {total_trades}")
        print()

        print("TRADE STATISTICS")
        print("-" * 70)
        print(f"Winning Trades:     {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades:      {losing_trades} ({100 - win_rate:.1f}%)")
        print(f"Win Rate:           {win_rate:.2f}%")
        print()
        print(f"Total Profit:       ${total_profit:,.2f}")
        print(f"Total Loss:         ${total_loss:,.2f}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print()
        print(f"Average Win:        ${avg_win:.2f}")
        print(f"Average Loss:       ${avg_loss:.2f}")
        print(f"Average Trade:      ${avg_trade:.2f}")
        print()

        print("RISK METRICS")
        print("-" * 70)
        print(f"Max Drawdown:       {max_drawdown:.2f}%")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Total Costs:        ${trades_df['costs'].sum():,.2f}")
        print()

        print("EXIT REASONS")
        print("-" * 70)
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"{reason:15s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Compile metrics dict
        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "net_profit": net_profit,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_costs": trades_df["costs"].sum(),
            "exit_reasons": exit_reasons.to_dict(),
        }

        return metrics

    def plot_results(self, output_dir: str = "results/backtest"):
        """Generate comprehensive plots"""
        print("Generating plots...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(self.trades) == 0:
            print("  No trades to plot!")
            return

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Equity curve
        axes[0, 0].plot(equity_df["time"], equity_df["equity"], linewidth=2)
        axes[0, 0].axhline(
            self.initial_capital, color="gray", linestyle="--", alpha=0.5
        )
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Equity ($)")
        axes[0, 0].set_title("Equity Curve")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Drawdown
        peak = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - peak) / peak * 100
        axes[0, 1].fill_between(equity_df["time"], drawdown, 0, alpha=0.3, color="red")
        axes[0, 1].plot(equity_df["time"], drawdown, color="red", linewidth=1)
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Trade P&L distribution
        axes[1, 0].hist(
            trades_df["net_profit_loss"], bins=30, edgecolor="black", alpha=0.7
        )
        axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Net P&L ($)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Trade P&L Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Cumulative trade results
        trades_df["cumulative_pl"] = trades_df["net_profit_loss"].cumsum()
        axes[1, 1].plot(range(len(trades_df)), trades_df["cumulative_pl"], linewidth=2)
        axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Trade Number")
        axes[1, 1].set_ylabel("Cumulative P&L ($)")
        axes[1, 1].set_title("Cumulative Trade Results")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "backtest_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Plots saved: {plot_path}")

    def save_results(self, metrics: Dict, output_dir: str = "results/backtest"):
        """Save backtest results"""
        print()
        print("Saving results...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trades
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_path = output_dir / "trades.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Trades saved: {trades_path}")

        # Save equity curve
        if len(self.equity_curve) > 0:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_path = output_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            print(f"  Equity curve saved: {equity_path}")

        # Save metrics
        import json

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved: {metrics_path}")
        print()


def main():
    """Run backtest from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest trading model")
    parser.add_argument("--model", "-m", required=True, help="Path to model (.pkl)")
    parser.add_argument("--scaler", "-s", required=True, help="Path to scaler (.pkl)")
    parser.add_argument("--data", "-d", required=True, help="Path to test data CSV")
    parser.add_argument(
        "--capital", "-c", type=float, default=10000.0, help="Initial capital ($)"
    )
    parser.add_argument(
        "--risk", "-r", type=float, default=0.01, help="Risk per trade (0.01 = 1%)"
    )
    parser.add_argument("--spread", type=float, default=3.0, help="Spread in pips")
    parser.add_argument("--slippage", type=float, default=1.0, help="Slippage in pips")
    parser.add_argument(
        "--confidence", type=float, default=0.65, help="Confidence threshold"
    )
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output", "-o", default="results/backtest", help="Output directory"
    )

    args = parser.parse_args()

    # Create backtester
    backtester = TradingBacktest(
        model_path=args.model,
        scaler_path=args.scaler,
        data_path=args.data,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        spread_pips=args.spread,
        slippage_pips=args.slippage,
    )

    # Run backtest
    metrics = backtester.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        confidence_threshold=args.confidence,
    )

    # Plot and save results
    if metrics:
        backtester.plot_results(args.output)
        backtester.save_results(metrics, args.output)

        print("=" * 70)
        print("BACKTEST COMPLETE!")
        print("=" * 70)
        print()
        print(f"Results saved to: {args.output}")
        print()


if __name__ == "__main__":
    main()
