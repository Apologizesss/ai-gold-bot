"""
Paper Trading System with Signal Logging
=========================================
Run inference on live data and log all signals for performance tracking.
This is the bridge between backtesting and live trading.

Features:
- Fetch live data from MT5
- Run inference every hour
- Log all signals to CSV
- Track hypothetical P&L
- Generate performance reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import MetaTrader5 as mt5

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference_pipeline import TradingInference


class PaperTrading:
    """
    Paper trading system with signal logging
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "H1",
        model_path: str = "results/xgboost/xgboost_model.pkl",
        scaler_path: str = "results/xgboost/xgboost_scaler.pkl",
        confidence_threshold: float = 0.70,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        log_dir: str = "logs/paper_trading",
    ):
        """
        Initialize paper trading system

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (H1, H4, D1, etc.)
            model_path: Path to trained model
            scaler_path: Path to scaler
            confidence_threshold: Minimum confidence to take trade
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (fraction)
            log_dir: Directory to save logs
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Trading state
        self.current_capital = initial_capital
        self.open_positions = []
        self.closed_trades = []

        # Initialize inference pipeline
        print("=" * 70)
        print("PAPER TRADING SYSTEM")
        print("=" * 70)
        print()
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Risk per Trade: {risk_per_trade * 100:.1f}%")
        print()

        self.inference = TradingInference(
            symbol=symbol,
            timeframe=timeframe,
            model_path=model_path,
            scaler_path=scaler_path,
            confidence_threshold=confidence_threshold,
        )

        # Initialize MT5
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")

        print("✓ MT5 connected")
        print("✓ Paper trading system ready")
        print()

    def fetch_data(self, bars: int = 300):
        """
        Fetch data from MT5

        Args:
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data
        """
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        tf = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        return df

    def run_inference(self):
        """
        Run inference on latest data

        Returns:
            Signal dictionary or None
        """
        # Fetch data
        df = self.fetch_data()
        if df is None:
            print("Failed to fetch data")
            return None

        # Calculate indicators
        df_with_features = self.inference.calculate_technical_indicators(df)
        if df_with_features is None:
            return None

        # Prepare features
        features = self.inference.prepare_features_for_prediction(df_with_features)
        if features is None:
            return None

        # Make prediction
        prob_up, predicted_class = self.inference.make_prediction(features)
        if prob_up is None:
            return None

        # Generate signal
        signal = self.inference.generate_signal(df_with_features, prob_up)

        # Add current market info
        latest = df.iloc[-1]
        signal["current_price"] = float(latest["close"])
        signal["current_time"] = latest["time"]
        signal["prob_up"] = float(prob_up)
        signal["predicted_class"] = int(predicted_class)

        return signal

    def check_open_positions(self):
        """
        Check and update open positions (simulate trade execution)
        """
        if not self.open_positions:
            return

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        current_price = tick.bid

        positions_to_close = []

        for i, pos in enumerate(self.open_positions):
            # Check stop loss
            if pos["direction"] == "LONG":
                if current_price <= pos["stop_loss"]:
                    # Stop loss hit
                    pnl = -pos["risk_amount"]
                    positions_to_close.append((i, "STOP_LOSS", pnl))
                elif current_price >= pos["take_profit"]:
                    # Take profit hit
                    pnl = pos["risk_amount"] * pos["reward_risk_ratio"]
                    positions_to_close.append((i, "TAKE_PROFIT", pnl))

            elif pos["direction"] == "SHORT":
                if current_price >= pos["stop_loss"]:
                    # Stop loss hit
                    pnl = -pos["risk_amount"]
                    positions_to_close.append((i, "STOP_LOSS", pnl))
                elif current_price <= pos["take_profit"]:
                    # Take profit hit
                    pnl = pos["risk_amount"] * pos["reward_risk_ratio"]
                    positions_to_close.append((i, "TAKE_PROFIT", pnl))

        # Close positions
        for idx, exit_reason, pnl in reversed(positions_to_close):
            pos = self.open_positions.pop(idx)
            pos["exit_time"] = datetime.now()
            pos["exit_price"] = current_price
            pos["exit_reason"] = exit_reason
            pos["pnl"] = pnl
            pos["final_capital"] = self.current_capital + pnl

            self.current_capital += pnl
            self.closed_trades.append(pos)

            print(f"  Position closed: {pos['direction']} {exit_reason}")
            print(f"  P&L: ${pnl:+.2f} | Capital: ${self.current_capital:,.2f}")
            print()

    def process_signal(self, signal):
        """
        Process trading signal and open position if valid

        Args:
            signal: Signal dictionary
        """
        if signal is None:
            return

        # Check if signal is valid
        if not signal.get("should_trade", False):
            print("  No entry signal")
            return

        # Check if we already have a position
        if len(self.open_positions) >= 1:
            print("  Already have open position - skipping")
            return

        direction = signal.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            print("  Neutral signal - no trade")
            return

        # Calculate position size
        risk_amount = self.current_capital * self.risk_per_trade
        entry_price = signal["current_price"]
        stop_loss = signal.get("stop_loss", entry_price * 0.99)
        take_profit = signal.get("take_profit", entry_price * 1.02)

        # Create position
        position = {
            "entry_time": datetime.now(),
            "entry_price": entry_price,
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_amount": risk_amount,
            "reward_risk_ratio": signal.get("reward_risk_ratio", 2.0),
            "prob_up": signal["prob_up"],
            "confidence": signal.get("confidence", "MEDIUM"),
        }

        self.open_positions.append(position)

        print(f"  📈 Position opened: {direction}")
        print(f"  Entry: {entry_price:.2f}")
        print(f"  SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        print(f"  Risk: ${risk_amount:.2f}")
        print()

    def log_signal(self, signal):
        """
        Log signal to CSV file

        Args:
            signal: Signal dictionary
        """
        if signal is None:
            return

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "current_time": signal.get("current_time", ""),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": signal.get("current_price", 0),
            "prob_up": signal.get("prob_up", 0),
            "direction": signal.get("direction", "NEUTRAL"),
            "action": signal.get("action", "NONE"),
            "confidence": signal.get("confidence", "NONE"),
            "entry_price": signal.get("entry_price", 0),
            "stop_loss": signal.get("stop_loss", 0),
            "take_profit": signal.get("take_profit", 0),
            "support": signal.get("support", 0),
            "resistance": signal.get("resistance", 0),
        }

        # Append to CSV
        log_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m')}.csv"
        df_log = pd.DataFrame([log_entry])

        if log_file.exists():
            df_log.to_csv(log_file, mode="a", header=False, index=False)
        else:
            df_log.to_csv(log_file, index=False)

    def log_trade(self, trade):
        """
        Log closed trade to CSV

        Args:
            trade: Trade dictionary
        """
        log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m')}.csv"

        # Prepare trade entry
        trade_entry = {
            "entry_time": trade["entry_time"],
            "exit_time": trade.get("exit_time", ""),
            "direction": trade["direction"],
            "entry_price": trade["entry_price"],
            "exit_price": trade.get("exit_price", 0),
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "exit_reason": trade.get("exit_reason", ""),
            "pnl": trade.get("pnl", 0),
            "final_capital": trade.get("final_capital", self.current_capital),
            "risk_amount": trade["risk_amount"],
            "confidence": trade.get("confidence", "MEDIUM"),
        }

        df_trade = pd.DataFrame([trade_entry])

        if log_file.exists():
            df_trade.to_csv(log_file, mode="a", header=False, index=False)
        else:
            df_trade.to_csv(log_file, index=False)

    def generate_report(self):
        """
        Generate performance report
        """
        if not self.closed_trades:
            print("No closed trades yet")
            return

        print("=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        print()

        total_trades = len(self.closed_trades)
        wins = len([t for t in self.closed_trades if t["pnl"] > 0])
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0

        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        total_profit = sum(t["pnl"] for t in self.closed_trades if t["pnl"] > 0)
        total_loss = abs(sum(t["pnl"] for t in self.closed_trades if t["pnl"] < 0))

        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        total_return = (
            (self.current_capital - self.initial_capital) / self.initial_capital * 100
        )

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {wins} ({win_rate * 100:.1f}%)")
        print(f"Losing Trades: {losses}")
        print()
        print(f"Total P&L: ${total_pnl:+,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print()
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Current Capital: ${self.current_capital:,.2f}")
        print()

        if self.open_positions:
            print(f"Open Positions: {len(self.open_positions)}")
            for pos in self.open_positions:
                print(f"  {pos['direction']} @ {pos['entry_price']:.2f}")
            print()

    def run(self, duration_minutes: int = None, check_interval_seconds: int = 60):
        """
        Run paper trading

        Args:
            duration_minutes: Duration to run (None = forever)
            check_interval_seconds: How often to check positions
        """
        print("=" * 70)
        print("STARTING PAPER TRADING")
        print("=" * 70)
        print()
        print(f"Check interval: {check_interval_seconds}s")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print("Duration: Indefinite (press Ctrl+C to stop)")
        print()

        start_time = datetime.now()
        last_inference_time = datetime.now() - timedelta(hours=2)

        try:
            iteration = 0
            while True:
                iteration += 1
                current_time = datetime.now()

                print(
                    f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Check #{iteration}"
                )

                # Check if we need to run inference (once per hour)
                time_since_last = (current_time - last_inference_time).total_seconds()
                if time_since_last >= 3600:  # 1 hour
                    print("  Running inference...")
                    signal = self.run_inference()

                    if signal:
                        print(f"  Signal: {signal.get('action', 'NONE')}")
                        self.log_signal(signal)
                        self.process_signal(signal)
                    else:
                        print("  No signal generated")

                    last_inference_time = current_time

                # Check open positions
                if self.open_positions:
                    print(f"  Checking {len(self.open_positions)} open position(s)...")
                    self.check_open_positions()

                    # Log closed trades
                    if self.closed_trades:
                        for trade in self.closed_trades[-10:]:  # Log last 10
                            self.log_trade(trade)

                # Check if we should stop
                if duration_minutes:
                    elapsed = (current_time - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print()
                        print("Duration reached - stopping")
                        break

                # Wait before next check
                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print()
            print("Stopped by user")

        finally:
            # Generate final report
            print()
            self.generate_report()

            # Save final state
            self.save_state()

            # Cleanup
            mt5.shutdown()
            print()
            print("✓ Paper trading stopped")

    def save_state(self):
        """
        Save current state to JSON
        """
        state_file = self.log_dir / "paper_trading_state.json"

        state = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_trades": len(self.closed_trades),
            "open_positions": len(self.open_positions),
            "closed_trades": self.closed_trades,
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"✓ State saved: {state_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Paper trading system")
    parser.add_argument(
        "--symbol", "-s", default="XAUUSD", help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--timeframe", "-t", default="H1", help="Timeframe (default: H1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold (default: 0.70)",
    )
    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=float,
        default=0.01,
        help="Risk per trade (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        help="Duration in minutes (default: run forever)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Create paper trading system
    paper_trader = PaperTrading(
        symbol=args.symbol,
        timeframe=args.timeframe,
        confidence_threshold=args.threshold,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
    )

    # Run paper trading
    paper_trader.run(
        duration_minutes=args.duration, check_interval_seconds=args.interval
    )


if __name__ == "__main__":
    main()
