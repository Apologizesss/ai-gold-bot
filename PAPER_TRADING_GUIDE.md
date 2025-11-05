# Paper Trading Guide

## 🎯 Overview

This guide explains how to use the paper trading system to test your trained XGBoost model in simulated real-time trading conditions **without risking real money**.

Paper trading connects to MetaTrader 5 to get live market data, makes predictions using your trained model, and simulates trades with virtual money.

---

## ⚠️ Prerequisites

### 1. Trained Model
You must have a trained XGBoost model in `results/xgboost/`:
- `xgboost_model.pkl` - The trained model
- `xgboost_scaler.pkl` - Feature scaler
- `feature_names.txt` - List of features used during training (96 features)

### 2. MetaTrader 5
- MT5 must be installed and running
- You must be logged into a demo or live account
- Symbol `XAUUSD` must be available in your broker

### 3. Feature Mismatch Fix Applied
The system now correctly loads the 96 features used during training. If you see errors about feature count mismatch, run:
```bash
python extract_feature_names.py
```

---

## 🚀 Quick Start

### Run Paper Trading

```bash
python paper_trading.py
```

Or on Windows:
```bash
python paper_trading.py
```

### What Happens

1. **Connects to MT5** - Fetches live market data
2. **Loads Model** - Loads your trained XGBoost model and scaler
3. **Starts Loop** - Checks market every 60 seconds
4. **Makes Predictions** - Uses live data to predict UP/DOWN
5. **Simulates Trades** - Opens/closes virtual positions
6. **Tracks Performance** - Logs all trades and P&L

---

## 📊 Configuration

Edit these settings in `paper_trading.py`:

```python
# Trading parameters
SYMBOL = "XAUUSD"                # Trading symbol
TIMEFRAME = "H1"                 # H1 = 1-hour candles
CONFIDENCE_THRESHOLD = 0.7       # Minimum confidence to trade (70%)
INITIAL_CAPITAL = 10000.0        # Starting capital ($10,000)
RISK_PER_TRADE = 0.01           # Risk 1% of capital per trade
CHECK_INTERVAL = 60              # Check every 60 seconds
```

### Key Parameters

- **CONFIDENCE_THRESHOLD**: Higher = fewer but more confident trades
  - 0.7 = Very selective (recommended)
  - 0.6 = Moderate
  - 0.5 = Trade every signal (not recommended)

- **RISK_PER_TRADE**: Percentage of capital to risk per trade
  - 0.01 = 1% (conservative, recommended)
  - 0.02 = 2% (moderate)
  - 0.03 = 3% (aggressive)

---

## 📈 Understanding the Output

### On Each Check

```
[2025-11-05 12:33:39] Check #1
  Running inference...
  Calculating technical indicators...
  [OK] Extracted 96 features

  🔍 Signal Evaluation:
    Prob UP: 0.2773 (27.73%)
    Prob DOWN: 0.7227 (72.27%)
    Threshold: 0.70
    Current Price: 3974.35
    ✅ SHORT Signal: 27.73% < 30.00%
  Signal: SHORT
  📈 Position opened: SHORT
  Entry: 3974.35
  SL: 3990.02 | TP: 3943.00
  Risk: $100.00
```

### Signal Types

- **LONG**: Model predicts price will go UP with high confidence
- **SHORT**: Model predicts price will go DOWN with high confidence
- **NONE**: No clear signal or confidence too low

### Trade Outcomes

```
  ❌ Position closed: SHORT at 3980.00 (Hit SL)
  Loss: -$60.23
  Balance: $9,939.77
```

or

```
  ✅ Position closed: SHORT at 3940.50 (Hit TP)
  Profit: +$150.45
  Balance: $10,150.45
```

---

## 📁 Output Files

### 1. Trade Logs
Location: `logs/paper_trading/paper_trading_YYYYMMDD.log`

Contains detailed logs of all activity:
- Timestamps
- Predictions
- Trade entries/exits
- P&L calculations

### 2. State File
Location: `logs/paper_trading/paper_trading_state.json`

Saves current state:
```json
{
  "balance": 10150.45,
  "open_positions": [...],
  "closed_trades": [...],
  "total_trades": 15,
  "winning_trades": 9,
  "losing_trades": 6
}
```

State is preserved between runs - you can stop and restart paper trading without losing history.

### 3. Performance Report

At the end (Ctrl+C to stop):
```
======================================================================
PAPER TRADING SUMMARY
======================================================================

Duration: 2h 15m
Total Checks: 135

Capital:
  Initial:   $10,000.00
  Final:     $10,234.56
  Net P&L:   $234.56 (2.35%)

Trades:
  Total:     15
  Long:      7 (46.7%)
  Short:     8 (53.3%)

Performance:
  Winners:   9 (60.0%)
  Losers:    6 (40.0%)
  Win Rate:  60.0%
  Avg Win:   $45.23
  Avg Loss:  -$28.67
  Profit Factor: 1.58
```

---

## 🔧 Troubleshooting

### Error: "X has 138 features, but StandardScaler is expecting 96 features"

**Solution**: Run the feature name extractor:
```bash
python extract_feature_names.py
```

This creates `results/xgboost/feature_names.txt` from your training data.

### Error: "MT5 initialization failed"

**Causes**:
1. MT5 is not running
2. Not logged into account
3. Wrong account credentials

**Solution**: 
- Start MT5
- Log into your demo account
- Verify connection in MT5 terminal

### Warning: "Missing X training columns"

**Cause**: Feature engineering creates slightly different features than training

**Impact**: Usually minor - missing features are filled with 0

**Solution**: If this affects performance, retrain the model:
```bash
python train_xgboost.py
```

### No Trades Happening

**Possible reasons**:
1. **Confidence too high** - Lower `CONFIDENCE_THRESHOLD` (try 0.6)
2. **No clear signal** - Market is ranging/choppy
3. **Model not confident** - Normal behavior, wait for clearer patterns

---

## 📊 Best Practices

### 1. Run for Sufficient Time
- **Minimum**: 1 week
- **Recommended**: 2-4 weeks
- **Why**: Need enough trades to assess true performance

### 2. Monitor Win Rate
- **Good**: 55-65%
- **Excellent**: 65%+
- **Warning**: <50% (may need retraining)

### 3. Check Profit Factor
- **Formula**: Total Wins / Total Losses
- **Acceptable**: >1.2
- **Good**: >1.5
- **Excellent**: >2.0

### 4. Review Trades Daily
Check `logs/paper_trading/` for:
- Unexpected behavior
- Correlation with market events
- Pattern recognition issues

### 5. Compare to Baseline
- Track buy-and-hold performance
- Compare your model's P&L to market direction
- Ensure model adds value over random trading

---

## 🎯 When to Move to Live Trading

Only proceed to live trading if:

✅ **Win rate ≥ 55%** over 2+ weeks
✅ **Profit factor ≥ 1.5**
✅ **Positive net P&L** (consistent profitability)
✅ **Drawdown < 10%** (maximum loss from peak)
✅ **No unexpected errors** or crashes
✅ **Understanding** of why model makes certain predictions

### Next Step: Live Trading

Once ready, review `live_trading.py` configuration:
- Start with **minimum position sizes**
- Use **strict stop losses**
- Monitor continuously for first few days
- Keep paper trading running in parallel for comparison

---

## 🛑 Stopping Paper Trading

### Graceful Shutdown
Press `Ctrl+C` to stop:
```
KeyboardInterrupt detected
Stopping paper trading...

[OK] Final report saved
[OK] State saved: logs/paper_trading/paper_trading_state.json
[OK] Paper trading stopped
```

### Force Stop
If system hangs, use `Ctrl+Break` or close terminal.

### Resume Trading
Run `python paper_trading.py` again - it will:
- Load previous state from JSON
- Continue with current balance
- Preserve trade history

---

## 📞 Support

### Check Diagnostics
```bash
python check_features.py      # Verify feature alignment
python validate_model.py       # Check model for data leakage
```

### Common Issues

1. **Feature count mismatch** → Run `extract_feature_names.py`
2. **MT5 connection failed** → Check MT5 is running and logged in
3. **No trades for hours** → Normal if market is choppy; consider lowering threshold
4. **Poor performance** → May need model retraining with more data

### Files to Check
- `logs/paper_trading/*.log` - Detailed execution logs
- `logs/paper_trading/paper_trading_state.json` - Current state
- `results/xgboost/feature_names.txt` - Training features (should have 96)

---

## 📝 Summary Checklist

Before running paper trading:
- [ ] XGBoost model trained and saved in `results/xgboost/`
- [ ] Feature names extracted (`feature_names.txt` exists with 96 features)
- [ ] MT5 installed, running, and logged in
- [ ] Demo account with XAUUSD symbol available
- [ ] Reviewed and adjusted configuration parameters

While running paper trading:
- [ ] Monitor logs daily
- [ ] Check win rate weekly
- [ ] Calculate profit factor
- [ ] Compare to baseline (buy-and-hold)
- [ ] Document any issues or patterns

Before live trading:
- [ ] Minimum 2 weeks of paper trading
- [ ] Win rate ≥ 55%
- [ ] Profit factor ≥ 1.5
- [ ] Positive net P&L
- [ ] Drawdown < 10%
- [ ] No critical errors

---

**Good luck with your paper trading! Remember: paper trading is for learning and validation, not for profit. Take your time and understand your model's behavior before risking real money.**