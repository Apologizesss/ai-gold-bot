# Live Trading Guide - Real Money Trading

**⚠️ CRITICAL WARNING ⚠️**

**THIS SYSTEM TRADES WITH REAL MONEY. YOU CAN LOSE EVERYTHING.**

---

## 🚨 BEFORE YOU START

### ⚠️ Mandatory Requirements

**DO NOT proceed unless ALL of these are TRUE:**

- [ ] ✅ Paper trading ran successfully for **minimum 2 weeks**
- [ ] ✅ Win rate in paper trading ≥ **55%**
- [ ] ✅ Profit factor ≥ **1.5**
- [ ] ✅ Positive net P&L consistently
- [ ] ✅ Maximum drawdown < **10%**
- [ ] ✅ No critical errors or crashes
- [ ] ✅ You understand every trade the model makes
- [ ] ✅ You have tested emergency stop procedures
- [ ] ✅ You are using a **DEMO account first** (not real account yet!)
- [ ] ✅ You can afford to lose 100% of the trading capital

**If ANY checkbox is unchecked, DO NOT proceed to live trading!**

---

## 🎯 Current Status

**Based on your conversation:**
- ❌ Paper trading **NOT validated** (just fixed bugs today)
- ❌ No performance history yet
- ❌ Model just started working correctly with M5 timeframe
- ⚠️ **You should NOT live trade yet!**

**Recommended:** Wait 1-2 weeks for paper trading validation first.

---

## 🛡️ Safety Configuration

### Absolute Minimum Safety Settings

If you insist on live trading, use these **STRICT** settings:

```python
# In live_trading.py
confidence_threshold = 0.80      # Very high threshold (80%)
risk_per_trade = 0.005          # 0.5% risk per trade (very small)
max_positions = 1               # Only 1 position at a time
max_daily_loss = 0.02           # Stop if lose 2% in one day
fixed_lot = 0.01                # Minimum lot size (0.01)
```

### Recommended Starting Capital

- **Demo Account:** Any amount (for testing)
- **Real Account (if you must):** $1,000 - $5,000 MAX
- **Never risk more than you can afford to lose**

---

## 📝 Step-by-Step Live Trading Setup

### Step 1: Verify Model is Working

```bash
# Check model files exist
ls results/xgboost/xgboost_model.pkl
ls results/xgboost/xgboost_scaler.pkl
ls results/xgboost/feature_names.txt

# Verify feature count (should be 96)
python check_features.py
```

### Step 2: Test on Demo Account FIRST

**NEVER skip this step!**

```bash
# Make sure MT5 is logged into DEMO account
# Then run live trading
python live_trading.py
```

Watch for:
- ✅ Orders execute successfully
- ✅ Stop loss and take profit set correctly
- ✅ Position sizes are correct
- ✅ Logs are being saved

### Step 3: Monitor Closely (First 24 Hours)

For the first day:
- **Check every 30 minutes**
- Monitor all trades in MT5 terminal
- Keep logs open: `logs/live_trading/`
- Be ready to emergency stop (Ctrl+C)

### Step 4: Review Daily

Every evening:
- Check win rate
- Review all trades
- Calculate profit/loss
- Look for unexpected behavior
- Adjust if needed

---

## 🔧 Configuration Options

### Edit `live_trading.py`

```python
class LiveTrading:
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",                    # ✅ Now correct!
        confidence_threshold: float = 0.80,       # ⚠️ Set HIGHER (80%)
        risk_per_trade: float = 0.005,           # ⚠️ Set LOWER (0.5%)
        max_positions: int = 1,                   # ⚠️ Only 1 position
        max_daily_loss: float = 0.02,            # ⚠️ Stop at 2% loss
        fixed_lot: float = 0.01,                 # ⚠️ Minimum lot size
    ):
```

### Safety Limits Explained

| Setting | Recommended | What It Does |
|---------|-------------|--------------|
| `confidence_threshold` | 0.80 | Only trade when model is 80%+ confident |
| `risk_per_trade` | 0.005 | Risk only 0.5% per trade (very conservative) |
| `max_positions` | 1 | Maximum 1 open position at a time |
| `max_daily_loss` | 0.02 | Stop trading if lose 2% in one day |
| `fixed_lot` | 0.01 | Use smallest possible lot size |

---

## 🚀 Running Live Trading

### Command

```bash
# With default settings
python live_trading.py

# With custom settings (safer)
python live_trading.py --threshold 0.85 --risk 0.003 --max-positions 1
```

### What You'll See

```
======================================================================
⚠️  LIVE TRADING SYSTEM - REAL MONEY AT RISK ⚠️
======================================================================

Symbol: XAUUSD
Timeframe: M5
Confidence Threshold: 0.80
Risk per Trade: 0.5%
Max Positions: 1
Max Daily Loss: 2.0%

Account Balance: $5,000.00
Account Equity: $5,000.00

[OK] Live trading system initialized

Starting live trading...
Press Ctrl+C to stop safely
```

### During Trading

The system will:
1. Check market every 30 seconds
2. Run inference every 5 minutes (M5 candle close)
3. Execute trades when confidence ≥ threshold
4. Set stop loss and take profit automatically
5. Monitor open positions
6. Close positions when SL/TP hit or opposite signal
7. Stop if daily loss limit reached

---

## 📊 Monitoring & Logs

### Real-Time Monitoring

1. **MT5 Terminal**
   - Open MT5 → Trade tab
   - Watch for new orders
   - See positions update in real-time

2. **System Logs**
   ```bash
   # View live logs
   tail -f logs/live_trading/live_trading_YYYYMMDD.log
   ```

3. **State File**
   ```bash
   # Check current state
   cat logs/live_trading/live_trading_state.json
   ```

### What to Monitor

**Every 30 minutes:**
- [ ] System is still running
- [ ] No error messages
- [ ] Positions are correct in MT5
- [ ] Balance/equity match expectations

**Daily:**
- [ ] Win rate
- [ ] P&L (profit/loss)
- [ ] Number of trades
- [ ] Largest win/loss
- [ ] Drawdown

**Weekly:**
- [ ] Overall profitability
- [ ] Compare to paper trading results
- [ ] Adjust strategy if needed

---

## 🛑 Emergency Stop Procedures

### Method 1: Graceful Stop (Preferred)

```bash
# Press Ctrl+C in the terminal
# System will:
# 1. Save current state
# 2. Log final report
# 3. Shutdown safely
# 4. Keep positions open (you manage manually)
```

### Method 2: Force Stop

```bash
# If system hangs
# Close terminal window or:
taskkill /F /IM python.exe  # Windows
```

### Method 3: Close All Positions

**In MT5:**
1. Right-click on position
2. Select "Close"
3. Confirm

**Or close all at once:**
- Right-click in Trade tab
- Select "Close All"

---

## ⚠️ Common Issues & Solutions

### Issue: No Trades Happening

**Causes:**
1. Confidence threshold too high (lower to 0.70-0.75)
2. Market is ranging/choppy (normal, wait)
3. Model not confident (normal, wait for clearer signals)

**Solution:** Wait or lower threshold slightly

### Issue: Too Many Losing Trades

**Causes:**
1. Model drift (market changed since training)
2. Bad market conditions (high volatility)
3. Model overfitted to training data

**Solutions:**
1. Stop trading immediately
2. Review paper trading results
3. Retrain model with recent data
4. Reduce position size

### Issue: System Crashed

**Immediate Actions:**
1. Check MT5 for open positions
2. Close any open positions manually
3. Check logs for error message
4. Fix error before restarting
5. Test on demo again

---

## 📈 Performance Expectations

### Realistic Expectations

Based on **84.59% training accuracy** and assuming real-world slippage:

**Good Performance:**
- Win Rate: 55-60%
- Profit Factor: 1.3-1.8
- Monthly Return: 3-8%
- Max Drawdown: 5-10%

**Warning Signs:**
- Win Rate: <50%
- Profit Factor: <1.2
- Monthly Return: Negative
- Max Drawdown: >15%

**If performance is worse than paper trading:**
- Stop immediately
- Review what changed
- Test more before continuing

---

## 💰 Position Sizing & Risk

### Risk Calculation

With `risk_per_trade = 0.005` (0.5%):

| Account Balance | Risk per Trade | Stop Loss Distance | Lot Size |
|----------------|----------------|-------------------|----------|
| $1,000 | $5 | 10 points | 0.05 lots |
| $2,000 | $10 | 10 points | 0.10 lots |
| $5,000 | $25 | 10 points | 0.25 lots |
| $10,000 | $50 | 10 points | 0.50 lots |

**For XAUUSD (Gold):**
- 1 lot = $100 per point
- 0.01 lot = $1 per point
- Use `fixed_lot = 0.01` to start (safest)

---

## 📝 Daily Routine

### Morning (Before Trading Starts)

```bash
# 1. Update data
python daily_update.py

# 2. Check model performance
python validate_model.py

# 3. Review yesterday's trades
cat logs/live_trading/live_trading_state.json

# 4. Start live trading
python live_trading.py
```

### During Trading Hours

- Check every 30-60 minutes
- Monitor MT5 terminal
- Watch for unusual behavior
- Be ready to intervene

### Evening (After Close)

- Review all trades
- Calculate daily P&L
- Update trading journal
- Compare to paper trading
- Adjust strategy if needed

---

## 🔐 Security & Best Practices

### Account Security

1. **Use Strong Password** on MT5 account
2. **Enable 2FA** if broker supports it
3. **Don't share credentials** with anyone
4. **Use VPS** if running 24/7 (optional)

### Code Security

1. **Backup regularly**
   ```bash
   # Backup models and logs
   cp -r results/ backups/results_YYYYMMDD/
   cp -r logs/ backups/logs_YYYYMMDD/
   ```

2. **Version control**
   ```bash
   git commit -am "Live trading session YYYYMMDD"
   git push origin main
   ```

3. **Don't modify code** during live trading

---

## 📞 Support & Help

### If Something Goes Wrong

1. **Stop trading immediately** (Ctrl+C)
2. **Close all positions** in MT5
3. **Check logs** for errors
4. **Review this guide**
5. **Test on demo** before retrying

### Files to Check

- `logs/live_trading/*.log` - Execution logs
- `logs/live_trading/live_trading_state.json` - Current state
- `results/xgboost/xgboost_results.json` - Model metrics

### Diagnostic Commands

```bash
python check_features.py      # Check feature alignment
python validate_model.py       # Validate model
python extract_feature_names.py  # Fix feature mismatch
```

---

## ⚖️ Legal Disclaimer

**IMPORTANT:**

- This is educational software only
- No guarantee of profits
- You can lose all your money
- Past performance ≠ future results
- You are responsible for all trading decisions
- Consult a financial advisor before trading
- Trading involves substantial risk
- Only trade with money you can afford to lose

**By using this software, you acknowledge:**
- You understand the risks
- You accept full responsibility
- You will not hold the developer liable for losses
- You have tested thoroughly on demo first

---

## 🎯 Final Checklist

### Before First Live Trade

- [ ] Paper trading completed (minimum 2 weeks)
- [ ] Win rate ≥ 55% in paper trading
- [ ] Profit factor ≥ 1.5 in paper trading
- [ ] Model validated (no data leakage)
- [ ] Feature alignment verified (96 features)
- [ ] M5 timeframe confirmed
- [ ] Tested on demo account successfully
- [ ] Safety limits configured (high threshold, low risk)
- [ ] Emergency stop procedures tested
- [ ] Monitoring plan in place
- [ ] You understand you can lose money

### Daily Pre-Trade Checklist

- [ ] Data updated (`python daily_update.py`)
- [ ] Model validated (`python validate_model.py`)
- [ ] MT5 connected and logged in
- [ ] Account balance verified
- [ ] No manual positions open
- [ ] Logs directory has space
- [ ] You are available to monitor

---

## 🚦 Traffic Light System

### 🟢 GREEN - Safe to Trade

- Paper trading profitable (2+ weeks)
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- System running smoothly
- All safety checks pass

### 🟡 YELLOW - Caution

- Win rate 50-55%
- Profit factor 1.2-1.5
- Some errors in logs
- Market volatility high
- **Reduce position size**

### 🔴 RED - STOP TRADING

- Win rate < 50%
- Profit factor < 1.2
- Daily loss limit reached
- System errors/crashes
- Unexpected behavior
- **Close all positions and stop immediately**

---

## 📖 Summary

### Key Points to Remember

1. **Paper trade first** - Minimum 2 weeks, ideally 4 weeks
2. **Start small** - Use minimum lot sizes (0.01)
3. **High threshold** - Set confidence to 80%+ initially
4. **Low risk** - Risk only 0.5% per trade
5. **Monitor closely** - Especially first week
6. **Stop if losing** - Don't chase losses
7. **Demo first** - Test everything on demo account
8. **You can lose money** - Only trade what you can afford to lose

### Recommended Path

1. ✅ **Weeks 1-2:** Paper trading (validation)
2. ✅ **Week 3:** Demo account with live_trading.py (testing)
3. ⚠️ **Week 4:** Small real account (if all goes well)
4. 📈 **Week 5+:** Scale up gradually (if profitable)

**NEVER skip steps 1-3!**

---

**Current Status: Paper trading just started - DO NOT live trade yet!**

**Next Action: Let paper trading run for 2 weeks minimum**

**Last Updated:** 2025-11-05

---

**Remember: Patience is more profitable than rushing. Give paper trading time to validate your system before risking real money.**