# Live Trading Monitoring Guide

**📊 How to Monitor Your Trading Bot Performance**

---

## 🎯 Overview

This guide shows you how to monitor your trading bot and track performance effectively.

---

## 📈 Quick Stats Commands

### View Current Performance

```bash
# View live trading statistics
python live_trading_stats.py

# Compare paper vs live trading
python compare_performance.py
```

### Check System Status

```bash
# View today's log
tail -50 logs/live_trading/live_trading_20251106.log

# View real-time (live updates)
tail -f logs/live_trading/live_trading_20251106.log

# Check current state
cat logs/live_trading/live_trading_state.json
```

---

## 📊 Daily Monitoring Routine

### Morning (Before Trading Starts)

**1. Update Data**
```bash
python daily_update.py
```

**2. Check Yesterday's Performance**
```bash
python live_trading_stats.py
```

Look for:
- ✅ Win rate ≥ 55%
- ✅ Profit factor ≥ 1.5
- ✅ Positive net P&L
- ⚠️ Any unusual patterns

**3. Start Trading Bot**
```bash
# Only if yesterday was good
python live_trading.py
```

### During Trading Hours

**Check every 1-2 hours:**

1. **MT5 Terminal**
   - Open positions count (should be ≤ max_positions)
   - Current P&L
   - Stop loss and take profit visible

2. **System Logs**
   ```bash
   tail -20 logs/live_trading/live_trading_20251106.log
   ```
   - No error messages
   - Trades executing correctly
   - Bot still running

3. **Account Balance**
   - Not dropping rapidly
   - Within daily loss limit

### Evening (After Market Close)

**1. View Daily Summary**
```bash
python live_trading_stats.py
```

**2. Record in Trading Journal**

Create a file: `trading_journal.txt` or Excel sheet

```
Date: 2025-11-06
Trades: 8
Wins: 5 (62.5%)
Losses: 3 (37.5%)
Net P&L: +$85.50
Balance: $5,085.50
Notes: Good day, strong trends, one false breakout
```

**3. Compare to Paper Trading**
```bash
python compare_performance.py
```

Check if live performance matches paper trading.

---

## 📋 Weekly Review

### Every Sunday Evening

**1. Calculate Weekly Metrics**

```bash
# View all stats
python live_trading_stats.py

# Export to CSV for analysis
# Answer 'y' when prompted, save as: week_01.csv
```

**2. Review Performance**

| Metric | Target | Action if Below |
|--------|--------|----------------|
| Win Rate | ≥ 55% | Review strategy |
| Profit Factor | ≥ 1.5 | Reduce risk or stop |
| Weekly P&L | Positive | Stop if negative 2 weeks |
| Max Drawdown | < 10% | Stop if > 15% |

**3. Compare to Paper Trading**

```bash
python compare_performance.py
```

Questions to ask:
- Is live performance similar to paper?
- If worse, why? (slippage, emotions, execution issues)
- If much worse, should I stop?

**4. Update Models (if needed)**

If performance declining:
```bash
# Collect more data
python daily_update.py

# Retrain model
python train_xgboost.py

# Validate new model
python validate_model.py
```

---

## 🚨 Red Flags - Stop Trading If You See

### Critical - Stop Immediately

1. **Daily loss limit hit** (e.g., -1% in one day)
2. **Win rate < 45%** (worse than random)
3. **System errors repeated** (crashes, failed orders)
4. **Negative P&L 3 days in a row**
5. **Drawdown > 15%** from peak

### Warning - Monitor Closely

1. **Win rate 45-50%** (barely breaking even)
2. **Profit factor < 1.2** (low profitability)
3. **Average loss > Average win** (poor risk/reward)
4. **Many failed order executions**
5. **Live performance much worse than paper**

---

## 📊 Key Metrics Explained

### Win Rate
```
Win Rate = (Winning Trades / Total Trades) × 100%
```
- **Target:** ≥ 55%
- **Good:** 60-70%
- **Warning:** < 50%

### Profit Factor
```
Profit Factor = Total Wins ÷ Total Losses
```
- **Target:** ≥ 1.5
- **Good:** 2.0+
- **Warning:** < 1.2

### Return on Investment (ROI)
```
ROI = ((Current Balance - Initial Balance) / Initial Balance) × 100%
```
- **Monthly Target:** 3-5%
- **Good:** 5-10%
- **Warning:** Negative

### Maximum Drawdown
```
Max Drawdown = (Peak Balance - Lowest Balance After Peak) / Peak Balance × 100%
```
- **Target:** < 10%
- **Warning:** > 15%
- **Critical:** > 20%

---

## 📝 Trading Journal Template

### Daily Entry Template

```
═══════════════════════════════════════════════════════════
DATE: 2025-11-06
═══════════════════════════════════════════════════════════

SUMMARY:
- Total Trades: 8
- Winning: 5 (62.5%)
- Losing: 3 (37.5%)
- Net P&L: +$85.50
- Balance: $5,085.50 (started: $5,000)

TOP TRADES:
1. LONG @ 2650.50 → 2652.30 = +$18 (Good momentum)
2. SHORT @ 2651.80 → 2650.20 = +$16 (Strong downtrend)

WORST TRADE:
- LONG @ 2650.00 → 2648.50 = -$15 (False breakout)

MARKET CONDITIONS:
- Trending / Ranging / Volatile
- News events: [None / List any]

EMOTIONS:
- Confident / Nervous / Excited
- Any manual interventions? [Yes/No]

LESSONS LEARNED:
- [What worked well today]
- [What could be improved]

NOTES:
- [Any observations]
- [System behavior]
- [Ideas for improvement]
═══════════════════════════════════════════════════════════
```

---

## 🔍 Performance Analysis Tools

### 1. View Statistics
```bash
python live_trading_stats.py
```

**Output:**
- Total trades, wins, losses
- Win rate, profit factor
- Average win/loss
- Largest win/loss
- Current balance & ROI
- Last 10 trades

### 2. Compare Paper vs Live
```bash
python compare_performance.py
```

**Output:**
- Side-by-side comparison
- Performance differences
- Slippage analysis
- Recommendations

### 3. Export Data
```bash
python live_trading_stats.py
# Answer 'y' to export
# Creates: logs/live_trading/trading_history.csv
```

Open in Excel for:
- Charts & graphs
- Pivot tables
- Custom analysis

---

## 📈 Excel/Google Sheets Tracking

### Create Performance Tracker

**Columns:**
| Date | Trades | Wins | Losses | Win% | P&L | Balance | Drawdown | Notes |
|------|--------|------|--------|------|-----|---------|----------|-------|

**Formulas:**
```excel
Win% = (Wins / Trades) * 100
Cumulative P&L = SUM(all P&L)
Drawdown = (Peak Balance - Current) / Peak
```

**Charts to Create:**
1. Balance over time (line chart)
2. Daily P&L (bar chart)
3. Win rate over time (line chart)
4. Win/Loss distribution (histogram)

---

## 🛑 When to Stop Trading

### Immediate Stop Conditions

**Stop and close all positions if:**

1. ❌ Daily loss limit reached (-1% or -$100)
2. ❌ Win rate drops below 40%
3. ❌ Losing streak ≥ 7 trades
4. ❌ System crashes repeatedly
5. ❌ Emotional trading (fear/greed taking over)
6. ❌ Unexpected market event (crash, news)

### Review & Adjust Conditions

**Pause and review if:**

1. ⚠️ Win rate 40-50% for 3+ days
2. ⚠️ Profit factor < 1.2
3. ⚠️ Negative P&L for 3+ days
4. ⚠️ Live performance < Paper performance
5. ⚠️ Drawdown > 10%

**Actions:**
- Analyze what changed
- Check model validity
- Review recent trades
- Adjust risk parameters
- Consider retraining model

---

## 📞 Troubleshooting

### Problem: Performance Worse Than Paper Trading

**Possible Causes:**
1. Slippage & spreads (normal, 1-3% difference)
2. Execution delays (upgrade VPS/internet)
3. Emotional interventions (stop manual trades)
4. Market conditions changed (retrain model)
5. Broker issues (consider switching)

**Solutions:**
- Compare paper vs live systematically
- Measure slippage per trade
- Check execution logs for delays
- Keep trading journal to spot patterns

### Problem: No Trades Happening

**Causes:**
1. Confidence threshold too high
2. No clear market signals
3. Model not confident

**Solutions:**
- Check if this is normal (high threshold = fewer trades)
- Review model predictions in logs
- Consider lowering threshold to 0.75 (carefully)
- Wait for better market conditions

### Problem: Many Losing Trades

**Causes:**
1. Market conditions changed
2. Model overfit or degraded
3. Poor risk management
4. Bad luck (short-term variance)

**Solutions:**
- Stop if win rate < 45% for 3+ days
- Retrain model with recent data
- Review stop loss and take profit levels
- Wait for 20-30 trades before judging

---

## 🎯 Performance Goals by Week

### Week 1 (Learning Phase)
- **Goal:** Don't lose more than 2%
- **Focus:** System stability, execution
- **Target:** Break even or small profit

### Week 2-3 (Validation Phase)
- **Goal:** Consistent performance
- **Focus:** Win rate ≥ 50%, positive P&L
- **Target:** 1-3% return

### Week 4+ (Growth Phase)
- **Goal:** Scale gradually if profitable
- **Focus:** Win rate ≥ 55%, profit factor ≥ 1.5
- **Target:** 3-5% monthly return

---

## 📋 Daily Checklist

### Morning
- [ ] Run `python daily_update.py`
- [ ] Check yesterday's stats: `python live_trading_stats.py`
- [ ] Review any errors in logs
- [ ] Verify MT5 is connected
- [ ] Start bot: `python live_trading.py`

### During Day (Every 1-2 Hours)
- [ ] Check bot is still running
- [ ] View open positions in MT5
- [ ] Check for errors in logs
- [ ] Verify balance is within limits

### Evening
- [ ] View daily summary: `python live_trading_stats.py`
- [ ] Update trading journal
- [ ] Compare to paper: `python compare_performance.py`
- [ ] Plan for tomorrow

### Weekly (Sunday)
- [ ] Calculate weekly metrics
- [ ] Export data to CSV
- [ ] Review performance vs targets
- [ ] Update models if needed
- [ ] Set goals for next week

---

## 📊 Sample Output

### Good Day Example
```
======================================================================
LIVE TRADING PERFORMANCE SUMMARY
======================================================================

OVERALL PERFORMANCE
----------------------------------------------------------------------
Total Trades:        8
Winning Trades:      5 (62.5%)
Losing Trades:       3 (37.5%)

PROFIT & LOSS
----------------------------------------------------------------------
Net P&L:             +$85.50
Total Wins:          +$150.00
Total Losses:        -$64.50
Average Win:         +$30.00
Average Loss:        -$21.50

PERFORMANCE METRICS
----------------------------------------------------------------------
Win Rate:            62.5%
Profit Factor:       2.33
Status:              [OK] GOOD - Keep trading

ACCOUNT
----------------------------------------------------------------------
Initial Balance:     $5,000.00
Current Balance:     $5,085.50
ROI:                 +1.71%
```

### Warning Day Example
```
OVERALL PERFORMANCE
----------------------------------------------------------------------
Total Trades:        10
Winning Trades:      4 (40.0%)
Losing Trades:       6 (60.0%)

Status:              [Warning] POOR - Consider stopping

RECOMMENDATION:
- Win rate below 50% (40%)
- Review recent trades
- Check if market conditions changed
- Consider stopping if continues tomorrow
```

---

## 🔗 Quick Reference

### Essential Commands
```bash
# View stats
python live_trading_stats.py

# Compare performance
python compare_performance.py

# View logs
tail -f logs/live_trading/live_trading_20251106.log

# Check state
cat logs/live_trading/live_trading_state.json
```

### Log Locations
- Logs: `logs/live_trading/`
- State: `logs/live_trading/live_trading_state.json`
- Exports: `logs/live_trading/*.csv`

### Important Files
- `live_trading.py` - Main bot
- `live_trading_stats.py` - Statistics viewer
- `compare_performance.py` - Comparison tool
- `LIVE_TRADING_GUIDE.md` - Full guide
- `SAFE_LIVE_TRADING_SETUP.md` - Safety setup

---

## 🎯 Summary

**Key Points:**
1. **Monitor daily** - Check stats every evening
2. **Compare regularly** - Paper vs live performance
3. **Keep journal** - Document everything
4. **Stop when necessary** - Don't chase losses
5. **Review weekly** - Analyze trends and adjust

**Success Metrics:**
- Win Rate ≥ 55%
- Profit Factor ≥ 1.5
- Positive monthly ROI
- Drawdown < 10%

**Remember:** Consistent monitoring and discipline are more important than individual trades!

---

**Last Updated:** 2025-11-05
**Status:** Live trading monitoring ready