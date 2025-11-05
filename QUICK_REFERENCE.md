# Quick Reference - TRADE Bot

## 🚀 Quick Start Commands

### Paper Trading (Test with Virtual Money)
```bash
python paper_trading.py
```
**Stop:** Press `Ctrl+C`

### Live Trading (Real Money - Use with Caution!)
```bash
python live_trading.py
```
**Stop:** Press `Ctrl+C`

---

## 📊 Model Training

### Train XGBoost (Recommended)
```bash
python train_xgboost.py
```

### Train LSTM Models
```bash
# Simple LSTM
python train_simple.py

# No early stopping (full epochs)
python train_no_stop.py

# Advanced architectures
python train_advanced.py

# Quick menu
train_better.bat
```

---

## 🔧 Data Management

### Daily Data Update
```bash
python daily_update.py
# or
daily_update.bat
```

### Collect Historical Data
```bash
python collect_more_data.py
# or
collect_more_data.bat
```

---

## 🐛 Troubleshooting

### Feature Mismatch Error
```bash
python extract_feature_names.py
```

### Check Model/Features
```bash
python check_features.py
```

### Validate Model (Check for Data Leakage)
```bash
python validate_model.py
```

### Clean Data Leakage
```bash
python clean_data_leakage.py
```

---

## 📁 Important Files

### Models
- `results/xgboost/xgboost_model.pkl` - Trained XGBoost model
- `results/xgboost/xgboost_scaler.pkl` - Feature scaler
- `results/xgboost/feature_names.txt` - Training features (96 features)
- `results/xgboost/feature_importance.csv` - Feature rankings

### Logs
- `logs/paper_trading/*.log` - Paper trading logs
- `logs/paper_trading/paper_trading_state.json` - Current state
- `logs/daily_reports/*.json` - Daily performance reports

### Data
- `data/processed/XAUUSD_M5_clean.csv` - Cleaned M5 training data
- `data/processed/XAUUSD_*_with_target.csv` - Data with targets

---

## ⚙️ Configuration

### Paper Trading Settings
Edit `paper_trading.py`:
```python
SYMBOL = "XAUUSD"                # Trading pair
TIMEFRAME = "H1"                 # 1-hour candles
CONFIDENCE_THRESHOLD = 0.7       # 70% confidence minimum
INITIAL_CAPITAL = 10000.0        # $10,000 start
RISK_PER_TRADE = 0.01           # 1% risk per trade
CHECK_INTERVAL = 60              # Check every 60 seconds
```

### Live Trading Settings
Edit `live_trading.py`:
```python
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
CONFIDENCE_THRESHOLD = 0.7
RISK_PER_TRADE = 0.01           # 1% risk
MAX_TRADES_PER_DAY = 3          # Safety limit
MAX_DAILY_LOSS_PCT = 0.02       # Stop if lose 2%
```

---

## 📈 Model Performance (Current)

**XGBoost Model:**
- Test Accuracy: 84.59%
- F1 Score: 0.7520
- AUC-ROC: 0.9129
- Features: 96
- Status: ✅ Production Ready

---

## 🎯 Trading Rules

### Entry Signals
- **LONG**: Model predicts UP with confidence ≥ threshold
- **SHORT**: Model predicts DOWN with confidence ≥ threshold
- **NONE**: Low confidence or no clear signal

### Risk Management
- Stop Loss: 1.5x ATR from entry
- Take Profit: 2.0x stop loss distance
- Risk/Reward: 1:2 minimum
- Position Size: Based on account risk percentage

### Exit Conditions
- Hit stop loss
- Hit take profit
- Opposite signal generated
- Maximum hold time exceeded

---

## 📊 Performance Metrics

### Good Performance Indicators
- Win Rate: 55-65%+
- Profit Factor: >1.5
- Sharpe Ratio: >1.0
- Max Drawdown: <10%

### Warning Signs
- Win Rate: <50%
- Profit Factor: <1.2
- Losing streak: >5 trades
- Drawdown: >15%

---

## 🔄 Daily Routine

1. **Morning (Before Market Open)**
   ```bash
   python daily_update.py
   ```
   - Collects new data
   - Updates features
   - Generates report

2. **During Trading Hours**
   ```bash
   python paper_trading.py
   # or (if validated)
   python live_trading.py
   ```

3. **Evening (After Market Close)**
   - Review logs in `logs/paper_trading/`
   - Check P&L and win rate
   - Note any issues or patterns

4. **Weekly**
   - Analyze performance report
   - Retrain model if drift detected:
     ```bash
     python train_xgboost.py
     ```

---

## ⚠️ Safety Checklist

### Before Paper Trading
- [ ] Model trained and validated
- [ ] Feature names extracted (96 features)
- [ ] MT5 connected (demo account)
- [ ] Configuration reviewed
- [ ] Logs directory exists

### Before Live Trading
- [ ] 2+ weeks successful paper trading
- [ ] Win rate ≥ 55%
- [ ] Profit factor ≥ 1.5
- [ ] Positive net P&L
- [ ] All safety limits configured
- [ ] Emergency stop procedure tested
- [ ] Start with minimum position size

---

## 🆘 Emergency Commands

### Stop All Trading Immediately
```bash
Ctrl+C  # In terminal running the bot
```

### Check Current Positions (MT5)
Open MT5 Terminal → Trade tab → View open positions

### Close All Positions Manually
In MT5: Right-click position → Close

### Reset State
```bash
# Backup first!
cp logs/paper_trading/paper_trading_state.json logs/backup_state.json

# Then delete to reset
del logs/paper_trading/paper_trading_state.json
```

---

## 📚 Documentation

- `README.md` - Project overview
- `PAPER_TRADING_GUIDE.md` - Complete paper trading guide
- `FIX_SUMMARY.md` - Feature mismatch fix documentation
- `DAILY_ROUTINE_GUIDE.md` - Daily update procedures
- `QUICK_REFERENCE.md` - This file

---

## 🔗 Important Links

- GitHub: https://github.com/Apologizesss/ai-gold-bot
- MT5 Download: https://www.metatrader5.com/

---

## 💡 Tips

1. **Always start with paper trading** - Never jump to live trading
2. **Monitor logs daily** - Catch issues early
3. **Keep good records** - Document all changes and results
4. **Update regularly** - Run `daily_update.py` every day
5. **Validate before deploying** - Use `validate_model.py` after training
6. **Backup everything** - Models, logs, state files
7. **Start small** - Use minimum position sizes in live trading
8. **Have an exit plan** - Know when to stop (max loss limit)

---

## 📞 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Feature mismatch (96 vs 138) | `python extract_feature_names.py` |
| MT5 won't connect | Check MT5 is running and logged in |
| No trades happening | Lower confidence threshold or wait |
| Poor performance | Retrain model with more/recent data |
| Unicode errors | Already fixed - update from Git |
| Model not found | Run `python train_xgboost.py` first |

---

**Last Updated:** 2025-11-05  
**Version:** 1.0  
**Status:** ✅ All Systems Operational