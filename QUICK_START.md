# QUICK START GUIDE - AI Gold Trading Bot

**⚡ Get up and running in 30 minutes!**

---

## 🎯 What You Need

### Minimum Requirements
- ✅ Computer with Windows/Linux/Mac
- ✅ 16GB RAM
- ✅ 50GB free disk space
- ✅ Stable internet connection
- ✅ $15-$50 for initial trading capital (after testing)

### Time Investment
- **Setup:** 1 week
- **Data Collection:** 2 weeks
- **Training:** 4 weeks
- **Testing:** 8 weeks
- **Total:** ~3.5 months to go live

---

## 🚀 5-STEP QUICK START

### STEP 1: Install Python (10 minutes)

**Windows:**
1. Download Python 3.9 or 3.10 from: https://www.python.org/downloads/
2. Run installer
3. ✅ CHECK "Add Python to PATH"
4. Click "Install Now"
5. Verify: Open CMD and type `python --version`

**Mac/Linux:**
```bash
# Mac (using Homebrew)
brew install python@3.9

# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3-pip

# Verify
python3 --version
```

---

### STEP 2: Setup Project (5 minutes)

**Open Terminal/CMD and run:**

```bash
# Navigate to desktop
cd C:\Users\ASUS\Desktop\TRADE

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) in your prompt now
```

---

### STEP 3: Install Dependencies (10 minutes)

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all packages (this takes 5-10 minutes)
pip install -r requirements.txt
```

**⚠️ Special: Install TA-Lib (Technical Indicators)**

**Windows:**
1. Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Choose: `TA_Lib‑0.4.26‑cp39‑cp39‑win_amd64.whl` (for Python 3.9)
3. Install: `pip install TA_Lib‑0.4.26‑cp39‑cp39‑win_amd64.whl`

**Mac/Linux:**
```bash
# Mac
brew install ta-lib
pip install TA-Lib

# Ubuntu/Debian
sudo apt install ta-lib
pip install TA-Lib
```

---

### STEP 4: Install MetaTrader 5 (5 minutes)

1. **Choose a Broker:**
   - Recommended: IC Markets, Pepperstone, XM, FBS
   - Look for: Low spreads on XAUUSD (<30 cents)

2. **Download MT5:**
   - Go to your broker's website
   - Download MetaTrader 5
   - Install it

3. **Open Demo Account:**
   - Launch MT5
   - File → Open an Account
   - Choose "Demo Account"
   - Balance: $100
   - Leverage: 1:100
   - Save your credentials!

4. **Enable Python API:**
   - MT5 → Tools → Options
   - Expert Advisors tab
   - ✅ Check "Allow DLL imports"
   - ✅ Check "Allow WebRequest"
   - Click OK

---

### STEP 5: Get API Keys (5 minutes)

**News API (Free):**
1. Go to: https://newsapi.org
2. Click "Get API Key"
3. Sign up (free 100 requests/day)
4. Copy your API key
5. Save it somewhere safe

**Telegram Bot (Optional):**
1. Open Telegram app
2. Search for "@BotFather"
3. Send: `/newbot`
4. Follow instructions
5. Copy your bot token
6. Get your chat ID:
   - Search for "@userinfobot"
   - Start chat
   - Copy your ID

---

## 📝 Configure Your Bot

### Create config/.env file

```bash
# Create config directory if it doesn't exist
mkdir config

# Create .env file (Windows)
notepad config\.env

# Create .env file (Mac/Linux)
nano config/.env
```

**Paste this into config/.env:**

```env
# MetaTrader 5 Credentials
MT5_LOGIN=your_demo_account_number
MT5_PASSWORD=your_demo_password
MT5_SERVER=your_broker_server

# News API
NEWS_API_KEY=your_newsapi_key_here

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Parameters
INITIAL_CAPITAL=100.0
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.15

# Database
DATABASE_PATH=data/trading.db

# Logging
LOG_LEVEL=INFO
```

**Save and close the file.**

**⚠️ NEVER commit .env to Git - it contains secrets!**

---

## ✅ Test Your Setup

**Create a test file: test_setup.py**

```python
# test_setup.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from transformers import pipeline

print("Testing all imports...")
print(f"✅ MetaTrader5: {mt5.__version__}")
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ XGBoost: {xgb.__version__}")
print("✅ Transformers: OK")

# Test MT5 connection
print("\nTesting MT5 connection...")
if mt5.initialize():
    print(f"✅ MT5 connected!")
    print(f"   Account: {mt5.account_info().login}")
    print(f"   Balance: ${mt5.account_info().balance}")
    mt5.shutdown()
else:
    print("❌ MT5 connection failed!")
    print("   Make sure MT5 is running and credentials are correct.")

print("\n🎉 Setup complete! You're ready to start.")
```

**Run the test:**

```bash
python test_setup.py
```

**Expected output:**
```
Testing all imports...
✅ MetaTrader5: 5.0.45
✅ Pandas: 2.0.0
✅ NumPy: 1.24.0
✅ TensorFlow: 2.13.0
✅ XGBoost: 2.0.0
✅ Transformers: OK

Testing MT5 connection...
✅ MT5 connected!
   Account: 12345678
   Balance: $100.0

🎉 Setup complete! You're ready to start.
```

---

## 📚 What's Next?

### Immediate Next Steps:

1. **Read the Documentation:**
   - [ ] Read `README.md` (full system overview)
   - [ ] Read `PROJECT_STATUS.md` (track progress)
   - [ ] Read `TODO.md` (detailed task list)

2. **Create Project Structure:**
   ```bash
   # Create all directories
   mkdir data data\raw data\processed data\labels
   mkdir models src config notebooks tests logs results scripts
   ```

3. **Start Phase 1: Data Collection**
   - Download 2+ years of XAUUSD data from MT5
   - Scrape Forex Factory news calendar
   - Collect sentiment data
   - See `TODO.md` for detailed steps

---

## 🎓 Learning Path

### Week 1: Setup & Learning
- [ ] Complete this quick start
- [ ] Learn MetaTrader 5 basics
- [ ] Understand the trading strategy (read README.md)
- [ ] Watch tutorials on LSTM, CNN, XGBoost

### Weeks 2-3: Data Collection
- [ ] Collect historical price data
- [ ] Scrape news events
- [ ] Generate sentiment scores
- [ ] Validate data quality

### Weeks 4-8: Model Training
- [ ] Engineer 65 features
- [ ] Train LSTM price predictor
- [ ] Train CNN pattern detector
- [ ] Train XGBoost news scorer
- [ ] Train Random Forest meta-learner

### Weeks 9-12: Backtesting
- [ ] Build backtesting engine
- [ ] Run historical backtest
- [ ] Stress test the system
- [ ] Optimize parameters

### Weeks 13-16: Paper Trading
- [ ] Run on demo account for 1 month
- [ ] Monitor daily performance
- [ ] Fix any bugs
- [ ] Validate profitability

### Week 17+: Live Trading
- [ ] Start with $15-$25
- [ ] Monitor closely
- [ ] Track all metrics
- [ ] Scale gradually if profitable

---

## 📖 Essential Reading

### Must Read First:
1. **README.md** - Complete system documentation
2. **PROJECT_STATUS.md** - Current progress tracking
3. **TODO.md** - Step-by-step task list

### Important Concepts:
- **Risk Management:** Never risk more than 2% per trade
- **Stop Losses:** ALWAYS use stop losses
- **Backtesting:** NEVER skip backtesting
- **Paper Trading:** Test on demo first
- **Patience:** Don't rush to live trading

---

## 🆘 Troubleshooting

### Problem: "python: command not found"
**Solution:** Python not installed or not in PATH
- Reinstall Python with "Add to PATH" checked
- Or manually add Python to system PATH

### Problem: "pip: command not found"
**Solution:** pip not installed
```bash
python -m ensurepip --upgrade
```

### Problem: TA-Lib installation fails
**Solution:** 
- Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Mac: `brew install ta-lib`
- Linux: `sudo apt install ta-lib`

### Problem: MT5 won't connect
**Solution:**
- Make sure MT5 is running
- Check credentials in .env file
- Enable DLL imports in MT5 settings
- Try restarting MT5

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Problem: GPU not detected (TensorFlow)
**Solution:**
- Check if you have NVIDIA GPU
- Install CUDA 11.8 and cuDNN 8.6
- Install: `pip install tensorflow-gpu==2.13.0`
- For CPU-only: Regular tensorflow is fine

---

## 💡 Pro Tips

### Development Best Practices:
1. **Always activate virtual environment** before working
2. **Commit code to Git regularly** (except .env and large files)
3. **Test small changes immediately** - don't write too much at once
4. **Keep detailed logs** - they're invaluable for debugging
5. **Document as you go** - future you will thank you

### Trading Best Practices:
1. **Start small** - $15-$25 is enough to begin
2. **Never risk more than 2%** per trade
3. **Always use stop losses** - no exceptions
4. **Monitor daily** - don't set and forget
5. **Keep a trading journal** - learn from every trade

### Performance Tips:
1. **Use SSD** for faster data loading
2. **16GB RAM minimum** for training models
3. **GPU optional** but speeds up training 10x
4. **VPS recommended** for 24/7 running (after paper trading)

---

## 🎯 Success Checklist

### Phase 0: Setup (This Week)
- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] MetaTrader 5 installed and configured
- [ ] Demo account opened
- [ ] API keys obtained
- [ ] config/.env file created
- [ ] Test script runs successfully
- [ ] Project structure created
- [ ] Git repository initialized

### Ready for Phase 1?
- [ ] All Phase 0 tasks complete ✅
- [ ] MT5 connects successfully ✅
- [ ] All imports work ✅
- [ ] Understand the system architecture ✅
- [ ] Read TODO.md for next steps ✅

---

## 📞 Need Help?

### Resources:
- **Documentation:** All .md files in project root
- **MetaTrader 5 Docs:** https://www.mql5.com/en/docs/python_metatrader5
- **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
- **TA-Lib Docs:** https://mrjbq7.github.io/ta-lib/

### Community:
- QuantConnect Forum
- Reddit: r/algotrading
- Stack Overflow: [metatrader5] tag

### Important Reminder:
**This is for educational purposes only. Trading involves risk. Never risk money you cannot afford to lose. This is not financial advice.**

---

## 🚀 Ready to Begin!

You've completed the Quick Start! Here's what to do now:

1. **Make sure all checkboxes above are checked** ✅
2. **Open TODO.md** and start Phase 1
3. **Update PROJECT_STATUS.md** as you progress
4. **Commit your progress to Git** regularly
5. **Stay patient and disciplined**

**Remember: The goal isn't to rush - it's to build a reliable, profitable trading system. Take your time and do it right.**

---

**Good luck! 🎉**

**Questions? Check the documentation or troubleshooting section above.**

---

**Last Updated:** 2024  
**Version:** 1.0  
**Status:** Ready to use

---

**⚠️ DISCLAIMER:** Trading forex and gold carries substantial risk of loss. Only trade with money you can afford to lose. Past performance does not guarantee future results. This software is for educational purposes only and is not financial advice.