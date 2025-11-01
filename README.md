# AI GOLD TRADING BOT - Complete System Documentation

**Version:** 1.0  
**Platform:** MetaTrader 5  
**Symbol:** XAUUSD (Gold/USD)  
**Min Capital:** $15-$50  
**Risk Level:** Moderate-Advanced  
**Win Rate Target:** 50-55%  
**Monthly Return Target:** 10-20%

---

## 📋 TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Data Sources](#data-sources)
3. [Features Engineering](#features-engineering)
4. [AI Models](#ai-models)
5. [Decision Process](#decision-process)
6. [Risk Management](#risk-management)
7. [Execution Flow](#execution-flow)
8. [Performance Monitoring](#performance-monitoring)
9. [Training Pipeline](#training-pipeline)
10. [Technical Requirements](#technical-requirements)
11. [Installation Guide](#installation-guide)
12. [Project Structure](#project-structure)
13. [Risk Warnings](#risk-warnings)

---

## 🎯 SYSTEM OVERVIEW

### Identity
You are an AI-powered automated gold trading system for XAU/USD in MetaTrader 5.

### Objective
Execute high-probability trades with:
- **70%+ AI confidence**
- **65+ technical score**
- **Minimum 1:3 Risk-Reward ratio**
- **Protect accounts from >15% drawdown**

### Core Philosophy
This system combines:
- 4 AI/ML models (LSTM, CNN, XGBoost, Random Forest)
- Traditional technical analysis (65 features)
- Smart Money Concepts
- News & Sentiment analysis
- Strict risk management

---

## 📊 DATA SOURCES

### 1. Price Data (MetaTrader 5 API)
**Symbols:**
- XAUUSD (primary)
- DXY (US Dollar Index)
- US10Y (10-Year Treasury Yield)
- SPY (S&P 500 ETF)

**Timeframes:**
- M1, M5, M15, M30 (primary), H1, H4, D1

**History:**
- Last 100 candles per timeframe

**Update Frequency:**
- Real-time tick data

### 2. News Data (Forex Factory Web Scraping)
**Source:** forexfactory.com/calendar

**Method:** BeautifulSoup/Selenium scraping

**Data Fields:**
- Event name
- Country
- Impact level (Low/Medium/High)
- Forecast value
- Previous value
- Actual value (when released)

**Update Frequency:** Every 15 minutes

**Lookback Window:** Next 2 hours of events

### 3. Sentiment Data (News API + BERT)
**Source:** newsapi.org (100 requests/day free tier)

**Keywords:** gold, XAU, inflation, Fed, USD, Treasury

**Processing:** BERT sentiment analysis model

**Output:** Sentiment score [-1.0 to +1.0]
- -1.0 = Very bearish
- 0.0 = Neutral
- +1.0 = Very bullish

**Update Frequency:** Every 30 minutes

### 4. Market Context (Yahoo Finance API)
**Source:** yfinance library (free)

**Symbols:**
- DXY (Dollar Index)
- ^TNX (US 10-Year Treasury)
- ^VIX (Volatility Index)
- SPY (S&P 500)

**Purpose:** Correlation analysis with gold

**Update Frequency:** Every 5 minutes

---

## 🧮 FEATURES ENGINEERING (65 Total Features)

### Group 1: Raw Price (5 features)
1. Open
2. High
3. Low
4. Close
5. Volume

### Group 2: Trend Indicators (10 features)
6. EMA 20
7. EMA 50
8. EMA 200
9. SMA 20
10. SMA 50
11. SMA 200
12. VWAP
13. Ichimoku Tenkan-sen
14. Ichimoku Kijun-sen
15. Ichimoku Senkou Span A
16. Ichimoku Senkou Span B

### Group 3: Momentum Indicators (8 features)
17. RSI (14)
18. MACD Line
19. MACD Signal
20. MACD Histogram
21. Stochastic %K
22. Stochastic %D
23. CCI (Commodity Channel Index)
24. Williams %R

### Group 4: Volatility Indicators (6 features)
25. ATR (Average True Range)
26. Bollinger Band Upper
27. Bollinger Band Middle
28. Bollinger Band Lower
29. Bollinger Band Width
30. Bollinger %B

### Group 5: Volume Indicators (4 features)
31. Volume Moving Average
32. OBV (On-Balance Volume)
33. Volume Rate of Change
34. Distance to VWAP

### Group 6: Fibonacci Levels (7 features)
35. Fib 0.236
36. Fib 0.382
37. Fib 0.500
38. Fib 0.618 (Golden Ratio)
39. Fib 0.705
40. Fib 0.786
41. Fib 1.000

### Group 7: Smart Money Concepts (10 features)
42. Order Block Support Level
43. Order Block Resistance Level
44. Order Block Volume
45. Break of Structure (Bullish Count)
46. Break of Structure (Bearish Count)
47. Market Structure Shift (Binary: 0/1)
48. Premium/Discount Zone (0-1 scale)
49. Fair Value Gap Size
50. Liquidity Sweep Bullish
51. Liquidity Sweep Bearish

### Group 8: News & Sentiment (3 features)
52. Next Event Impact (0=none, 1=low, 2=medium, 3=high)
53. Time to Next Event (minutes)
54. BERT Sentiment Score (-1 to +1)

### Group 9: Market Correlation (5 features)
55. DXY Direction (-1/0/+1)
56. US10Y Movement (% change)
57. VIX Level
58. Gold-DXY Correlation (rolling 20 periods)
59. Risk Sentiment Score

### Group 10: Time Features (3 features)
60. Hour of Day (0-23)
61. Day of Week (0-6)
62. Trading Session (0=Asia, 1=Europe, 2=NY, 3=Overlap)

### Group 11: Pattern Recognition (4 features)
63. CNN Pattern Type (0-9)
64. CNN Pattern Reliability (0-1)
65. Candlestick Pattern
66. Pattern Confirmation (0/1)

---

## 🤖 AI MODELS (4 Models Ensemble)

### Model 1: LSTM Price Predictor
**Purpose:** Predict next 30-minute price direction

**Input Shape:** (100 candles × 65 features)

**Architecture:**
```
Input Layer: (100, 65)
├── LSTM(128 units, return_sequences=True)
├── Dropout(0.3)
├── LSTM(64 units, return_sequences=True)
├── Dropout(0.3)
├── LSTM(32 units, return_sequences=False)
├── Dense(16, activation='relu')
└── Dense(3, activation='softmax')
```

**Output:** [SELL probability, HOLD probability, BUY probability]

**Training:**
- Epochs: 100
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Categorical crossentropy
- Data: 2 years historical M30 candles

**Target Accuracy:** >60%

**Labels:**
- SELL: Price drops >10 pips in next 30 min
- HOLD: Price moves <10 pips
- BUY: Price rises >10 pips in next 30 min

---

### Model 2: CNN Pattern Detector
**Purpose:** Recognize chart patterns from candlestick images

**Input Shape:** (128, 128, 1) grayscale image

**Architecture:**
```
Input Layer: (128, 128, 1)
├── Conv2D(32, 3x3, activation='relu')
├── MaxPooling2D(2x2)
├── Conv2D(64, 3x3, activation='relu')
├── MaxPooling2D(2x2)
├── Conv2D(128, 3x3, activation='relu')
├── MaxPooling2D(2x2)
├── Flatten()
├── Dense(128, activation='relu')
├── Dropout(0.5)
└── Dense(10, activation='softmax')
```

**Output:** 10 pattern types
1. Head and Shoulders
2. Inverse Head and Shoulders
3. Double Top
4. Double Bottom
5. Ascending Triangle
6. Descending Triangle
7. Order Block (Bullish)
8. Order Block (Bearish)
9. Consolidation
10. No Pattern

**Training:**
- Epochs: 50
- Batch size: 16
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical crossentropy
- Data: 10,000+ labeled chart images
- Augmentation: Rotation, zoom, shift

**Target Accuracy:** >55%

---

### Model 3: XGBoost News Impact Scorer
**Purpose:** Predict price movement magnitude from news events

**Input Features:** 770 features
- 768 BERT embeddings from news headline
- News impact level (1-3)
- Surprise factor (actual - forecast)

**Parameters:**
```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1
)
```

**Output:** Expected price move in pips [-50 to +50]

**Training:**
- Cross-validation: 5-fold
- Data: 5000+ news events with price reactions
- Evaluation metric: RMSE

**Target RMSE:** <15 pips

---

### Model 4: Random Forest Signal Filter (Meta-Learner)
**Purpose:** Final trade decision combining all models

**Input Features:** 9 meta-features
1. LSTM Direction (0=SELL, 1=HOLD, 2=BUY)
2. LSTM Confidence (0-1, max probability)
3. CNN Pattern Type (0-9)
4. CNN Reliability (0-1)
5. XGBoost News Impact (-50 to +50 pips)
6. Technical Confluence Score (0-100)
7. ATR Normalized (current ATR / 20-period ATR avg)
8. Time to Next News (minutes)
9. Current Drawdown (%)

**Parameters:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt'
)
```

**Output:** 
- Trade Decision (0=NO TRADE, 1=TRADE)
- Win Probability (0-1)

**Training:**
- Data: All historical signals + outcomes
- Class balancing: SMOTE oversampling
- Evaluation: Precision, Recall, F1-Score

**Target Accuracy:** >70%

---

## 🎯 DECISION PROCESS (4-Layer System)

### Layer 1: Pre-Trade Filters (ALL Must Pass)

**Trading Hours:**
- **Active:** 14:00-22:00 GMT+7 (London/NY sessions)
- **Reason:** Highest liquidity, tightest spreads

**News Filter:**
- **Rule:** No high-impact news in next 30 minutes
- **Reason:** Avoid unpredictable volatility spikes

**Daily Loss Limit:**
- **Rule:** Loss today <5% of account balance
- **Action:** Stop trading until next day

**Maximum Positions:**
- **Rule:** <2 open trades simultaneously
- **Reason:** Limit exposure and correlation risk

**Spread Check:**
- **Rule:** Current spread <40 cents
- **Reason:** High spreads eat into profits

**Margin Level:**
- **Rule:** Free margin >70%
- **Reason:** Ensure buffer for adverse moves

**Weekend Risk:**
- **Rule:** Don't trade Friday after 20:00 GMT+7
- **Reason:** Avoid weekend gap risk

**Consecutive Losses:**
- **Rule:** <5 losses in a row
- **Action:** Pause trading, review system

**❌ If ANY filter fails → STOP, don't trade**

---

### Layer 2: AI Predictions

**Step 1: LSTM Prediction**
- Input: Last 100 candles × 65 features
- Output: [SELL prob, HOLD prob, BUY prob]
- Extract: Direction (argmax) + Confidence (max prob)

**Step 2: CNN Pattern Detection**
- Input: 128×128 candlestick chart image
- Output: Pattern type (0-9) + Reliability (0-1)

**Step 3: XGBoost News Impact**
- Input: Next news headline + BERT embeddings
- Output: Expected price move in pips

**Requirement:** 
- **LSTM Confidence >70%**
- If <70%, skip this cycle

---

### Layer 3: Traditional Technical Analysis

**Calculate Confluence Score (0-100 points):**

#### A. Fibonacci Golden Pocket (25 points max)
- ✅ 25 pts: Price in 0.618-0.786 zone + rejection candle
- ⚠️ 15 pts: Price near 0.618 or 0.786
- ❌ 0 pts: Price outside key levels

#### B. Order Block (20 points max)
- ✅ 20 pts: OB detected + strong price reaction + volume spike
- ⚠️ 10 pts: OB detected but weak reaction
- ❌ 0 pts: No order block

#### C. Trend Alignment (15 points max)
- ✅ 15 pts: M30, H1, H4 all same direction
- ⚠️ 10 pts: M30 and H1 aligned
- ⚠️ 5 pts: Only M30 trend clear
- ❌ 0 pts: Conflicting trends

#### D. Volume/VWAP Confirmation (10 points max)
- ✅ 10 pts: Volume >150% average + price above/below VWAP
- ⚠️ 5 pts: Volume high OR VWAP aligned (not both)
- ❌ 0 pts: Low volume and no VWAP confirmation

#### E. Indicator Confluence (10 points max)
- 4 pts: MACD crossover aligned with direction
- 3 pts: RSI in optimal zone (BUY: 40-50, SELL: 50-60)
- 3 pts: Price above/below Ichimoku cloud

#### F. Market Structure (15 points max)
- 8 pts: Break of Structure (BOS) in direction
- 7 pts: Market Structure Shift (MSS) confirmed

**Requirement:** Score ≥65

---

### Layer 4: Meta-Decision (Random Forest)

**Process:**
1. Collect 9 meta-features from Layers 1-3
2. Feed into trained Random Forest model
3. Get Trade Decision + Win Probability

**Final Requirements (ALL must pass):**
- ✅ AI Confidence >70%
- ✅ Technical Score ≥65
- ✅ Win Probability >75%

**Decision:**
- **ALL 3 pass** → ENTER TRADE
- **ANY fails** → WAIT for next cycle

---

## 💰 RISK MANAGEMENT

### Position Sizing Formula

**Base Formula:**
```
Lot Size = (Account Balance × Risk%) / SL_pips
```

**Adaptive Risk:**
- **After 0 losses:** 2% per trade
- **After 1 loss:** 1.5% per trade
- **After 2 losses:** 1% per trade
- **After 3+ losses:** 0.5% per trade (defensive mode)

**Constraints:**
- Minimum lot: 0.01
- Maximum lot: 0.10 (for $15-$50 accounts)
- Never risk >2% on any single trade

**Example:**
```
Account: $50
Risk: 2%
SL: 20 pips
Lot = ($50 × 0.02) / 20 = 0.05 lots
```

---

### Entry/Exit Levels

#### Entry
**Method:** Market order at current price
- Ensures immediate execution
- Avoid limit orders (may miss opportunities)

#### Stop Loss Calculation

**For BUY Orders:**
```
SL = min(
    Swing Low - (0.5 × ATR),
    Order Block Support - (0.5 × ATR)
)
```

**For SELL Orders:**
```
SL = max(
    Swing High + (0.5 × ATR),
    Order Block Resistance + (0.5 × ATR)
)
```

**Constraints:**
- Minimum SL: 10 pips
- Maximum SL: 50 pips
- Must be logical level (not arbitrary)

#### Take Profit Calculation

**TP1 (First Target - 50% position):**
```
TP1 = Entry ± (SL_distance × 2.0)
```
- Fibonacci -0.27 extension (preferred)
- Minimum RR: 1:2

**TP2 (Second Target - Remaining 50%):**
```
TP2 = Entry ± (SL_distance × 3.0)
```
- Fibonacci -0.618 extension (preferred)
- Minimum RR: 1:3

**Requirements:**
- TP1 must be ≥1:2 ratio
- TP2 must be ≥1:3 ratio
- Both TPs must be beyond key resistance/support

---

### Position Management

#### At TP1 Hit:
1. ✅ Close 50% of position (lock in profit)
2. ✅ Move SL to breakeven (entry price)
3. ✅ Let remaining 50% run to TP2
4. ✅ Log partial profit

#### After Breakeven:
1. **Trail Stop Loss by 0.5 × ATR**
   - Update every new favorable candle
   - Lock in increasing profit
   - Let winners run

2. **Monitor Emergency Exit Triggers**

#### Emergency Exit Triggers (Close Immediately):

**1. Strong Opposite AI Signal**
- LSTM confidence >85% in opposite direction
- Pattern reversal detected by CNN

**2. Order Block Broken**
- Support/resistance OB violated
- No reaction from price

**3. Unexpected High-Impact News**
- News released during position
- Contradicts trade direction

**4. Volatility Spike**
- ATR >2.5× entry ATR
- Potential for rapid adverse move

**5. Market Structure Shift**
- MSS against position
- Trend reversal confirmed

**6. Position Stalled**
- >4 hours open
- <5 pips profit
- No momentum

---

### Daily Limits (Hard Stops)

**Maximum Daily Loss:** 5% of account
- **Action:** Stop all trading until next day
- **Reason:** Protect from revenge trading

**Maximum Daily Trades:** 5
- **Action:** No more entries today
- **Reason:** Prevent overtrading

**Maximum Drawdown:** 15%
- **Action:** PAUSE all trading
- **Manual Review:** Required before resuming
- **Reason:** Preserve capital

**Maximum Concurrent Positions:** 2
- **Action:** No new trades until one closes
- **Reason:** Limit exposure

---

## 🔄 EXECUTION FLOW

### Main Loop (Every 1 Minute)

```
┌─────────────────────────────────────┐
│  1. Pre-Trade Checks                │
│     └─ FAIL → Skip cycle            │
│     └─ PASS → Continue              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Collect Data                    │
│     - MT5 price data                │
│     - News events                   │
│     - Sentiment scores              │
│     - Market context (DXY, VIX...)  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. Engineer Features (65)          │
│     - Calculate all indicators      │
│     - Detect patterns               │
│     - Compute correlations          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  4. AI Predictions                  │
│     - LSTM: Direction + Confidence  │
│     - CNN: Pattern + Reliability    │
│     - XGBoost: News impact          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  5. Technical Analysis              │
│     - Calculate confluence (0-100)  │
│     - Check Fibonacci levels        │
│     - Verify order blocks           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  6. Meta-Decision (Random Forest)   │
│     - Combine all signals           │
│     - Output: Trade or Wait         │
│     - Win probability               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  7. Log Decision                    │
│     - Save to database              │
│     - Record all features           │
│     - Timestamp                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  8. Execute Trade (if approved)     │
│     - Calculate lot size            │
│     - Set SL/TP levels              │
│     - Send order to MT5             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  9. Manage Existing Positions       │
│     - Check TP1 hit                 │
│     - Trail stop loss               │
│     - Monitor exit triggers         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  10. Update Metrics                 │
│      - Win rate, profit factor      │
│      - Drawdown                     │
│      - Check pause triggers         │
└─────────────────────────────────────┘
```

---

### Trade Execution Sequence

```
TRADE APPROVED
       ↓
Calculate Entry Price
       ↓
Calculate SL (10-50 pips)
       ↓
Calculate TP1 (RR ≥1:2)
Calculate TP2 (RR ≥1:3)
       ↓
Verify RR Ratio ≥1:3
       ↓
Calculate Lot Size (adaptive risk)
       ↓
Send Market Order to MT5
       ↓
Verify Order Executed
       ↓
Set TP1 as Initial TP
       ↓
Log Trade Details
       ↓
MONITOR EVERY MINUTE
       ↓
┌──────────────────────┐
│  TP1 Hit?            │
│  YES → Close 50%     │
│     → Move SL to BE  │
│  NO → Continue       │
└──────────────────────┘
       ↓
Trail SL After Breakeven
       ↓
Check Emergency Exits
       ↓
TP2 Hit or SL Hit
       ↓
Close Position
       ↓
Log Results
```

---

## 📈 PERFORMANCE MONITORING

### Real-Time Metrics (Tracked Continuously)

**Trade Statistics:**
- Total trades executed
- Winning trades
- Losing trades
- Win rate % (target: >50%)
- Current streak (wins/losses)

**Profit Metrics:**
- Net profit/loss ($)
- Total gross profit ($)
- Total gross loss ($)
- Profit factor (target: >1.5)
- Average win ($)
- Average loss ($)
- Largest win ($)
- Largest loss ($)

**Risk Metrics:**
- Current drawdown (%)
- Maximum drawdown (%)
- Sharpe ratio (target: >1.0)
- Risk-adjusted return
- Recovery factor

**AI Performance:**
- LSTM accuracy (rolling 30 days)
- CNN pattern detection rate
- XGBoost RMSE
- Random Forest precision

---

### Auto-Pause Triggers

**System will STOP trading if:**

1. **5 Consecutive Losses**
   - Action: Pause 24 hours
   - Reason: Possible model drift or market regime change

2. **15% Drawdown Reached**
   - Action: Pause indefinitely
   - Reason: Capital preservation
   - Resume: Manual review required

3. **Win Rate <35%** (after 20+ trades)
   - Action: Enter defensive mode (0.5% risk)
   - Reason: System underperforming

4. **Daily Loss ≥5%**
   - Action: Stop until next day
   - Reason: Daily limit reached

---

### Model Drift Detection

**Monitor Monthly:**
- LSTM prediction accuracy
- CNN pattern recognition rate
- XGBoost news impact RMSE
- Random Forest win probability calibration

**If accuracy drops >10% below baseline:**
1. ⚠️ Increase technical analysis weight to 60%
2. ⚠️ Reduce AI signal weight to 40%
3. 🔄 Retrain models on recent data
4. ✅ Resume normal operation after validation

---

## 🎓 TRAINING PIPELINE

### Phase 1: Data Collection (2 weeks)

**Week 1: Historical Price Data**
- Source: MetaTrader 5
- Symbol: XAUUSD
- Timeframe: M1 (primary collection)
- Period: 2+ years (2022-2024)
- Records: ~1 million candles
- Storage: SQLite database

**Week 2: News & Sentiment Data**
- Forex Factory: Scrape 2023-2025 news calendar
- News API: Collect gold-related headlines
- BERT Processing: Generate sentiment scores
- Storage: JSON + SQLite

**Week 2: Market Context Data**
- Yahoo Finance: DXY, US10Y, VIX, SPY
- Period: 2+ years
- Frequency: Daily + Intraday
- Storage: CSV + SQLite

**Labeling:**
- BUY: Price rises >10 pips in next 30 min
- SELL: Price drops >10 pips in next 30 min
- HOLD: Price moves <10 pips in next 30 min

---

### Phase 2: Model Training (4 weeks)

**Week 1: LSTM Price Predictor**
- Prepare sequences: 100 candles × 65 features
- Train/Val/Test split: 70/15/15%
- Epochs: 100
- Early stopping: Patience 10
- Target: >60% accuracy on test set

**Week 2: CNN Pattern Detector**
- Generate chart images: 128×128 grayscale
- Augmentation: Rotation (±10°), zoom (0.9-1.1×)
- Train/Val/Test split: 70/15/15%
- Epochs: 50
- Target: >55% accuracy

**Week 3: XGBoost News Impact Scorer**
- Generate BERT embeddings (768 dimensions)
- Add impact level + surprise factor
- 5-fold cross-validation
- Hyperparameter tuning: GridSearchCV
- Target: RMSE <15 pips

**Week 4: Random Forest Meta-Learner**
- Collect all model outputs + technical scores
- Generate historical trading signals
- Balance classes with SMOTE
- Train: 200 trees, depth 10
- Target: >70% accuracy, >0.75 precision

---

### Phase 3: Backtesting (4 weeks)

**Week 1: Historical Backtest**
- Period: 6 months out-of-sample data
- Method: Walk-forward analysis
- Window: Rolling 30 days
- Metrics: Win rate, profit factor, drawdown

**Week 2: Stress Testing**
- High volatility periods (news events)
- Low volatility periods (consolidation)
- Trending markets
- Ranging markets

**Week 3: Monte Carlo Simulation**
- Runs: 1000 simulations
- Randomize: Trade order, entry timing
- Analyze: Probability of ruin, expected return

**Week 4: Optimization & Refinement**
- Adjust thresholds (AI confidence, technical score)
- Fine-tune risk parameters
- Validate changes

**Targets:**
- Win rate: >45%
- Profit factor: >1.5
- Max drawdown: <15%
- Sharpe ratio: >1.0

---

### Phase 4: Paper Trading (4 weeks)

**Setup:**
- MT5 demo account: $100
- All systems live except real money
- Real-time execution

**Week 1-2: Monitoring**
- Track every trade
- Log all signals
- Monitor execution quality

**Week 3-4: Validation**
- Calculate performance metrics
- Compare to backtest results
- Fix any bugs or issues

**Go/No-Go Criteria:**
- ✅ Win rate >45%
- ✅ Profit factor >1.5
- ✅ No major bugs
- ✅ Consistent daily performance

---

### Phase 5: Live Trading

**Week 1: Minimum Capital**
- Start: $15-$25
- Risk: 2% per trade (max $0.50)
- Lot size: 0.01 minimum
- Monitor: Every day

**Month 2-3: Observation**
- Track all metrics
- Verify profitability
- Document any issues

**Month 4+: Scale (If Profitable)**
- Add capital gradually
- Increase lot sizes
- Maintain same risk %

---

## ⚙️ TECHNICAL REQUIREMENTS

### Hardware Requirements

**Minimum:**
- CPU: Intel Core i5 / AMD Ryzen 5 (4 cores)
- RAM: 16GB
- Storage: 50GB SSD
- Internet: Stable broadband (10 Mbps+)
- OS: Windows 10/11, Linux, or macOS

**Recommended:**
- CPU: Intel Core i7 / AMD Ryzen 7 (8 cores)
- RAM: 32GB
- Storage: 100GB NVMe SSD
- GPU: NVIDIA GTX 1660+ (for training only)
- Internet: Fiber optic connection

**VPS (Optional):**
- Provider: Vultr, DigitalOcean, AWS EC2
- Specs: 2 vCPU, 4GB RAM, 50GB SSD
- Location: Near broker server (latency <50ms)
- Cost: $20-40/month

---

### Software Requirements

**Python Environment:**
- Python: 3.9+
- Package manager: pip or conda

**AI/ML Libraries:**
```
tensorflow==2.13.0          # LSTM, CNN models
keras==2.13.0              # High-level API
xgboost==2.0.0             # News impact scorer
scikit-learn==1.3.0        # Random Forest, preprocessing
transformers==4.30.0       # BERT sentiment analysis
torch==2.0.0               # PyTorch (alternative)
```

**Trading Platform:**
```
MetaTrader5==5.0.45        # MT5 Python API
```

**Data Science:**
```
pandas==2.0.0              # Data manipulation
numpy==1.24.0              # Numerical computing
ta-lib==0.4.26             # Technical indicators
matplotlib==3.7.0          # Plotting
seaborn==0.12.0            # Statistical visualization
```

**Data Collection:**
```
beautifulsoup4==4.12.0     # Web scraping
selenium==4.10.0           # Browser automation
requests==2.31.0           # HTTP requests
yfinance==0.2.0            # Yahoo Finance API
newsapi-python==0.2.7      # News API
```

**Database:**
```
sqlite3                     # Built-in Python
sqlalchemy==2.0.0          # ORM
```

**Utilities:**
```
python-telegram-bot==20.3  # Telegram alerts
schedule==1.2.0            # Task scheduling
python-dotenv==1.0.0       # Environment variables
loguru==0.7.0              # Logging
```

---

### MetaTrader 5 Setup

**Broker Requirements:**
- Symbol: XAUUSD available
- Spread: <30 cents average
- Account type: Standard or ECN
- Minimum deposit: $15-$50
- Leverage: 1:100 or higher

**Recommended Brokers:**
- IC Markets
- Pepperstone
- FBS
- XM

**MT5 Installation:**
1. Download from broker website
2. Install on Windows/Linux/Mac
3. Login with demo/live account
4. Enable Python API: Tools → Options → Expert Advisors → Allow DLL imports

---

## 📁 PROJECT STRUCTURE

```
TRADE/
│
├── README.md                      # This file
├── PROJECT_STATUS.md              # Track progress
├── TODO.md                        # Task list
├── CHANGELOG.md                   # Version history
│
├── data/
│   ├── raw/                       # Raw collected data
│   │   ├── mt5_xauusd_m1.csv
│   │   ├── news_events.json
│   │   └── sentiment_scores.csv
│   ├── processed/                 # Engineered features
│   │   └── features_65.csv
│   └── labels/                    # Training labels
│       └── labels_buy_sell_hold.csv
│
├── models/
│   ├── lstm_price_predictor.h5    # Trained LSTM
│   ├── cnn_pattern_detector.h5    # Trained CNN
│   ├── xgboost_news_scorer.pkl    # Trained XGBoost
│   ├── rf_meta_learner.pkl        # Trained Random Forest
│   └── bert_sentiment/            # BERT model
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── mt5_collector.py       # Collect price data
│   │   ├── news_scraper.py        # Scrape Forex Factory
│   │   ├── sentiment_analyzer.py  # BERT sentiment
│   │   └── market_context.py      # DXY, VIX, etc.
│   │
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py # Calculate 65 features
│   │   ├── smart_money.py          # Order blocks, BOS
│   │   └── pattern_recognition.py  # Candlestick patterns
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM implementation
│   │   ├── cnn_model.py            # CNN implementation
│   │   ├── xgboost_model.py        # XGBoost implementation
│   │   └── rf_model.py             # Random Forest meta-learner
│   │
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── signal_generator.py     # Generate trade signals
│   │   ├── risk_manager.py         # Position sizing, SL/TP
│   │   ├── executor.py             # Execute trades in MT5
│   │   └── position_manager.py     # Manage open positions
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py      # Backtest framework
│   │   ├── performance_metrics.py  # Calculate metrics
│   │   └── visualization.py        # Plot results
│   │
│   └── utils/
│       ├── __init__.py
│       ├── database.py             # SQLite operations
│       ├── logger.py               # Logging setup
│       ├── telegram_bot.py         # Telegram notifications
│       └── config.py               # Configuration
│
├── config/
│   ├── config.yaml                 # Main configuration
│   ├── model_params.yaml           # Model hyperparameters
│   └── .env                        # API keys (not in git)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── tests/
│   ├── test_data_collection.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_trading.py
│
├── logs/
│   ├── trades.log                  # Trade execution logs
│   ├── errors.log                  # Error logs
│   └── performance.log             # Daily metrics
│
├── results/
│   ├── backtest_results.csv
│   ├── live_trades.csv
│   └── performance_charts/
│
├── scripts/
│   ├── train_all_models.py         # Train pipeline
│   ├── run_backtest.py             # Run backtest
│   ├── run_live.py                 # Live trading
│   └── data_update.py              # Update data
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── .gitignore                      # Git ignore rules
```

---

## 🚀 INSTALLATION GUIDE

### Step 1: Clone Repository
```bash
cd C:\Users\ASUS\Desktop
git clone https://github.com/yourusername/ai-gold-trading-bot.git TRADE
cd TRADE
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install TA-Lib (Technical Indicators)
**Windows:**
```bash
# Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.26‑cp39‑cp39‑win_amd64.whl
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y ta-lib
pip install TA-Lib
```

### Step 5: Setup MetaTrader 5
1. Download MT5 from broker
2. Install and login
3. Enable Python API in settings
4. Note your MT5 installation path

### Step 6: Configure Environment
```bash
cp config/.env.example config/.env
# Edit .env with your API keys
```

**config/.env:**
```
# MetaTrader 5
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# News API
NEWS_API_KEY=your_newsapi_key

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database
DATABASE_PATH=data/trading.db

# Risk Parameters
INITIAL_CAPITAL=50.0
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
```

### Step 7: Download Historical Data
```bash
python scripts/data_update.py --symbol XAUUSD --start 2022-01-01 --end 2024-12-31
```

### Step 8: Train Models
```bash
python scripts/train_all_models.py
```

### Step 9: Run Backtest
```bash
python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-30
```

### Step 10: Start Live Trading
```bash
# Paper trading (demo account)
python scripts/run_live.py --mode demo

# Live trading (real account)
python scripts/run_live.py --mode live
```

---

## ⚠️ RISK WARNINGS

### 1. AI Model Limitations
- **Model Drift:** AI models degrade over time as market conditions change
- **Black Swan Events:** Models cannot predict unprecedented events (COVID-19, wars)
- **Overfitting:** Models may perform well on historical data but fail live
- **Data Quality:** Garbage in = garbage out

### 2. Market Risks
- **Volatility:** Gold extremely volatile, can move 50+ pips in seconds
- **Slippage:** Actual execution price may differ from expected
- **Spread Widening:** Spreads can spike during news (10 cents → 100 cents)
- **Gap Risk:** Weekend gaps can bypass stop losses
- **Liquidity:** Low liquidity periods (Asian session) = higher costs

### 3. Technical Risks
- **API Failures:** MT5 API, News API, data sources can fail
- **Connection Drops:** Internet outage during open position = risk
- **Code Bugs:** Software bugs can cause incorrect trades
- **Server Downtime:** VPS or broker server maintenance
- **Latency:** Slow execution = worse prices

### 4. Small Account Challenges
- **High Cost Ratio:** Spreads eat larger % of profits on small accounts
- **Limited Lot Sizes:** 0.01 lot minimum = less flexibility
- **Psychological Pressure:** Watching small balance fluctuate
- **Compounding Time:** Takes longer to grow small accounts

### 5. Regulatory & Broker Risks
- **Broker Reliability:** Some brokers manipulate prices or reject trades
- **Account Restrictions:** Limitations on scalping, hedging
- **Withdrawal Issues:** Difficulty getting money out
- **Regulatory Changes:** Laws affecting forex trading

### 6. Model-Specific Risks
- **LSTM:** Sensitive to input sequence length, requires large data
- **CNN:** Image quality affects pattern recognition
- **XGBoost:** Overfits easily without proper tuning
- **BERT:** Misinterprets sarcasm, context in news

### 7. System Risks
- **Over-Optimization:** Curve-fitting to historical data
- **Look-Ahead Bias:** Using future data in training
- **Survivorship Bias:** Only studying successful periods
- **Confirmation Bias:** Ignoring contradictory signals

---

## 🛡️ RISK MITIGATION STRATEGIES

✅ **Never risk more than 2% per trade**
✅ **Use stop losses on EVERY trade**
✅ **Maintain 15% max drawdown limit**
✅ **Diversify: Don't put all capital in one account**
✅ **Regular model retraining (monthly)**
✅ **Monitor performance metrics daily**
✅ **Have kill switch for emergencies**
✅ **Keep detailed logs for post-analysis**
✅ **Start with demo account**
✅ **Scale up slowly after proven results**

---

## 📊 EXPECTED PERFORMANCE

### Realistic Targets (for $15-$50 account)

| Metric | Conservative | Moderate | Aggressive |
|--------|-------------|----------|------------|
| **Win Rate** | 50-52% | 52-55% | 55-58% |
| **Profit Factor** | 1.5-1.8 | 1.8-2.2 | 2.2-2.5 |
| **Monthly Return** | 5-10% | 10-20% | 20-30% |
| **Max Drawdown** | <10% | <15% | <20% |
| **Sharpe Ratio** | 1.0-1.5 | 1.5-2.0 | 2.0-2.5 |
| **Avg RR** | 1:2 | 1:2.5 | 1:3 |

### Trade Frequency
- **Daily:** 1-3 trades
- **Weekly:** 5-15 trades
- **Monthly:** 20-60 trades

### Time to Double Account
- **Conservative:** 7-14 months
- **Moderate:** 4-7 months
- **Aggressive:** 2-4 months

### Model Performance Benchmarks
| Model | Metric | Target | Good | Excellent |
|-------|--------|--------|------|-----------|
| **LSTM** | Accuracy | >60% | 62-65% | >65% |
| **CNN** | Accuracy | >55% | 57-60% | >60% |
| **XGBoost** | RMSE | <15 pips | <12 pips | <10 pips |
| **Random Forest** | Accuracy | >70% | 72-75% | >75% |

---

## 📞 SUPPORT & MAINTENANCE

### Daily Tasks
- ✅ Check system health
- ✅ Review overnight trades
- ✅ Monitor drawdown
- ✅ Check for software updates

### Weekly Tasks
- ✅ Analyze performance metrics
- ✅ Review losing trades
- ✅ Update news database
- ✅ Backup database

### Monthly Tasks
- ✅ Retrain AI models (if drift detected)
- ✅ Comprehensive performance review
- ✅ Optimize parameters (if needed)
- ✅ Generate monthly report

### Emergency Procedures
1. **System Crash:** Manually close all positions in MT5
2. **Data Feed Loss:** Switch to backup data source
3. **Model Error:** Disable AI, use technical analysis only
4. **Broker Issues:** Contact broker support immediately

---

## 📝 LICENSE

This project is for educational purposes only. Use at your own risk.

**Disclaimer:** Trading forex and gold involves substantial risk of loss. Past performance does not guarantee future results. This is not financial advice.

---

## 🤝 CONTRIBUTING

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 📧 CONTACT

- **Email:** your-email@example.com
- **Telegram:** @yourusername
- **GitHub:** github.com/yourusername

---

**Last Updated:** 2024
**Version:** 1.0.0
**Status:** Development Phase

---

## 🎯 QUICK START SUMMARY

1. ✅ Install Python 3.9+
2. ✅ Install MetaTrader 5
3. ✅ Clone repository
4. ✅ Install dependencies
5. ✅ Configure .env file
6. ✅ Download historical data (2+ years)
7. ✅ Train all 4 AI models (~4 weeks)
8. ✅ Run backtest (6 months data)
9. ✅ Paper trade (1 month demo)
10. ✅ Go live with $15-$50

**Total Timeline:** ~3 months from start to live trading

---

**🚀 Good luck and trade responsibly!**