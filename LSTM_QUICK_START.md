# 🧠 LSTM Training - Quick Start Guide

**Get your LSTM model trained in 5 minutes!** ⚡

---

## ⚡ Super Quick Start

```bash
# Install dependencies (if needed)
pip install tensorflow keras scikit-learn

# Train model NOW! (15-30 minutes)
python train_models.py
```

**That's it!** ✅

---

## 📊 What You'll Get

After training completes:

```
✅ Trained LSTM model → models/lstm_best_model.h5
✅ Performance metrics → results/lstm/lstm_results.json  
✅ Training charts → results/lstm/lstm_training_results.png
✅ Scaler for predictions → models/lstm_scaler.pkl
```

**Expected Accuracy:** 55-60% (anything >52% is tradeable!)

---

## 🎯 Training Options

### 1. Default (Recommended) ⭐
```bash
python train_models.py
```
- **Time:** 15-30 min
- **Architecture:** Bidirectional LSTM
- **Features:** Top 70
- **Best for:** Production trading

---

### 2. Quick Test (Fast)
```bash
python train_models.py --epochs 10 --batch-size 128
```
- **Time:** 2-3 min
- **Best for:** Testing setup

---

### 3. Compare All Architectures
```bash
python train_models.py --mode all
```
- **Time:** 1-2 hours
- **Compares:** Simple, Stacked, Bidirectional, Attention
- **Best for:** Finding optimal architecture

---

### 4. Different Timeframes
```bash
# Short-term (5-min candles)
python train_models.py --timeframe M5 --sequence-length 120

# Medium-term (15-min candles) ⭐ DEFAULT
python train_models.py --timeframe M15 --sequence-length 60

# Long-term (1-hour candles)
python train_models.py --timeframe H1 --sequence-length 30
```

---

### 5. Custom Configuration
```bash
python train_models.py \
  --mode bidirectional \
  --timeframe M15 \
  --sequence-length 120 \
  --batch-size 64 \
  --epochs 100 \
  --learning-rate 0.001 \
  --features 70
```

---

## 📈 During Training

You'll see:
```
================================================================================
🧠 LSTM Training Pipeline Initialized
================================================================================
📂 Data: data/processed/XAUUSD_M15_features_complete.csv
📊 Sequence Length: 60
🎯 Prediction Horizon: 1
🔢 Batch Size: 64
🔁 Epochs: 100
📈 Learning Rate: 0.001
✨ Top Features: 70
================================================================================

📥 Loading data...
✅ Loaded 5,962 rows × 143 columns

🎯 Selecting top 70 features...
✅ Selected 70 features

📊 Target Distribution:
   Down (0): 2,954 (49.55%)
   Up (1):   3,008 (50.45%)

🔄 Creating sequences...
✅ Created 5,902 sequences

🚀 Starting training...

Epoch 1/100
67/67 [==============================] - 12s - loss: 0.6923 - accuracy: 0.5124
Epoch 2/100
67/67 [==============================] - 10s - loss: 0.6901 - accuracy: 0.5234
...
Epoch 45/100
67/67 [==============================] - 10s - loss: 0.4532 - accuracy: 0.7856

Early stopping triggered. Best weights restored.

✅ Training completed!

================================================================================
🎯 PERFORMANCE METRICS
================================================================================
Train Accuracy: 0.7856 (78.56%)
Val Accuracy:   0.5673 (56.73%)
Test Accuracy:  0.5593 (55.93%)  ← THIS IS WHAT MATTERS!

📈 Test Set Metrics:
   Precision: 0.5812
   Recall:    0.5423
   F1-Score:  0.5611

✅ PIPELINE COMPLETED SUCCESSFULLY!
⏱️ Total time: 0:23:45
```

---

## 🎯 What's Good?

**Good Results ✅:**
- Test accuracy > 52% ✅
- Test accuracy > 55% 🎉
- Test accuracy > 60% 🚀
- Precision & Recall balanced (diff < 0.10) ✅
- Train/Test gap < 15% ✅

**Bad Results ❌:**
- Test accuracy ≤ 50% (random guessing)
- Train accuracy 95%, Test 51% (severe overfitting)
- Only predicts one class
- Model crashes/errors

---

## 🔧 Troubleshooting

### Problem: Out of Memory
```bash
# Solution: Reduce batch size
python train_models.py --batch-size 16 --features 30
```

---

### Problem: Too Slow
```bash
# Solution: Larger batches, fewer features
python train_models.py --batch-size 128 --features 50 --epochs 50
```

---

### Problem: Not Learning (Stuck at 50%)
```bash
# Solution: More features, longer sequences
python train_models.py --features 0 --sequence-length 120 --learning-rate 0.005
```

---

### Problem: Severe Overfitting
```bash
# Solution: Simpler model, fewer features
python train_models.py --mode simple --features 30 --learning-rate 0.0005
```

---

## 📊 After Training

### 1. Check Results
```bash
# Windows
type results\lstm\lstm_results.json

# Mac/Linux
cat results/lstm/lstm_results.json
```

---

### 2. View Charts
```bash
# Windows
start results\lstm\lstm_training_results.png

# Mac/Linux
open results/lstm/lstm_training_results.png
```

---

### 3. Compare with XGBoost
```bash
python src/models/compare_models.py
```

---

### 4. Run Backtesting
```bash
python src/backtesting/run_backtest.py --model lstm
```

---

## 💡 Command Reference

| Command | Purpose | Time |
|---------|---------|------|
| `python train_models.py` | Default training | 15-30 min |
| `--epochs 10` | Quick test | 2-3 min |
| `--mode all` | Compare architectures | 1-2 hours |
| `--timeframe M5` | 5-min candles | 20-40 min |
| `--timeframe H1` | 1-hour candles | 5-10 min |
| `--batch-size 128` | Faster training | Varies |
| `--batch-size 32` | More stable | Varies |
| `--features 30` | Fewer features | Faster |
| `--features 0` | All features | Slower |
| `--learning-rate 0.0005` | Lower LR | More stable |
| `--learning-rate 0.005` | Higher LR | Less stable |

---

## 🚀 Next Steps

After successful training:

1. **Compare Models**
   ```bash
   python src/models/compare_models.py
   ```

2. **Backtest Strategy**
   ```bash
   python src/backtesting/run_backtest.py
   ```

3. **Paper Trading**
   ```bash
   python src/trading/paper_trading.py
   ```

4. **Train Other Timeframes**
   ```bash
   python train_models.py --timeframe M5
   python train_models.py --timeframe H1
   python train_models.py --timeframe H4
   ```

5. **Build Ensemble**
   - Combine LSTM + XGBoost
   - Weight predictions: `0.6*LSTM + 0.4*XGBoost`

---

## 📚 Full Documentation

For detailed info, see:
- **Complete Guide:** `LSTM_TRAINING_GUIDE.md` (820 lines)
- **Phase Status:** `PHASE_D_LSTM_READY.md`
- **Feature Analysis:** `PHASE_C_SUMMARY.md`

---

## ❓ FAQ

**Q: How long does it take?**  
A: 15-30 minutes (CPU), 3-5 minutes (GPU)

**Q: What accuracy is good?**  
A: >55% is good, >60% is excellent, >52% is tradeable

**Q: Can I stop training early?**  
A: Yes! Press `Ctrl+C`. Best model already saved.

**Q: Need GPU?**  
A: No, but 5-10x faster with GPU

**Q: Which timeframe?**  
A: Start with M15 (best balance)

**Q: Which architecture?**  
A: Bidirectional LSTM (best for time-series)

---

## 🎉 You're Ready!

Everything is set up. Just run:

```bash
python train_models.py
```

And wait for your model! ☕

---

**Good luck! May your accuracy be high! 📈**