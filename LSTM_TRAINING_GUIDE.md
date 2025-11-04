# 🧠 LSTM Model Training Guide

**Last Updated:** 2024  
**Status:** Ready for Training  
**Purpose:** Train deep learning LSTM models for Gold price prediction

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [What is LSTM?](#what-is-lstm)
3. [Training Options](#training-options)
4. [Architecture Comparison](#architecture-comparison)
5. [Configuration Guide](#configuration-guide)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## 🚀 Quick Start

### Option 1: Default Training (RECOMMENDED)
Train bidirectional LSTM with optimal settings:

```bash
python train_models.py
```

This will:
- Use M15 timeframe data
- Train bidirectional LSTM (best for time-series)
- Use top 70 features
- Train for 100 epochs with early stopping
- Save model to `models/` directory
- Save results to `results/lstm/`

**Expected time:** 10-30 minutes depending on your hardware

---

### Option 2: Quick Test (Fast)
For quick testing with fewer epochs:

```bash
python train_models.py --epochs 20 --batch-size 128
```

**Expected time:** 3-5 minutes

---

### Option 3: Train All Architectures
Compare all 4 LSTM architectures:

```bash
python train_models.py --mode all
```

This trains:
1. Simple LSTM
2. Stacked LSTM
3. Bidirectional LSTM ⭐ (recommended)
4. LSTM with Attention

**Expected time:** 1-2 hours

---

## 🤔 What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) designed for sequential data like time-series.

### Why LSTM for Gold Trading?

✅ **Captures Temporal Patterns** - Remembers past price movements  
✅ **Handles Long Dependencies** - Connects events far apart in time  
✅ **Sequential Processing** - Processes data in order (like candles)  
✅ **Non-linear Relationships** - Captures complex market dynamics  

### vs. XGBoost

| Feature | LSTM | XGBoost |
|---------|------|---------|
| **Data Type** | Sequential | Tabular |
| **Memory** | Built-in (remembers past) | None (each row independent) |
| **Training Time** | Slower (hours) | Faster (minutes) |
| **Interpretability** | Low (black box) | High (feature importance) |
| **Best For** | Temporal patterns | Feature interactions |

**Our Strategy:** Use BOTH! LSTM for patterns + XGBoost for features = Ensemble 🎯

---

## 🎛️ Training Options

### Basic Syntax

```bash
python train_models.py [OPTIONS]
```

### Available Options

#### 1. **Mode** (Architecture Type)
```bash
--mode simple          # Single LSTM layer (fast, basic)
--mode stacked         # Multiple LSTM layers (deeper)
--mode bidirectional   # Process forward+backward (BEST) ⭐
--mode attention       # LSTM + attention mechanism (advanced)
--mode all             # Train all 4 and compare
--mode custom          # Use custom parameters
```

#### 2. **Timeframe**
```bash
--timeframe M5    # 5-minute candles (more data, noisy)
--timeframe M15   # 15-minute candles (RECOMMENDED) ⭐
--timeframe M30   # 30-minute candles
--timeframe H1    # 1-hour candles
--timeframe H4    # 4-hour candles (less data, smoother)
--timeframe D1    # Daily candles (very little data)
```

**Recommendation:** Start with M15 (good balance of data volume and noise)

#### 3. **Sequence Length** (Lookback Window)
```bash
--sequence-length 30    # Look back 30 candles (7.5 hours for M15)
--sequence-length 60    # Look back 60 candles (15 hours) ⭐ DEFAULT
--sequence-length 120   # Look back 120 candles (30 hours)
--sequence-length 240   # Look back 240 candles (60 hours)
```

**Rule of thumb:**
- Shorter (30-60): Better for scalping, faster predictions
- Longer (120-240): Better for swing trading, captures trends

#### 4. **Batch Size**
```bash
--batch-size 32    # Small batch (slower, more stable)
--batch-size 64    # Medium batch (RECOMMENDED) ⭐
--batch-size 128   # Large batch (faster, less stable)
```

**GPU users:** Use larger batch sizes (128, 256)  
**CPU users:** Use smaller batch sizes (32, 64)

#### 5. **Epochs**
```bash
--epochs 20     # Quick test
--epochs 50     # Medium training
--epochs 100    # Full training (DEFAULT) ⭐
--epochs 200    # Extended training
```

**Note:** Early stopping will stop training if no improvement for 10 epochs.

#### 6. **Learning Rate**
```bash
--learning-rate 0.0001   # Slow, stable
--learning-rate 0.001    # DEFAULT ⭐
--learning-rate 0.01     # Fast, risky
```

#### 7. **Features**
```bash
--features 30    # Top 30 features only (faster)
--features 70    # Top 70 features (RECOMMENDED) ⭐
--features 0     # Use all features (slower, may overfit)
```

---

## 📊 Training Examples

### Example 1: Quick Test Run
```bash
python train_models.py --epochs 10 --batch-size 128
```
**Use when:** Testing setup, debugging code  
**Time:** 2-3 minutes

---

### Example 2: Production Training (Recommended)
```bash
python train_models.py --mode bidirectional --timeframe M15 --sequence-length 60 --epochs 100 --features 70
```
**Use when:** Final model for trading  
**Time:** 15-30 minutes

---

### Example 3: Multi-Timeframe Analysis
```bash
# Train on M5 (short-term signals)
python train_models.py --timeframe M5 --sequence-length 120

# Train on H1 (medium-term signals)
python train_models.py --timeframe H1 --sequence-length 60

# Train on H4 (long-term signals)
python train_models.py --timeframe H4 --sequence-length 30
```
**Use when:** Building multi-timeframe ensemble  
**Time:** 1 hour total

---

### Example 4: Architecture Comparison
```bash
python train_models.py --mode all --epochs 50
```
**Use when:** Finding best architecture  
**Time:** 45-60 minutes

---

### Example 5: High-Performance Training (GPU)
```bash
python train_models.py --batch-size 256 --epochs 200 --learning-rate 0.002
```
**Use when:** Have powerful GPU, want best accuracy  
**Time:** 30-60 minutes

---

## 🏗️ Architecture Comparison

### 1. **Simple LSTM**
```
Input → LSTM(64) → Dropout → Dense(32) → Output
```
**Pros:** Fast, simple, good baseline  
**Cons:** Limited capacity, may underfit  
**Use when:** Quick testing, limited data

---

### 2. **Stacked LSTM**
```
Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Output
```
**Pros:** More capacity, captures complex patterns  
**Cons:** Slower, prone to overfitting  
**Use when:** Have lots of data, need deep model

---

### 3. **Bidirectional LSTM** ⭐ RECOMMENDED
```
Input → BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(32) → Output
```
**Pros:** Best for time-series, processes forward+backward  
**Cons:** 2x parameters, slightly slower  
**Use when:** Production model, want best accuracy

**Why best?** Gold prices influenced by both past AND future events (news releases, patterns forming). BiLSTM sees both directions!

---

### 4. **LSTM with Attention**
```
Input → LSTM(64) → Attention Layer → Dense(32) → Output
```
**Pros:** Focuses on important timesteps, interpretable  
**Cons:** More complex, harder to tune  
**Use when:** Research, want to visualize what model focuses on

---

## ⚙️ Configuration Guide

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 8 GB | 16 GB | 32 GB |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **GPU** | None (CPU only) | GTX 1060 6GB | RTX 3080 12GB |
| **Storage** | 5 GB free | 10 GB free | 20 GB+ free |

**Note:** Training works on CPU but is 5-10x slower than GPU.

---

### Hyperparameter Tuning Tips

#### If Model Overfits (Train >> Test Accuracy):
```bash
# Increase regularization
--learning-rate 0.0005   # Lower learning rate
--batch-size 32          # Smaller batches
--features 50            # Fewer features
```

Add more dropout in `src/models/train_lstm.py`:
```python
layers.Dropout(0.5)  # Increase from 0.3 to 0.5
```

---

#### If Model Underfits (Both Accuracies Low):
```bash
# Increase capacity
--sequence-length 120    # Longer sequences
--epochs 200             # More training
--features 0             # Use all features
```

Use stacked or attention architecture:
```bash
--mode stacked
```

---

#### If Training is Too Slow:
```bash
--batch-size 128      # Larger batches
--epochs 50           # Fewer epochs
--features 30         # Fewer features
--timeframe H1        # Higher timeframe (less data)
```

---

#### If Want Best Accuracy (Don't Care About Speed):
```bash
--mode bidirectional
--sequence-length 120
--batch-size 32
--epochs 200
--learning-rate 0.0005
--features 0
```

---

## 📈 Expected Results

### Target Metrics (Test Set)

| Metric | Minimum Target | Good | Excellent |
|--------|----------------|------|-----------|
| **Accuracy** | > 52% | > 55% | > 60% |
| **Precision** | > 0.53 | > 0.58 | > 0.65 |
| **Recall** | > 0.50 | > 0.55 | > 0.62 |
| **F1-Score** | > 0.51 | > 0.56 | > 0.63 |

**Note:** 50% is random guessing. Anything above 52% is tradeable with proper risk management!

---

### Typical Training Output

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

🔧 Handling missing values...

📊 Target Distribution:
   Down (0): 2,954 (49.55%)
   Up (1):   3,008 (50.45%)

🔄 Creating sequences...
📏 Scaling features...
✅ Created 5,902 sequences
   Shape: (5902, 60, 70) (samples, timesteps, features)

📂 Dataset splits:
   Train: 4,250 sequences (72.0%)
   Val:   472 sequences (8.0%)
   Test:  1,180 sequences (20.0%)

🏗️ Building bidirectional LSTM model...

📋 Model Architecture:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional (Bidirectional (None, 60, 128)          69,120    
_________________________________________________________________
dropout (Dropout)            (None, 60, 128)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64)                41,216    
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense (Dense)                (None, 32)                2,080     
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 112,449
Trainable params: 112,449
Non-trainable params: 0

🚀 Starting training...

Epoch 1/100
67/67 [==============================] - 12s - loss: 0.6923 - accuracy: 0.5124
Epoch 2/100
67/67 [==============================] - 10s - loss: 0.6901 - accuracy: 0.5234
...
Epoch 45/100
67/67 [==============================] - 10s - loss: 0.4532 - accuracy: 0.7856
Epoch 46/100
67/67 [==============================] - 10s - loss: 0.4489 - accuracy: 0.7912

Early stopping triggered. Best weights restored.

✅ Training completed!

================================================================================
🎯 PERFORMANCE METRICS
================================================================================
Train Accuracy: 0.7912 (79.12%)
Val Accuracy:   0.5673 (56.73%)
Test Accuracy:  0.5593 (55.93%)

📈 Test Set Metrics:
   Precision: 0.5812
   Recall:    0.5423
   F1-Score:  0.5611

📋 Classification Report (Test Set):
              precision    recall  f1-score   support

    Down (0)       0.54      0.58      0.56       582
      Up (1)       0.58      0.54      0.56       598

    accuracy                           0.56      1180
   macro avg       0.56      0.56      0.56      1180
weighted avg       0.56      0.56      0.56      1180

✅ PIPELINE COMPLETED SUCCESSFULLY!
⏱️ Total time: 0:23:45
🎯 Test Accuracy: 0.5593
📊 Test F1-Score: 0.5611
```

---

### Understanding the Output

**Good Signs ✅:**
- Test accuracy > 52%
- Train accuracy not too far from test (< 15% difference)
- Precision and recall balanced (difference < 0.10)
- Training completed without errors

**Bad Signs ❌:**
- Test accuracy ≤ 50% (no better than random)
- Train accuracy >> Test accuracy (overfitting)
- Precision or recall < 0.45 (poor performance)
- Model predicts only one class

---

## 🛠️ Troubleshooting

### Issue 1: Out of Memory (OOM)
```
Error: ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
```bash
# Reduce batch size
--batch-size 16

# Use fewer features
--features 30

# Shorter sequences
--sequence-length 30
```

---

### Issue 2: Training Too Slow
```
Epoch 1/100 taking 5+ minutes...
```

**Solutions:**
```bash
# Larger batches (if GPU available)
--batch-size 128

# Fewer features
--features 50

# Use simpler architecture
--mode simple
```

---

### Issue 3: Model Not Learning (Accuracy Stuck ~50%)
```
Accuracy not improving after many epochs
```

**Solutions:**
```bash
# Increase learning rate
--learning-rate 0.005

# More features
--features 0

# Longer sequences
--sequence-length 120

# Try different architecture
--mode all  # Find which works best
```

---

### Issue 4: Severe Overfitting
```
Train Accuracy: 95%
Test Accuracy: 51%
```

**Solutions:**
- Add more dropout (edit `train_lstm.py`)
- Reduce model complexity: `--mode simple`
- Fewer features: `--features 30`
- More regularization (L2 penalty)

---

### Issue 5: File Not Found
```
Error: FileNotFoundError: data/processed/XAUUSD_M15_features_complete.csv
```

**Solution:**
Make sure you ran feature engineering first:
```bash
python process_all_timeframes.py
```

---

### Issue 6: TensorFlow/Keras Errors
```
Error: No module named 'tensorflow'
```

**Solution:**
Install dependencies:
```bash
pip install tensorflow==2.13.0
pip install keras==2.13.1
```

For GPU support:
```bash
pip install tensorflow-gpu==2.13.0
```

---

## 📊 Analyzing Results

After training, check these files:

### 1. Training History Plot
```
results/lstm/lstm_training_results.png
```
Shows:
- Loss curves (should decrease)
- Accuracy curves (should increase)
- Confusion matrix
- Prediction distribution

**What to look for:**
- Validation loss should follow training loss
- No huge gap between train and val accuracy
- Confusion matrix should be balanced

---

### 2. Results JSON
```
results/lstm/lstm_results.json
```
Contains:
```json
{
  "train_accuracy": 0.7912,
  "val_accuracy": 0.5673,
  "test_accuracy": 0.5593,
  "test_precision": 0.5812,
  "test_recall": 0.5423,
  "test_f1": 0.5611,
  "predictions": [...],
  "probabilities": [...],
  "true_labels": [...]
}
```

---

### 3. Model Files
```
models/lstm_best_model.h5      # Best weights during training
models/lstm_final_model.h5     # Final model
models/lstm_scaler.pkl         # Feature scaler
models/lstm_features.json      # Feature list
models/lstm_config.json        # Configuration
```

---

### 4. Training History CSV
```
results/lstm/lstm_training_history.csv
```
Epoch-by-epoch metrics for analysis.

---

## 🎯 Next Steps After Training

### 1. Compare with XGBoost
```bash
python src/models/compare_models.py
```
Generates comparison plots and report.

---

### 2. Run Backtesting
```bash
python src/backtesting/run_backtest.py --model lstm
```
Test on historical data with realistic trading.

---

### 3. Feature Analysis
```bash
python analyze_features.py --model lstm
```
See which features are most important.

---

### 4. Ensemble Models
Combine LSTM + XGBoost for better results:
```python
# In your trading script
lstm_prob = lstm_model.predict(X)
xgb_prob = xgb_model.predict_proba(X)[:, 1]
ensemble_prob = 0.6 * lstm_prob + 0.4 * xgb_prob
```

---

### 5. Paper Trading
Deploy to demo account and monitor:
```bash
python src/trading/paper_trading.py --model lstm
```

---

## 📚 Advanced Topics

### Multi-Timeframe LSTM Ensemble

Train on different timeframes:
```bash
# Short-term (M5)
python train_models.py --timeframe M5 --sequence-length 120
# Medium-term (M15)
python train_models.py --timeframe M15 --sequence-length 60
# Long-term (H1)
python train_models.py --timeframe H1 --sequence-length 30
```

Combine predictions:
```python
signal = 0.2*M5_pred + 0.5*M15_pred + 0.3*H1_pred
```

---

### Custom Architecture

Edit `src/models/train_lstm.py` to create your own:
```python
def build_custom_model():
    model = Sequential([
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
```

---

### Transfer Learning

Fine-tune on different symbols:
```python
# Load pre-trained gold model
base_model = keras.models.load_model('models/lstm_best_model.h5')

# Freeze early layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# Train on silver (XAGUSD) data
model.fit(xagusd_data, epochs=20)
```

---

## 🎓 Resources

### Documentation
- TensorFlow LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- Keras Guide: https://keras.io/guides/sequential_model/

### Papers
- Original LSTM Paper: Hochreiter & Schmidhuber (1997)
- Bidirectional RNN: Schuster & Paliwal (1997)

### Tutorials
- Time Series Forecasting with LSTM
- Stock Price Prediction using Deep Learning

---

## ❓ FAQ

**Q: How long does training take?**  
A: 10-30 minutes with default settings (CPU). 3-5 minutes with GPU.

**Q: Can I stop training early?**  
A: Yes! Press `Ctrl+C`. The best model will still be saved.

**Q: What accuracy is good enough?**  
A: > 55% is good, > 60% is excellent. Even 52% is tradeable!

**Q: Should I use GPU?**  
A: Highly recommended but not required. 5-10x faster.

**Q: How often should I retrain?**  
A: Every 1-3 months as market conditions change.

**Q: Can I use this for other assets?**  
A: Yes! Change data source and retrain.

---

## 🏆 Pro Tips

1. **Always start with default settings** - They're optimized for gold trading
2. **Monitor validation loss** - Should decrease with training loss
3. **Use ensemble** - Combine LSTM + XGBoost for best results
4. **Keep training logs** - Track what works and what doesn't
5. **Test multiple timeframes** - Different timeframes capture different patterns
6. **Don't overtrain** - More epochs ≠ better model
7. **Check class balance** - Should be close to 50/50
8. **Save best models** - Automatically done by callbacks

---

## 📞 Support

**Issues:** Check `TROUBLESHOOTING.md`  
**Questions:** See `FAQ.md`  
**Updates:** Check `CHANGELOG.md`

---

**Good luck with your LSTM training! 🚀**

**Remember:** Machine learning is iterative. Don't expect perfect results on first try. Experiment, analyze, improve! 💪

---

**Last Updated:** 2024  
**Version:** 1.0  
**Status:** Production Ready ✅