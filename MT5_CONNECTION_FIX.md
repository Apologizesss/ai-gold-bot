# 🔧 MT5 CONNECTION FIX - Error Code -6

## ⚠️ PROBLEM
```
❌ ไม่สามารถเชื่อมต่อ MT5 ได้
Error: Terminal: Authorization failed (Error code: -6)
```

## ✅ SOLUTION (Takes 30 seconds)

### **THE #1 FIX - Enable Algo Trading Button** ⭐

This fixes 90% of connection issues!

1. **Open MetaTrader 5**
2. **Look at the toolbar** (below menu bar: File, View, Insert...)
3. **Find the "Algo Trading" button** (might be called "AutoTrading" or show a robot icon 🤖)
4. **Click it to turn it GREEN**
   - 🔴 RED or ⚫ GRAY = Disabled (Python CANNOT connect)
   - 🟢 GREEN = Enabled (Python CAN connect)
5. **Test the connection:**
   ```bash
   python test_mt5_simple.py
   ```

**Keyboard shortcut:** Press `Ctrl+E` or `F8` to toggle

---

## 🎯 VERIFICATION

After enabling the button, you should see:

```
✅ SUCCESS! MT5 Connected!
📊 Account Information:
   Login:     5123456
   Server:    VantageInternational-Demo
   Balance:   10000.0 USD
```

---

## 🔧 IF STILL NOT WORKING

### Step 1: Check Expert Advisors Settings

1. In MT5: **Tools → Options → Expert Advisors**
2. Make sure **ALL** are checked:
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports
   - ✅ Allow imports of external experts
3. Click **OK**
4. **Restart MT5 completely**
5. **Re-enable the green Algo Trading button** (it resets after restart)
6. Test again: `python test_mt5_simple.py`

### Step 2: Verify You're Logged In

- **Top-right corner** of MT5 should show: `[Account] - [Server]`
  - Example: `5123456 - VantageInternational-Demo`
- **Bottom-right corner** should show **green connection bars** 📶

If not logged in:
1. **File → Login to Trade Account**
2. Enter your credentials
3. Click **Login**
4. Enable Algo Trading button (green)
5. Test again

### Step 3: Close Other MT5 Instances

1. Close ALL MT5 windows
2. Press `Ctrl+Shift+Esc` → Task Manager
3. End any `terminal64.exe` processes
4. Open only ONE MT5
5. Login and enable Algo Trading
6. Test again

---

## 📋 QUICK CHECKLIST

Before running Python scripts, verify:

- ✅ MT5 is open and running
- ✅ Logged in (top-right shows account number)
- ✅ Connected (bottom-right shows green bars)
- ✅ **Algo Trading button is GREEN** ← Most important!
- ✅ Tools → Options → Expert Advisors → "Allow algorithmic trading" checked

---

## 🧪 TEST COMMANDS

### Quick Test (Recommended)
```bash
python test_mt5_simple.py
```

### Full Diagnostic Test
```bash
python debug_mt5.py
```

### Windows Batch File
```bash
test_connection.bat
```

---

## 📚 DETAILED GUIDES

- **ENABLE_ALGO_TRADING.txt** - Visual guide to find the button
- **FIX_MT5_AUTH.md** - Complete troubleshooting guide (7 solutions)
- **debug_mt5.py** - Full diagnostic script with 8 tests

---

## 💡 WHY THIS ERROR HAPPENS

MetaTrader 5 blocks all external automation (including Python API) **by default** for security. The Algo Trading button is a manual safety switch.

Think of it as:
- 🔴 Red/Gray button = "Door locked - Python cannot enter"
- 🟢 Green button = "Door open - Python can connect"

The error `-6 Authorization failed` specifically means MT5 found your terminal but refuses to let Python connect because this safety switch is off.

---

## ✅ SUCCESS CRITERIA

You know it's working when:

```
✅ SUCCESS! MT5 Connected!
✅ Account Info: [shows your account]
✅ Can access symbols
✅ XAUUSD found - Bid: 2645.23, Ask: 2645.45
🎉 ALL TESTS PASSED!
```

---

## 🚀 NEXT STEPS AFTER FIXING

Once connection works:

1. ✅ Verify with `python test_mt5_simple.py`
2. 📖 Continue to `QUICK_START.md` → Step 4 (Data Collection)
3. 🔨 Start building the trading bot components

---

## 🆘 STILL STUCK?

Run the full diagnostic and share the output:

```bash
python debug_mt5.py > mt5_debug_output.txt
```

Then check:
1. The output file `mt5_debug_output.txt`
2. Take screenshots of:
   - MT5 toolbar (showing Algo Trading button color)
   - MT5 top-right corner (account/server)
   - Tools → Options → Expert Advisors

---

**Remember:** The Algo Trading button must be GREEN every time you want to run Python scripts with MT5! 🟢