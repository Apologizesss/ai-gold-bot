# 🔧 FIX MT5 AUTHORIZATION ERROR (-6)

**Error:** `Terminal: Authorization failed` (Error code: -6)

**Status:** MT5 terminal found ✅ | Python package installed ✅ | **Authorization FAILED ❌**

---

## ✅ SOLUTION CHECKLIST (Follow in Order)

### 1️⃣ **ENABLE ALGO TRADING BUTTON** ⭐ (99% of the time this fixes it!)

**This is the #1 most common cause!**

1. Open MetaTrader 5 terminal
2. Look at the **toolbar** (top of the window, below menu bar)
3. Find the **"Algo Trading"** or **"AutoTrading"** button
   - Icon looks like: 📊🤖 (chart with robot/automation symbol)
   - Might say "Algo Trading" or just show an icon
4. **Click it to turn it GREEN**
   - 🔴 RED or ⚫ GRAY = DISABLED (Python cannot connect)
   - 🟢 GREEN = ENABLED (Python can connect)
5. Run `python debug_mt5.py` again

**Alternative:** Press **Ctrl+E** or **F8** keyboard shortcut to toggle

---

### 2️⃣ **CHECK EXPERT ADVISORS SETTINGS**

1. In MT5: **Tools → Options**
2. Go to **"Expert Advisors"** tab
3. Make sure **ALL THREE** are checked:
   ```
   ✅ Allow algorithmic trading
   ✅ Allow DLL imports  
   ✅ Allow imports of external experts
   ```
4. Click **OK**
5. **RESTART MT5** (close completely and reopen)
6. **Re-enable Algo Trading button** (step 1 above)
7. Run `python debug_mt5.py` again

---

### 3️⃣ **VERIFY ACCOUNT IS LOGGED IN**

1. Check **top-right corner** of MT5 window
   - Should show: `[Account Number] - [Server Name]`
   - Example: `5123456 - VantageInternational-Demo`
2. Check **bottom-right corner** (connection indicator)
   - Should show green signal bars (📶 green)
   - If red or shows "No connection": you're not connected

**If not logged in:**
1. **File → Login to Trade Account**
2. Enter:
   - Login: `VZE***` (your account number)
   - Password: (your password)
   - Server: `VantageInternational-Demo`
3. Click **Login**
4. Wait for connection (bottom-right should show green)
5. Run `python debug_mt5.py` again

---

### 4️⃣ **CHECK FOR MULTIPLE MT5 INSTANCES**

Sometimes multiple MT5 terminals confuse the Python API:

1. **Close ALL MT5 windows** completely
2. Press **Ctrl+Shift+Esc** → Task Manager
3. Look for any `terminal64.exe` or `MetaTrader 5` processes
4. **End Task** on all of them
5. **Open ONLY ONE MT5** terminal
6. Log in to your account
7. Enable Algo Trading button (green)
8. Run `python debug_mt5.py` again

---

### 5️⃣ **RUN AS ADMINISTRATOR (if nothing else works)**

Sometimes Windows permissions block the connection:

1. **Close MT5 completely**
2. Right-click **MetaTrader 5** icon
3. Choose **"Run as administrator"**
4. Log in to your account
5. Enable Algo Trading (green button)
6. **Also run Python as admin:**
   - Right-click Command Prompt or PowerShell
   - Choose "Run as administrator"
   - `cd C:\Users\ASUS\Desktop\TRADE`
   - `python debug_mt5.py`

---

### 6️⃣ **REINSTALL MT5 PYTHON PACKAGE (if corrupted)**

```bash
# In your activated virtual environment:
pip uninstall MetaTrader5 -y
pip install --no-cache-dir MetaTrader5
python debug_mt5.py
```

---

### 7️⃣ **TRY DIFFERENT SERVER**

Your current server: `VantageInternational-Demo`

Sometimes the server name needs to be exact:

1. In MT5: **File → Login to Trade Account**
2. Click on **Server** dropdown
3. Note the **EXACT** server name (case-sensitive, spaces matter)
4. Update `config/.env`:
   ```env
   MT5_SERVER=VantageInternational-Demo
   # Try variations if needed:
   # MT5_SERVER=Vantage-Demo
   # MT5_SERVER=VantageInt-Demo
   ```
5. Run `python debug_mt5.py` again

---

## 🎯 QUICK TEST AFTER EACH FIX

After trying any solution above, run:

```bash
python debug_mt5.py
```

**Success looks like:**
```
✅ MT5 initialized successfully!
✅ Account Info:
  Login: 5123456
  Server: VantageInternational-Demo
  Balance: 10000.0
✅ CONNECTION TEST PASSED!
```

---

## 📸 IF STILL FAILING - SEND SCREENSHOTS

Take screenshots of:

1. **MT5 top-right corner** (showing account number and server)
2. **MT5 toolbar** (showing Algo Trading button - is it green?)
3. **MT5 bottom-right** (showing connection status)
4. **MT5: Tools → Options → Expert Advisors** (showing all checkboxes)
5. **Output of** `python debug_mt5.py`

---

## 🚀 MOST LIKELY FIX

**90% of the time, the issue is:**
- ❌ Algo Trading button is RED or GRAY
- ✅ Click it to make it GREEN
- ✅ Run script again → WORKS!

---

## 📝 NOTES

- The Python API can **detect** MT5 terminal (found at `C:\Program Files\MetaTrader 5\terminal64.exe` ✅)
- But it cannot **connect** due to authorization/permission
- This is almost always the Algo Trading button or Expert Advisors settings
- Error code `-6` specifically means "authorization denied by terminal"

---

**After fixing, proceed to:** `QUICK_START.md` → Step 4 (Data Collection)