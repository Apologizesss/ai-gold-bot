# 🔑 HOW TO GET YOUR MT5 ACCOUNT CREDENTIALS

## ⚠️ PROBLEM FOUND

Your `.env` file has:
```
MT5_LOGIN=VZERO
```

**This is WRONG!** ❌

MT5 login must be a **NUMERIC account number**, not text.

---

## ✅ HOW TO FIND YOUR CORRECT MT5 CREDENTIALS

### Step 1: Open MetaTrader 5

### Step 2: Check If You're Logged In

Look at the **top-right corner** of MT5:
- If you see a number like `51234567 - VantageInternational-Demo` → You're logged in ✅
- If it's empty or says "No connection" → You need to login first

### Step 3: Find Your Account Number

**Method A: From MT5 Window (if logged in)**
- Top-right corner shows: `[ACCOUNT NUMBER] - [SERVER]`
- Example: `51234567 - VantageInternational-Demo`
- Your login = `51234567` (the number part)

**Method B: From Account List**
1. In MT5: **File → Login to Trade Account**
2. You'll see a list of accounts
3. Each account shows:
   ```
   [Account Number]
   [Broker Name]
   [Server Name]
   ```
4. Your login = the account number (all digits)

**Method C: From Mailbox**
1. In MT5: View → Toolbox (or press Ctrl+T)
2. Click the **"Mailbox"** tab
3. Look for the welcome email when you created the account
4. It will show your account number

**Method D: From Broker Email**
- Check your email inbox
- Look for email from your broker (Vantage)
- Subject might be "Account Created" or "Demo Account Details"
- Email contains your account number and server

---

## 📝 WHAT YOUR CREDENTIALS SHOULD LOOK LIKE

### ✅ CORRECT FORMAT:

```env
# MT5 Demo Account Credentials
MT5_LOGIN=51234567
MT5_PASSWORD=YourPassword123
MT5_SERVER=VantageInternational-Demo
```

**Notes:**
- `MT5_LOGIN` = **NUMBERS ONLY** (6-8 digits typically)
- `MT5_PASSWORD` = Your password (can have letters/numbers/symbols)
- `MT5_SERVER` = Exact server name (case-sensitive, copy from MT5)

### ❌ WRONG FORMAT:

```env
MT5_LOGIN=VZERO                    # ❌ Not text!
MT5_LOGIN="51234567"               # ❌ No quotes needed
MT5_LOGIN= 51234567                # ❌ No space after =
MT5_SERVER=Vantage Demo            # ❌ Check exact name in MT5
```

---

## 🔍 HOW TO GET THE EXACT SERVER NAME

1. In MT5: **File → Login to Trade Account**
2. Click on the **Server** dropdown
3. You'll see a list like:
   ```
   VantageInternational-Demo
   VantageInternational-Live
   VantageInternational-Real
   ```
4. Note the **EXACT** name (including dashes, capitalization)
5. Use this exact name in your `.env` file

**Common server name variations:**
- `VantageInternational-Demo`
- `Vantage-Demo`
- `VantageInt-Demo`
- `VantageFX-Demo`

**The server name MUST match exactly!**

---

## 🆕 DON'T HAVE A DEMO ACCOUNT YET?

### Create a Vantage Demo Account:

1. In MT5: **File → Open an Account**
2. Find **"Vantage"** or search for it
3. Click **"Open a demo account"**
4. Fill in the form:
   - Name: (any name)
   - Email: your email
   - Phone: your phone
   - Account Type: **Standard** or **Raw ECN**
   - Currency: **USD**
   - Leverage: **1:500** (recommended for testing)
   - Deposit: **10000** (demo money, doesn't matter)
5. Click **Next** / **Register**
6. MT5 will create your account and show:
   ```
   ✅ Account Created!
   Login: 51234567
   Password: abc123XYZ
   Server: VantageInternational-Demo
   ```
7. **SAVE THESE CREDENTIALS!**
8. Put them in your `config/.env` file

---

## 📋 UPDATE YOUR .ENV FILE

### Step 1: Open your .env file
```bash
# Location: C:\Users\ASUS\Desktop\TRADE\config\.env
notepad config\.env
```

### Step 2: Find the MT5 section
Look for:
```env
# MT5 Demo Account Credentials
MT5_LOGIN=VZERO
MT5_PASSWORD=yourpassword
MT5_SERVER=VantageInternational-Demo
```

### Step 3: Replace with your REAL credentials
```env
# MT5 Demo Account Credentials
MT5_LOGIN=51234567                        # ← Replace with YOUR account number
MT5_PASSWORD=YourActualPassword           # ← Replace with YOUR password
MT5_SERVER=VantageInternational-Demo      # ← Verify this is correct
```

### Step 4: Save the file (Ctrl+S)

### Step 5: Make sure file is saved as UTF-8
- In Notepad: File → Save As → Encoding: **UTF-8**

---

## ✅ VERIFY YOUR CHANGES

### Step 1: Check .env file is correct
```bash
type config\.env | findstr MT5_
```

Should show something like:
```
MT5_LOGIN=51234567
MT5_PASSWORD=YourPassword
MT5_SERVER=VantageInternational-Demo
```

### Step 2: Login to MT5 manually
1. Open MetaTrader 5
2. File → Login to Trade Account
3. Enter the SAME credentials from your .env
4. Click Login
5. Top-right should show your account number
6. Bottom-right should show green connection bars

### Step 3: Test Python connection
```bash
python fix_login.py
```

**Success looks like:**
```
✅ LOGIN SUCCESSFUL!
Account: 51234567
Server: VantageInternational-Demo
Balance: 10000.0 USD
🎉 SUCCESS! MT5 Connection Working!
```

---

## 🆘 STILL HAVING ISSUES?

### Issue 1: "Account not found"
- Check the **Server** name is EXACTLY correct (case-sensitive)
- Try the variations listed above

### Issue 2: "Invalid credentials"
- Verify your password is correct
- Try logging in manually in MT5 first
- Your demo account might have expired (create a new one)

### Issue 3: "Authorization failed"
- Make sure MT5 is open AND logged in
- Check Algo Trading button is GREEN
- Restart MT5 and try again

---

## 📞 NEXT STEPS

After updating your `.env` with correct credentials:

1. **Save** the `.env` file
2. **Login to MT5** manually with the same credentials
3. Make sure **Algo Trading button is GREEN**
4. Run: `python fix_login.py`
5. If successful, run: `python test_mt5_simple.py`

---

## 💡 EXAMPLE COMPLETE .ENV

```env
# =============================================================================
# MT5 DEMO ACCOUNT CREDENTIALS
# =============================================================================

# Your MT5 demo account login (NUMBERS ONLY - no text!)
MT5_LOGIN=51234567

# Your MT5 demo account password
MT5_PASSWORD=MySecurePass123

# Your MT5 server name (must match EXACTLY from MT5)
MT5_SERVER=VantageInternational-Demo

# Path to MT5 terminal (usually auto-detected, but can specify if needed)
# MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

---

**Remember:** Your login is ALWAYS a number (6-8 digits), never text like "VZERO"! 🔢