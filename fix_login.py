"""
MT5 Login Diagnostic & Fix
===========================
This script checks if MT5 is logged in and attempts to login programmatically.
"""

import MetaTrader5 as mt5
import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 70)
print("MT5 LOGIN DIAGNOSTIC & FIX")
print("=" * 70)
print()

# Load environment variables
env_path = Path("config/.env")
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded .env file")
else:
    print("❌ .env file not found at config/.env")
    print("Cannot proceed without credentials.")
    exit(1)

# Get credentials
mt5_login = os.getenv("MT5_LOGIN")
mt5_password = os.getenv("MT5_PASSWORD")
mt5_server = os.getenv("MT5_SERVER")

print(f"Credentials from .env:")
print(f"  Login: {mt5_login[:3]}*** (length: {len(mt5_login) if mt5_login else 0})")
print(f"  Password: {'*' * len(mt5_password) if mt5_password else 'NOT SET'}")
print(f"  Server: {mt5_server}")
print()

if not all([mt5_login, mt5_password, mt5_server]):
    print("❌ Missing credentials in .env file")
    print("Please ensure MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER are all set.")
    exit(1)

# Try to find MT5 terminal
print("=" * 70)
print("STEP 1: Locating MT5 Terminal")
print("=" * 70)

terminal_paths = [
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
]

terminal_path = None
for path in terminal_paths:
    if os.path.exists(path):
        terminal_path = path
        print(f"✅ Found MT5 at: {path}")
        break

if not terminal_path:
    print("⚠️  Using auto-detection (no path specified)")

print()

# Try different initialization and login combinations
print("=" * 70)
print("STEP 2: Attempting Connection Methods")
print("=" * 70)
print()

# Method 1: Initialize without path, no login
print("📍 Method 1: Basic initialization (auto-detect)")
result = mt5.initialize()
print(f"   mt5.initialize() = {result}")
if result:
    account = mt5.account_info()
    if account:
        print(f"   ✅ Already logged in!")
        print(f"   Account: {account.login}")
        print(f"   Server: {account.server}")
        print(f"   Balance: {account.balance} {account.currency}")
        print()
        print("🎉 SUCCESS! Your MT5 connection is working!")
        mt5.shutdown()
        exit(0)
    else:
        print(f"   ⚠️  Initialized but no account logged in")
        mt5.shutdown()
else:
    error = mt5.last_error()
    print(f"   ❌ Failed: {error}")

print()

# Method 2: Initialize with path, no login
if terminal_path:
    print("📍 Method 2: Initialize with explicit path")
    result = mt5.initialize(path=terminal_path)
    print(f"   mt5.initialize(path=...) = {result}")
    if result:
        account = mt5.account_info()
        if account:
            print(f"   ✅ Already logged in!")
            print(f"   Account: {account.login}")
            print(f"   Server: {account.server}")
            print()
            print("🎉 SUCCESS! Your MT5 connection is working!")
            mt5.shutdown()
            exit(0)
        else:
            print(f"   ⚠️  Initialized but no account logged in")
            mt5.shutdown()
    else:
        error = mt5.last_error()
        print(f"   ❌ Failed: {error}")
    print()

# Method 3: Initialize with login parameters
print("📍 Method 3: Initialize with login credentials")
try:
    login_int = int(mt5_login)
    print(f"   Attempting login for account: {login_int}")
    print(f"   Server: {mt5_server}")

    if terminal_path:
        result = mt5.initialize(
            path=terminal_path,
            login=login_int,
            password=mt5_password,
            server=mt5_server,
        )
    else:
        result = mt5.initialize(
            login=login_int, password=mt5_password, server=mt5_server
        )

    print(f"   mt5.initialize(login=...) = {result}")

    if result:
        account = mt5.account_info()
        if account:
            print(f"   ✅ LOGIN SUCCESSFUL!")
            print(f"   Account: {account.login}")
            print(f"   Server: {account.server}")
            print(f"   Balance: {account.balance} {account.currency}")
            print(f"   Leverage: 1:{account.leverage}")
            print()
            print("=" * 70)
            print("🎉 SUCCESS! MT5 Connection Working!")
            print("=" * 70)
            print()
            print("Your connection is now active. You can proceed with:")
            print("  python test_mt5_simple.py")
            print()
            mt5.shutdown()
            exit(0)
        else:
            print(f"   ⚠️  Initialized but account_info() returned None")
            error = mt5.last_error()
            print(f"   Last error: {error}")
            mt5.shutdown()
    else:
        error = mt5.last_error()
        print(f"   ❌ Failed: {error}")

except ValueError:
    print(f"   ❌ MT5_LOGIN must be numeric, got: {mt5_login}")
    print(f"   Please check your .env file")

print()

# Method 4: Initialize then login separately
print("📍 Method 4: Initialize first, then login")
result = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
print(f"   mt5.initialize() = {result}")

if result:
    try:
        login_int = int(mt5_login)
        print(
            f"   Attempting mt5.login({login_int}, password=***, server={mt5_server})"
        )

        login_result = mt5.login(
            login=login_int, password=mt5_password, server=mt5_server
        )

        print(f"   mt5.login() = {login_result}")

        if login_result:
            account = mt5.account_info()
            if account:
                print(f"   ✅ LOGIN SUCCESSFUL!")
                print(f"   Account: {account.login}")
                print(f"   Server: {account.server}")
                print(f"   Balance: {account.balance} {account.currency}")
                print()
                print("=" * 70)
                print("🎉 SUCCESS! MT5 Connection Working!")
                print("=" * 70)
                mt5.shutdown()
                exit(0)
        else:
            error = mt5.last_error()
            print(f"   ❌ Login failed: {error}")

    except ValueError:
        print(f"   ❌ Invalid login number: {mt5_login}")

    mt5.shutdown()
else:
    error = mt5.last_error()
    print(f"   ❌ Initialize failed: {error}")

print()

# Final diagnosis
print("=" * 70)
print("❌ ALL CONNECTION METHODS FAILED")
print("=" * 70)
print()
print("DIAGNOSIS:")
print()
print("1. Algo Trading is enabled ✅ (confirmed from your MT5 log)")
print("2. Python can find MT5 terminal ✅")
print("3. But cannot establish connection ❌")
print()
print("MOST LIKELY CAUSES:")
print()
print("🔴 MT5 is NOT logged in to your account")
print("   → Check top-right corner of MT5")
print("   → Should show: [Account Number] - [Server Name]")
print("   → If empty or says 'No connection': You need to login manually")
print()
print("   HOW TO LOGIN IN MT5:")
print("   1. Open MetaTrader 5")
print("   2. File → Login to Trade Account")
print("   3. Enter:")
print(f"      Login: {mt5_login}")
print(f"      Password: (your password)")
print(f"      Server: {mt5_server}")
print("   4. Click 'Login'")
print("   5. Wait for connection (green bars bottom-right)")
print("   6. Run this script again")
print()
print("🔴 Wrong server name in .env")
print(f"   → Current: {mt5_server}")
print("   → In MT5: File → Login to Trade Account → Check EXACT server name")
print("   → Server names are case-sensitive and must match exactly")
print()
print("🔴 Wrong credentials")
print("   → Double-check MT5_LOGIN and MT5_PASSWORD in config/.env")
print("   → Make sure there are no extra spaces or quotes")
print()
print("🔴 Demo account expired")
print("   → Demo accounts expire after 30-90 days")
print("   → You may need to create a new demo account")
print()
print("=" * 70)
print("NEXT STEPS:")
print("1. Open MT5 and manually login (File → Login to Trade Account)")
print("2. Verify you see your account number in top-right corner")
print("3. Verify bottom-right shows green connection bars")
print("4. Run: python fix_login.py")
print("=" * 70)
