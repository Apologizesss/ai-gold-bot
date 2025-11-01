"""
MT5 Connection Debug Script
============================
This script performs comprehensive MT5 connection diagnostics.
Run this to identify why MT5 connection is failing.
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("MT5 CONNECTION DEBUG SCRIPT")
print("=" * 70)
print()

# Step 1: Check Python environment
print("1️⃣  PYTHON ENVIRONMENT")
print("-" * 70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {sys.platform}")
print(f"Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")
print()

# Step 2: Check MetaTrader5 package
print("2️⃣  METATRADER5 PACKAGE")
print("-" * 70)
try:
    import MetaTrader5 as mt5

    print("✅ MetaTrader5 package imported successfully")
    print(f"MT5 module location: {mt5.__file__}")
    try:
        version = mt5.version()
        if version:
            print(f"MT5 API version: {version}")
        else:
            print(
                "MT5 API version: (version() returned None - this is normal before initialize)"
            )
    except Exception as e:
        print(f"Could not get version: {e}")
except ImportError as e:
    print(f"❌ Failed to import MetaTrader5: {e}")
    print("\nPlease install: pip install MetaTrader5")
    sys.exit(1)
print()

# Step 3: Check .env file
print("3️⃣  CONFIGURATION FILE")
print("-" * 70)
env_path = Path("config/.env")
if env_path.exists():
    print(f"✅ Found .env at: {env_path.absolute()}")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"File has {len(lines)} lines")

        # Parse and display (masked)
        mt5_config = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key.startswith("MT5_"):
                        mt5_config[key] = value
                        # Mask sensitive data
                        if "PASSWORD" in key:
                            display_value = "*" * len(value)
                        elif "LOGIN" in key:
                            display_value = (
                                value[:3] + "*" * (len(value) - 3)
                                if len(value) > 3
                                else "***"
                            )
                        else:
                            display_value = value
                        print(f"  {key} = {display_value}")

        if not mt5_config:
            print("⚠️  No MT5_* variables found in .env")

    except Exception as e:
        print(f"⚠️  Error reading .env: {e}")
else:
    print(f"❌ .env file not found at: {env_path.absolute()}")
    print("Please copy config/.env.example to config/.env and fill in your credentials")
print()

# Step 4: Find MT5 terminal
print("4️⃣  LOCATE MT5 TERMINAL")
print("-" * 70)
possible_paths = [
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    r"C:\Program Files\MetaTrader 5 IC Markets\terminal64.exe",
    r"C:\Program Files\XM MT5\terminal64.exe",
    r"C:\Program Files\Exness MT5\terminal64.exe",
    r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe",
]

found_terminals = []
for path in possible_paths:
    if os.path.exists(path):
        found_terminals.append(path)
        print(f"✅ Found: {path}")

if not found_terminals:
    print("⚠️  No MT5 terminal found in common locations")
    print("Please enter your MT5 terminal path manually when testing below")
else:
    print(f"\nTotal found: {len(found_terminals)}")
print()

# Step 5: Test MT5 initialization (without path)
print("5️⃣  TEST INITIALIZE (Auto-detect)")
print("-" * 70)
try:
    result = mt5.initialize()
    print(f"mt5.initialize() returned: {result}")

    if result:
        print("✅ MT5 initialized successfully!")

        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"\nTerminal Info:")
            print(f"  Company: {terminal_info.company}")
            print(f"  Name: {terminal_info.name}")
            print(f"  Path: {terminal_info.path}")
            print(f"  Data Path: {terminal_info.data_path}")
            print(f"  Connected: {terminal_info.connected}")
            print(f"  Trade Allowed: {terminal_info.trade_allowed}")

        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"\n✅ Account Info:")
            print(f"  Login: {account_info.login}")
            print(f"  Server: {account_info.server}")
            print(f"  Balance: {account_info.balance}")
            print(f"  Currency: {account_info.currency}")
            print(f"  Leverage: {account_info.leverage}")
            print(f"  Trade Allowed: {account_info.trade_allowed}")
        else:
            print("\n⚠️  account_info() returned None")
            print("This means MT5 terminal is not logged in")

        mt5.shutdown()
        print("\n✅ CONNECTION TEST PASSED!")
        print("Your MT5 connection is working correctly.")
        sys.exit(0)
    else:
        error = mt5.last_error()
        print(f"❌ Initialization failed")
        print(f"Error code: {error[0]}")
        print(f"Error message: {error[1]}")
        print("\nProceeding to advanced diagnostics...")

except Exception as e:
    print(f"❌ Exception during initialization: {e}")
    import traceback

    traceback.print_exc()
print()

# Step 6: Test with explicit path
print("6️⃣  TEST INITIALIZE (With explicit path)")
print("-" * 70)
for terminal_path in found_terminals:
    print(f"Trying: {terminal_path}")
    try:
        result = mt5.initialize(path=terminal_path)
        print(f"  Result: {result}")

        if result:
            print("  ✅ Success with explicit path!")

            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"  Connected: {terminal_info.connected}")
                print(f"  Path: {terminal_info.path}")

            account_info = mt5.account_info()
            if account_info:
                print(
                    f"  ✅ Logged in as: {account_info.login} on {account_info.server}"
                )
                mt5.shutdown()
                print("\n✅ CONNECTION SUCCESSFUL!")
                print(f"\n💡 SOLUTION: Use this path in your code:")
                print(f'   mt5.initialize(path=r"{terminal_path}")')
                sys.exit(0)
            else:
                print("  ⚠️  Not logged in to any account")
                mt5.shutdown()
        else:
            error = mt5.last_error()
            print(f"  Error: {error}")

    except Exception as e:
        print(f"  Exception: {e}")
    print()

# Step 7: Manual login test
print("7️⃣  MANUAL LOGIN TEST")
print("-" * 70)
print("If MT5 terminal is running but not logged in, we can try programmatic login")

if env_path.exists():
    try:
        # Re-read credentials
        from dotenv import load_dotenv

        load_dotenv(env_path)

        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        if login and password and server:
            print(f"Attempting login with:")
            print(f"  Login: {login[:3]}***")
            print(f"  Server: {server}")

            # Initialize first
            init_result = mt5.initialize()
            if not init_result and found_terminals:
                init_result = mt5.initialize(path=found_terminals[0])

            if init_result:
                # Try login
                login_result = mt5.login(int(login), password=password, server=server)
                print(f"Login result: {login_result}")

                if login_result:
                    print("✅ LOGIN SUCCESSFUL!")
                    account_info = mt5.account_info()
                    if account_info:
                        print(f"Account: {account_info.login} on {account_info.server}")
                    mt5.shutdown()
                    sys.exit(0)
                else:
                    error = mt5.last_error()
                    print(f"❌ Login failed: {error}")
            else:
                print("❌ Could not initialize MT5 for login test")
        else:
            print(
                "⚠️  Missing credentials in .env (MT5_LOGIN, MT5_PASSWORD, or MT5_SERVER)"
            )

    except Exception as e:
        print(f"Error during login test: {e}")
        import traceback

        traceback.print_exc()
else:
    print("⚠️  No .env file found for login credentials")
print()

# Step 8: Final diagnostics
print("8️⃣  DIAGNOSTIC SUMMARY")
print("=" * 70)
print("❌ Could not establish MT5 connection")
print()
print("POSSIBLE CAUSES:")
print()
print("1. MT5 Terminal not running")
print("   → Open MetaTrader 5 application")
print()
print("2. MT5 not logged in")
print("   → In MT5: File → Login to Trade Account")
print("   → Enter your account number, password, and select server")
print()
print("3. MT5 API/DLL imports not enabled")
print("   → In MT5: Tools → Options → Expert Advisors")
print("   → Check: ✓ Allow DLL imports")
print("   → Check: ✓ Allow automated trading")
print()
print("4. MT5 terminal path not auto-detected")
print("   → Find your terminal64.exe location")
print("   → Use: mt5.initialize(path=r'C:\\path\\to\\terminal64.exe')")
print()
print("5. Wrong credentials in config/.env")
print("   → Verify MT5_LOGIN, MT5_PASSWORD, MT5_SERVER match your MT5 account")
print()
print("6. Permission issues")
print("   → Run MT5 as regular user (not Administrator)")
print("   → Run Python script as same user")
print()
print("7. Firewall/Antivirus blocking")
print("   → Temporarily disable and test")
print()
print("=" * 70)
print("NEXT STEPS:")
print("1. Ensure MT5 is running and logged in (check top-right corner of MT5)")
print("2. Check Tools → Options → Expert Advisors → Allow DLL imports")
print("3. Run this script again: python debug_mt5.py")
print("4. If still failing, send screenshot of:")
print("   - MT5 top-right corner (showing account/server)")
print("   - MT5 Tools → Options → Expert Advisors tab")
print("   - This script's output")
print("=" * 70)
