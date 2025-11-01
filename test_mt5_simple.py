"""
Simple MT5 Connection Test
===========================
Run this after applying the fixes from FIX_MT5_AUTH.md

This is a simplified test that gives you quick feedback.
"""

import MetaTrader5 as mt5

print("=" * 60)
print("SIMPLE MT5 CONNECTION TEST")
print("=" * 60)
print()

print("Step 1: Initializing MT5...")
result = mt5.initialize()

if result:
    print("✅ SUCCESS! MT5 Connected!")
    print()

    # Get account info
    account = mt5.account_info()
    if account:
        print("📊 Account Information:")
        print(f"   Login:     {account.login}")
        print(f"   Server:    {account.server}")
        print(f"   Balance:   {account.balance} {account.currency}")
        print(f"   Leverage:  1:{account.leverage}")
        print()

        # Get terminal info
        terminal = mt5.terminal_info()
        if terminal:
            print("💻 Terminal Information:")
            print(f"   Company:   {terminal.company}")
            print(f"   Name:      {terminal.name}")
            print(f"   Connected: {terminal.connected}")
            print()

        # Try to get a symbol to verify data access
        print("Testing data access...")
        symbols = mt5.symbols_get()
        if symbols:
            print(f"✅ Can access {len(symbols)} symbols")

            # Try to get XAUUSD
            xauusd = mt5.symbol_info("XAUUSD")
            if xauusd:
                print(f"✅ XAUUSD found - Bid: {xauusd.bid}, Ask: {xauusd.ask}")
            else:
                print("⚠️  XAUUSD not found (might be named differently on your broker)")
                # Show first 5 gold-related symbols
                gold_symbols = [
                    s.name
                    for s in symbols
                    if "GOLD" in s.name.upper() or "XAU" in s.name.upper()
                ]
                if gold_symbols:
                    print(f"   Gold symbols available: {', '.join(gold_symbols[:5])}")

        print()
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your MT5 connection is working correctly.")
        print("You can now proceed with the trading bot setup.")
        print()

    else:
        print("⚠️  Connected but no account info")
        print("Make sure you're logged in to MT5")

    mt5.shutdown()

else:
    # Failed
    error = mt5.last_error()
    print("❌ FAILED to connect")
    print(f"   Error Code: {error[0]}")
    print(f"   Error Message: {error[1]}")
    print()

    if error[0] == -6:
        print("💡 ERROR -6: Authorization Failed")
        print()
        print("QUICK FIX:")
        print("1. Open MetaTrader 5")
        print("2. Click the 'Algo Trading' button in the toolbar")
        print("   (Make sure it turns GREEN)")
        print("3. Run this script again")
        print()
        print("If that doesn't work, check: FIX_MT5_AUTH.md")
    elif error[0] == -10002:
        print("💡 ERROR -10002: Terminal not found")
        print()
        print("QUICK FIX:")
        print("1. Make sure MetaTrader 5 is installed")
        print("2. Open MT5 application")
        print("3. Run this script again")
    else:
        print("💡 For detailed diagnostics, run:")
        print("   python debug_mt5.py")

    print()
    print("=" * 60)
