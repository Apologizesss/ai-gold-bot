@echo off
echo ====================================================================
echo MT5 CONNECTION TEST - Quick Launcher
echo ====================================================================
echo.

REM Check if we're in the right directory
if not exist "test_mt5_simple.py" (
    echo ERROR: test_mt5_simple.py not found!
    echo Please run this batch file from the TRADE directory.
    echo.
    pause
    exit /b 1
)

echo Running MT5 connection test...
echo.
echo BEFORE RUNNING THIS TEST:
echo   1. Open MetaTrader 5
echo   2. Make sure you're logged in
echo   3. Click the "Algo Trading" button to make it GREEN
echo.
echo Press any key to continue...
pause > nul
echo.

REM Run the simple test
python test_mt5_simple.py

echo.
echo ====================================================================
echo.
echo If you saw "SUCCESS" above, your MT5 connection is working!
echo.
echo If you saw "FAILED":
echo   - Open ENABLE_ALGO_TRADING.txt for step-by-step instructions
echo   - Run: python debug_mt5.py (for detailed diagnostics)
echo   - Read: FIX_MT5_AUTH.md (complete troubleshooting guide)
echo.
echo ====================================================================
pause
