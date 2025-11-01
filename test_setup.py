# test_setup.py
# สคริปต์ทดสอบการติดตั้งและการตั้งค่าสำหรับ AI Gold Trading Bot

import sys
import os


def print_header(text):
    """แสดงหัวข้อสวยงาม"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def test_python_version():
    """ทดสอบ Python version"""
    print_header("🐍 ทดสอบ Python Version")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 9:
        print("✅ Python version ถูกต้อง (≥3.9)")
        return True
    else:
        print("❌ Python version ต้อง ≥3.9")
        print("   กรุณาติดตั้ง Python 3.9 หรือ 3.10")
        return False


def test_imports():
    """ทดสอบการ import libraries ทั้งหมด"""
    print_header("📚 ทดสอบการ Import Libraries")

    results = {}

    # Core libraries
    try:
        import pandas as pd

        print(f"✅ pandas: {pd.__version__}")
        results["pandas"] = True
    except ImportError as e:
        print(f"❌ pandas: ไม่พบ - {e}")
        results["pandas"] = False

    try:
        import numpy as np

        print(f"✅ numpy: {np.__version__}")
        results["numpy"] = True
    except ImportError as e:
        print(f"❌ numpy: ไม่พบ - {e}")
        results["numpy"] = False

    # AI/ML libraries
    try:
        import tensorflow as tf

        print(f"✅ tensorflow: {tf.__version__}")
        results["tensorflow"] = True
    except ImportError as e:
        print(f"❌ tensorflow: ไม่พบ - {e}")
        results["tensorflow"] = False

    try:
        import xgboost as xgb

        print(f"✅ xgboost: {xgb.__version__}")
        results["xgboost"] = True
    except ImportError as e:
        print(f"❌ xgboost: ไม่พบ - {e}")
        results["xgboost"] = False

    try:
        from sklearn import __version__ as sklearn_version

        print(f"✅ scikit-learn: {sklearn_version}")
        results["sklearn"] = True
    except ImportError as e:
        print(f"❌ scikit-learn: ไม่พบ - {e}")
        results["sklearn"] = False

    try:
        import transformers

        print(f"✅ transformers: {transformers.__version__}")
        results["transformers"] = True
    except ImportError as e:
        print(f"❌ transformers: ไม่พบ - {e}")
        results["transformers"] = False

    # Trading library
    try:
        import MetaTrader5 as mt5

        print(f"✅ MetaTrader5: {mt5.__version__}")
        results["MetaTrader5"] = True
    except ImportError as e:
        print(f"❌ MetaTrader5: ไม่พบ - {e}")
        results["MetaTrader5"] = False

    # Technical indicators
    try:
        import talib

        print(f"✅ TA-Lib: {talib.__version__}")
        results["talib"] = True
    except ImportError as e:
        print(f"❌ TA-Lib: ไม่พบ - {e}")
        print("   ⚠️  TA-Lib ต้องติดตั้งแยก: ดู QUICK_START.md")
        results["talib"] = False

    # Data collection
    try:
        import requests

        print(f"✅ requests: {requests.__version__}")
        results["requests"] = True
    except ImportError as e:
        print(f"❌ requests: ไม่พบ - {e}")
        results["requests"] = False

    try:
        import yfinance

        print(f"✅ yfinance: {yfinance.__version__}")
        results["yfinance"] = True
    except ImportError as e:
        print(f"❌ yfinance: ไม่พบ - {e}")
        results["yfinance"] = False

    try:
        from bs4 import BeautifulSoup

        print(f"✅ beautifulsoup4: OK")
        results["beautifulsoup4"] = True
    except ImportError as e:
        print(f"❌ beautifulsoup4: ไม่พบ - {e}")
        results["beautifulsoup4"] = False

    return results


def test_mt5_connection():
    """ทดสอบการเชื่อมต่อ MetaTrader 5"""
    print_header("🔌 ทดสอบการเชื่อมต่อ MetaTrader 5")

    try:
        import MetaTrader5 as mt5

        # พยายามเชื่อมต่อ MT5
        if mt5.initialize():
            account_info = mt5.account_info()

            if account_info is None:
                print("❌ ไม่สามารถดึงข้อมูลบัญชีได้")
                mt5.shutdown()
                return False

            print("✅ เชื่อมต่อ MT5 สำเร็จ!")
            print(f"\n📊 ข้อมูลบัญชี:")
            print(f"   Account Number: {account_info.login}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Equity: ${account_info.equity:.2f}")
            print(f"   Server: {account_info.server}")
            print(f"   Currency: {account_info.currency}")
            print(f"   Leverage: 1:{account_info.leverage}")
            print(f"   Company: {account_info.company}")

            # ทดสอบดึงข้อมูล XAUUSD
            print(f"\n💰 ทดสอบดึงข้อมูล XAUUSD:")
            symbol_info = mt5.symbol_info("XAUUSD")

            if symbol_info is None:
                print("❌ ไม่พบสัญลักษณ์ XAUUSD")
                print("   ตรวจสอบว่าโบรกเกอร์รองรับ XAUUSD หรือไม่")
            else:
                print(f"✅ พบสัญลักษณ์ XAUUSD")
                print(f"   Bid: {symbol_info.bid}")
                print(f"   Ask: {symbol_info.ask}")
                print(f"   Spread: {symbol_info.spread} points")

            mt5.shutdown()
            return True
        else:
            print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
            print("\nสาเหตุที่เป็นไปได้:")
            print("1. MT5 ไม่ได้เปิดอยู่")
            print("2. ยังไม่ได้ login บัญชี MT5")
            print("3. ยังไม่ได้เปิดใช้งาน Python API ใน MT5")
            print("   (Tools → Options → Expert Advisors → Allow DLL imports)")
            print("4. ยังไม่ได้สร้างไฟล์ config/.env")
            return False

    except ImportError:
        print("❌ ไม่พบ MetaTrader5 library")
        print("   ติดตั้งด้วย: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return False


def test_directory_structure():
    """ทดสอบโครงสร้างโฟลเดอร์"""
    print_header("📁 ทดสอบโครงสร้างโฟลเดอร์")

    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/labels",
        "models",
        "src",
        "config",
        "notebooks",
        "tests",
        "logs",
        "results",
        "scripts",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - ไม่พบ")
            all_exist = False

    return all_exist


def test_config_file():
    """ทดสอบไฟล์ config/.env"""
    print_header("⚙️  ทดสอบไฟล์ Config")

    env_path = "config/.env"

    if os.path.exists(env_path):
        print(f"✅ พบไฟล์ {env_path}")

        # อ่านไฟล์และตรวจสอบว่ามีค่าที่จำเป็นหรือไม่
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_keys = ["MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "NEWS_API_KEY"]

        print("\nตรวจสอบค่าที่จำเป็น:")
        for key in required_keys:
            if key in content:
                print(f"✅ {key} พบในไฟล์")
            else:
                print(f"❌ {key} ไม่พบในไฟล์")

        return True
    else:
        print(f"❌ ไม่พบไฟล์ {env_path}")
        print("\nสร้างไฟล์ด้วยคำสั่ง:")
        print("   notepad config\\.env")
        print("\nหรือคัดลอกจาก config/.env.example (ถ้ามี)")
        return False


def test_git_repository():
    """ทดสอบ Git repository"""
    print_header("🔧 ทดสอบ Git Repository")

    if os.path.exists(".git"):
        print("✅ Git repository ถูก initialize แล้ว")
        return True
    else:
        print("❌ ยังไม่ได้ initialize Git repository")
        print("\nสร้างด้วยคำสั่ง:")
        print("   git init")
        print("   git add .")
        print('   git commit -m "Initial commit"')
        return False


def generate_summary(results):
    """สรุปผลการทดสอบ"""
    print_header("📊 สรุปผลการทดสอบ")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    print(f"\nผลการทดสอบ: {passed_tests}/{total_tests} ผ่าน")
    print(f"เปอร์เซ็นต์: {(passed_tests / total_tests) * 100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ยินดีด้วย! ระบบพร้อมใช้งาน 100%")
        print("\n📋 ขั้นตอนถัดไป:")
        print("1. อ่าน TODO.md เพื่อดูงานที่ต้องทำ")
        print("2. เริ่ม Phase 1: Data Collection")
        print("3. Update PROJECT_STATUS.md ตามความคืบหน้า")
    else:
        print("\n⚠️  ยังมีบางส่วนที่ต้องแก้ไข")
        print("\nสิ่งที่ต้องทำ:")

        if not results.get("python_version", True):
            print("❌ ติดตั้ง Python 3.9 หรือ 3.10")

        failed_imports = [
            k
            for k, v in results.items()
            if not v
            and k
            not in [
                "python_version",
                "mt5_connection",
                "directory_structure",
                "config_file",
                "git_repo",
            ]
        ]
        if failed_imports:
            print(f"❌ ติดตั้ง packages ที่ขาดหายไป:")
            print("   pip install -r requirements.txt")

        if not results.get("mt5_connection", True):
            print("❌ ตั้งค่า MetaTrader 5 และเชื่อมต่อ")

        if not results.get("config_file", True):
            print("❌ สร้างไฟล์ config/.env")

        if not results.get("git_repo", True):
            print("❌ Initialize Git repository: git init")


def main():
    """ฟังก์ชันหลัก"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🤖 AI GOLD TRADING BOT - SETUP TEST SCRIPT 🤖        ║
    ║                                                              ║
    ║              ทดสอบการติดตั้งและการตั้งค่า                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # ทดสอบ Python version
    results["python_version"] = test_python_version()

    # ทดสอบ imports
    import_results = test_imports()
    results.update(import_results)

    # ทดสอบโครงสร้างโฟลเดอร์
    results["directory_structure"] = test_directory_structure()

    # ทดสอบไฟล์ config
    results["config_file"] = test_config_file()

    # ทดสอบ Git
    results["git_repo"] = test_git_repository()

    # ทดสอบ MT5 connection (ถ้า MetaTrader5 ติดตั้งแล้ว)
    if results.get("MetaTrader5", False):
        results["mt5_connection"] = test_mt5_connection()

    # สรุปผล
    generate_summary(results)

    print("\n" + "=" * 60)
    print("  ทดสอบเสร็จสิ้น!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
