#!/usr/bin/env python3
"""
Setup script for HALCON MCP Server
Helps users get up and running quickly.
"""
import sys
import subprocess
from pathlib import Path

def check_database():
    """Check if the database exists and is valid."""
    db_path = Path("halcon_operators.db")
    if not db_path.exists():
        return False, "Database file not found"
    
    # Check size (should be around 50MB)
    size_mb = db_path.stat().st_size / (1024 * 1024)
    if size_mb < 10:
        return False, f"Database file too small ({size_mb:.1f}MB - expected ~50MB)"
    
    # Check if we can connect and count records
    try:
        import sqlite3
        con = sqlite3.connect(db_path)
        count = con.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        con.close()
        
        if count < 1000:
            return False, f"Database has too few operators ({count} - expected ~2395)"
        
        return True, f"Database valid with {count} operators ({size_mb:.1f}MB)"
    except Exception as e:
        return False, f"Database error: {e}"

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import mcp
        import rapidfuzz
        return True, "Core dependencies available"
    except ImportError as e:
        return False, f"Missing dependency: {e}"

def install_dependencies():
    """Install dependencies using UV or pip."""
    print("Installing dependencies...")
    
    # Try UV first
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("Using UV to install dependencies...")
        subprocess.run(["uv", "sync"], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fall back to pip
    try:
        print("Using pip to install dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "mcp[cli]>=1.9.4", "rapidfuzz>=3.13.0"
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 HALCON MCP Server Setup")
    print("=" * 40)
    
    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    print(f"Dependencies: {deps_msg}")
    
    if not deps_ok:
        print("\n📦 Installing missing dependencies...")
        if install_dependencies():
            print("✅ Dependencies installed successfully")
        else:
            print("❌ Failed to install dependencies")
            print("Please manually install: pip install mcp[cli] rapidfuzz")
            return 1
    
    # Check database
    db_ok, db_msg = check_database()
    print(f"Database: {db_msg}")
    
    if not db_ok:
        print("\n📊 Database not found or invalid.")
        print("You have two options:")
        print("1. Get the pre-built halcon_operators.db file (~50MB)")
        print("2. Build it yourself with: python build_halcon_db.py")
        print("   (Requires additional deps: pip install requests beautifulsoup4 lxml)")
        print("   (Takes ~15-20 minutes)")
        return 1
    
    # Test import
    try:
        import HalconTest
        print("✅ HalconTest module loads successfully")
    except ImportError as e:
        print(f"❌ Cannot import HalconTest: {e}")
        return 1
    
    print("\n🎉 Setup complete!")
    print("\nTo start the MCP server:")
    print("  uv run python HalconTest.py    # with UV")
    print("  python HalconTest.py           # with pip")
    print("\nThe server will run on stdio and wait for MCP client connections.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 