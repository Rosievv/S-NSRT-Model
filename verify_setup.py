#!/usr/bin/env python3
"""
Lightweight setup verification (no pandas dependency)
"""

import os
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has API keys"""
    print("\n" + "="*60)
    print("Checking Environment Configuration")
    print("="*60)
    
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        print("✗ .env file not found")
        print("  Run: cp .env.example .env")
        return False
    
    print("✓ .env file exists")
    
    # Read .env
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Check Census API key
    census_configured = False
    fred_configured = False
    
    for line in content.split('\n'):
        if line.startswith('US_CENSUS_API_KEY='):
            value = line.split('=', 1)[1].strip()
            if value and value != 'your_census_api_key_here':
                print(f"✓ US Census API Key configured: {value[:10]}...{value[-10:]}")
                census_configured = True
            else:
                print("✗ US Census API Key not set")
        
        elif line.startswith('FRED_API_KEY='):
            value = line.split('=', 1)[1].strip()
            if value and value != 'your_fred_api_key_here':
                print(f"✓ FRED API Key configured: {value[:10]}...{value[-10:]}")
                fred_configured = True
            else:
                print("⚠ FRED API Key not set (optional)")
    
    return census_configured


def check_project_structure():
    """Verify project directory structure"""
    print("\n" + "="*60)
    print("Checking Project Structure")
    print("="*60)
    
    required_dirs = [
        'src/collectors',
        'src/utils',
        'config',
        'data/raw',
        'data/processed',
        'logs',
        'tests'
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check configuration files"""
    print("\n" + "="*60)
    print("Checking Configuration Files")
    print("="*60)
    
    base_path = Path(__file__).parent
    
    required_files = [
        'config/config.yaml',
        'src/config_manager.py',
        'src/collectors/base_collector.py',
        'src/collectors/us_census_collector.py',
        'main.py',
        'requirements.txt'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def check_python_version():
    """Check Python version"""
    print("\n" + "="*60)
    print("Checking Python Environment")
    print("="*60)
    
    import sys
    version = sys.version_info
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version compatible (3.8+)")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("SCRAM Setup Verification")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Project Structure': check_project_structure(),
        'Config Files': check_config_files(),
        'Environment': check_env_file()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ Setup Complete!")
        print("="*60)
        print("\nYour SCRAM data collection module is ready.")
        print("\nNext: Install dependencies (if not already done):")
        print("  pip install -r requirements.txt")
        print("\nThen run data collection:")
        print("  python main.py --phase train --sources all")
    else:
        print("\n" + "="*60)
        print("✗ Setup Issues Found")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
