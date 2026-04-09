#!/usr/bin/env python3
"""
Quick test script to verify API keys and data collection setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config_manager import config


def test_census_api():
    """Test US Census API key configuration"""
    print("\n" + "="*60)
    print("Testing US Census API Configuration")
    print("="*60)
    
    api_key = config.get('data_sources.us_census.api_key')
    
    if api_key and api_key != 'your_census_api_key_here':
        print(f"✓ US Census API Key found: {api_key[:10]}...{api_key[-10:]}")
        print("✓ Configuration valid")
        return True
    else:
        print("✗ US Census API Key not configured")
        print("  Please set US_CENSUS_API_KEY in .env file")
        return False


def test_fred_api():
    """Test FRED API key configuration"""
    print("\n" + "="*60)
    print("Testing FRED API Configuration")
    print("="*60)
    
    api_key = config.get('data_sources.fred.api_key')
    
    if api_key and api_key != 'your_fred_api_key_here':
        print(f"✓ FRED API Key found: {api_key[:10]}...{api_key[-10:]}")
        print("✓ Configuration valid")
        return True
    else:
        print("⚠ FRED API Key not configured (optional)")
        print("  System will use mock data for PMI indicators")
        print("  To enable: Set FRED_API_KEY in .env file")
        return False


def test_collectors():
    """Test collector initialization"""
    print("\n" + "="*60)
    print("Testing Data Collectors")
    print("="*60)
    
    try:
        from src.collectors import USCensusCollector, USGSCollector, MacroIndicatorCollector
        
        print("✓ USCensusCollector imported successfully")
        print("✓ USGSCollector imported successfully")
        print("✓ MacroIndicatorCollector imported successfully")
        
        # Test instantiation
        census = USCensusCollector(
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        print("✓ Collectors can be instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Error importing collectors: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        # Test time ranges
        train_range = config.get_time_range('train')
        test_range = config.get_time_range('test')
        
        print(f"✓ Training period: {train_range['start']} to {train_range['end']}")
        print(f"✓ Testing period: {test_range['start']} to {test_range['end']}")
        
        # Test HS codes
        hs_codes = config.get_hs_codes('integrated_circuits')
        print(f"✓ Found {len(hs_codes)} integrated circuit HS codes")
        
        # Test storage config
        storage = config.get_storage_config()
        print(f"✓ Data format: {storage.get('file_format')}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    required_packages = [
        'pandas',
        'numpy',
        'yaml',
        'requests',
        'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'dotenv':
                from dotenv import load_dotenv
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCRAM Data Collection Module - System Check")
    print("="*60)
    
    results = {
        'Dependencies': test_dependencies(),
        'Configuration': test_config(),
        'Census API': test_census_api(),
        'FRED API': test_fred_api(),
        'Collectors': test_collectors()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    # Overall status
    critical_passed = results['Dependencies'] and results['Census API'] and results['Collectors']
    
    if critical_passed:
        print("\n✓ System is ready for data collection!")
        print("\nNext steps:")
        print("  1. Run: python main.py --phase train --sources census")
        print("  2. Check: data/raw/ for collected data")
        print("  3. Review: logs/ for execution details")
    else:
        print("\n✗ Please fix the issues above before proceeding")
    
    return 0 if critical_passed else 1


if __name__ == '__main__':
    sys.exit(main())
