"""
Example usage of SCRAM data collectors
Demonstrates different collection scenarios
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.collectors import USCensusCollector, USGSCollector, MacroIndicatorCollector
from src.config_manager import config


def example_1_collect_census_data():
    """Example 1: Collect US Census trade data for integrated circuits"""
    print("\n" + "=" * 80)
    print("Example 1: Collecting US Census Trade Data")
    print("=" * 80)
    
    # Initialize collector for training period
    collector = USCensusCollector(
        start_date='2010-01-01',
        end_date='2019-12-31',
        trade_type='imports'
    )
    
    # Collect data for integrated circuits category
    df = collector.collect(
        category='integrated_circuits',
        save_data=True,
        validate_data=True
    )
    
    print(f"\nCollected {len(df)} records")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Show statistics
    if not df.empty:
        print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique HS codes: {df['hs_code'].nunique()}")
        print(f"Unique countries: {df['country'].nunique()}")
        print(f"Total import value: ${df['value_usd'].sum():,.0f}")


def example_2_collect_usgs_data():
    """Example 2: Collect USGS mineral data with interpolation"""
    print("\n" + "=" * 80)
    print("Example 2: Collecting USGS Mineral Data")
    print("=" * 80)
    
    # Initialize collector
    collector = USGSCollector(
        start_date='2010-01-01',
        end_date='2019-12-31'
    )
    
    # Collect annual data
    df_annual = collector.collect(
        save_data=True,
        validate_data=True
    )
    
    print(f"\nCollected {len(df_annual)} annual records")
    
    # Interpolate to monthly
    if not df_annual.empty:
        df_monthly = collector.interpolate_to_monthly(df_annual, method='linear')
        print(f"Interpolated to {len(df_monthly)} monthly records")
        print(f"\nMonthly data sample:")
        print(df_monthly.head())


def example_3_collect_macro_indicators():
    """Example 3: Collect macro indicators and calculate PMI gap"""
    print("\n" + "=" * 80)
    print("Example 3: Collecting Macro Indicators")
    print("=" * 80)
    
    # Initialize collector
    collector = MacroIndicatorCollector(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Collect GSCPI and PMI data
    df = collector.collect(
        indicators=['gscpi', 'pmi'],
        save_data=True,
        validate_data=True
    )
    
    print(f"\nCollected {len(df)} indicator records")
    
    # Calculate PMI gap
    if not df.empty:
        pmi_data = df[df['indicator'].str.contains('pmi', case=False)]
        if not pmi_data.empty:
            gap_df = collector.calculate_pmi_gap(pmi_data)
            print(f"\nPMI Gap calculation:")
            print(gap_df.head())


def example_4_custom_date_range():
    """Example 4: Collect data for custom date range"""
    print("\n" + "=" * 80)
    print("Example 4: Custom Date Range Collection")
    print("=" * 80)
    
    # Collect data for COVID period only
    collector = USCensusCollector(
        start_date='2020-03-01',
        end_date='2021-12-31',
        trade_type='imports'
    )
    
    # Collect specific HS codes
    df = collector.collect(
        hs_codes=['854231', '854232'],  # Processors and memories
        save_data=False,  # Don't save, just analyze
        validate_data=True
    )
    
    print(f"\nCollected {len(df)} records for COVID period")
    
    if not df.empty:
        # Analyze year-over-year changes
        df['year'] = df['date'].dt.year
        yearly_imports = df.groupby('year')['value_usd'].sum()
        print(f"\nYearly import values:")
        print(yearly_imports)


def example_5_batch_collection():
    """Example 5: Batch collection of all sources"""
    print("\n" + "=" * 80)
    print("Example 5: Batch Collection of All Sources")
    print("=" * 80)
    
    # Define collection parameters
    start_date = '2010-01-01'
    end_date = '2019-12-31'
    
    results = {}
    
    # Census data
    print("\n1. Collecting Census data...")
    census = USCensusCollector(start_date=start_date, end_date=end_date)
    try:
        results['census'] = census.collect(category='integrated_circuits')
        print(f"   ✓ Census: {len(results['census'])} records")
    except Exception as e:
        print(f"   ✗ Census failed: {e}")
    
    # USGS data
    print("\n2. Collecting USGS data...")
    usgs = USGSCollector(start_date=start_date, end_date=end_date)
    try:
        results['usgs'] = usgs.collect()
        print(f"   ✓ USGS: {len(results['usgs'])} records")
    except Exception as e:
        print(f"   ✗ USGS failed: {e}")
    
    # Macro indicators
    print("\n3. Collecting macro indicators...")
    macro = MacroIndicatorCollector(start_date=start_date, end_date=end_date)
    try:
        results['macro'] = macro.collect(indicators=['gscpi', 'pmi'])
        print(f"   ✓ Macro: {len(results['macro'])} records")
    except Exception as e:
        print(f"   ✗ Macro failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Collection Summary")
    print("=" * 80)
    total_records = sum(len(df) for df in results.values() if df is not None)
    print(f"Total records collected: {total_records:,}")
    print(f"Data sources: {len(results)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SCRAM Usage Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Example number to run (1-5). If not specified, runs all examples.'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SCRAM Data Collection Examples")
    print("=" * 80)
    print("\nNote: These examples demonstrate the API. Actual data collection")
    print("requires valid API keys and may take significant time.")
    print("=" * 80)
    
    if args.example:
        # Run specific example
        examples = {
            1: example_1_collect_census_data,
            2: example_2_collect_usgs_data,
            3: example_3_collect_macro_indicators,
            4: example_4_custom_date_range,
            5: example_5_batch_collection
        }
        examples[args.example]()
    else:
        # Run all examples
        print("\nRunning all examples...")
        example_1_collect_census_data()
        example_2_collect_usgs_data()
        example_3_collect_macro_indicators()
        example_4_custom_date_range()
        example_5_batch_collection()
    
    print("\n" + "=" * 80)
    print("Examples completed")
    print("=" * 80)
