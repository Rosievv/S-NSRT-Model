"""
Main orchestrator for SCRAM data collection
Run this script to collect all data sources
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.collectors import USCensusCollector, USGSCollector, MacroIndicatorCollector
from src.config_manager import config
from src.utils import setup_logger


def collect_census_data(args):
    """Collect US Census trade data"""
    logger.info("=" * 80)
    logger.info("Collecting US Census Trade Data")
    logger.info("=" * 80)
    
    collector = USCensusCollector(
        start_date=args.start_date,
        end_date=args.end_date,
        trade_type=args.trade_type
    )
    
    # Collect data for specified category or all
    df = collector.collect(
        category=args.category if args.category != 'all' else None,
        save_data=True,
        validate_data=True
    )
    
    logger.info(f"Collected {len(df)} trade records")
    return df


def collect_usgs_data(args):
    """Collect USGS mineral data"""
    logger.info("=" * 80)
    logger.info("Collecting USGS Mineral Data")
    logger.info("=" * 80)
    
    collector = USGSCollector(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    df = collector.collect(save_data=True, validate_data=True)
    
    logger.info(f"Collected {len(df)} mineral records")
    return df


def collect_macro_data(args):
    """Collect macro indicator data"""
    logger.info("=" * 80)
    logger.info("Collecting Macro Indicator Data")
    logger.info("=" * 80)
    
    collector = MacroIndicatorCollector(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    df = collector.collect(
        indicators=args.indicators,
        save_data=True,
        validate_data=True
    )
    
    logger.info(f"Collected {len(df)} macro indicator records")
    return df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='SCRAM Data Collection Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all data for training period
  python main.py --phase train --sources all
  
  # Collect only Census data for custom period
  python main.py --sources census --start-date 2020-01-01 --end-date 2024-12-31
  
  # Collect specific HS code category
  python main.py --sources census --category integrated_circuits
  
  # Collect macro indicators only
  python main.py --sources macro --indicators gscpi pmi
        """
    )
    
    # Phase selection
    parser.add_argument(
        '--phase',
        choices=['train', 'test', 'custom'],
        default='train',
        help='Data collection phase (train: 2010-2019, test: 2020-2024, custom: specify dates)'
    )
    
    # Date range (for custom phase)
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD) for custom phase'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD) for custom phase'
    )
    
    # Data sources
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['census', 'usgs', 'macro', 'all'],
        default=['all'],
        help='Data sources to collect'
    )
    
    # Census-specific options
    parser.add_argument(
        '--trade-type',
        choices=['imports', 'exports'],
        default='imports',
        help='Trade type for Census data'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='all',
        help='HS code category (e.g., integrated_circuits, manufacturing_equipment)'
    )
    
    # Macro-specific options
    parser.add_argument(
        '--indicators',
        nargs='+',
        choices=['gscpi', 'pmi'],
        default=None,
        help='Macro indicators to collect'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup global logger
    global logger
    log_file = config.resolve_path(f"logs/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger('SCRAM_Main', str(log_file), args.log_level)
    
    logger.info("=" * 80)
    logger.info("SCRAM Data Collection Pipeline Started")
    logger.info("=" * 80)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Sources: {args.sources}")
    
    # Determine date range
    if args.phase == 'custom':
        if not args.start_date or not args.end_date:
            logger.error("Custom phase requires --start-date and --end-date")
            sys.exit(1)
    else:
        time_range = config.get_time_range(args.phase)
        args.start_date = time_range['start']
        args.end_date = time_range['end']
    
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Collect data from specified sources
    results = {}
    
    if 'all' in args.sources or 'census' in args.sources:
        try:
            results['census'] = collect_census_data(args)
        except Exception as e:
            logger.error(f"Failed to collect Census data: {e}", exc_info=True)
    
    if 'all' in args.sources or 'usgs' in args.sources:
        try:
            results['usgs'] = collect_usgs_data(args)
        except Exception as e:
            logger.error(f"Failed to collect USGS data: {e}", exc_info=True)
    
    if 'all' in args.sources or 'macro' in args.sources:
        try:
            results['macro'] = collect_macro_data(args)
        except Exception as e:
            logger.error(f"Failed to collect macro data: {e}", exc_info=True)
    
    # Summary
    logger.info("=" * 80)
    logger.info("Data Collection Summary")
    logger.info("=" * 80)
    for source, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"{source.upper()}: {len(df)} records collected")
        else:
            logger.info(f"{source.upper()}: No data collected")
    
    logger.info("=" * 80)
    logger.info("SCRAM Data Collection Pipeline Completed")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
