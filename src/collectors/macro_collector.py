"""
Macro Indicator Collector
Collects GSCPI and ISM PMI data
"""

import requests
import pandas as pd
from typing import Optional, Dict
from datetime import datetime
from io import StringIO

from .base_collector import BaseCollector
from ..config_manager import config
from ..utils import retry_with_backoff


class MacroIndicatorCollector(BaseCollector):
    """Collector for macroeconomic indicators (GSCPI, ISM PMI)"""
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize macro indicator collector
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        super().__init__(
            collector_name='MacroIndicatorCollector',
            data_source='macro_indicators',
            start_date=start_date,
            end_date=end_date
        )
        
        self.fed_config = config.get_data_source_config('fed_ny')
        self.ism_config = config.get_data_source_config('ism')
    
    @retry_with_backoff
    def fetch_gscpi(self) -> pd.DataFrame:
        """
        Fetch Global Supply Chain Pressure Index from NY Fed
        
        Returns:
            DataFrame with GSCPI data
        """
        self.logger.info("Fetching GSCPI data from NY Fed...")
        
        # NY Fed GSCPI data URL
        # Note: Actual URL may vary - check NY Fed website
        gscpi_url = "https://www.newyorkfed.org/medialibrary/media/research/policy/gscpi/data/gscpi.csv"
        
        try:
            response = requests.get(gscpi_url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Expected columns: date, gscpi
            if 'date' not in df.columns:
                # Try to find date column
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: 'date'})
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'])
            
            # Add metadata
            df['indicator'] = 'gscpi'
            df['data_source'] = 'ny_fed'
            
            self.logger.info(f"Fetched {len(df)} GSCPI records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch GSCPI: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def fetch_ism_pmi(self) -> pd.DataFrame:
        """
        Fetch ISM PMI data
        
        Returns:
            DataFrame with PMI data
        """
        self.logger.info("Fetching ISM PMI data...")
        
        # Note: ISM data typically requires subscription
        # This is a placeholder showing the structure
        # Alternative: Use FRED API (Federal Reserve Economic Data)
        
        try:
            # Example using FRED API for PMI data
            # Requires FRED API key
            fred_api_key = config.get('data_sources.fred.api_key')
            
            if not fred_api_key:
                self.logger.warning(
                    "No FRED API key found. ISM PMI data collection limited. "
                    "Set FRED_API_KEY in .env file."
                )
                return self._fetch_pmi_mock_data()
            
            # FRED series IDs for PMI components
            series_ids = {
                'MANEMP': 'manufacturing_pmi',
                'NEWORDERS': 'new_orders',
                'DELIVERIES': 'supplier_deliveries',
                'INVENTORY': 'inventories'
            }
            
            all_data = []
            
            for series_id, name in series_ids.items():
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': fred_api_key,
                    'file_type': 'json',
                    'observation_start': self.start_date or '2010-01-01',
                    'observation_end': self.end_date or datetime.now().strftime('%Y-%m-%d')
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'observations' in data:
                    obs_df = pd.DataFrame(data['observations'])
                    obs_df['indicator'] = name
                    obs_df['series_id'] = series_id
                    all_data.append(obs_df)
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df['data_source'] = 'fred'
                
                self.logger.info(f"Fetched {len(df)} PMI records")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to fetch ISM PMI: {e}")
            return pd.DataFrame()
    
    def _fetch_pmi_mock_data(self) -> pd.DataFrame:
        """
        Create mock PMI data structure for testing
        
        Returns:
            Mock DataFrame
        """
        self.logger.warning("Using mock PMI data for demonstration")
        
        date_range = self.get_date_range()
        dates = pd.date_range(
            start=date_range['start'],
            end=date_range['end'],
            freq='MS'
        )
        
        # Create mock data
        data = []
        for date in dates:
            data.append({
                'date': date,
                'indicator': 'manufacturing_pmi',
                'value': 50.0,  # Neutral value
                'data_source': 'mock'
            })
        
        return pd.DataFrame(data)
    
    def calculate_pmi_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PMI gap (new orders vs supplier deliveries)
        
        Args:
            df: DataFrame with PMI data
            
        Returns:
            DataFrame with gap calculation
        """
        self.logger.info("Calculating PMI gap...")
        
        # Pivot to get indicators as columns
        pivot_df = df.pivot_table(
            index='date',
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Calculate gap
        if 'new_orders' in pivot_df.columns and 'supplier_deliveries' in pivot_df.columns:
            # Supplier deliveries is inverse (higher = slower)
            # So gap = new_orders - (100 - supplier_deliveries)
            pivot_df['pmi_gap'] = (
                pivot_df['new_orders'] - (100 - pivot_df['supplier_deliveries'])
            )
            
            self.logger.info("PMI gap calculated successfully")
        else:
            self.logger.warning("Cannot calculate PMI gap - missing required indicators")
        
        return pivot_df
    
    def fetch(self, indicators: Optional[list] = None) -> pd.DataFrame:
        """
        Fetch macro indicator data
        
        Args:
            indicators: List of indicators to fetch ['gscpi', 'pmi'] (default: all)
            
        Returns:
            Combined DataFrame with all indicators
        """
        if indicators is None:
            indicators = ['gscpi', 'pmi']
        
        all_data = []
        
        if 'gscpi' in indicators:
            gscpi_df = self.fetch_gscpi()
            if not gscpi_df.empty:
                all_data.append(gscpi_df)
        
        if 'pmi' in indicators:
            pmi_df = self.fetch_ism_pmi()
            if not pmi_df.empty:
                all_data.append(pmi_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = self._transform_data(combined_df)
            return combined_df
        else:
            self.logger.warning("No macro indicator data fetched")
            return pd.DataFrame()
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform macro indicator data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.logger.info("Transforming macro indicator data...")
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Ensure value column is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Add collection timestamp
        df['collected_at'] = datetime.now()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Filter to date range
        date_range = self.get_date_range()
        start = pd.to_datetime(date_range['start'])
        end = pd.to_datetime(date_range['end'])
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        self.logger.info(f"Transformation complete. Final shape: {df.shape}")
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate macro indicator data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'indicator', 'value']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values in critical columns
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found: {null_counts.to_dict()}")
        
        # Check value ranges (PMI typically 0-100, GSCPI can be negative)
        if 'indicator' in df.columns and 'value' in df.columns:
            for indicator in df['indicator'].unique():
                indicator_data = df[df['indicator'] == indicator]['value']
                if 'pmi' in indicator.lower():
                    if (indicator_data < 0).any() or (indicator_data > 100).any():
                        self.logger.warning(f"{indicator} has values outside 0-100 range")
        
        self.logger.info("Validation passed")
        return True
