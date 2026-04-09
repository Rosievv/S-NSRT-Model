"""
USGS Mineral Data Collector
Collects mineral production, trade, and price data from USGS
"""

import re
import requests
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup

from .base_collector import BaseCollector
from ..config_manager import config
from ..utils import retry_with_backoff


class USGSCollector(BaseCollector):
    """Collector for USGS mineral commodity data"""
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize USGS collector
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        super().__init__(
            collector_name='USGSCollector',
            data_source='usgs',
            start_date=start_date,
            end_date=end_date
        )
        
        self.usgs_config = config.get_data_source_config('usgs')
        self.minerals = self.usgs_config.get('minerals', [])
        self.data_types = self.usgs_config.get('data_types', [])
    
    @retry_with_backoff
    def _download_file(self, url: str) -> bytes:
        """
        Download file from URL
        
        Args:
            url: File URL
            
        Returns:
            File content as bytes
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    
    def fetch_mineral_data(
        self,
        mineral: str,
        year_start: int,
        year_end: int
    ) -> pd.DataFrame:
        """
        Fetch data for a specific mineral
        
        Args:
            mineral: Mineral name (e.g., 'gallium', 'germanium')
            year_start: Start year
            year_end: End year
            
        Returns:
            DataFrame with mineral data
        """
        self.logger.info(f"Fetching {mineral} data from {year_start} to {year_end}")
        
        # Note: USGS data structure varies by publication
        # This is a simplified example showing the approach
        
        # For demonstration, we create a mock structure
        # In practice, you would parse actual USGS Excel/PDF files
        
        all_data = []
        
        for year in range(year_start, year_end + 1):
            try:
                # Simulate fetching annual data
                # Real implementation would download and parse USGS Mineral Commodity Summaries
                data_point = self._fetch_annual_mineral_data(mineral, year)
                if data_point:
                    all_data.append(data_point)
            except Exception as e:
                self.logger.warning(f"Failed to fetch {mineral} for {year}: {e}")
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()
    
    def _fetch_annual_mineral_data(
        self,
        mineral: str,
        year: int
    ) -> Optional[Dict]:
        """
        Fetch annual data for a mineral
        
        Args:
            mineral: Mineral name
            year: Year
            
        Returns:
            Dictionary with mineral statistics
        """
        # This is a placeholder implementation
        # Real implementation would:
        # 1. Download the appropriate USGS publication (PDF or Excel)
        # 2. Parse the tables for the specific mineral
        # 3. Extract key metrics
        
        self.logger.debug(f"Fetching {mineral} for year {year}")
        
        # Example structure of what would be extracted
        data = {
            'year': year,
            'mineral': mineral,
            'us_production': None,  # Would be extracted from USGS tables
            'us_imports': None,
            'us_exports': None,
            'world_production': None,
            'average_price': None,
            'import_dependence_pct': None,
            'major_sources': None  # Top import source countries
        }
        
        # Note: Actual implementation would use pandas.read_excel() or PyPDF2
        # to extract data from USGS publications
        
        return data
    
    def fetch(
        self,
        minerals: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch USGS mineral data
        
        Args:
            minerals: List of minerals to fetch (uses config if None)
            
        Returns:
            Combined DataFrame with mineral data
        """
        if minerals is None:
            minerals = self.minerals
        
        self.logger.info(f"Fetching data for minerals: {minerals}")
        
        # Get year range
        date_range = self.get_date_range()
        year_start = datetime.strptime(date_range['start'], '%Y-%m-%d').year
        year_end = datetime.strptime(date_range['end'], '%Y-%m-%d').year
        
        all_data = []
        
        for mineral in minerals:
            df = self.fetch_mineral_data(mineral, year_start, year_end)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = self._transform_data(combined_df)
            return combined_df
        else:
            self.logger.warning("No USGS data fetched")
            return pd.DataFrame()
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw USGS data into standardized format
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.logger.info("Transforming USGS data...")
        
        # Create date column (use January 1st for annual data)
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
        
        # Add metadata
        df['data_source'] = 'usgs'
        df['collected_at'] = datetime.now()
        df['frequency'] = 'annual'
        
        # Standardize mineral names
        df['mineral'] = df['mineral'].str.lower().str.strip()
        
        # Ensure numeric columns
        numeric_cols = [
            'us_production', 'us_imports', 'us_exports',
            'world_production', 'average_price', 'import_dependence_pct'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info(f"Transformation complete. Final shape: {df.shape}")
        return df
    
    def interpolate_to_monthly(
        self,
        df: pd.DataFrame,
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Interpolate annual data to monthly frequency
        
        Args:
            df: DataFrame with annual data
            method: Interpolation method ('linear', 'forward_fill', 'spline')
            
        Returns:
            DataFrame with monthly data
        """
        self.logger.info(f"Interpolating to monthly using {method} method")
        
        # Group by mineral
        result_dfs = []
        
        for mineral, group in df.groupby('mineral'):
            # Create monthly date range
            date_range = pd.date_range(
                start=group['date'].min(),
                end=group['date'].max(),
                freq='MS'
            )
            
            # Create monthly DataFrame
            monthly_df = pd.DataFrame({'date': date_range})
            monthly_df['mineral'] = mineral
            
            # Merge with annual data
            group_reindexed = group.set_index('date')
            monthly_df = monthly_df.set_index('date')
            
            # Perform interpolation for numeric columns
            numeric_cols = group_reindexed.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                if method == 'linear':
                    monthly_df[col] = group_reindexed[col].reindex(
                        monthly_df.index
                    ).interpolate(method='linear')
                elif method == 'forward_fill':
                    monthly_df[col] = group_reindexed[col].reindex(
                        monthly_df.index
                    ).fillna(method='ffill')
                elif method == 'spline':
                    monthly_df[col] = group_reindexed[col].reindex(
                        monthly_df.index
                    ).interpolate(method='spline', order=2)
            
            monthly_df = monthly_df.reset_index()
            monthly_df['interpolated'] = True
            result_dfs.append(monthly_df)
        
        result = pd.concat(result_dfs, ignore_index=True)
        self.logger.info(f"Interpolation complete. Shape: {result.shape}")
        
        return result
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate USGS mineral data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False
        
        # Check required columns
        required_cols = ['year', 'mineral', 'data_source']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check year range
        if 'year' in df.columns:
            min_year = df['year'].min()
            max_year = df['year'].max()
            if min_year < 1900 or max_year > datetime.now().year:
                self.logger.warning(f"Unusual year range: {min_year}-{max_year}")
        
        # Check for expected minerals
        if 'mineral' in df.columns:
            found_minerals = set(df['mineral'].unique())
            expected_minerals = set(self.minerals)
            missing = expected_minerals - found_minerals
            if missing:
                self.logger.warning(f"Missing minerals: {missing}")
        
        self.logger.info("Validation passed")
        return True
