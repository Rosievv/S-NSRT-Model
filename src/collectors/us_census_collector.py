"""
US Census Bureau Trade Data Collector
Collects import/export data for semiconductor supply chain HS codes
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from urllib.parse import urlencode

from .base_collector import BaseCollector
from ..config_manager import config
from ..utils import retry_with_backoff, chunk_list, standardize_country_name, generate_date_range


class USCensusCollector(BaseCollector):
    """Collector for US Census Bureau international trade data"""
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trade_type: str = 'imports'
    ):
        """
        Initialize US Census collector
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            trade_type: 'imports' or 'exports'
        """
        super().__init__(
            collector_name='USCensusCollector',
            data_source='us_census',
            start_date=start_date,
            end_date=end_date
        )
        
        self.trade_type = trade_type
        self.census_config = config.get_data_source_config('us_census')
        self.api_key = self.census_config.get('api_key')
        
        if not self.api_key:
            self.logger.warning(
                "No API key found. Set US_CENSUS_API_KEY in .env file. "
                "Some features may be limited."
            )
        
        # Rate limiting
        self.rate_limit = self.census_config.get('rate_limit', {})
        self.requests_per_minute = self.rate_limit.get('requests_per_minute', 30)
        self.retry_attempts = self.rate_limit.get('retry_attempts', 3)
        self.retry_delay = self.rate_limit.get('retry_delay', 5)
        
        # Track request timing for rate limiting
        self.request_times: List[float] = []
    
    def _wait_for_rate_limit(self) -> None:
        """Implement rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < 60
        ]
        
        # If we've hit the limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.request_times = []
        
        self.request_times.append(current_time)
    
    @retry_with_backoff
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict:
        """
        Make HTTP request with retry logic
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response
        """
        self._wait_for_rate_limit()
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def fetch_hs_code_data(
        self,
        hs_code: str,
        year: int,
        month: int
    ) -> pd.DataFrame:
        """
        Fetch data for a specific HS code and time period
        
        Args:
            hs_code: 6-digit HS code
            year: Year
            month: Month (1-12)
            
        Returns:
            DataFrame with trade data
        """
        self.logger.debug(f"Fetching {hs_code} for {year}-{month:02d}")
        
        # Note: US Census API structure varies by endpoint
        # This is a simplified example - actual API may differ
        base_url = self.census_config.get('base_url')
        endpoint = self.census_config['endpoints'][self.trade_type]
        
        params = {
            'get': 'I_COMMODITY,CTY_CODE,CTY_NAME,GEN_VAL_MO,CON_VAL_MO,GEN_QY1_MO',
            'COMM_LVL': 'HS6',
            'I_COMMODITY': hs_code,
            'time': f"{year}-{month:02d}",
            'key': self.api_key
        }
        
        try:
            url = f"{base_url}{endpoint}"
            data = self._make_request(url, params)
            
            # Parse response (format depends on actual API)
            if isinstance(data, list) and len(data) > 1:
                # First row is headers
                headers = data[0]
                rows = data[1:]
                
                df = pd.DataFrame(rows, columns=headers)
                df['year'] = year
                df['month'] = month
                # Don't add hs_code here as I_COMMODITY will be renamed to hs_code in transform
                
                return df
            else:
                self.logger.warning(f"No data returned for {hs_code} {year}-{month:02d}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {hs_code}: {e}")
            return pd.DataFrame()
    
    def fetch(
        self,
        hs_codes: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch trade data for specified HS codes
        
        Args:
            hs_codes: List of HS codes to fetch (if None, uses config)
            category: Category from config (e.g., 'integrated_circuits')
            
        Returns:
            Combined DataFrame with all fetched data
        """
        # Get HS codes from config if not provided
        if hs_codes is None:
            if category:
                codes_config = config.get_hs_codes(category)
                # When category is specified, codes_config is a list directly
                if isinstance(codes_config, list):
                    hs_codes = [item['code'] for item in codes_config]
                else:
                    hs_codes = []
            else:
                codes_config = config.get_hs_codes()
                # When no category, codes_config is a dict of categories
                hs_codes = []
                for cat_name, cat_codes in codes_config.items():
                    if isinstance(cat_codes, list):
                        hs_codes.extend([item['code'] for item in cat_codes])
        
        self.logger.info(f"Fetching data for {len(hs_codes)} HS codes")
        
        # Generate date range
        date_range = self.get_date_range()
        dates = generate_date_range(
            date_range['start'],
            date_range['end'],
            freq='MS'
        )
        
        # Convert to year-month tuples
        year_months = [
            (datetime.strptime(d, '%Y-%m-%d').year,
             datetime.strptime(d, '%Y-%m-%d').month)
            for d in dates
        ]
        
        all_data = []
        total_requests = len(hs_codes) * len(year_months)
        completed = 0
        
        # Fetch data for each HS code and time period
        for hs_code in hs_codes:
            self.logger.info(f"Processing HS code: {hs_code}")
            
            for year, month in year_months:
                df = self.fetch_hs_code_data(hs_code, year, month)
                if not df.empty:
                    all_data.append(df)
                
                completed += 1
                if completed % 10 == 0:
                    progress = (completed / total_requests) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({completed}/{total_requests})")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = self._transform_data(combined_df)
            return combined_df
        else:
            self.logger.warning("No data fetched")
            return pd.DataFrame()
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw Census data into standardized format
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            Transformed DataFrame
        """
        self.logger.info("Transforming data...")
        
        # Create date column
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
        
        # Standardize column names (adjust based on actual API response)
        column_mapping = {
            'CTY_NAME': 'country',
            'CTY_CODE': 'country_code',
            'GEN_VAL_MO': 'value_usd',
            'GEN_QY1_MO': 'quantity',
            'I_COMMODITY': 'hs_code'
        }
        
        # Rename columns that exist
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)
        
        # Remove duplicate columns (keep the first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Standardize country names
        if 'country' in df.columns:
            df['country'] = df['country'].apply(standardize_country_name)
        
        # Convert numeric columns
        numeric_cols = ['value_usd', 'quantity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata
        df['trade_type'] = self.trade_type
        df['data_source'] = 'us_census'
        df['collected_at'] = datetime.now()
        
        # Select and order columns
        standard_cols = [
            'date', 'hs_code', 'country', 'country_code',
            'value_usd', 'quantity', 'trade_type', 'data_source', 'collected_at'
        ]
        existing_cols = [col for col in standard_cols if col in df.columns]
        df = df[existing_cols]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'hs_code', 'country'])
        
        self.logger.info(f"Transformation complete. Final shape: {df.shape}")
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate Census trade data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        required_cols = self.validation_rules.get('required_columns', {}).get('trade_data', [])
        
        # Basic validation
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False
        
        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for reasonable value ranges
        if 'value_usd' in df.columns:
            negative_values = (df['value_usd'] < 0).sum()
            if negative_values > 0:
                self.logger.warning(f"Found {negative_values} negative values")
        
        # Check date continuity
        if 'date' in df.columns:
            date_gaps = df['date'].sort_values().diff().dt.days
            large_gaps = (date_gaps > 45).sum()  # More than 45 days
            if large_gaps > 0:
                self.logger.warning(f"Found {large_gaps} large date gaps")
        
        self.logger.info("Validation passed")
        return True
