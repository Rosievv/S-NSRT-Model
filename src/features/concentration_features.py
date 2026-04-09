"""
Concentration Feature Extractor

Calculates supply chain concentration metrics including HHI (Herfindahl-Hirschman Index)
to measure supplier concentration risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base_feature import BaseFeatureExtractor


class ConcentrationFeatureExtractor(BaseFeatureExtractor):
    """
    Extract supply chain concentration features
    
    Features:
    - HHI (Herfindahl-Hirschman Index): Market concentration by country
    - Top N concentration: Share of top N suppliers
    - Gini coefficient: Inequality measure of trade distribution
    - Number of active suppliers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize concentration feature extractor
        
        Args:
            config: Configuration with parameters:
                - top_n: Number of top suppliers to track (default: 5)
                - rolling_window: Window size for rolling features (default: 12)
        """
        default_config = {
            'top_n': 5,
            'rolling_window': 12,
            'feature_names': [
                'hhi', 'hhi_3m', 'hhi_6m', 'hhi_12m',
                'top1_share', 'top3_share', 'top5_share',
                'gini_coefficient', 'n_suppliers',
                'hhi_change_mom', 'hhi_change_yoy'
            ]
        }
        if config:
            default_config.update(config)
        
        super().__init__('ConcentrationFeatures', default_config)
    
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract concentration features
        
        Args:
            df: DataFrame with columns: date, hs_code, country, value_usd
            **kwargs: Additional parameters
                - group_by: Grouping columns (default: ['date', 'hs_code'])
                
        Returns:
            DataFrame with concentration features
        """
        self.validate_input(df, ['date', 'hs_code', 'country', 'value_usd'])
        
        group_by = kwargs.get('group_by', ['date', 'hs_code'])
        
        self.logger.info(f"Extracting concentration features, grouping by {group_by}")
        
        # Calculate features for each group
        results = []
        
        for group_keys, group_df in df.groupby(group_by):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            
            # Create base feature dict
            features = dict(zip(group_by, group_keys))
            
            # Calculate HHI
            features['hhi'] = self._calculate_hhi(group_df)
            
            # Calculate top N concentration
            top_n = self.config['top_n']
            features.update(self._calculate_top_n_share(group_df, top_n))
            
            # Calculate Gini coefficient
            features['gini_coefficient'] = self._calculate_gini(group_df)
            
            # Number of suppliers
            features['n_suppliers'] = group_df['country'].nunique()
            
            results.append(features)
        
        result_df = pd.DataFrame(results)
        
        # Calculate rolling HHI features
        if 'date' in group_by and 'hs_code' in group_by:
            result_df = self._add_rolling_features(result_df)
            result_df = self._add_change_features(result_df)
        
        self.logger.info(f"Extracted {len(result_df)} concentration feature records")
        return result_df
    
    def _calculate_hhi(self, df: pd.DataFrame) -> float:
        """
        Calculate Herfindahl-Hirschman Index
        
        HHI = sum((market_share_i)^2) * 10000
        
        Args:
            df: DataFrame with value_usd and country
            
        Returns:
            HHI value (0-10000, higher = more concentrated)
        """
        total_value = df['value_usd'].sum()
        
        if total_value == 0:
            return 0.0
        
        # Calculate market share for each country
        country_shares = df.groupby('country')['value_usd'].sum() / total_value
        
        # HHI = sum of squared market shares * 10000
        hhi = (country_shares ** 2).sum() * 10000
        
        return float(hhi)
    
    def _calculate_top_n_share(self, df: pd.DataFrame, n: int) -> Dict[str, float]:
        """
        Calculate concentration of top N suppliers
        
        Args:
            df: DataFrame with value_usd and country
            n: Number of top suppliers
            
        Returns:
            Dictionary with top1_share, top3_share, top5_share, etc.
        """
        total_value = df['value_usd'].sum()
        
        if total_value == 0:
            return {f'top{i}_share': 0.0 for i in [1, 3, 5] if i <= n}
        
        # Get top N countries by value
        top_values = df.groupby('country')['value_usd'].sum().nlargest(n)
        
        results = {}
        for i in [1, 3, 5]:
            if i <= n and i <= len(top_values):
                results[f'top{i}_share'] = float(top_values.head(i).sum() / total_value)
            elif i <= n:
                results[f'top{i}_share'] = 0.0
        
        return results
    
    def _calculate_gini(self, df: pd.DataFrame) -> float:
        """
        Calculate Gini coefficient for trade distribution
        
        Gini = 0 means perfect equality, 1 means perfect inequality
        
        Args:
            df: DataFrame with value_usd
            
        Returns:
            Gini coefficient (0-1)
        """
        values = df.groupby('country')['value_usd'].sum().values
        
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
        
        return float(gini)
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling HHI features
        
        Args:
            df: DataFrame with date, hs_code, and hhi
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['hs_code', 'date'])
        
        # Calculate rolling HHI for different windows
        for window in [3, 6, 12]:
            col_name = f'hhi_{window}m'
            df[col_name] = df.groupby('hs_code')['hhi'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return df
    
    def _add_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add HHI change features (MoM, YoY)
        
        Args:
            df: DataFrame with date, hs_code, and hhi
            
        Returns:
            DataFrame with change features added
        """
        df = df.copy()
        df = df.sort_values(['hs_code', 'date'])
        
        # Month-over-month change
        df['hhi_change_mom'] = df.groupby('hs_code')['hhi'].diff()
        
        # Year-over-year change (12 months)
        df['hhi_change_yoy'] = df.groupby('hs_code')['hhi'].diff(periods=12)
        
        return df
