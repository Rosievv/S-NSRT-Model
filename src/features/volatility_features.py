"""
Volatility Feature Extractor

Calculates volatility and stability metrics for supply chain risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_feature import BaseFeatureExtractor


class VolatilityFeatureExtractor(BaseFeatureExtractor):
    """
    Extract volatility and stability features
    
    Features:
    - CoV (Coefficient of Variation): Relative volatility
    - Standard deviation: Absolute volatility
    - Rolling volatility: Short-term and long-term
    - Price/quantity volatility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize volatility feature extractor
        
        Args:
            config: Configuration with parameters:
                - windows: Rolling windows to calculate (default: [3, 6, 12])
        """
        default_config = {
            'windows': [3, 6, 12],
            'feature_names': [
                'value_std', 'value_cov', 
                'value_std_3m', 'value_std_6m', 'value_std_12m',
                'value_cov_3m', 'value_cov_6m', 'value_cov_12m',
                'quantity_std', 'quantity_cov',
                'volatility_trend', 'stability_score'
            ]
        }
        if config:
            default_config.update(config)
        
        super().__init__('VolatilityFeatures', default_config)
    
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract volatility features
        
        Args:
            df: DataFrame with columns: date, hs_code, country, value_usd, quantity
            **kwargs: Additional parameters
                - group_by: Grouping columns (default: ['hs_code'])
                
        Returns:
            DataFrame with volatility features
        """
        self.validate_input(df, ['date', 'hs_code', 'value_usd'])
        
        group_by = kwargs.get('group_by', ['hs_code'])
        
        self.logger.info(f"Extracting volatility features, grouping by {group_by}")
        
        # Aggregate by date and group
        agg_cols = list(set(group_by + ['date']))
        df_agg = df.groupby(agg_cols).agg({
            'value_usd': 'sum',
            'quantity': 'sum' if 'quantity' in df.columns else 'count'
        }).reset_index()
        
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.sort_values(agg_cols)
        
        # Calculate features for each group
        results = []
        
        for group_keys, group_df in df_agg.groupby(group_by):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            
            features = dict(zip(group_by, group_keys))
            
            # Basic volatility
            features['value_std'] = float(group_df['value_usd'].std())
            features['value_cov'] = self._calculate_cov(group_df['value_usd'])
            
            if 'quantity' in group_df.columns:
                features['quantity_std'] = float(group_df['quantity'].std())
                features['quantity_cov'] = self._calculate_cov(group_df['quantity'])
            
            # Rolling volatility
            for window in self.config['windows']:
                features[f'value_std_{window}m'] = float(
                    group_df['value_usd'].rolling(window=window, min_periods=1).std().iloc[-1]
                )
                features[f'value_cov_{window}m'] = self._calculate_rolling_cov(
                    group_df['value_usd'], window
                )
            
            # Volatility trend (is volatility increasing or decreasing?)
            features['volatility_trend'] = self._calculate_volatility_trend(group_df['value_usd'])
            
            # Stability score (inverse of normalized CoV, 0-100)
            features['stability_score'] = self._calculate_stability_score(group_df['value_usd'])
            
            results.append(features)
        
        result_df = pd.DataFrame(results)
        
        self.logger.info(f"Extracted {len(result_df)} volatility feature records")
        return result_df
    
    def _calculate_cov(self, series: pd.Series) -> float:
        """
        Calculate Coefficient of Variation
        
        CoV = (std / mean) * 100
        
        Args:
            series: Data series
            
        Returns:
            CoV value (percentage)
        """
        mean = series.mean()
        if mean == 0 or pd.isna(mean):
            return 0.0
        
        std = series.std()
        return float((std / mean) * 100)
    
    def _calculate_rolling_cov(self, series: pd.Series, window: int) -> float:
        """
        Calculate rolling CoV for last window
        
        Args:
            series: Data series
            window: Rolling window size
            
        Returns:
            Rolling CoV value
        """
        rolling_mean = series.rolling(window=window, min_periods=1).mean().iloc[-1]
        rolling_std = series.rolling(window=window, min_periods=1).std().iloc[-1]
        
        if rolling_mean == 0 or pd.isna(rolling_mean):
            return 0.0
        
        return float((rolling_std / rolling_mean) * 100)
    
    def _calculate_volatility_trend(self, series: pd.Series) -> float:
        """
        Calculate volatility trend
        
        Positive value means volatility is increasing
        Negative value means volatility is decreasing
        
        Args:
            series: Data series
            
        Returns:
            Volatility trend value
        """
        if len(series) < 24:
            return 0.0
        
        # Split into two halves
        mid = len(series) // 2
        first_half_vol = series.iloc[:mid].std()
        second_half_vol = series.iloc[mid:].std()
        
        if first_half_vol == 0:
            return 0.0
        
        # Percentage change in volatility
        trend = ((second_half_vol - first_half_vol) / first_half_vol) * 100
        
        return float(trend)
    
    def _calculate_stability_score(self, series: pd.Series) -> float:
        """
        Calculate stability score (0-100)
        
        Higher score = more stable
        
        Args:
            series: Data series
            
        Returns:
            Stability score (0-100)
        """
        cov = self._calculate_cov(series)
        
        # Normalize CoV to 0-100 scale (inverse)
        # CoV of 0% = 100 stability, CoV of 100%+ = 0 stability
        stability = max(0, min(100, 100 - cov))
        
        return float(stability)
