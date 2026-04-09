"""
Temporal Feature Extractor

Extracts time-series features including trends, seasonality, and lagged features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base_feature import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    Extract temporal and time-series features
    
    Features:
    - Trend: Linear trend over time
    - Seasonality: Monthly, quarterly patterns
    - Lagged features: Previous periods' values
    - Moving averages: SMA, EMA
    - Momentum: Rate of change
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temporal feature extractor
        
        Args:
            config: Configuration with parameters:
                - lags: List of lag periods (default: [1, 3, 6, 12])
                - ma_windows: Moving average windows (default: [3, 6, 12])
        """
        default_config = {
            'lags': [1, 3, 6, 12],
            'ma_windows': [3, 6, 12],
            'feature_names': [
                'month', 'quarter', 'year',
                'value_trend', 'value_ma_3m', 'value_ma_6m', 'value_ma_12m',
                'value_lag_1m', 'value_lag_3m', 'value_lag_6m', 'value_lag_12m',
                'momentum_1m', 'momentum_3m', 'momentum_12m',
                'seasonality_index', 'is_peak_season'
            ]
        }
        if config:
            default_config.update(config)
        
        super().__init__('TemporalFeatures', default_config)
    
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract temporal features
        
        Args:
            df: DataFrame with columns: date, hs_code, value_usd
            **kwargs: Additional parameters
                - group_by: Grouping columns (default: ['hs_code'])
                
        Returns:
            DataFrame with temporal features
        """
        self.validate_input(df, ['date', 'hs_code', 'value_usd'])
        
        group_by = kwargs.get('group_by', ['hs_code'])
        
        self.logger.info(f"Extracting temporal features, grouping by {group_by}")
        
        # Aggregate by date and group
        agg_cols = list(set(group_by + ['date']))
        df_agg = df.groupby(agg_cols).agg({
            'value_usd': 'sum'
        }).reset_index()
        
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.sort_values(agg_cols)
        
        # Extract date features
        df_agg['month'] = df_agg['date'].dt.month
        df_agg['quarter'] = df_agg['date'].dt.quarter
        df_agg['year'] = df_agg['date'].dt.year
        
        # Calculate features for each group
        for group_keys, group_indices in df_agg.groupby(group_by).groups.items():
            group_df = df_agg.loc[group_indices].copy()
            
            # Trend
            df_agg.loc[group_indices, 'value_trend'] = self._calculate_trend(group_df['value_usd'])
            
            # Moving averages
            for window in self.config['ma_windows']:
                df_agg.loc[group_indices, f'value_ma_{window}m'] = (
                    group_df['value_usd'].rolling(window=window, min_periods=1).mean()
                )
            
            # Lagged features
            for lag in self.config['lags']:
                df_agg.loc[group_indices, f'value_lag_{lag}m'] = group_df['value_usd'].shift(lag)
            
            # Momentum (rate of change)
            df_agg.loc[group_indices, 'momentum_1m'] = group_df['value_usd'].pct_change(periods=1) * 100
            df_agg.loc[group_indices, 'momentum_3m'] = group_df['value_usd'].pct_change(periods=3) * 100
            df_agg.loc[group_indices, 'momentum_12m'] = group_df['value_usd'].pct_change(periods=12) * 100
            
            # Seasonality
            seasonality_df = self._calculate_seasonality(group_df)
            df_agg.loc[group_indices, 'seasonality_index'] = seasonality_df['seasonality_index'].values
            df_agg.loc[group_indices, 'is_peak_season'] = seasonality_df['is_peak_season'].values
        
        self.logger.info(f"Extracted temporal features for {len(df_agg)} records")
        return df_agg
    
    def _calculate_trend(self, series: pd.Series) -> pd.Series:
        """
        Calculate linear trend
        
        Args:
            series: Time series data
            
        Returns:
            Series with trend values
        """
        if len(series) < 2:
            return pd.Series([0] * len(series), index=series.index)
        
        # Linear regression: y = ax + b
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return pd.Series([0] * len(series), index=series.index)
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        # Return trend for all points
        trend = pd.Series([slope * i for i in x], index=series.index)
        
        return trend
    
    def _calculate_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonality index
        
        Args:
            df: DataFrame with date and value_usd
            
        Returns:
            DataFrame with seasonality_index and is_peak_season
        """
        df = df.copy()
        
        # Calculate average by month
        df['month'] = pd.to_datetime(df['date']).dt.month
        month_avg = df.groupby('month')['value_usd'].mean()
        overall_avg = df['value_usd'].mean()
        
        if overall_avg == 0:
            df['seasonality_index'] = 1.0
            df['is_peak_season'] = False
            return df[['seasonality_index', 'is_peak_season']]
        
        # Seasonality index: month_avg / overall_avg
        seasonality_map = (month_avg / overall_avg).to_dict()
        df['seasonality_index'] = df['month'].map(seasonality_map)
        
        # Peak season: seasonality index > 1.1
        df['is_peak_season'] = df['seasonality_index'] > 1.1
        
        return df[['seasonality_index', 'is_peak_season']]
