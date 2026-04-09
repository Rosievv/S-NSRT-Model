"""
Growth Feature Extractor

Calculates growth rates and change metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_feature import BaseFeatureExtractor


class GrowthFeatureExtractor(BaseFeatureExtractor):
    """
    Extract growth and change features
    
    Features:
    - MoM (Month-over-Month) growth
    - YoY (Year-over-Year) growth
    - QoQ (Quarter-over-Quarter) growth
    - CAGR (Compound Annual Growth Rate)
    - Growth acceleration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize growth feature extractor
        
        Args:
            config: Configuration parameters
        """
        default_config = {
            'feature_names': [
                'growth_mom', 'growth_qoq', 'growth_yoy',
                'growth_3m_avg', 'growth_6m_avg', 'growth_12m_avg',
                'growth_acceleration', 'is_growing',
                'cagr_3y', 'cumulative_growth'
            ]
        }
        if config:
            default_config.update(config)
        
        super().__init__('GrowthFeatures', default_config)
    
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract growth features
        
        Args:
            df: DataFrame with columns: date, hs_code, value_usd
            **kwargs: Additional parameters
                - group_by: Grouping columns (default: ['hs_code'])
                
        Returns:
            DataFrame with growth features
        """
        self.validate_input(df, ['date', 'hs_code', 'value_usd'])
        
        group_by = kwargs.get('group_by', ['hs_code'])
        
        self.logger.info(f"Extracting growth features, grouping by {group_by}")
        
        # Aggregate by date and group
        agg_cols = list(set(group_by + ['date']))
        df_agg = df.groupby(agg_cols).agg({
            'value_usd': 'sum'
        }).reset_index()
        
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.sort_values(agg_cols)
        
        # Calculate features for each group
        for group_keys, group_indices in df_agg.groupby(group_by).groups.items():
            group_df = df_agg.loc[group_indices].copy()
            
            # MoM growth
            df_agg.loc[group_indices, 'growth_mom'] = (
                group_df['value_usd'].pct_change(periods=1) * 100
            )
            
            # QoQ growth (3 months)
            df_agg.loc[group_indices, 'growth_qoq'] = (
                group_df['value_usd'].pct_change(periods=3) * 100
            )
            
            # YoY growth (12 months)
            df_agg.loc[group_indices, 'growth_yoy'] = (
                group_df['value_usd'].pct_change(periods=12) * 100
            )
            
            # Average growth rates
            df_agg.loc[group_indices, 'growth_3m_avg'] = (
                group_df['value_usd'].pct_change(periods=1).rolling(3, min_periods=1).mean() * 100
            )
            df_agg.loc[group_indices, 'growth_6m_avg'] = (
                group_df['value_usd'].pct_change(periods=1).rolling(6, min_periods=1).mean() * 100
            )
            df_agg.loc[group_indices, 'growth_12m_avg'] = (
                group_df['value_usd'].pct_change(periods=1).rolling(12, min_periods=1).mean() * 100
            )
            
            # Growth acceleration (change in growth rate)
            growth_mom = group_df['value_usd'].pct_change(periods=1) * 100
            df_agg.loc[group_indices, 'growth_acceleration'] = growth_mom.diff()
            
            # Is growing (positive growth in last 3 months)
            df_agg.loc[group_indices, 'is_growing'] = (
                group_df['value_usd'].pct_change(periods=3) > 0
            ).astype(int)
            
            # CAGR (3 years = 36 months)
            df_agg.loc[group_indices, 'cagr_3y'] = self._calculate_cagr(
                group_df['value_usd'], periods=36
            )
            
            # Cumulative growth from start
            df_agg.loc[group_indices, 'cumulative_growth'] = (
                (group_df['value_usd'] / group_df['value_usd'].iloc[0] - 1) * 100
                if group_df['value_usd'].iloc[0] != 0 else 0
            )
        
        self.logger.info(f"Extracted growth features for {len(df_agg)} records")
        return df_agg
    
    def _calculate_cagr(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Calculate Compound Annual Growth Rate
        
        CAGR = ((End Value / Start Value) ^ (1 / years)) - 1
        
        Args:
            series: Time series data
            periods: Number of periods for CAGR calculation
            
        Returns:
            Series with CAGR values
        """
        result = []
        
        for i in range(len(series)):
            if i < periods:
                result.append(np.nan)
            else:
                start_val = series.iloc[i - periods]
                end_val = series.iloc[i]
                
                if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
                    result.append(np.nan)
                else:
                    years = periods / 12  # Convert months to years
                    cagr = (np.power(end_val / start_val, 1 / years) - 1) * 100
                    result.append(cagr)
        
        return pd.Series(result, index=series.index)
