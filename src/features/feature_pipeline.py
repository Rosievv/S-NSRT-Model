"""
Feature Pipeline

Orchestrates the feature extraction process by combining multiple feature extractors.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .base_feature import BaseFeatureExtractor
from .concentration_features import ConcentrationFeatureExtractor
from .volatility_features import VolatilityFeatureExtractor
from .temporal_features import TemporalFeatureExtractor
from .growth_features import GrowthFeatureExtractor


class FeaturePipeline:
    """
    Feature engineering pipeline that orchestrates multiple feature extractors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("SCRAM.FeaturePipeline")
        
        # Initialize feature extractors
        self.extractors = {
            'concentration': ConcentrationFeatureExtractor(),
            'volatility': VolatilityFeatureExtractor(),
            'temporal': TemporalFeatureExtractor(),
            'growth': GrowthFeatureExtractor()
        }
        
        self.logger.info("Initialized Feature Pipeline with 4 extractors")
    
    def extract_all(
        self, 
        df: pd.DataFrame, 
        extractors: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract all features from input data
        
        Args:
            df: Input DataFrame with trade data
            extractors: List of extractors to use (None = all)
            save_path: Path to save feature DataFrame
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("="*80)
        self.logger.info("Starting Feature Extraction Pipeline")
        self.logger.info("="*80)
        self.logger.info(f"Input data shape: {df.shape}")
        
        if extractors is None:
            extractors = list(self.extractors.keys())
        
        # Start with original data
        features_df = df.copy()
        
        # Extract concentration features (date + hs_code level)
        if 'concentration' in extractors:
            self.logger.info("\n--- Extracting Concentration Features ---")
            conc_features = self.extractors['concentration'].extract(df)
            features_df = self._merge_features(features_df, conc_features, ['date', 'hs_code'])
        
        # Extract volatility features (hs_code level, aggregated)
        if 'volatility' in extractors:
            self.logger.info("\n--- Extracting Volatility Features ---")
            vol_features = self.extractors['volatility'].extract(df)
            features_df = self._merge_features(features_df, vol_features, ['hs_code'], how='left')
        
        # Extract temporal features (date + hs_code level)
        if 'temporal' in extractors:
            self.logger.info("\n--- Extracting Temporal Features ---")
            temp_features = self.extractors['temporal'].extract(df)
            features_df = self._merge_features(features_df, temp_features, ['date', 'hs_code'])
        
        # Extract growth features (date + hs_code level)
        if 'growth' in extractors:
            self.logger.info("\n--- Extracting Growth Features ---")
            growth_features = self.extractors['growth'].extract(df)
            features_df = self._merge_features(features_df, growth_features, ['date', 'hs_code'])
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Feature Extraction Complete")
        self.logger.info(f"Final shape: {features_df.shape}")
        self.logger.info(f"Total features: {len(features_df.columns)}")
        self.logger.info("="*80)
        
        # Save if path provided
        if save_path:
            self._save_features(features_df, save_path)
        
        return features_df
    
    def extract_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only concentration features"""
        return self.extractors['concentration'].extract(df)
    
    def extract_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only volatility features"""
        return self.extractors['volatility'].extract(df)
    
    def extract_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only temporal features"""
        return self.extractors['temporal'].extract(df)
    
    def extract_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only growth features"""
        return self.extractors['growth'].extract(df)
    
    def _merge_features(
        self, 
        base_df: pd.DataFrame, 
        feature_df: pd.DataFrame, 
        merge_on: List[str],
        how: str = 'left'
    ) -> pd.DataFrame:
        """
        Merge feature DataFrame with base DataFrame
        
        Args:
            base_df: Base DataFrame
            feature_df: Features to merge
            merge_on: Columns to merge on
            how: Merge type ('left', 'inner', etc.)
            
        Returns:
            Merged DataFrame
        """
        # Remove duplicate columns from feature_df (except merge keys)
        feature_cols = [col for col in feature_df.columns if col not in base_df.columns or col in merge_on]
        feature_df_clean = feature_df[feature_cols]
        
        # Merge
        merged = base_df.merge(feature_df_clean, on=merge_on, how=how)
        
        self.logger.info(f"Merged {len(feature_cols) - len(merge_on)} new features")
        
        return merged
    
    def _save_features(self, df: pd.DataFrame, path: str):
        """
        Save features to file
        
        Args:
            df: Features DataFrame
            path: Output path
        """
        # Determine format from extension
        if path.endswith('.parquet'):
            df.to_parquet(path, compression='snappy')
            self.logger.info(f"Saved features to {path} (Parquet format)")
        elif path.endswith('.csv'):
            df.to_csv(path, index=False)
            self.logger.info(f"Saved features to {path} (CSV format)")
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of all available features
        
        Returns:
            Dictionary with feature information
        """
        summary = {
            'extractors': {},
            'total_features': 0
        }
        
        for name, extractor in self.extractors.items():
            info = extractor.get_info()
            summary['extractors'][name] = {
                'type': info['type'],
                'n_features': len(info['features']),
                'features': info['features']
            }
            summary['total_features'] += len(info['features'])
        
        return summary
