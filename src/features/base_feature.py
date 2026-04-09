"""
Base Feature Extractor

Abstract base class for all feature extractors in SCRAM.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import logging


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors
    
    All feature extractors should inherit from this class and implement
    the extract() method.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature extractor
        
        Args:
            name: Name of the feature extractor
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"SCRAM.Features.{name}")
        self.logger.info(f"Initialized {name}")
    
    @abstractmethod
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract features from input DataFrame
        
        Args:
            df: Input DataFrame with trade data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with extracted features
        """
        pass
    
    def validate_input(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate input DataFrame has required columns
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces
        
        Returns:
            List of feature names
        """
        return self.config.get('feature_names', [])
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this feature extractor
        
        Returns:
            Dictionary with extractor information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'features': self.get_feature_names()
        }
