"""Base model class for SCRAM models"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import json
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all SCRAM models"""
    
    def __init__(
        self, 
        model_name: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model
        
        Args:
            model_name: Name of the model
            model_type: Type of model (time_series, classifier, anomaly_detector)
            config: Model configuration parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.metrics = {}
        self.feature_importance = {}
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f"SCRAM.{model_name}")
        
    @abstractmethod
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction
        
        Args:
            df: Input dataframe with features
            
        Returns:
            X, y: Features and targets
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def save_model(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix('.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_type(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            return obj
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'config': convert_to_python_type(self.config),
            'metrics': convert_to_python_type(self.metrics),
            'feature_importance': convert_to_python_type(self.feature_importance),
            'training_history': convert_to_python_type(self.training_history),
            'saved_at': datetime.now().isoformat()
        }
        
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        
    def load_model(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        # Load model
        model_path = path.with_suffix('.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.config = metadata.get('config', {})
                self.metrics = metadata.get('metrics', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.training_history = metadata.get('training_history', [])
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """
        Get feature names for modeling
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        # Exclude metadata columns
        exclude_cols = [
            'date', 'hs_code', 'country', 'country_code', 
            'data_source', 'collected_at', 'trade_type'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def log_metrics(self, metrics: Dict[str, float], phase: str = "evaluation"):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            phase: Training phase (train/val/test)
        """
        self.logger.info(f"\n{phase.upper()} METRICS:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")
        
        # Store metrics
        self.metrics[phase] = metrics
