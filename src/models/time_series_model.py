"""Time series forecasting models for supply chain risk prediction"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import logging

from .base_model import BaseModel


def mean_absolute_error(y_true, y_pred):
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """Calculate MSE"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class StandardScaler:
    """Simple standard scaler"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return (X - self.mean_) / self.std_
    
    def transform(self, X):
        """Transform"""
        if self.mean_ is None:
            raise ValueError("Scaler not fitted yet")
        return (X - self.mean_) / self.std_


class TimeSeriesForecaster(BaseModel):
    """Time series forecasting model using gradient boosting"""
    
    def __init__(
        self,
        target_variable: str = 'value_usd',
        forecast_horizon: int = 1,
        model_type: str = 'xgboost',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize time series forecaster
        
        Args:
            target_variable: Variable to forecast (value_usd, hhi, etc.)
            forecast_horizon: Number of periods ahead to forecast
            model_type: 'xgboost' or 'lightgbm'
            config: Model hyperparameters
        """
        super().__init__(
            model_name=f"TimeSeries_{target_variable}_{model_type}",
            model_type="time_series",
            config=config
        )
        
        self.target_variable = target_variable
        self.forecast_horizon = forecast_horizon
        self.model_algorithm = model_type
        self.scaler = StandardScaler()
        
        # Default configs
        if model_type == 'xgboost':
            self.default_config = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        else:  # lightgbm
            self.default_config = {
                'objective': 'regression',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
        
        # Merge with user config
        self.config = {**self.default_config, **self.config}
        
    def prepare_data(
        self, 
        df: pd.DataFrame,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for training
        
        Args:
            df: Input dataframe with features
            scale_features: Whether to scale features
            
        Returns:
            X, y: Features and targets
        """
        # Sort by date
        df = df.sort_values('date').copy()
        
        # Get target variable
        y = df[self.target_variable].values
        
        # Get feature columns
        feature_cols = self.get_feature_names(df)
        
        # Remove target from features if present
        if self.target_variable in feature_cols:
            feature_cols.remove(self.target_variable)
        
        # Only keep numeric columns
        X_df = df[feature_cols].select_dtypes(include=[np.number])
        X = X_df.values.astype(np.float64)
        
        # Scale features
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        # Create forecast targets (shift by horizon)
        y_forecast = np.roll(y, -self.forecast_horizon)
        
        # Remove last N samples (no future target)
        X = X[:-self.forecast_horizon]
        y_forecast = y_forecast[:-self.forecast_horizon]
        
        self.logger.info(f"Prepared data: X shape {X.shape}, y shape {y_forecast.shape}")
        
        return X, y_forecast
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the time series model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        self.logger.info(f"Training {self.model_algorithm} model...")
        
        # Initialize model
        if self.model_algorithm == 'xgboost':
            self.model = xgb.XGBRegressor(**self.config)
            
            # Train with validation set if provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
                
        else:  # lightgbm
            self.model = lgb.LGBMRegressor(**self.config)
            
            # Train with validation set if provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            else:
                self.model.fit(X_train, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = {
                f'feature_{i}': imp 
                for i, imp in enumerate(self.model.feature_importances_)
            }
        
        self.logger.info("Training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        self.log_metrics(metrics, "test")
        
        return metrics
    
    def forecast_future(
        self,
        df_recent: pd.DataFrame,
        n_periods: int = 6
    ) -> pd.DataFrame:
        """
        Forecast future values
        
        Args:
            df_recent: Recent historical data
            n_periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        forecasts = []
        current_data = df_recent.copy()
        
        for i in range(n_periods):
            # Prepare features
            X, _ = self.prepare_data(current_data, scale_features=True)
            X_latest = X[-1:] if len(X) > 0 else X
            
            # Make prediction
            pred = self.predict(X_latest)[0]
            forecasts.append(pred)
            
            # Update data for next iteration (simplified)
            # In practice, you'd update all time-dependent features
            
        forecast_df = pd.DataFrame({
            'period': range(1, n_periods + 1),
            'forecast': forecasts
        })
        
        return forecast_df
