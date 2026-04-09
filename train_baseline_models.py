#!/usr/bin/env python3
"""
Baseline model training script for SCRAM
Trains simple time series forecasting models on the prepared features
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import TimeSeriesForecaster


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCRAM.Training")


def load_data():
    """Load training and test features"""
    logger.info("Loading feature data...")
    
    train_df = pd.read_parquet('data/processed/features_train_full.parquet')
    test_df = pd.read_parquet('data/processed/features_test_full.parquet')
    
    logger.info(f"Training data: {len(train_df):,} rows")
    logger.info(f"Test data: {len(test_df):,} rows")
    
    return train_df, test_df


def train_value_forecaster(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train a model to forecast trade values"""
    logger.info("\n" + "="*70)
    logger.info("TASK 1: Trade Value Forecasting")
    logger.info("="*70)
    
    # Initialize model
    model = TimeSeriesForecaster(
        target_variable='value_usd',
        forecast_horizon=1,  # 1 month ahead
        model_type='xgboost',
        config={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    )
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train_full, y_train_full = model.prepare_data(train_df)
    
    # Split into train/validation (80/20, keeping temporal order)
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    logger.info(f"Train set: {X_train.shape[0]:,} samples")
    logger.info(f"Validation set: {X_val.shape[0]:,} samples")
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation
    val_metrics = model.evaluate(X_val, y_val)
    
    # Prepare test data
    logger.info("\nPreparing test data...")
    X_test, y_test = model.prepare_data(test_df)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model_dir = Path('models/trained')
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / 'value_forecaster_xgb')
    
    return model, test_metrics


def train_hhi_forecaster(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train a model to forecast HHI (concentration)"""
    logger.info("\n" + "="*70)
    logger.info("TASK 2: HHI (Concentration) Forecasting")
    logger.info("="*70)
    
    # Initialize model
    model = TimeSeriesForecaster(
        target_variable='hhi',
        forecast_horizon=1,
        model_type='lightgbm',
        config={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    )
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train_full, y_train_full = model.prepare_data(train_df)
    
    # Split into train/validation (80/20, keeping temporal order)
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    logger.info(f"Train set: {X_train.shape[0]:,} samples")
    logger.info(f"Validation set: {X_val.shape[0]:,} samples")
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation
    val_metrics = model.evaluate(X_val, y_val)
    
    # Prepare test data
    logger.info("\nPreparing test data...")
    X_test, y_test = model.prepare_data(test_df)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model_dir = Path('models/trained')
    model.save_model(model_dir / 'hhi_forecaster_lgb')
    
    return model, test_metrics


def generate_summary(results: dict):
    """Generate training summary"""
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE:  {metrics['mae']:,.2f}")
        logger.info(f"  RMSE: {metrics['rmse']:,.2f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Baseline models trained successfully!")
    logger.info("  Models saved to: models/trained/")
    logger.info("="*70)


def main():
    """Main training pipeline"""
    logger.info("Starting baseline model training...")
    logger.info("="*70)
    
    # Load data
    train_df, test_df = load_data()
    
    # Train models
    results = {}
    
    # Model 1: Trade value forecasting
    try:
        value_model, value_metrics = train_value_forecaster(train_df, test_df)
        results['Trade Value Forecaster (XGBoost)'] = value_metrics
    except Exception as e:
        logger.error(f"Error training value forecaster: {e}")
        import traceback
        traceback.print_exc()
    
    # Model 2: HHI forecasting
    try:
        hhi_model, hhi_metrics = train_hhi_forecaster(train_df, test_df)
        results['HHI Forecaster (LightGBM)'] = hhi_metrics
    except Exception as e:
        logger.error(f"Error training HHI forecaster: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate summary
    if results:
        generate_summary(results)
    else:
        logger.error("No models were successfully trained!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
