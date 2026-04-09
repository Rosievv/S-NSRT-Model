"""
Feature Engineering Example

This script demonstrates how to use the feature engineering module
with the training data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import logging
from features import FeaturePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run feature engineering example"""
    
    print("\n" + "="*80)
    print("SCRAM Feature Engineering Example")
    print("="*80)
    
    # Load training data
    data_path = 'data/raw/us_census_20260215_201556.parquet'
    print(f"\nLoading data from: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize feature pipeline
    print("\n" + "-"*80)
    print("Initializing Feature Pipeline...")
    print("-"*80)
    
    pipeline = FeaturePipeline()
    
    # Get feature summary
    summary = pipeline.get_feature_summary()
    print(f"\nAvailable Features:")
    for extractor_name, info in summary['extractors'].items():
        print(f"\n{extractor_name.upper()}:")
        print(f"  Type: {info['type']}")
        print(f"  Features: {info['n_features']}")
        print(f"  Names: {', '.join(info['features'][:5])}...")
    
    print(f"\nTotal features available: {summary['total_features']}")
    
    # Extract features (using a sample for demo)
    print("\n" + "-"*80)
    print("Extracting Features (using sample data)...")
    print("-"*80)
    
    # Use first 6 months of data for demo
    df_sample = df[df['date'] < '2010-07-01'].copy()
    print(f"\nSample size: {len(df_sample):,} records")
    
    # Extract all features
    features_df = pipeline.extract_all(
        df_sample,
        save_path='data/processed/features_sample.parquet'
    )
    
    # Display results
    print("\n" + "="*80)
    print("Feature Extraction Results")
    print("="*80)
    
    print(f"\nOutput shape: {features_df.shape}")
    print(f"Number of features: {len(features_df.columns)}")
    
    print("\nSample of extracted features:")
    print(features_df[['date', 'hs_code', 'hhi', 'top1_share', 'value_cov', 'growth_mom']].head(10))
    
    print("\nFeature statistics:")
    print(features_df[['hhi', 'value_cov', 'growth_mom', 'value_std']].describe())
    
    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)
    print(f"\nFeatures saved to: data/processed/features_sample.parquet")
    print("\nNext steps:")
    print("1. Extract features from full training dataset")
    print("2. Analyze feature distributions and correlations")
    print("3. Use features for model training")

if __name__ == '__main__':
    main()
