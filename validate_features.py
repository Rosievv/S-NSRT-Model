#!/usr/bin/env python3
"""验证特征数据质量"""

import pandas as pd
import numpy as np
import os

def validate_features():
    print("Feature Data Quality Report")
    print("=" * 70)
    
    # Load feature data
    train_features = pd.read_parquet('data/processed/features_train_full.parquet')
    test_features = pd.read_parquet('data/processed/features_test_full.parquet')
    
    print("\n1. Data Size")
    print(f"   Training set: {len(train_features):,} rows x {len(train_features.columns)} columns")
    print(f"   Test set: {len(test_features):,} rows x {len(test_features.columns)} columns")
    
    print("\n2. Time Range")
    print(f"   Training: {train_features.date.min()} to {train_features.date.max()}")
    print(f"   Test: {test_features.date.min()} to {test_features.date.max()}")
    
    print("\n3. Feature Columns:")
    for i, col in enumerate(train_features.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n4. Data Quality Check")
    print("   Training set:")
    print(f"      Missing values: {train_features.isnull().sum().sum():,}")
    numeric_train = train_features.select_dtypes(include=[np.number])
    print(f"      Infinite values: {np.isinf(numeric_train).sum().sum():,}")
    print(f"      Duplicate rows: {train_features.duplicated().sum():,}")
    
    print("   Test set:")
    print(f"      Missing values: {test_features.isnull().sum().sum():,}")
    numeric_test = test_features.select_dtypes(include=[np.number])
    print(f"      Infinite values: {np.isinf(numeric_test).sum().sum():,}")
    print(f"      Duplicate rows: {test_features.duplicated().sum():,}")
    
    print("\n5. File Size")
    train_size = os.path.getsize('data/processed/features_train_full.parquet')
    test_size = os.path.getsize('data/processed/features_test_full.parquet')
    print(f"   Training: {train_size/1024/1024:.2f} MB")
    print(f"   Test: {test_size/1024/1024:.2f} MB")
    
    print("\n6. Numeric Feature Statistics (first 10):")
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns[:10]
    print(train_features[numeric_cols].describe().round(2))
    
    print("\n" + "=" * 70)
    print("✓ Feature data quality validation complete!")

if __name__ == "__main__":
    validate_features()
