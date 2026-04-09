"""
Utility functions for SCRAM project
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import hashlib


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_date_range(
    start_date: str,
    end_date: str,
    freq: str = 'MS'
) -> List[str]:
    """
    Generate list of dates between start and end
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        freq: Frequency ('MS' for month start, 'D' for daily)
        
    Returns:
        List of date strings
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in date_range]


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    date_column: Optional[str] = None,
    max_missing_pct: float = 0.2
) -> Dict[str, Any]:
    """
    Validate DataFrame quality
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        date_column: Name of date column for continuity check
        max_missing_pct: Maximum allowed percentage of missing values
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check data completeness
    missing_pct = df.isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > max_missing_pct]
    if not high_missing.empty:
        results['warnings'].append(
            f"Columns with >20% missing: {high_missing.to_dict()}"
        )
    
    # Check date continuity
    if date_column and date_column in df.columns:
        df_sorted = df.sort_values(date_column)
        dates = pd.to_datetime(df_sorted[date_column])
        expected_freq = pd.infer_freq(dates)
        if expected_freq is None:
            results['warnings'].append("Date column has irregular frequency")
    
    # Calculate statistics
    results['stats'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return results


def detect_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 5.0
) -> pd.Series:
    """
    Detect outliers in a numeric series
    
    Args:
        series: Pandas Series to check
        method: 'zscore' or 'iqr'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = 'parquet',
    **kwargs
) -> None:
    """
    Save DataFrame to file with specified format
    
    Args:
        df: DataFrame to save
        filepath: Destination file path
        format: File format ('parquet', 'csv', 'feather')
        **kwargs: Additional arguments for save function
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(filepath, **kwargs)
    elif format == 'csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif format == 'feather':
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load DataFrame from file
    
    Args:
        filepath: Source file path
        format: File format (auto-detected if None)
        **kwargs: Additional arguments for load function
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lstrip('.')
    
    if format == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif format == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif format == 'feather':
        return pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def calculate_hash(data: Union[str, bytes, pd.DataFrame]) -> str:
    """
    Calculate MD5 hash for data deduplication
    
    Args:
        data: String, bytes, or DataFrame to hash
        
    Returns:
        MD5 hash string
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_json(orient='records').encode()
    elif isinstance(data, str):
        data = data.encode()
    
    return hashlib.md5(data).hexdigest()


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplication factor for delay
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    import time
                    time.sleep(delay)
                    delay *= backoff_factor
        
        raise last_exception
    
    return wrapper


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def standardize_country_name(country: str) -> str:
    """
    Standardize country names for consistency
    
    Args:
        country: Country name string
        
    Returns:
        Standardized country name
    """
    # Common mappings
    mappings = {
        'china, peoples republic of': 'China',
        'korea, republic of': 'South Korea',
        'korea, democratic peoples republic of': 'North Korea',
        'taiwan': 'Taiwan',
        'hong kong': 'Hong Kong',
        'united kingdom': 'United Kingdom',
        'united states': 'United States'
    }
    
    country_lower = country.lower().strip()
    return mappings.get(country_lower, country.title())


def calculate_hhi(shares: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index for concentration
    
    Args:
        shares: Series of market shares (as percentages, 0-100)
        
    Returns:
        HHI value (0-10000)
    """
    return (shares ** 2).sum()


def calculate_cov(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Calculate rolling Coefficient of Variation
    
    Args:
        series: Time series data
        window: Rolling window size
        
    Returns:
        Series of CoV values
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return rolling_std / rolling_mean
