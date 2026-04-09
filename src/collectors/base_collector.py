"""
Base Collector class for all data collectors
Provides common interface and functionality for data collection
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from ..config_manager import config
from ..utils import setup_logger, save_dataframe, validate_dataframe


class BaseCollector(ABC):
    """Abstract base class for all data collectors"""
    
    def __init__(
        self,
        collector_name: str,
        data_source: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize base collector
        
        Args:
            collector_name: Name of the collector (for logging)
            data_source: Data source key in config
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.collector_name = collector_name
        self.data_source = data_source
        
        # Setup logging
        log_config = config.get_logging_config()
        log_file = config.resolve_path(
            f"logs/{collector_name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        self.logger = setup_logger(
            name=collector_name,
            log_file=str(log_file),
            level=log_config.get('level', 'INFO')
        )
        
        # Date range
        self.start_date = start_date
        self.end_date = end_date
        
        # Storage configuration
        self.storage_config = config.get_storage_config()
        self.raw_data_path = config.resolve_path(
            self.storage_config.get('raw_data_path', 'data/raw')
        )
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Validation rules
        self.validation_rules = config.get_validation_rules()
        
        self.logger.info(f"Initialized {collector_name}")
    
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch data from source
        
        Returns:
            DataFrame with fetched data
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate fetched data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        pass
    
    def save(
        self,
        df: pd.DataFrame,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save DataFrame to storage
        
        Args:
            df: DataFrame to save
            filename: Filename without extension
            subdirectory: Optional subdirectory under raw_data_path
            
        Returns:
            Path to saved file
        """
        # Determine save path
        if subdirectory:
            save_path = self.raw_data_path / subdirectory
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.raw_data_path
        
        # Add file extension based on format
        file_format = self.storage_config.get('file_format', 'parquet')
        filepath = save_path / f"{filename}.{file_format}"
        
        # Save with compression
        compression = self.storage_config.get('compression', 'snappy')
        if file_format == 'parquet':
            save_dataframe(df, filepath, format=file_format, compression=compression)
        else:
            save_dataframe(df, filepath, format=file_format)
        
        self.logger.info(f"Saved data to {filepath}")
        return filepath
    
    def collect(
        self,
        save_data: bool = True,
        validate_data: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Main collection workflow: fetch -> validate -> save
        
        Args:
            save_data: Whether to save fetched data
            validate_data: Whether to validate data before saving
            **kwargs: Additional arguments passed to fetch()
            
        Returns:
            Collected DataFrame
        """
        self.logger.info(f"Starting data collection for {self.collector_name}")
        
        try:
            # Fetch data
            self.logger.info("Fetching data...")
            df = self.fetch(**kwargs)
            self.logger.info(f"Fetched {len(df)} records")
            
            # Validate if requested
            if validate_data:
                self.logger.info("Validating data...")
                is_valid = self.validate(df)
                if not is_valid:
                    self.logger.warning("Data validation failed, but continuing...")
            
            # Save if requested
            if save_data:
                filename = self._generate_filename()
                self.save(df, filename)
            
            self.logger.info("Data collection completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}", exc_info=True)
            raise
    
    def _generate_filename(self) -> str:
        """
        Generate filename for saved data
        
        Returns:
            Filename string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.data_source}_{timestamp}"
    
    def get_date_range(self) -> Dict[str, str]:
        """
        Get date range for collection
        
        Returns:
            Dictionary with start and end dates
        """
        if self.start_date and self.end_date:
            return {'start': self.start_date, 'end': self.end_date}
        
        # Default to training period
        return config.get_time_range('train')
    
    def log_stats(self, df: pd.DataFrame, description: str = "Dataset") -> None:
        """
        Log basic statistics about DataFrame
        
        Args:
            df: DataFrame to analyze
            description: Description for logging
        """
        self.logger.info(f"{description} statistics:")
        self.logger.info(f"  - Shape: {df.shape}")
        self.logger.info(f"  - Columns: {list(df.columns)}")
        self.logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}" 
                        if 'date' in df.columns else "  - No date column")
        self.logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
        self.logger.info(f"  - Duplicate rows: {df.duplicated().sum()}")
