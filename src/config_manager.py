"""
Configuration Manager for SCRAM Project
Handles loading and accessing configuration from YAML and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Singleton configuration manager"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables"""
        # Get project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
        
        # Load environment variables
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Load YAML configuration
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Inject environment variables
        self._inject_env_vars()
    
    def _inject_env_vars(self) -> None:
        """Inject sensitive data from environment variables"""
        if self._config is None:
            return
        
        # Add US Census API key
        census_key = os.getenv('US_CENSUS_API_KEY')
        if census_key:
            self._config['data_sources']['us_census']['api_key'] = census_key
        
        # Add FRED API key
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            if 'fred' not in self._config.get('data_sources', {}):
                self._config['data_sources']['fred'] = {}
            self._config['data_sources']['fred']['api_key'] = fred_key
        
        # Add proxy settings if present
        http_proxy = os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('HTTPS_PROXY')
        if http_proxy or https_proxy:
            self._config['proxy'] = {
                'http': http_proxy,
                'https': https_proxy
            }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config key (e.g., 'data_sources.us_census.base_url')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            return default
        
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_hs_codes(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get HS codes configuration
        
        Args:
            category: Optional category filter (e.g., 'integrated_circuits')
            
        Returns:
            Dictionary of HS codes
        """
        hs_codes = self.get('hs_codes', {})
        if category:
            return hs_codes.get(category, {})
        return hs_codes
    
    def get_time_range(self, phase: str = 'train') -> Dict[str, str]:
        """
        Get time range for data collection
        
        Args:
            phase: 'train' or 'test'
            
        Returns:
            Dictionary with start and end dates
        """
        time_range = self.get('time_range', {})
        if phase == 'train':
            return {
                'start': time_range.get('train_start'),
                'end': time_range.get('train_end')
            }
        elif phase == 'test':
            return {
                'start': time_range.get('test_start'),
                'end': time_range.get('test_end')
            }
        else:
            return time_range
    
    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """
        Get configuration for a specific data source
        
        Args:
            source: Data source name (e.g., 'us_census', 'usgs')
            
        Returns:
            Data source configuration dictionary
        """
        return self.get(f'data_sources.{source}', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.get('storage', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get data validation rules"""
        return self.get('validation', {})
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolve relative path to absolute path from project root
        
        Args:
            relative_path: Relative path string
            
        Returns:
            Absolute Path object
        """
        return self.project_root / relative_path


# Global config instance
config = ConfigManager()
