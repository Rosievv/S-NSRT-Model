"""SCRAM source package"""

from .config_manager import config, ConfigManager
from .collectors import (
    BaseCollector,
    USCensusCollector,
    USGSCollector,
    MacroIndicatorCollector
)
from .utils import *

__version__ = '0.1.0'

__all__ = [
    'config',
    'ConfigManager',
    'BaseCollector',
    'USCensusCollector',
    'USGSCollector',
    'MacroIndicatorCollector'
]
