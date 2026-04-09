"""SCRAM — Supply Chain Risk Analysis Model"""

from .config_manager import config, ConfigManager
from .collectors import (
    BaseCollector,
    USCensusCollector,
    USGSCollector,
    MacroIndicatorCollector,
    FREDCollector,
)
from .utils import *

__version__ = '0.2.0'

__all__ = [
    'config',
    'ConfigManager',
    'BaseCollector',
    'USCensusCollector',
    'USGSCollector',
    'MacroIndicatorCollector',
    'FREDCollector',
]
