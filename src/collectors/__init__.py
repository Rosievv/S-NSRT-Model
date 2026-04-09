"""Collectors package for SCRAM project"""

from .base_collector import BaseCollector
from .us_census_collector import USCensusCollector
from .usgs_collector import USGSCollector
from .macro_collector import MacroIndicatorCollector

__all__ = [
    'BaseCollector',
    'USCensusCollector',
    'USGSCollector',
    'MacroIndicatorCollector'
]
