"""
Data Integration Module for SCRAM

Provides a mixed-data logic layer that manages heterogeneous data
sources (internal / public / proxy), tracks provenance, and degrades
gracefully when preferred sources are unavailable.
"""

from .data_registry import DataRegistry, DataSourceMeta
from .mixed_data_loader import MixedDataLoader
from .data_quality import DataQualityChecker

__all__ = [
    "DataRegistry",
    "DataSourceMeta",
    "MixedDataLoader",
    "DataQualityChecker",
]
