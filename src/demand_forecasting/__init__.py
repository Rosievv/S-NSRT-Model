"""
Demand Forecasting Module for SCRAM

Provides quantile-based supply forecasting, shortage-risk detection,
and adaptive calibration for prediction intervals.
"""

from .quantile_forecaster import QuantileForecaster
from .shortage_detector import ShortageDetector
from .adaptive_calibration import AdaptiveCalibrator

__all__ = [
    "QuantileForecaster",
    "ShortageDetector",
    "AdaptiveCalibrator",
]
