"""
Demand Forecasting Module for SCRAM

Provides quantile-based supply forecasting, shortage-risk detection,
and adaptive calibration for prediction intervals.
"""

from .quantile_forecaster import QuantileForecaster
from .shortage_detector import ShortageDetector
from .adaptive_calibration import AdaptiveCalibrator
from .risk_integration import (
    add_shortage_metrics,
    apply_gap_elasticity,
    apply_risk_gating,
    adjust_supply_quantiles,
    attach_module1_risk,
    build_inventory_scenarios,
)

__all__ = [
    "QuantileForecaster",
    "ShortageDetector",
    "AdaptiveCalibrator",
    "add_shortage_metrics",
    "apply_gap_elasticity",
    "apply_risk_gating",
    "adjust_supply_quantiles",
    "attach_module1_risk",
    "build_inventory_scenarios",
]
