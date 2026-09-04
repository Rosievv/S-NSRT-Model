"""
Unit tests for AdaptiveCalibrator (Module 3: Demand & Shortage Forecasting)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.demand_forecasting.adaptive_calibration import AdaptiveCalibrator


@pytest.fixture
def well_calibrated_data():
    """q50 predictions equal to actuals -> coverage should be ~0.5 at q50."""
    n = 200
    actuals = pd.Series(np.arange(n, dtype=float))
    predictions = pd.DataFrame({
        'q10': actuals - 10,
        'q50': actuals,
        'q90': actuals + 10,
    })
    return actuals, predictions


def test_compute_calibration_error_schema(well_calibrated_data):
    actuals, predictions = well_calibrated_data
    calibrator = AdaptiveCalibrator(quantiles=[0.10, 0.50, 0.90])

    result = calibrator.compute_calibration_error(actuals, predictions)

    assert set(result['quantile']) == {0.10, 0.50, 0.90}
    for col in ('target_coverage', 'observed_coverage', 'calibration_error'):
        assert col in result.columns


def test_compute_calibration_error_well_calibrated_q50(well_calibrated_data):
    actuals, predictions = well_calibrated_data
    calibrator = AdaptiveCalibrator(quantiles=[0.50])

    result = calibrator.compute_calibration_error(actuals, predictions)

    # actuals <= q50 (== actuals) is always true -> observed coverage should be 1.0
    # since q50 == actuals exactly (using <=), this demonstrates the metric definition
    assert result.iloc[0]['observed_coverage'] == pytest.approx(1.0)


def test_compute_calibration_error_appends_to_history(well_calibrated_data):
    actuals, predictions = well_calibrated_data
    calibrator = AdaptiveCalibrator(quantiles=[0.50])

    assert calibrator.calibration_history == []
    calibrator.compute_calibration_error(actuals, predictions)
    assert len(calibrator.calibration_history) == 1
    assert 'avg_abs_error' in calibrator.calibration_history[0]


def test_recalibrate_predictions_enforces_monotonicity():
    calibrator = AdaptiveCalibrator(quantiles=[0.10, 0.50, 0.90])
    predictions = pd.DataFrame({
        'q10': [10.0, 20.0],
        'q50': [15.0, 25.0],
        'q90': [30.0, 40.0],
    })
    # Force a large under-coverage correction on q10 to test monotonicity re-enforcement
    calibration_df = pd.DataFrame({
        'quantile': [0.10, 0.50, 0.90],
        'target_coverage': [0.10, 0.50, 0.90],
        'observed_coverage': [0.02, 0.50, 0.90],
        'calibration_error': [-0.08, 0.0, 0.0],
    })

    adjusted = calibrator.recalibrate_predictions(predictions, calibration_df)

    # q10 <= q50 <= q90 must still hold after adjustment
    assert (adjusted['q10'] <= adjusted['q50']).all()
    assert (adjusted['q50'] <= adjusted['q90']).all()


def test_recalibrate_predictions_skips_well_calibrated_quantiles():
    calibrator = AdaptiveCalibrator(quantiles=[0.50])
    predictions = pd.DataFrame({'q50': [15.0, 25.0]})
    calibration_df = pd.DataFrame({
        'quantile': [0.50],
        'target_coverage': [0.50],
        'observed_coverage': [0.505],
        'calibration_error': [0.005],  # below the 0.02 threshold -> no change
    })

    adjusted = calibrator.recalibrate_predictions(predictions, calibration_df)

    pd.testing.assert_series_equal(adjusted['q50'], predictions['q50'])
