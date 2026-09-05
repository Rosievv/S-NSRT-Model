"""
Unit tests for CostAlertSystem (Module 4: Cost-Push Monitoring)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.cost_monitoring.alert_system import CostAlertSystem


@pytest.fixture
def flat_indicator_df():
    """12 months of a perfectly flat indicator -> zero std -> z-score 0 -> green."""
    idx = pd.date_range('2023-01-01', periods=12, freq='MS')
    return pd.DataFrame({'ppi_semiconductor': [100.0] * 12}, index=idx)


@pytest.fixture
def spiking_indicator_df():
    """11 months at baseline, then a final month with a huge spike -> red."""
    idx = pd.date_range('2023-01-01', periods=12, freq='MS')
    values = [100.0] * 11 + [1000.0]
    return pd.DataFrame({'ppi_semiconductor': values}, index=idx)


def test_compute_thresholds_schema(flat_indicator_df):
    system = CostAlertSystem(rolling_window=12)
    thresholds = system.compute_thresholds(flat_indicator_df)

    assert not thresholds.empty
    for col in ('indicator', 'latest_value', 'rolling_mean', 'rolling_std', 'z_score',
                'threshold_yellow', 'threshold_orange', 'threshold_red'):
        assert col in thresholds.columns


def test_check_alerts_flat_series_is_green(flat_indicator_df):
    system = CostAlertSystem(rolling_window=12)
    alerts = system.check_alerts(flat_indicator_df)

    assert len(alerts) == 1
    assert alerts[0].level == 'green'


def test_check_alerts_spike_triggers_red(spiking_indicator_df):
    system = CostAlertSystem(rolling_window=12)
    alerts = system.check_alerts(spiking_indicator_df)

    assert len(alerts) == 1
    assert alerts[0].level == 'red'
    assert alerts[0].z_score > 2.0


def test_generate_alert_report_structure(spiking_indicator_df):
    system = CostAlertSystem(rolling_window=12)
    report = system.generate_alert_report(indicator_df=spiking_indicator_df)

    assert report['n_indicators'] == 1
    assert report['n_elevated'] == 1
    assert 'ppi_semiconductor' in report['summary']['red']
    assert len(report['details']) == 1


def test_generate_alert_report_requires_alerts_or_indicator_df():
    system = CostAlertSystem()
    with pytest.raises(ValueError):
        system.generate_alert_report()
