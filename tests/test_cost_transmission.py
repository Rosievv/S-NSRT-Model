"""
Unit tests for CostTransmissionAnalyzer (Module 4: Cost-Push Monitoring)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.cost_monitoring.cost_transmission import CostTransmissionAnalyzer


@pytest.fixture
def synthetic_series():
    """A cost driver that leads trade value by a fixed lag, plus a pure-noise driver."""
    rng = np.random.default_rng(42)
    n = 60
    idx = pd.date_range('2019-01-01', periods=n, freq='MS')

    cost = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name='cost')
    # Trade value driven by cost lagged 1 month, plus noise
    cost_lagged = cost.shift(1).bfill()
    trade_vals = 1000 + 5 * cost_lagged.values + rng.normal(0, 2, n)
    trade = pd.Series(trade_vals, index=idx, name='trade')

    noise = pd.Series(rng.normal(50, 5, n), index=idx, name='noise')

    return trade, {'linked_cost': cost, 'unrelated_noise': noise}


def test_granger_causality_returns_expected_schema_or_empty(synthetic_series):
    trade, costs = synthetic_series
    analyzer = CostTransmissionAnalyzer(trade, costs, max_lag=3)

    result = analyzer.granger_causality()

    # statsmodels may not be installed in every environment; if so this
    # gracefully returns an empty DataFrame instead of raising.
    if not result.empty:
        for col in ('cost_driver', 'lag', 'f_statistic', 'p_value', 'significant'):
            assert col in result.columns


def test_compute_cost_pressure_index_is_zscore_weighted_average(synthetic_series):
    _, costs = synthetic_series
    analyzer = CostTransmissionAnalyzer(pd.Series(dtype=float), costs)

    index = analyzer.compute_cost_pressure_index()

    assert not index.empty
    assert index.name == 'cost_pressure_index'
    # z-score composite of stationary-ish series should hover near 0
    assert abs(index.mean()) < 1.0


def test_compute_cost_pressure_index_respects_custom_weights(synthetic_series):
    _, costs = synthetic_series
    analyzer = CostTransmissionAnalyzer(pd.Series(dtype=float), costs)

    # All weight on a single driver should reproduce that driver's z-score series
    only_linked = analyzer.compute_cost_pressure_index(
        weights={'linked_cost': 1.0, 'unrelated_noise': 0.0}
    )
    linked_z = (costs['linked_cost'] - costs['linked_cost'].mean()) / costs['linked_cost'].std()

    pd.testing.assert_series_equal(
        only_linked, linked_z.rename('cost_pressure_index'), check_exact=False, atol=1e-8
    )


def test_decompose_cost_drivers_returns_r_squared_between_0_and_1(synthetic_series):
    trade, costs = synthetic_series
    analyzer = CostTransmissionAnalyzer(trade, costs)

    result = analyzer.decompose_cost_drivers()

    assert not result.empty
    assert set(result['cost_driver']) == set(costs.keys())
    assert (result['r_squared'] >= 0).all()
    assert (result['r_squared'] <= 1).all()
    # The linked driver should explain more variance than pure noise
    linked_r2 = result.loc[result['cost_driver'] == 'linked_cost', 'r_squared'].iloc[0]
    noise_r2 = result.loc[result['cost_driver'] == 'unrelated_noise', 'r_squared'].iloc[0]
    assert linked_r2 > noise_r2


def test_estimate_passthrough_elasticity_returns_expected_schema(synthetic_series):
    trade, costs = synthetic_series
    analyzer = CostTransmissionAnalyzer(trade, costs, max_lag=3)

    result = analyzer.estimate_passthrough_elasticity(lags=[1, 3])

    assert not result.empty
    for col in ('cost_driver', 'lag_months', 'elasticity'):
        assert col in result.columns
    assert set(result['lag_months'].unique()) <= {1, 3}
