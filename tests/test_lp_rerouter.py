"""
Unit tests for LPReRouter (Module 2: Transportation Resilience)
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.transportation.rl_optimizer import LPReRouter


@pytest.fixture
def lane_summary():
    return pd.DataFrame({
        'region': ['East Asia', 'Europe', 'Americas'],
        'capacity': [1_000_000_000.0, 300_000_000.0, 100_000_000.0],
        'unit_cost': [1.0, 0.9, 0.5],
        'transit_days': [22, 16, 5],
        'reliability': [0.8, 0.9, 0.95],
    })


def test_optimise_respects_capacity_bounds(lane_summary):
    router = LPReRouter(lane_summary)
    demand = 500_000_000.0

    result = router.optimise(demand=demand)

    assert len(result) == len(lane_summary)
    # No lane can be allocated more than its own capacity
    assert (result['allocation'] <= result['capacity'] + 1e-6).all()
    # No negative allocations
    assert (result['allocation'] >= -1e-6).all()


def test_optimise_meets_total_demand(lane_summary):
    router = LPReRouter(lane_summary)
    demand = 500_000_000.0

    result = router.optimise(demand=demand)

    assert result['allocation'].sum() >= demand - 1.0  # allow tiny float slack


def test_optimise_allocation_pct_sums_to_100(lane_summary):
    router = LPReRouter(lane_summary)
    result = router.optimise(demand=200_000_000.0)

    assert result['allocation_pct'].sum() == pytest.approx(100.0, abs=1e-6)


def test_optimise_applies_region_disruption(lane_summary):
    router = LPReRouter(lane_summary)
    demand = 200_000_000.0

    normal = router.optimise(demand=demand)
    disrupted = router.optimise(
        demand=demand, disrupted_regions={'East Asia': 0.1}
    )

    normal_ea = normal.loc[normal['region'] == 'East Asia', 'capacity'].iloc[0]
    disrupted_ea = disrupted.loc[disrupted['region'] == 'East Asia', 'capacity'].iloc[0]

    assert disrupted_ea == pytest.approx(normal_ea * 0.1)
