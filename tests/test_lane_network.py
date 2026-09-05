"""
Unit tests for LogisticsNetwork (Module 2: Transportation Resilience)
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.transportation.lane_network import LogisticsNetwork


@pytest.fixture
def trade_df():
    dates = pd.date_range('2022-01-01', periods=6, freq='MS')
    rows = []
    for d in dates:
        rows.append({'date': d, 'country': 'China', 'value_usd': 1_000_000_000})
        rows.append({'date': d, 'country': 'Germany', 'value_usd': 200_000_000})
        rows.append({'date': d, 'country': 'Mexico', 'value_usd': 50_000_000})
    return pd.DataFrame(rows)


def test_build_creates_region_nodes_and_usa_sink(trade_df):
    net = LogisticsNetwork(trade_df)
    G = net.build()

    assert 'USA' in G.nodes
    assert 'East Asia' in G.nodes  # China
    assert 'Europe' in G.nodes     # Germany
    assert 'Americas' in G.nodes   # Mexico
    # Every supply region has a directed edge into USA
    for region in ('East Asia', 'Europe', 'Americas'):
        assert G.has_edge(region, 'USA')


def test_get_lane_summary_has_expected_columns(trade_df):
    net = LogisticsNetwork(trade_df)
    G = net.build()
    summary = net.get_lane_summary(G)

    assert not summary.empty
    for col in ('region', 'destination', 'capacity', 'unit_cost', 'transit_days', 'reliability'):
        assert col in summary.columns
    assert (summary['destination'] == 'USA').all()


def test_get_lane_summary_builds_graph_if_none_passed(trade_df):
    """get_lane_summary() should lazily build the graph when called with no arg."""
    net = LogisticsNetwork(trade_df)
    summary = net.get_lane_summary()

    assert not summary.empty


def test_get_vulnerable_lanes_filters_by_reliability_threshold(trade_df):
    net = LogisticsNetwork(trade_df)
    G = net.build()
    summary = net.get_lane_summary(G)

    vulnerable = net.get_vulnerable_lanes(G, reliability_threshold=1.1)
    # With threshold above the max possible reliability (1.0), every lane qualifies
    assert len(vulnerable) == len(summary)

    none_vulnerable = net.get_vulnerable_lanes(G, reliability_threshold=-0.1)
    assert none_vulnerable.empty
