"""
Logistics Lane Network

Models the bipartite transport graph:  supplier-regions → shipping lanes → USA.
Each lane carries attributes (capacity, unit cost, reliability) derived from
real freight indices and historical trade-flow variability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx>=3.1 required")

logger = logging.getLogger("SCRAM.Transportation.LaneNetwork")


# Pre-defined major shipping regions and associated proxy transit attributes
# (transit_days is a very rough approximation for US-bound ocean freight)
REGION_DEFAULTS: Dict[str, Dict] = {
    "East Asia": {
        "countries": [
            "China", "Taiwan", "Japan", "Korea, South",
            "Malaysia", "Vietnam", "Thailand", "Singapore",
            "Philippines", "Indonesia",
        ],
        "transit_days": 22,
        "base_cost_factor": 1.0,
    },
    "Europe": {
        "countries": [
            "Germany", "Netherlands", "United Kingdom",
            "France", "Ireland", "Italy", "Belgium",
        ],
        "transit_days": 16,
        "base_cost_factor": 0.9,
    },
    "Americas": {
        "countries": [
            "Mexico", "Canada", "Costa Rica", "Brazil",
        ],
        "transit_days": 5,
        "base_cost_factor": 0.5,
    },
    "South Asia": {
        "countries": ["India", "Bangladesh"],
        "transit_days": 28,
        "base_cost_factor": 1.1,
    },
    "Middle East": {
        "countries": ["Israel", "United Arab Emirates"],
        "transit_days": 25,
        "base_cost_factor": 1.05,
    },
}


def _country_to_region(country: str) -> str:
    """Map a country name to a shipping region."""
    for region, meta in REGION_DEFAULTS.items():
        if country in meta["countries"]:
            return region
    return "Other"


class LogisticsNetwork:
    """
    Construct and query a logistics-lane network from trade data.

    Nodes
    -----
    - **region** nodes (East Asia, Europe, …)
    - **USA** demand-sink node

    Edges
    -----
    Directed: region → USA, carrying:
    ``capacity`` (total value_usd from that region),
    ``unit_cost``, ``transit_days``, ``reliability_score``.
    """

    def __init__(self, trade_df: pd.DataFrame, freight_df: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        trade_df : pd.DataFrame
            US Census trade data (date, country, value_usd, …).
        freight_df : pd.DataFrame, optional
            FRED / external freight-index data (date, series_id, value).
            If supplied, lane costs are calibrated to freight-index levels.
        """
        self.trade_df = trade_df.copy()
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        self.trade_df["region"] = self.trade_df["country"].apply(_country_to_region)
        self.freight_df = freight_df
        self._graph: Optional[nx.DiGraph] = None

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def build(self, period: str = "all") -> nx.DiGraph:
        """
        Build the logistics graph for a given period.

        Parameters
        ----------
        period : str
            ``"all"`` or ``"YYYY-MM"``.
        """
        df = self.trade_df
        if period != "all":
            df = df[df["date"].dt.to_period("M").astype(str) == period]

        region_agg = (
            df.groupby("region")
            .agg(
                capacity=("value_usd", "sum"),
                n_countries=("country", "nunique"),
                value_std=("value_usd", "std"),
            )
            .reset_index()
        )
        total = region_agg["capacity"].sum()

        G = nx.DiGraph()
        G.add_node("USA", role="demand_sink")

        for _, row in region_agg.iterrows():
            region = row["region"]
            defaults = REGION_DEFAULTS.get(region, {"transit_days": 30, "base_cost_factor": 1.2})

            # Reliability: inverse of coefficient of variation (low variability = high reliability)
            cov = (row["value_std"] / row["capacity"]) if row["capacity"] > 0 else 1.0
            reliability = max(0.0, min(1.0, 1.0 - cov))

            # Cost: base factor × freight-index ratio if available
            cost_factor = defaults.get("base_cost_factor", 1.0)
            if self.freight_df is not None and not self.freight_df.empty:
                cost_factor *= self._latest_freight_ratio()

            G.add_node(
                region,
                role="supply_region",
                capacity=float(row["capacity"]),
                share=float(row["capacity"] / total) if total > 0 else 0,
                n_countries=int(row["n_countries"]),
            )
            G.add_edge(
                region,
                "USA",
                capacity=float(row["capacity"]),
                unit_cost=float(cost_factor),
                transit_days=int(defaults.get("transit_days", 30)),
                reliability=round(float(reliability), 3),
            )

        G.graph["period"] = period
        G.graph["total_capacity"] = float(total)
        self._graph = G
        logger.info(
            "Logistics network (%s): %d regions, $%.1fB total capacity",
            period, G.number_of_nodes() - 1, total / 1e9,
        )
        return G

    # ------------------------------------------------------------------ #
    #  Freight-index helpers
    # ------------------------------------------------------------------ #

    def _latest_freight_ratio(self) -> float:
        """Scale factor from freight index relative to historical median."""
        if self.freight_df is None or self.freight_df.empty:
            return 1.0
        freight = self.freight_df
        if "series_id" in freight.columns:
            # Use GSCPI as default freight signal
            gscpi = freight[freight["series_id"] == "GSCPI"]
            if gscpi.empty:
                gscpi = freight
            series = gscpi["value"]
        else:
            series = freight.iloc[:, -1]
        median = series.median()
        latest = series.iloc[-1]
        if median == 0:
            return 1.0
        return float(latest / median)

    # ------------------------------------------------------------------ #
    #  Query
    # ------------------------------------------------------------------ #

    def get_lane_summary(self, G: Optional[nx.DiGraph] = None) -> pd.DataFrame:
        """Return a DataFrame summarising each lane in the network."""
        if G is None:
            G = self._graph or self.build()

        rows = []
        for u, v, data in G.edges(data=True):
            rows.append({"region": u, "destination": v, **data})
        return pd.DataFrame(rows)

    def get_vulnerable_lanes(
        self, G: Optional[nx.DiGraph] = None, reliability_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Lanes with reliability below threshold."""
        summary = self.get_lane_summary(G)
        return summary[summary["reliability"] < reliability_threshold]
