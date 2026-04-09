"""
Supply Chain Network Graph Builder

Constructs directed weighted graphs from US Census trade data where:
  - Nodes = supplier countries (+ "USA" as the demand sink)
  - Edges = trade flows weighted by value_usd
  - Node attributes include centrality scores, market share, etc.

Uses NetworkX for graph operations and supports temporal snapshots
(one graph per month) for longitudinal disruption analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx>=3.1")

logger = logging.getLogger("SCRAM.RiskPropagation.GraphNetwork")


class SupplyChainNetwork:
    """
    Build and analyse a directed supply-chain network for a given
    set of HS codes and time period.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, trade_df: pd.DataFrame, hs_codes: Optional[List[str]] = None):
        """
        Parameters
        ----------
        trade_df : pd.DataFrame
            Raw US Census trade data with at least:
            ``date, hs_code, country, value_usd``
        hs_codes : list[str], optional
            Subset of HS codes to include.  Default = all.
        """
        required = {"date", "hs_code", "country", "value_usd"}
        missing = required - set(trade_df.columns)
        if missing:
            raise ValueError(f"trade_df missing columns: {missing}")

        self.trade_df = trade_df.copy()
        if hs_codes:
            self.trade_df = self.trade_df[
                self.trade_df["hs_code"].astype(str).isin(hs_codes)
            ]
        # Ensure date is datetime
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        logger.info(
            "SupplyChainNetwork: %d records, %d countries, %d HS codes",
            len(self.trade_df),
            self.trade_df["country"].nunique(),
            self.trade_df["hs_code"].nunique(),
        )

    # ------------------------------------------------------------------ #
    #  Snapshot graph for a single period
    # ------------------------------------------------------------------ #

    def build_network(
        self, period: str = "all", min_value: float = 0
    ) -> nx.DiGraph:
        """
        Build a directed graph for a given time period.

        Parameters
        ----------
        period : str
            ``"all"`` for aggregate, or ``"YYYY-MM"`` for monthly snapshot.
        min_value : float
            Minimum cumulative ``value_usd`` to include an edge.

        Returns
        -------
        nx.DiGraph
            Nodes = countries, edges point supplier → "USA".
        """
        df = self.trade_df
        if period != "all":
            df = df[df["date"].dt.to_period("M").astype(str) == period]

        if df.empty:
            logger.warning("No data for period %s", period)
            return nx.DiGraph()

        # Aggregate by country (across HS codes for the period)
        agg = (
            df.groupby("country")
            .agg(total_value=("value_usd", "sum"), n_hs_codes=("hs_code", "nunique"))
            .reset_index()
        )
        total_supply = agg["total_value"].sum()

        G = nx.DiGraph()
        G.add_node("USA", role="demand_sink")

        for _, row in agg.iterrows():
            if row["total_value"] < min_value:
                continue
            country = row["country"]
            share = row["total_value"] / total_supply if total_supply > 0 else 0
            G.add_node(
                country,
                role="supplier",
                total_value=float(row["total_value"]),
                n_hs_codes=int(row["n_hs_codes"]),
                market_share=float(share),
            )
            G.add_edge(
                country,
                "USA",
                weight=float(row["total_value"]),
                share=float(share),
            )

        G.graph["period"] = period
        G.graph["total_supply"] = float(total_supply)
        logger.info(
            "Graph for %s: %d suppliers, total value $%.1fB",
            period,
            G.number_of_nodes() - 1,
            total_supply / 1e9,
        )
        return G

    # ------------------------------------------------------------------ #
    #  Centrality & importance metrics
    # ------------------------------------------------------------------ #

    def compute_centrality(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Compute node-level centrality & importance metrics.

        Returns a DataFrame indexed by country with columns:
        ``market_share, degree, betweenness, eigenvector, pagerank``
        """
        if G.number_of_nodes() < 2:
            return pd.DataFrame()

        records: List[Dict] = []
        pagerank = nx.pagerank(G, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight")
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eigenvector = {n: 0 for n in G.nodes()}

        for node in G.nodes():
            if node == "USA":
                continue
            records.append(
                {
                    "country": node,
                    "market_share": G.nodes[node].get("market_share", 0),
                    "total_value": G.nodes[node].get("total_value", 0),
                    "degree": G.degree(node),
                    "betweenness": betweenness.get(node, 0),
                    "eigenvector": eigenvector.get(node, 0),
                    "pagerank": pagerank.get(node, 0),
                }
            )

        df = pd.DataFrame(records).sort_values("market_share", ascending=False)
        return df.reset_index(drop=True)

    def identify_critical_nodes(
        self, G: nx.DiGraph, threshold: float = 0.05
    ) -> List[str]:
        """
        Return suppliers whose removal would cause more than
        ``threshold`` (fraction) loss of total supply value.
        """
        total_supply = G.graph.get("total_supply", 0)
        if total_supply == 0:
            return []

        critical: List[str] = []
        for node in G.nodes():
            if node == "USA":
                continue
            val = G.nodes[node].get("total_value", 0)
            if val / total_supply >= threshold:
                critical.append(node)
        return critical

    # ------------------------------------------------------------------ #
    #  Temporal snapshots
    # ------------------------------------------------------------------ #

    def build_temporal_snapshots(
        self, freq: str = "M", min_value: float = 0
    ) -> Dict[str, nx.DiGraph]:
        """
        Build one graph per ``freq`` period across the full time range.

        Returns ``{period_str: DiGraph}`` dict.
        """
        periods = self.trade_df["date"].dt.to_period(freq).unique().sort_values()
        snapshots: Dict[str, nx.DiGraph] = {}
        for p in periods:
            snapshots[str(p)] = self.build_network(str(p), min_value=min_value)
        return snapshots

    def track_concentration_over_time(self) -> pd.DataFrame:
        """
        For each monthly snapshot, compute HHI and top-supplier share.
        Returns a time-series DataFrame.
        """
        snapshots = self.build_temporal_snapshots()
        rows = []
        for period, G in snapshots.items():
            shares = [
                G.nodes[n].get("market_share", 0)
                for n in G.nodes()
                if n != "USA"
            ]
            if not shares:
                continue
            hhi = sum(s ** 2 for s in shares) * 10_000
            top1 = max(shares) if shares else 0
            top3 = sum(sorted(shares, reverse=True)[:3])
            rows.append(
                {
                    "period": period,
                    "hhi": hhi,
                    "top1_share": top1,
                    "top3_share": top3,
                    "n_suppliers": len(shares),
                    "total_supply": G.graph.get("total_supply", 0),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  HS-code-level sub-networks
    # ------------------------------------------------------------------ #

    def build_hs_network(
        self, hs_code: str, period: str = "all", min_value: float = 0
    ) -> nx.DiGraph:
        """Build a network for a single HS code."""
        df = self.trade_df[self.trade_df["hs_code"].astype(str) == hs_code]
        if df.empty:
            return nx.DiGraph()
        sub = SupplyChainNetwork(df)
        return sub.build_network(period=period, min_value=min_value)
