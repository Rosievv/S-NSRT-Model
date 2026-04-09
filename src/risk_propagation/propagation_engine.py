"""
Disruption Propagation Engine

Simulates how localised supply-chain shocks (e.g., a key supplier going
offline) cascade through the Tier-N semiconductor import network.

Disruption model
----------------
1. **Node shock** — a supplier country's capacity is reduced by a given
   severity (0–1).  Direct supply from that country drops proportionally.
2. **Substitution effect** — remaining suppliers may partially absorb the
   shortfall, but only up to a configurable ``substitution_elasticity``.
3. **Cascading impact** — for HS codes where the shocked country was a
   dominant supplier, the effective loss can exceed the direct share loss
   because substitution options are limited (high-HHI products).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required")

logger = logging.getLogger("SCRAM.RiskPropagation.PropagationEngine")


@dataclass
class DisruptionResult:
    """Container for a single disruption simulation result."""
    scenario_name: str
    shocked_nodes: List[str]
    severity: float
    original_supply: float
    disrupted_supply: float
    supply_gap_pct: float
    most_affected_hs: List[Dict]
    substitution_absorbed_pct: float
    details: Dict = field(default_factory=dict)


class PropagationEngine:
    """
    Run disruption simulations on a :class:`SupplyChainNetwork` graph.
    """

    def __init__(
        self,
        trade_df: pd.DataFrame,
        substitution_elasticity: float = 0.3,
    ):
        """
        Parameters
        ----------
        trade_df : pd.DataFrame
            Raw trade data (``date, hs_code, country, value_usd``).
        substitution_elasticity : float
            Fraction [0, 1] of the lost supply that remaining suppliers
            can absorb.  0 = no substitution, 1 = perfect substitution.
        """
        self.trade_df = trade_df.copy()
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        self.substitution_elasticity = substitution_elasticity
        # Pre-compute per-HS, per-country aggregates for speed
        self._agg = (
            self.trade_df.groupby(["hs_code", "country"])["value_usd"]
            .sum()
            .reset_index()
        )
        self._total_by_hs = (
            self._agg.groupby("hs_code")["value_usd"].sum().to_dict()
        )
        self._total = self._agg["value_usd"].sum()

    # ------------------------------------------------------------------ #
    #  Core simulation
    # ------------------------------------------------------------------ #

    def simulate_node_shock(
        self,
        countries: List[str],
        severity: float = 1.0,
        scenario_name: str = "node_shock",
    ) -> DisruptionResult:
        """
        Simulate one or more supplier countries losing capacity.

        Parameters
        ----------
        countries : list[str]
            Countries whose capacity is reduced.
        severity : float
            Fraction of capacity lost (0=none, 1=total shutdown).
        scenario_name : str
            Label for the scenario.

        Returns
        -------
        DisruptionResult
        """
        countries_upper = [c.upper() for c in countries]
        agg = self._agg.copy()
        agg["country_upper"] = agg["country"].str.upper()

        # Direct loss per HS code
        mask = agg["country_upper"].isin(countries_upper)
        direct_loss_total = agg.loc[mask, "value_usd"].sum() * severity

        # Per-HS-code impact
        hs_impacts: List[Dict] = []
        for hs_code, hs_total in self._total_by_hs.items():
            hs_loss = agg.loc[
                mask & (agg["hs_code"] == hs_code), "value_usd"
            ].sum() * severity

            if hs_loss == 0:
                continue

            # Remaining supplier capacity for this HS code
            remaining = hs_total - hs_loss
            n_remaining = agg.loc[
                ~mask & (agg["hs_code"] == hs_code)
            ].shape[0]

            # Substitution: remaining suppliers absorb part of the loss
            substitution = min(hs_loss, remaining * self.substitution_elasticity)
            net_loss = hs_loss - substitution
            loss_pct = (net_loss / hs_total * 100) if hs_total > 0 else 0

            hs_impacts.append({
                "hs_code": str(hs_code),
                "direct_loss_usd": float(hs_loss),
                "substitution_usd": float(substitution),
                "net_loss_usd": float(net_loss),
                "loss_pct": round(float(loss_pct), 2),
                "remaining_suppliers": int(n_remaining),
            })

        hs_impacts.sort(key=lambda x: x["loss_pct"], reverse=True)

        # Aggregate
        total_substitution = sum(h["substitution_usd"] for h in hs_impacts)
        total_net_loss = sum(h["net_loss_usd"] for h in hs_impacts)
        gap_pct = (total_net_loss / self._total * 100) if self._total > 0 else 0
        sub_absorbed = (
            (total_substitution / direct_loss_total * 100)
            if direct_loss_total > 0
            else 0
        )

        return DisruptionResult(
            scenario_name=scenario_name,
            shocked_nodes=countries,
            severity=severity,
            original_supply=float(self._total),
            disrupted_supply=float(self._total - total_net_loss),
            supply_gap_pct=round(float(gap_pct), 2),
            most_affected_hs=hs_impacts[:10],
            substitution_absorbed_pct=round(float(sub_absorbed), 2),
            details={
                "direct_loss_usd": float(direct_loss_total),
                "substitution_usd": float(total_substitution),
                "net_loss_usd": float(total_net_loss),
            },
        )

    # ------------------------------------------------------------------ #
    #  Multi-node cascade
    # ------------------------------------------------------------------ #

    def simulate_regional_disruption(
        self,
        region_countries: List[str],
        severity: float = 0.5,
        scenario_name: str = "regional_disruption",
    ) -> DisruptionResult:
        """
        Simulate a regional event affecting multiple countries
        (e.g., "East Asia disruption").
        """
        return self.simulate_node_shock(
            countries=region_countries,
            severity=severity,
            scenario_name=scenario_name,
        )

    # ------------------------------------------------------------------ #
    #  Convenience: pre-built common scenarios
    # ------------------------------------------------------------------ #

    def top_supplier_failure(self, n: int = 1, severity: float = 1.0) -> DisruptionResult:
        """Shock the top-N suppliers by total value."""
        top_n = (
            self._agg.groupby("country")["value_usd"]
            .sum()
            .nlargest(n)
            .index.tolist()
        )
        return self.simulate_node_shock(
            countries=top_n,
            severity=severity,
            scenario_name=f"top_{n}_supplier_failure",
        )

    EAST_ASIA = [
        "China", "Taiwan", "Japan", "Korea, South",
        "Malaysia", "Vietnam", "Thailand", "Singapore",
    ]

    def east_asia_disruption(self, severity: float = 0.5) -> DisruptionResult:
        """Simulate partial disruption across major East Asian suppliers."""
        return self.simulate_regional_disruption(
            region_countries=self.EAST_ASIA,
            severity=severity,
            scenario_name="east_asia_disruption",
        )

    # ------------------------------------------------------------------ #
    #  Run all standard scenarios
    # ------------------------------------------------------------------ #

    def run_standard_scenarios(self) -> List[DisruptionResult]:
        """Run a battery of built-in stress-test scenarios."""
        results = [
            self.top_supplier_failure(n=1, severity=1.0),
            self.top_supplier_failure(n=3, severity=0.5),
            self.east_asia_disruption(severity=0.3),
            self.east_asia_disruption(severity=0.7),
        ]
        return results

    # ------------------------------------------------------------------ #
    #  Reporting
    # ------------------------------------------------------------------ #

    @staticmethod
    def results_to_dataframe(results: List[DisruptionResult]) -> pd.DataFrame:
        """Convert a list of DisruptionResults to a summary DataFrame."""
        rows = []
        for r in results:
            rows.append({
                "scenario": r.scenario_name,
                "shocked_nodes": ", ".join(r.shocked_nodes),
                "severity": r.severity,
                "original_supply_B": round(r.original_supply / 1e9, 2),
                "disrupted_supply_B": round(r.disrupted_supply / 1e9, 2),
                "supply_gap_pct": r.supply_gap_pct,
                "substitution_absorbed_pct": r.substitution_absorbed_pct,
                "most_affected_hs": r.most_affected_hs[0]["hs_code"]
                if r.most_affected_hs
                else "N/A",
            })
        return pd.DataFrame(rows)
