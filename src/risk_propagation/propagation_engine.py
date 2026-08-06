"""
Disruption Propagation Engine

Simulates how localised supply-chain shocks (e.g., a key supplier going
offline) cascade through the Tier-N semiconductor import network.

Disruption model
----------------
1. **Node shock** — a supplier country's capacity is reduced by a given
   severity (0–1).  Direct supply from that country drops proportionally.
2. **Substitution effect** — remaining suppliers may partially absorb the
   shortfall.  Elasticity is looked up from ``HS_ELASTICITY_MAP`` first;
   if no specific entry exists the global ``substitution_elasticity``
   fallback is used.  This means CPUs/logic (8542.31, low substitutability)
   and commodity memory (8542.32, high substitutability) are treated
   differently, giving more accurate Taiwan / advanced-node risk exposure.
3. **Cascading impact** — for HS codes where the shocked country was a
   dominant supplier, the effective loss can exceed the direct share loss
   because substitution options are limited (high-HHI products).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required")

logger = logging.getLogger("SCRAM.RiskPropagation.PropagationEngine")


# ---------------------------------------------------------------------------
# Per-HS-code substitution elasticity lookup
# ---------------------------------------------------------------------------
# Values represent the fraction [0, 1] of lost supply that remaining suppliers
# can realistically absorb in the short-to-medium term.
#
# Rationale:
#   8542.31  Advanced logic / CPUs / SoCs — highly fab-specific, long lead
#            times, almost no short-run substitution → 0.10
#   8542.32  Memory (DRAM / NAND) — multiple commodity producers, faster
#            capacity ramp → 0.50
#   8542.33  Programmable logic (FPGAs) — moderate lock-in → 0.25
#   8542.39  Other ICs — mixed; default-ish → 0.35
#   8541.xx  Discrete semiconductors — commodity-like, moderate → 0.40
#   2804.61  Silicon — bulk commodity, highly substitutable → 0.70
#   2804.69  Other rare-earth-related → 0.30
#
# Keys may be 4-digit (HS chapter+heading), 6-digit, or 8-digit strings.
# The lookup tries the most-specific match first, then progressively shorter
# prefixes, then falls back to the global ``substitution_elasticity``.
HS_ELASTICITY_MAP: Dict[str, float] = {
    # --- Advanced logic (CPUs, GPUs, SoCs, ASICs) ---
    "854231": 0.10,
    "85423100": 0.10,
    # --- Memory (DRAM, NAND flash) ---
    "854232": 0.50,
    "85423200": 0.50,
    # --- Programmable logic (FPGAs, CPLDs) ---
    "854233": 0.25,
    "85423300": 0.25,
    # --- Other ICs ---
    "854239": 0.35,
    "85423900": 0.35,
    # --- Discrete semiconductors ---
    "8541": 0.40,
    # --- Silicon (bulk) ---
    "280461": 0.70,
    "280469": 0.30,
}


def _resolve_hs_elasticity(
    hs_code: str,
    hs_map: Dict[str, float],
    fallback: float,
) -> float:
    """Return the elasticity for *hs_code* via longest-prefix match."""
    code = str(hs_code).replace(".", "").strip()
    # Try lengths: 8, 6, 4 digits
    for length in (8, 6, 4):
        key = code[:length]
        if key in hs_map:
            return hs_map[key]
    return fallback


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
        hs_elasticity_map: Optional[Dict[str, float]] = None,
        use_weighted_substitution: bool = True,
        concentration_penalty_lambda: float = 0.5,
        geo_penalty_factor: float = 0.6,
    ):
        """
        Parameters
        ----------
        trade_df : pd.DataFrame
            Raw trade data (``date, hs_code, country, value_usd``).
        substitution_elasticity : float
            Global fallback elasticity [0, 1] used when an HS code has no
            entry in *hs_elasticity_map*.
        hs_elasticity_map : dict[str, float], optional
            Per-HS-code elasticity overrides.  Keys are HS code strings
            (dots removed, any digit length).  Defaults to
            ``HS_ELASTICITY_MAP``.
        use_weighted_substitution : bool
            If True, scale substitution capacity by reliability,
            concentration, and geographic-correlation penalties.
        concentration_penalty_lambda : float
            Strength of concentration penalty on substitution [0, 1].
        geo_penalty_factor : float
            Penalty multiplier [0, 1] for remaining suppliers in the
            same disrupted region during regional events.
        """
        self.trade_df = trade_df.copy()
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        self.substitution_elasticity = substitution_elasticity
        self.hs_elasticity_map: Dict[str, float] = (
            hs_elasticity_map if hs_elasticity_map is not None else HS_ELASTICITY_MAP
        )
        self.use_weighted_substitution = use_weighted_substitution
        self.concentration_penalty_lambda = float(
            min(max(concentration_penalty_lambda, 0.0), 1.0)
        )
        self.geo_penalty_factor = float(min(max(geo_penalty_factor, 0.0), 1.0))
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

        # Monthly table for reliability/activity weights.
        monthly = self.trade_df.copy()
        monthly["period"] = monthly["date"].dt.to_period("M")
        self._monthly_agg = (
            monthly.groupby(["period", "hs_code", "country"]) ["value_usd"]
            .sum()
            .reset_index()
        )
        self._n_periods = int(monthly["period"].nunique())

        self._reliability_by_hs_country = self._build_reliability_weights()
        self._activity_by_hs_country = self._build_activity_weights()
        self._concentration_by_hs = self._build_concentration_weights()

    def _build_reliability_weights(self) -> Dict[tuple, float]:
        """Compute a stability score per (hs_code, country) from monthly CoV."""
        if self._monthly_agg.empty:
            return {}

        stats = (
            self._monthly_agg.groupby(["hs_code", "country"])["value_usd"]
            .agg(["mean", "std"])
            .reset_index()
        )
        out: Dict[tuple, float] = {}
        for row in stats.itertuples(index=False):
            mean_val = float(row.mean)
            std_val = float(0.0 if pd.isna(row.std) else row.std)
            cov = (std_val / mean_val) if mean_val > 0 else 0.0
            # Higher CoV means lower reliability.
            reliability = 1.0 / (1.0 + cov)
            out[(str(row.hs_code), row.country)] = float(
                min(max(reliability, 0.1), 1.0)
            )
        return out

    def _build_activity_weights(self) -> Dict[tuple, float]:
        """Approximate ramp/lead-time readiness via active-month ratio."""
        if self._monthly_agg.empty or self._n_periods == 0:
            return {}

        counts = (
            self._monthly_agg.groupby(["hs_code", "country"])["period"]
            .nunique()
            .reset_index(name="active_months")
        )
        out: Dict[tuple, float] = {}
        for row in counts.itertuples(index=False):
            ratio = float(row.active_months) / float(self._n_periods)
            # Keep a floor to avoid zeroing viable suppliers entirely.
            out[(str(row.hs_code), row.country)] = float(min(max(ratio, 0.2), 1.0))
        return out

    def _build_concentration_weights(self) -> Dict[str, float]:
        """Compute HS-level concentration penalty from normalized HHI."""
        out: Dict[str, float] = {}
        if self._agg.empty:
            return out

        for hs_code, grp in self._agg.groupby("hs_code"):
            total_val = float(grp["value_usd"].sum())
            if total_val <= 0:
                out[str(hs_code)] = 1.0
                continue
            shares = (grp["value_usd"] / total_val).astype(float).tolist()
            n = len(shares)
            if n <= 1:
                out[str(hs_code)] = 0.5
                continue
            hhi = float(sum(s * s for s in shares))
            # Normalize HHI to [0, 1].
            hhi_norm = (hhi - (1.0 / n)) / (1.0 - (1.0 / n))
            hhi_norm = float(min(max(hhi_norm, 0.0), 1.0))
            penalty = 1.0 - (self.concentration_penalty_lambda * hhi_norm)
            out[str(hs_code)] = float(min(max(penalty, 0.2), 1.0))
        return out

    # ------------------------------------------------------------------ #
    #  Core simulation
    # ------------------------------------------------------------------ #

    def simulate_node_shock(
        self,
        countries: List[str],
        severity: float = 1.0,
        scenario_name: str = "node_shock",
        correlated_region_countries: Optional[List[str]] = None,
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
        region_upper: Set[str] = {
            c.upper() for c in (correlated_region_countries or [])
        }
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
            remaining_rows = agg.loc[
                ~mask & (agg["hs_code"] == hs_code)
            ].copy()
            n_remaining = int(remaining_rows.shape[0])

            # Per-HS elasticity (longest-prefix match, then global fallback)
            elasticity = _resolve_hs_elasticity(
                hs_code, self.hs_elasticity_map, self.substitution_elasticity
            )

            # Substitution: remaining suppliers absorb part of the loss
            if not self.use_weighted_substitution:
                substitution_capacity = remaining * elasticity
            else:
                hs_key = str(hs_code)
                concentration_weight = self._concentration_by_hs.get(hs_key, 1.0)
                weighted_capacity = 0.0
                for row in remaining_rows.itertuples(index=False):
                    rel_weight = self._reliability_by_hs_country.get(
                        (hs_key, row.country), 0.5
                    )
                    activity_weight = self._activity_by_hs_country.get(
                        (hs_key, row.country), 0.5
                    )
                    geo_weight = 1.0
                    if region_upper and row.country_upper in region_upper:
                        geo_weight = self.geo_penalty_factor

                    combined_weight = (
                        rel_weight * activity_weight * concentration_weight * geo_weight
                    )
                    weighted_capacity += (
                        float(row.value_usd)
                        * elasticity
                        * combined_weight
                    )

                substitution_capacity = weighted_capacity

            substitution = min(hs_loss, substitution_capacity)
            net_loss = hs_loss - substitution
            loss_pct = (net_loss / hs_total * 100) if hs_total > 0 else 0

            hs_impacts.append({
                "hs_code": str(hs_code),
                "direct_loss_usd": float(hs_loss),
                "substitution_capacity_usd": float(substitution_capacity),
                "substitution_usd": float(substitution),
                "net_loss_usd": float(net_loss),
                "loss_pct": round(float(loss_pct), 2),
                "remaining_suppliers": int(n_remaining),
                "elasticity_used": round(float(elasticity), 3),
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
                "substitution_mode": (
                    "weighted" if self.use_weighted_substitution else "simple"
                ),
                "substitution_elasticity": float(self.substitution_elasticity),
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
            correlated_region_countries=region_countries,
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
