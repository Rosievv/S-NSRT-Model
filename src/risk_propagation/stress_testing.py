"""
Scenario-Based Stress Testing

Provides a runner that executes predefined or custom disruption scenarios
against the supply-chain network and compares simulated impacts with
historical actuals (backtesting).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .propagation_engine import PropagationEngine, DisruptionResult

logger = logging.getLogger("SCRAM.RiskPropagation.StressTesting")


# --------------------------------------------------------------------------- #
#  Built-in scenario library
# --------------------------------------------------------------------------- #

SCENARIO_LIBRARY: List[Dict] = [
    {
        "name": "single_source_failure",
        "description": "Top-1 supplier goes completely offline",
        "type": "top_supplier",
        "n": 1,
        "severity": 1.0,
    },
    {
        "name": "top3_partial",
        "description": "Top-3 suppliers lose 50 % capacity simultaneously",
        "type": "top_supplier",
        "n": 3,
        "severity": 0.5,
    },
    {
        "name": "east_asia_moderate",
        "description": "Moderate disruption across East Asian suppliers (30 %)",
        "type": "regional",
        "region": "east_asia",
        "severity": 0.3,
    },
    {
        "name": "east_asia_severe",
        "description": "Severe East Asian disruption (70 %)",
        "type": "regional",
        "region": "east_asia",
        "severity": 0.7,
    },
    {
        "name": "china_decoupling",
        "description": "Full decoupling from China supply",
        "type": "node_shock",
        "countries": ["China"],
        "severity": 1.0,
    },
    {
        "name": "taiwan_crisis",
        "description": "Taiwan capacity drops by 80 %",
        "type": "node_shock",
        "countries": ["Taiwan"],
        "severity": 0.8,
    },
]

# Known historical disruption events for backtesting
HISTORICAL_EVENTS = [
    {
        "name": "covid_q1_2020",
        "description": "COVID-19 initial lockdowns",
        "date_range": ("2020-01", "2020-06"),
        "affected_countries": ["China", "Malaysia", "Vietnam"],
        "estimated_severity": 0.3,
    },
    {
        "name": "japan_earthquake_2011",
        "description": "Tōhoku earthquake and tsunami",
        "date_range": ("2011-03", "2011-09"),
        "affected_countries": ["Japan"],
        "estimated_severity": 0.4,
    },
    {
        "name": "thai_flood_2011",
        "description": "Thailand flooding (HDD/memory supply)",
        "date_range": ("2011-10", "2012-03"),
        "affected_countries": ["Thailand"],
        "estimated_severity": 0.5,
    },
]


class StressTestRunner:
    """
    Execute scenario-based stress tests and optionally back-test
    against historical disruption events.
    """

    def __init__(self, trade_df: pd.DataFrame, substitution_elasticity: float = 0.3):
        self.trade_df = trade_df.copy()
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        self.engine = PropagationEngine(
            trade_df=self.trade_df,
            substitution_elasticity=substitution_elasticity,
        )

    # ------------------------------------------------------------------ #
    #  Run scenarios
    # ------------------------------------------------------------------ #

    def run_scenario(self, scenario: Dict) -> DisruptionResult:
        """Run a single scenario dict and return the result."""
        stype = scenario.get("type", "node_shock")

        if stype == "top_supplier":
            return self.engine.top_supplier_failure(
                n=scenario.get("n", 1),
                severity=scenario.get("severity", 1.0),
            )
        elif stype == "regional":
            region = scenario.get("region", "east_asia")
            countries = (
                PropagationEngine.EAST_ASIA
                if region == "east_asia"
                else scenario.get("countries", [])
            )
            return self.engine.simulate_regional_disruption(
                region_countries=countries,
                severity=scenario.get("severity", 0.5),
                scenario_name=scenario.get("name", "regional"),
            )
        else:  # node_shock
            return self.engine.simulate_node_shock(
                countries=scenario.get("countries", []),
                severity=scenario.get("severity", 1.0),
                scenario_name=scenario.get("name", "node_shock"),
            )

    def run_all_standard(self) -> List[DisruptionResult]:
        """Run every scenario in the built-in library."""
        results = []
        for s in SCENARIO_LIBRARY:
            logger.info("Running scenario: %s", s["name"])
            results.append(self.run_scenario(s))
        return results

    # ------------------------------------------------------------------ #
    #  Historical back-testing
    # ------------------------------------------------------------------ #

    def backtest_event(self, event: Dict) -> Dict:
        """
        Compare a simulated disruption against actual trade data for the
        same period.  Returns both the simulation result and the observed
        trade change.
        """
        start, end = event["date_range"]

        # Simulated impact (using pre-event aggregate as baseline)
        pre_event_df = self.trade_df[
            self.trade_df["date"] < pd.Timestamp(start + "-01")
        ]
        if pre_event_df.empty:
            logger.warning("No pre-event data for %s", event["name"])
            return {"event": event["name"], "status": "insufficient_data"}

        pre_engine = PropagationEngine(
            trade_df=pre_event_df,
            substitution_elasticity=self.engine.substitution_elasticity,
        )
        sim_result = pre_engine.simulate_node_shock(
            countries=event["affected_countries"],
            severity=event["estimated_severity"],
            scenario_name=f"backtest_{event['name']}",
        )

        # Observed change
        pre_val = pre_event_df["value_usd"].sum()
        event_df = self.trade_df[
            (self.trade_df["date"] >= pd.Timestamp(start + "-01"))
            & (self.trade_df["date"] <= pd.Timestamp(end + "-28"))
        ]
        event_val = event_df["value_usd"].sum()

        # Normalise to monthly
        n_pre_months = pre_event_df["date"].dt.to_period("M").nunique()
        n_event_months = event_df["date"].dt.to_period("M").nunique()
        pre_monthly = pre_val / max(n_pre_months, 1)
        event_monthly = event_val / max(n_event_months, 1)
        observed_change_pct = (
            (event_monthly - pre_monthly) / pre_monthly * 100
            if pre_monthly > 0
            else 0
        )

        return {
            "event": event["name"],
            "description": event["description"],
            "simulated_gap_pct": sim_result.supply_gap_pct,
            "observed_change_pct": round(float(observed_change_pct), 2),
            "simulation_vs_actual_diff": round(
                float(sim_result.supply_gap_pct - abs(observed_change_pct)), 2
            ),
        }

    def backtest_all(self) -> pd.DataFrame:
        """Run back-tests for all known historical events."""
        rows = [self.backtest_event(e) for e in HISTORICAL_EVENTS]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save_report(
        self,
        results: List[DisruptionResult],
        path: str = "reports/stress_test_report.json",
    ) -> Path:
        """Serialize scenario results to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "scenario_name": r.scenario_name,
                "shocked_nodes": r.shocked_nodes,
                "severity": r.severity,
                "original_supply": r.original_supply,
                "disrupted_supply": r.disrupted_supply,
                "supply_gap_pct": r.supply_gap_pct,
                "substitution_absorbed_pct": r.substitution_absorbed_pct,
                "most_affected_hs": r.most_affected_hs,
                "details": r.details,
            }
            for r in results
        ]
        out.write_text(json.dumps(data, indent=2))
        logger.info("Stress test report saved to %s", out)
        return out
