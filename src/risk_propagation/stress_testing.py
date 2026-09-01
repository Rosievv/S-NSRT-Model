"""
Scenario-Based Stress Testing

Provides a runner that executes predefined or custom disruption scenarios
against the supply-chain network and compares simulated impacts with
historical actuals (backtesting).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        "affected_countries": ["China"],
        "affected_hs_codes": ["854231", "854232", "854239"],
        "event_type": "pandemic",
        "estimated_severity": 0.15,
    },
    {
        "name": "japan_earthquake_2011",
        "description": "Tōhoku earthquake and tsunami",
        "date_range": ("2011-03", "2011-09"),
        "affected_countries": ["Japan", "Taiwan", "Korea, South"],
        "affected_hs_codes": ["848620", "381800", "854239", "854231", "854232"],
        "event_type": "earthquake",
        "estimated_severity": 0.55,
    },
    {
        "name": "thai_flood_2011",
        "description": "Thailand flooding (HDD/memory supply)",
        "date_range": ("2011-10", "2012-03"),
        "affected_countries": ["Thailand"],
        "affected_hs_codes": ["854232", "381800"],
        "event_type": "flood",
        "estimated_severity": 0.5,
    },
    {
        "name": "japan_export_controls_2019",
        "description": "Japan-Korea export control tensions",
        "date_range": ("2019-07", "2020-02"),
        "affected_countries": ["Japan", "Korea, South"],
        "affected_hs_codes": ["854231", "854232", "854239"],
        "event_type": "export_control",
        "estimated_severity": 0.2,
    },
    {
        "name": "china_power_shortage_2021",
        "description": "China manufacturing power curtailments",
        "date_range": ("2021-09", "2022-02"),
        "affected_countries": ["China", "Taiwan", "Korea, South"],
        "affected_hs_codes": ["854239", "848620", "381800", "854231", "854232"],
        "event_type": "power_shortage",
        "estimated_severity": 0.45,
    },
    {
        "name": "taiwan_drought_2021",
        "description": "Taiwan drought affecting semiconductor fabs",
        "date_range": ("2021-03", "2021-08"),
        "affected_countries": ["Taiwan"],
        "affected_hs_codes": ["854231", "854232", "854239", "848620"],
        "event_type": "drought",
        "estimated_severity": 0.25,
    },
    {
        "name": "red_sea_disruption_2024",
        "description": "Shipping rerouting disruptions via Red Sea",
        "date_range": ("2024-01", "2024-06"),
        "affected_countries": ["China", "Taiwan", "Korea, South", "Japan"],
        "affected_hs_codes": ["854231", "854232", "854239", "848620", "381800"],
        "event_type": "logistics",
        "estimated_severity": 0.15,
    },
]

FORWARD_SCENARIO_LIBRARY = [
    {
        "name": "baseline_moderate",
        "description": "Moderate ongoing East Asia friction",
        "type": "regional",
        "region": "east_asia",
        "severity": 0.15,
    },
    {
        "name": "top_1_supplier_failure",
        "description": "Top-1 supplier goes completely offline",
        "type": "top_supplier",
        "n": 1,
        "severity": 1.0,
    },
    {
        "name": "top_3_supplier_failure",
        "description": "Top-3 suppliers lose 50 % capacity simultaneously",
        "type": "top_supplier",
        "n": 3,
        "severity": 0.5,
    },
    {
        "name": "east_asia_severe",
        "description": "Severe East Asian disruption (70 %)",
        "type": "regional",
        "region": "east_asia",
        "severity": 0.7,
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
            use_weighted_substitution=True,
            concentration_penalty_lambda=0.5,
            geo_penalty_factor=0.6,
        )

    @staticmethod
    def _month_range(start: str, end: str) -> pd.PeriodIndex:
        start_period = pd.Period(start, freq="M")
        end_period = pd.Period(end, freq="M")
        return pd.period_range(start=start_period, end=end_period, freq="M")

    @staticmethod
    def _severity_bin(value: float) -> str:
        if value < 2.0:
            return "low"
        if value < 8.0:
            return "medium"
        return "high"

    @staticmethod
    def _event_substitution_multiplier(event: Dict) -> float:
        """Reduce effective substitution for event types with slower rerouting."""
        event_type = event.get("event_type", "generic")
        base = {
            "pandemic": 0.9,
            "earthquake": 0.35,
            "flood": 0.35,
            "export_control": 0.5,
            "power_shortage": 0.25,
            "drought": 0.5,
            "logistics": 0.75,
            "generic": 0.6,
        }.get(event_type, 0.6)

        months = len(StressTestRunner._month_range(*event["date_range"]))
        ramp = min(1.0, 0.35 + 0.1 * max(months - 1, 0))
        return float(min(max(base * ramp, 0.1), 1.0))

    @staticmethod
    def _event_severity_multiplier(event: Dict) -> float:
        """Amplify event severity where direct production loss is typically undercounted."""
        return {
            "pandemic": 1.0,
            "earthquake": 1.4,
            "flood": 1.2,
            "export_control": 1.1,
            "power_shortage": 1.8,
            "drought": 1.3,
            "logistics": 1.0,
            "generic": 1.0,
        }.get(event.get("event_type", "generic"), 1.0)

    @staticmethod
    def _event_month_profiles(event: Dict) -> List[Tuple[float, float]]:
        """
        Return month-by-month (severity_scale, substitution_scale) pairs.

        Lower substitution_scale in early months encodes rerouting lag.
        """
        months = len(StressTestRunner._month_range(*event["date_range"]))
        if months <= 0:
            return [(1.0, 1.0)]

        event_type = event.get("event_type", "generic")
        profiles: List[Tuple[float, float]] = []

        for idx in range(months):
            progress = idx / max(months - 1, 1)

            if event_type in {"earthquake", "flood"}:
                # Acute shock, strongest at onset, substitution ramps slowly.
                severity_scale = max(0.55, 1.15 - 0.55 * progress)
                substitution_scale = min(1.0, 0.2 + 0.5 * progress)
            elif event_type == "power_shortage":
                # Persistent production constraints.
                severity_scale = max(0.8, 1.05 - 0.1 * progress)
                substitution_scale = min(1.0, 0.2 + 0.3 * progress)
            elif event_type == "export_control":
                # Policy friction tends to persist.
                severity_scale = max(0.9, 1.0 - 0.05 * progress)
                substitution_scale = min(1.0, 0.45 + 0.35 * progress)
            elif event_type == "pandemic":
                # Initial acute dislocation then adaptation.
                severity_scale = max(0.6, 0.9 - 0.25 * progress)
                substitution_scale = min(1.0, 0.75 + 0.25 * progress)
            elif event_type == "drought":
                severity_scale = max(0.75, 1.0 - 0.2 * progress)
                substitution_scale = min(1.0, 0.45 + 0.45 * progress)
            elif event_type == "logistics":
                severity_scale = max(0.8, 0.95 - 0.1 * progress)
                substitution_scale = min(1.0, 0.5 + 0.4 * progress)
            else:
                severity_scale = max(0.8, 1.0 - 0.15 * progress)
                substitution_scale = min(1.0, 0.45 + 0.45 * progress)

            profiles.append((float(severity_scale), float(substitution_scale)))

        return profiles

    def _build_observed_supply_gap(
        self,
        event: Dict,
    ) -> Dict[str, float]:
        """Create a supply-focused observed gap using trend counterfactual."""
        start, end = event["date_range"]
        event_months = self._month_range(start, end)
        countries = {c.upper() for c in event.get("affected_countries", [])}
        hs_codes = {str(c) for c in event.get("affected_hs_codes", [])}

        slice_df = self.trade_df.copy()
        slice_df["country_upper"] = slice_df["country"].str.upper()
        if hs_codes:
            slice_df = slice_df[slice_df["hs_code"].astype(str).isin(hs_codes)]
        if countries:
            slice_df = slice_df[slice_df["country_upper"].isin(countries)]

        if slice_df.empty:
            return {
                "observed_supply_gap_pct": 0.0,
                "counterfactual_supply_usd": 0.0,
                "observed_supply_usd": 0.0,
            }

        monthly = (
            slice_df.groupby(slice_df["date"].dt.to_period("M"))["value_usd"]
            .sum()
            .sort_index()
        )
        event_series = monthly.reindex(event_months, fill_value=0.0)

        pre_series = monthly[monthly.index < event_months[0]]
        if pre_series.empty:
            return {
                "observed_supply_gap_pct": 0.0,
                "counterfactual_supply_usd": 0.0,
                "observed_supply_usd": float(event_series.sum()),
            }

        # Use up to 24 pre-event months to avoid overly stale trend fitting.
        pre_tail = pre_series.tail(24)
        y = pre_tail.values.astype(float)
        x = np.arange(len(y), dtype=float)
        if len(y) >= 2:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_future = np.arange(len(y), len(y) + len(event_months), dtype=float)
            forecast = intercept + slope * x_future
        else:
            forecast = np.repeat(float(y.mean()), len(event_months))

        counterfactual = np.clip(forecast, a_min=0.0, a_max=None)
        counterfactual_supply_usd = float(counterfactual.sum())
        observed_supply_usd = float(event_series.sum())

        if counterfactual_supply_usd <= 0:
            gap_pct = 0.0
        else:
            gap_pct = max(
                0.0,
                (counterfactual_supply_usd - observed_supply_usd)
                / counterfactual_supply_usd
                * 100.0,
            )

        return {
            "observed_supply_gap_pct": float(round(gap_pct, 2)),
            "counterfactual_supply_usd": counterfactual_supply_usd,
            "observed_supply_usd": observed_supply_usd,
        }

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
        Compare a simulated disruption against a supply-focused observed
        gap for the same period.
        """
        start, end = event["date_range"]
        affected_hs_codes = [str(code) for code in event.get("affected_hs_codes", [])]

        # Simulated impact (using pre-event aggregate as baseline)
        pre_event_df = self.trade_df[
            self.trade_df["date"] < pd.Timestamp(start + "-01")
        ]
        if pre_event_df.empty:
            logger.warning("No pre-event data for %s", event["name"])
            return {"event": event["name"], "status": "insufficient_data"}

        sim_df = pre_event_df.copy()
        if affected_hs_codes:
            sim_df = sim_df[sim_df["hs_code"].astype(str).isin(affected_hs_codes)]
        if sim_df.empty:
            logger.warning("No pre-event HS-slice data for %s", event["name"])
            return {"event": event["name"], "status": "insufficient_data"}

        effective_elasticity = (
            self.engine.substitution_elasticity
            * self._event_substitution_multiplier(event)
        )
        base_severity = min(
            1.0,
            float(event["estimated_severity"]) * self._event_severity_multiplier(event),
        )

        monthly_profiles = self._event_month_profiles(event)
        month_gaps: List[float] = []
        month_sub_absorbed: List[float] = []
        final_result: Optional[DisruptionResult] = None

        for month_idx, (severity_scale, substitution_scale) in enumerate(monthly_profiles, start=1):
            month_severity = min(1.0, base_severity * severity_scale)
            month_elasticity = max(0.02, min(1.0, effective_elasticity * substitution_scale))

            month_engine = PropagationEngine(
                trade_df=sim_df,
                substitution_elasticity=month_elasticity,
                use_weighted_substitution=self.engine.use_weighted_substitution,
                concentration_penalty_lambda=self.engine.concentration_penalty_lambda,
                geo_penalty_factor=self.engine.geo_penalty_factor,
            )
            month_result = month_engine.simulate_node_shock(
                countries=event["affected_countries"],
                severity=month_severity,
                scenario_name=f"backtest_{event['name']}_m{month_idx}",
            )
            month_gaps.append(float(month_result.supply_gap_pct))
            month_sub_absorbed.append(float(month_result.substitution_absorbed_pct))
            final_result = month_result

        if final_result is None:
            logger.warning("Monthly simulation failed for %s", event["name"])
            return {"event": event["name"], "status": "insufficient_data"}

        predicted_gap = float(round(float(np.mean(month_gaps)), 2))
        predicted_sub_absorbed = float(round(float(np.mean(month_sub_absorbed)), 2))

        observed = self._build_observed_supply_gap(event)
        observed_gap = float(observed["observed_supply_gap_pct"])
        pred_bin = self._severity_bin(predicted_gap)
        obs_bin = self._severity_bin(observed_gap)

        return {
            "event": event["name"],
            "description": event["description"],
            "predicted_supply_gap_pct": round(predicted_gap, 2),
            "observed_supply_gap_pct": round(observed_gap, 2),
            "counterfactual_supply_usd": round(
                float(observed["counterfactual_supply_usd"]), 2
            ),
            "observed_supply_usd": round(float(observed["observed_supply_usd"]), 2),
            "supply_gap_diff_pct": round(predicted_gap - observed_gap, 2),
            "abs_error_pct": round(abs(predicted_gap - observed_gap), 2),
            "predicted_substitution_absorbed_pct": predicted_sub_absorbed,
            "severity_bin_predicted": pred_bin,
            "severity_bin_observed": obs_bin,
            "directional_hit": int(pred_bin == obs_bin),
        }

    def backtest_all(self) -> pd.DataFrame:
        """Run back-tests for all known historical events."""
        rows = [self.backtest_event(e) for e in HISTORICAL_EVENTS]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        valid_df = df.dropna(subset=["predicted_supply_gap_pct", "observed_supply_gap_pct"]).copy()
        if valid_df.empty:
            df["directional_accuracy_pct"] = np.nan
            df["pairwise_ranking_accuracy_pct"] = np.nan
            return df

        directional_accuracy_pct = float(valid_df["directional_hit"].mean() * 100.0)

        valid = valid_df[["predicted_supply_gap_pct", "observed_supply_gap_pct"]].copy()
        pair_hits: List[int] = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pred_diff = (
                    valid.iloc[i]["predicted_supply_gap_pct"]
                    - valid.iloc[j]["predicted_supply_gap_pct"]
                )
                obs_diff = (
                    valid.iloc[i]["observed_supply_gap_pct"]
                    - valid.iloc[j]["observed_supply_gap_pct"]
                )
                pair_hits.append(int((pred_diff == 0 and obs_diff == 0) or (pred_diff * obs_diff > 0)))

        pairwise_accuracy_pct = float(np.mean(pair_hits) * 100.0) if pair_hits else 100.0
        df["directional_accuracy_pct"] = round(directional_accuracy_pct, 2)
        df["pairwise_ranking_accuracy_pct"] = round(pairwise_accuracy_pct, 2)
        return df

    # ------------------------------------------------------------------ #
    #  Forward projections
    # ------------------------------------------------------------------ #

    def _project_monthly_trade_panel(
        self,
        start: str = "2025-01",
        end: str = "2026-12",
        lookback_months: int = 24,
    ) -> pd.DataFrame:
        """Project monthly trade values for the requested forward window."""
        future_periods = pd.period_range(start=start, end=end, freq="M")
        if self.trade_df.empty or future_periods.empty:
            return pd.DataFrame(columns=list(self.trade_df.columns))

        monthly = self.trade_df.copy()
        monthly["period"] = monthly["date"].dt.to_period("M")

        future_rows: List[Dict] = []
        for (hs_code, country), grp in monthly.groupby(["hs_code", "country"], dropna=False):
            series = grp.groupby("period")["value_usd"].sum().sort_index()
            if series.empty:
                continue

            tail = series.tail(lookback_months)
            y = tail.values.astype(float)
            if len(y) == 0:
                continue

            x = np.arange(len(y), dtype=float)
            if len(y) >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_future = np.arange(len(y), len(y) + len(future_periods), dtype=float)
                trend_forecast = intercept + slope * x_future
            else:
                trend_forecast = np.repeat(float(y.mean()), len(future_periods))

            seasonal_adjustment = np.ones(len(future_periods))
            overall_mean = float(series.mean())
            if len(series) >= 12 and overall_mean > 0:
                month_means = series.groupby(series.index.month).mean()
                seasonal_adjustment = np.array(
                    [
                        float(month_means.get(period.month, overall_mean)) / overall_mean
                        for period in future_periods
                    ],
                    dtype=float,
                )

            projected_values = np.clip(trend_forecast * seasonal_adjustment, 0.0, None)
            for period, value in zip(future_periods, projected_values):
                future_rows.append(
                    {
                        "date": period.to_timestamp(how="start"),
                        "hs_code": str(hs_code),
                        "country": country,
                        "value_usd": float(value),
                        "quantity": np.nan,
                        "projection_source": "trend_extrapolation",
                        "projection_method": "24m_trend_with_seasonality",
                    }
                )

        projected_df = pd.DataFrame(future_rows)
        if projected_df.empty:
            return projected_df

        projected_df["date"] = pd.to_datetime(projected_df["date"])
        return projected_df

    def _run_projection_case(
        self,
        trade_df: pd.DataFrame,
        scenario: Dict,
        severity_multiplier: float,
        elasticity_multiplier: float,
    ) -> DisruptionResult:
        """Run a single forward projection case on a projected trade panel."""
        engine = PropagationEngine(
            trade_df=trade_df,
            substitution_elasticity=max(
                0.02, float(self.engine.substitution_elasticity * elasticity_multiplier)
            ),
            use_weighted_substitution=self.engine.use_weighted_substitution,
            concentration_penalty_lambda=self.engine.concentration_penalty_lambda,
            geo_penalty_factor=self.engine.geo_penalty_factor,
        )

        severity = min(1.0, float(scenario.get("severity", 1.0)) * severity_multiplier)
        stype = scenario.get("type", "node_shock")

        if stype == "top_supplier":
            return engine.top_supplier_failure(
                n=int(scenario.get("n", 1)),
                severity=severity,
            )
        if stype == "regional":
            region = scenario.get("region", "east_asia")
            countries = (
                PropagationEngine.EAST_ASIA
                if region == "east_asia"
                else scenario.get("countries", [])
            )
            return engine.simulate_regional_disruption(
                region_countries=countries,
                severity=severity,
                scenario_name=scenario.get("name", "regional"),
            )
        return engine.simulate_node_shock(
            countries=scenario.get("countries", []),
            severity=severity,
            scenario_name=scenario.get("name", "node_shock"),
        )

    def generate_forward_projections_2025_2026(self) -> pd.DataFrame:
        """Generate quarterly out-of-sample projections for 2025-2026."""
        projected_panel = self._project_monthly_trade_panel()
        if projected_panel.empty:
            logger.warning("No projected forward panel could be generated")
            return pd.DataFrame()

        projected_panel = projected_panel.copy()
        projected_panel["quarter"] = projected_panel["date"].dt.to_period("Q")

        projection_rows: List[Dict] = []
        bound_cases = {
            "lower": (0.85, 1.15),
            "expected": (1.0, 1.0),
            "upper": (1.15, 0.85),
        }

        for quarter in pd.period_range("2025Q1", "2026Q4", freq="Q"):
            quarter_df = projected_panel.loc[projected_panel["quarter"] == quarter].copy()
            if quarter_df.empty:
                continue

            for scenario in FORWARD_SCENARIO_LIBRARY:
                case_results: Dict[str, DisruptionResult] = {}
                for bound_name, (severity_multiplier, elasticity_multiplier) in bound_cases.items():
                    case_results[bound_name] = self._run_projection_case(
                        trade_df=quarter_df,
                        scenario=scenario,
                        severity_multiplier=severity_multiplier,
                        elasticity_multiplier=elasticity_multiplier,
                    )

                lower_gap = min(
                    case_results["lower"].supply_gap_pct,
                    case_results["expected"].supply_gap_pct,
                    case_results["upper"].supply_gap_pct,
                )
                upper_gap = max(
                    case_results["lower"].supply_gap_pct,
                    case_results["expected"].supply_gap_pct,
                    case_results["upper"].supply_gap_pct,
                )
                expected_result = case_results["expected"]

                projection_rows.append(
                    {
                        "quarter": str(quarter),
                        "quarter_start": quarter.start_time.strftime("%Y-%m-%d"),
                        "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
                        "scenario": scenario["name"],
                        "description": scenario["description"],
                        "scenario_type": scenario["type"],
                        "shocked_nodes": ", ".join(expected_result.shocked_nodes),
                        "severity": scenario.get("severity", 1.0),
                        "lower_gap_pct": round(float(lower_gap), 2),
                        "expected_gap_pct": round(float(expected_result.supply_gap_pct), 2),
                        "upper_gap_pct": round(float(upper_gap), 2),
                        "projection_band_width_pct": round(float(upper_gap - lower_gap), 2),
                        "original_supply_B": round(float(expected_result.original_supply / 1e9), 2),
                        "disrupted_supply_B": round(float(expected_result.disrupted_supply / 1e9), 2),
                        "substitution_absorbed_pct": round(
                            float(expected_result.substitution_absorbed_pct), 2
                        ),
                        "most_affected_hs": (
                            expected_result.most_affected_hs[0]["hs_code"]
                            if expected_result.most_affected_hs
                            else "N/A"
                        ),
                        "scenario_status": "out_of_sample",
                    }
                )

        return pd.DataFrame(projection_rows)

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
