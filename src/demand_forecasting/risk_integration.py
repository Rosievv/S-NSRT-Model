"""Integrate Module 1 supply-gap risk with Module 3 supply forecasts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def attach_module1_risk(
    forecasts: pd.DataFrame,
    module1_validation: pd.DataFrame,
    scenario: str = "top_1_supplier_failure",
) -> pd.DataFrame:
    """Attach the risk estimate available for each forecast origin quarter."""
    required = {"origin_date", "q10", "q50", "q90"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts missing required columns: {sorted(missing)}")

    risk = module1_validation.loc[
        module1_validation["scenario"].eq(scenario),
        ["quarter_end", "predicted_gap_pct", "lower_gap_pct", "upper_gap_pct"],
    ].copy()
    risk["quarter"] = pd.to_datetime(risk["quarter_end"]).dt.to_period("Q")
    risk = risk.drop_duplicates("quarter", keep="last").drop(columns="quarter_end")

    result = forecasts.copy()
    result["origin_date"] = pd.to_datetime(result["origin_date"])
    result["quarter"] = result["origin_date"].dt.to_period("Q")
    result = result.merge(risk, on="quarter", how="left", validate="many_to_one")
    for column in ("predicted_gap_pct", "lower_gap_pct", "upper_gap_pct"):
        result[column] = result[column].fillna(0.0).clip(0.0, 100.0)
    return result.drop(columns="quarter")


def apply_risk_gating(
    forecasts_with_risk: pd.DataFrame,
    threshold_pct: float = 2.0,
    mode: str = "hard_gate",
) -> pd.DataFrame:
    """Apply trigger logic to risk gap inputs before stress mapping."""
    if mode not in {"hard_gate", "none"}:
        raise ValueError("mode must be 'hard_gate' or 'none'")

    result = forecasts_with_risk.copy()
    trigger_active = result["predicted_gap_pct"] >= threshold_pct
    result["trigger_threshold_pct"] = float(threshold_pct)
    result["trigger_active"] = trigger_active

    if mode == "none":
        result["effective_predicted_gap_pct"] = result["predicted_gap_pct"]
        result["effective_lower_gap_pct"] = result["lower_gap_pct"]
        result["effective_upper_gap_pct"] = result["upper_gap_pct"]
        return result

    result["effective_predicted_gap_pct"] = np.where(
        trigger_active, result["predicted_gap_pct"], 0.0
    )
    result["effective_lower_gap_pct"] = np.where(
        trigger_active, result["lower_gap_pct"], 0.0
    )
    result["effective_upper_gap_pct"] = np.where(
        trigger_active, result["upper_gap_pct"], 0.0
    )
    return result


def apply_gap_elasticity(
    forecasts_with_risk: pd.DataFrame,
    elasticity_by_horizon: dict[int, float] | None = None,
    default_elasticity: float = 0.0,
) -> pd.DataFrame:
    """Reduce effective risk gap as substitution capacity recovers by horizon."""
    result = forecasts_with_risk.copy()
    mapping = elasticity_by_horizon or {}

    if "horizon_months" in result.columns:
        horizon = result["horizon_months"].fillna(1).astype(int)
        elasticity = horizon.map(lambda h: mapping.get(h, default_elasticity)).astype(float)
    else:
        elasticity = pd.Series(default_elasticity, index=result.index, dtype=float)

    elasticity = elasticity.clip(0.0, 0.95)
    result["substitution_elasticity"] = elasticity

    for src_col, out_col in (
        ("effective_predicted_gap_pct", "elastic_predicted_gap_pct"),
        ("effective_lower_gap_pct", "elastic_lower_gap_pct"),
        ("effective_upper_gap_pct", "elastic_upper_gap_pct"),
    ):
        if src_col not in result.columns:
            fallback = src_col.replace("effective_", "")
            if fallback in result.columns:
                result[src_col] = result[fallback]
            else:
                raise ValueError(f"Missing required risk gap column: {src_col}")

        result[out_col] = (result[src_col] * (1.0 - elasticity)).clip(0.0, 100.0)

    return result


def adjust_supply_quantiles(forecasts_with_risk: pd.DataFrame) -> pd.DataFrame:
    """Apply Module 1 gap bounds to Module 3 supply quantiles."""
    result = forecasts_with_risk.copy()

    def _choose(preferred: str, effective: str, raw: str) -> str:
        if preferred in result.columns:
            return preferred
        if effective in result.columns:
            return effective
        return raw

    upper_col = _choose("elastic_upper_gap_pct", "effective_upper_gap_pct", "upper_gap_pct")
    median_col = _choose("elastic_predicted_gap_pct", "effective_predicted_gap_pct", "predicted_gap_pct")
    lower_col = _choose("elastic_lower_gap_pct", "effective_lower_gap_pct", "lower_gap_pct")
    result["risk_q10"] = result["q10"] * (1.0 - result[upper_col] / 100.0)
    result["risk_q50"] = result["q50"] * (1.0 - result[median_col] / 100.0)
    result["risk_q90"] = result["q90"] * (1.0 - result[lower_col] / 100.0)
    result["risk_q10"] = result["risk_q10"].clip(lower=0.0)
    result["risk_q50"] = np.maximum(result["risk_q50"], result["risk_q10"])
    result["risk_q90"] = np.maximum(result["risk_q90"], result["risk_q50"])
    return result


def add_shortage_metrics(forecasts: pd.DataFrame, demand_col: str = "demand_proxy") -> pd.DataFrame:
    """Compare baseline and risk-adjusted supply with a lagged demand proxy."""
    if demand_col not in forecasts.columns:
        raise ValueError(f"Forecasts require demand column '{demand_col}'")
    result = forecasts.copy()
    result["demand_proxy_used"] = result[demand_col]
    demand = result[demand_col].replace(0.0, np.nan)
    result["baseline_supply_demand_ratio"] = result["q50"] / demand
    result["risk_supply_demand_ratio"] = result["risk_q50"] / demand
    result["actual_supply_demand_ratio"] = result["actual_supply"] / demand
    result["baseline_shortfall_usd"] = (result[demand_col] - result["q50"]).clip(lower=0.0)
    result["risk_shortfall_usd"] = (result[demand_col] - result["risk_q50"]).clip(lower=0.0)
    result["actual_shortfall_usd"] = (result[demand_col] - result["actual_supply"]).clip(lower=0.0)
    result["baseline_shortage_flag"] = result["baseline_supply_demand_ratio"] < 0.80
    result["risk_shortage_flag"] = result["risk_supply_demand_ratio"] < 0.80
    result["actual_shortage_flag"] = result["actual_supply_demand_ratio"] < 0.80
    return result


def build_inventory_scenarios(
    forecasts: pd.DataFrame,
    initial_months: Iterable[float] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Simulate recursive monthly inventory balances under standard buffers."""
    rows = []
    for months in initial_months:
        scenario = forecasts.copy()
        horizon = scenario.get("horizon_months", pd.Series(1.0, index=scenario.index)).astype(float)
        scenario["initial_inventory_months"] = float(months)
        scenario["initial_inventory_usd"] = scenario["demand_proxy_used"] * float(months)

        for prefix, supply_col in (
            ("baseline", "q50"),
            ("risk", "risk_q50"),
            ("actual", "actual_supply"),
        ):
            ending_inventory = []
            cumulative_unmet = []
            first_stockout_step = []
            for row in scenario.itertuples(index=False):
                demand = float(getattr(row, "demand_proxy_used"))
                supply = float(getattr(row, supply_col))
                start_inventory = float(getattr(row, "initial_inventory_usd"))
                steps = int(max(round(float(getattr(row, "horizon_months", 1.0))), 1))
                current_inventory = start_inventory
                unmet_total = 0.0
                stockout_step = None
                for step in range(1, steps + 1):
                    available = current_inventory + supply
                    unmet = max(0.0, demand - available)
                    if unmet > 0 and stockout_step is None:
                        stockout_step = step
                    unmet_total += unmet
                    current_inventory = max(0.0, available - demand)
                ending_inventory.append(current_inventory)
                cumulative_unmet.append(unmet_total)
                first_stockout_step.append(stockout_step)

            scenario[f"{prefix}_ending_inventory_usd"] = ending_inventory
            scenario[f"{prefix}_cumulative_shortfall_usd"] = cumulative_unmet
            scenario[f"{prefix}_stockout_step"] = first_stockout_step
            scenario[f"{prefix}_stockout_flag"] = scenario[f"{prefix}_stockout_step"].notna()

        if "origin_date" in scenario.columns:
            origin = pd.to_datetime(scenario["origin_date"])
            scenario["risk_first_stockout_date"] = scenario["risk_stockout_step"].apply(
                lambda x: pd.NaT if pd.isna(x) else int(x)
            )
            scenario["risk_first_stockout_date"] = [
                pd.NaT if pd.isna(step) else origin.iloc[i] + pd.DateOffset(months=int(step))
                for i, step in enumerate(scenario["risk_stockout_step"])
            ]

        scenario["risk_ending_inventory_months"] = np.where(
            scenario["demand_proxy_used"] > 0,
            scenario["risk_ending_inventory_usd"] / scenario["demand_proxy_used"],
            np.nan,
        )
        rows.append(scenario)
    return pd.concat(rows, ignore_index=True)