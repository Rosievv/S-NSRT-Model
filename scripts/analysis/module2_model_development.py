"""
Module2 spatial allocation and resilience rerouting model.

This script implements a three-leg logistics network optimization with:
- end-to-end flow conservation (overseas -> port -> DC -> state demand)
- multimodal routing (rail, truck, air)
- stress rerouting from West Coast to Gulf/East ports
- state-level fill-rate and shortfall outputs
- bottleneck and shadow-price diagnostics
- strategy tradeoff dashboard (cost, lead time, fill rate, air premium share)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "module2"

sys.path.insert(0, str(ROOT))
from src.transportation.weather_disruption import WeatherDisruptionDetector  # noqa: E402

CORE_STATES = {"CA", "TX", "MI"}

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}

WEST_STATES = {
    "AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY",
}
CENTRAL_STATES = {
    "IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE", "OH", "OK", "SD", "TX", "WI",
}


@dataclass(frozen=True)
class StrategyConfig:
    key: str
    name: str
    stress: bool
    delay_penalty: float
    unmet_penalty: float
    core_unmet_multiplier: float
    noncore_unmet_multiplier: float
    air_cap_multiplier: float
    reroute_discount: float
    min_core_air_share: float


@dataclass(frozen=True)
class OptimizationPreference:
    mode: str = "strategy"
    service_weight: float = 0.5
    min_fill_rate: Optional[float] = None


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def _normalize_state_name(value: object) -> str:
    text = str(value).strip().upper().replace("STATE_", "")
    return text.replace("_", " ")


def _state_to_region(state_abbr: str) -> str:
    if state_abbr in WEST_STATES:
        return "WEST"
    if state_abbr in CENTRAL_STATES:
        return "CENTRAL"
    return "EAST"


def _build_state_demand_panel(demand: pd.DataFrame) -> pd.DataFrame:
    data = demand.copy()
    if "state" in data.columns:
        data["state_name"] = data["state"].map(_normalize_state_name)
    elif "demand_zone" in data.columns:
        data["state_name"] = data["demand_zone"].map(_normalize_state_name)
    else:
        raise ValueError("State demand input must include state or demand_zone")

    if "hs_code" not in data.columns:
        data["hs_code"] = "854231"

    data["demand_value_usd"] = pd.to_numeric(data["demand_value_usd"], errors="coerce")
    data = data.dropna(subset=["state_name", "demand_value_usd"])

    grouped = (
        data.groupby("state_name", as_index=False)
        .agg(demand_usd=("demand_value_usd", "sum"))
        .sort_values("demand_usd", ascending=False)
    )
    grouped["state_abbr"] = grouped["state_name"].map(STATE_NAME_TO_ABBR)
    grouped = grouped.dropna(subset=["state_abbr"]).copy()
    grouped["region"] = grouped["state_abbr"].map(_state_to_region)
    grouped["demand_zone"] = "STATE_" + grouped["state_name"].str.replace(" ", "_", regex=False)
    return grouped.reset_index(drop=True)


def _lane_geography_summary(lanes: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [c for c in ["faf_state", "faf_zone", "border_state_1", "border_state_2", "faf_country"] if c in lanes.columns]
    if not geo_cols:
        return pd.DataFrame()
    grouped = (
        lanes.groupby(geo_cols, as_index=False)
        .agg(
            lane_count=("capacity", "size"),
            capacity=("capacity", "sum"),
            unit_cost=("unit_cost", "mean"),
            transit_days=("transit_days", "mean"),
            reliability=("reliability", "mean"),
        )
        .sort_values("capacity", ascending=False)
    )
    total_capacity = float(grouped["capacity"].sum())
    grouped["capacity_share"] = np.where(total_capacity > 0, grouped["capacity"] / total_capacity, 0.0)
    grouped["capacity_share_pct"] = grouped["capacity_share"] * 100.0
    return grouped


def _node_capacity_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        nodes.groupby("state", as_index=False)
        .agg(
            handling_capacity=("handling_capacity", "sum"),
            storage_capacity=("storage_capacity", "sum"),
            node_count=("node_id", "count"),
        )
        .sort_values("handling_capacity", ascending=False)
    )
    total = float(summary["handling_capacity"].sum())
    summary["handling_share"] = np.where(total > 0, summary["handling_capacity"] / total, 0.0)
    summary["handling_share_pct"] = summary["handling_share"] * 100.0
    return summary


def _build_network(
    state_demand: pd.DataFrame,
    lanes: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_demand = float(state_demand["demand_usd"].sum())
    lane_capacity_usd = float(lanes["capacity"].sum()) * 60000.0
    network_capacity = min(max(total_demand * 0.95, lane_capacity_usd), total_demand * 1.20)

    ports = pd.DataFrame(
        [
            {"port": "PORT_LA_LB", "capacity": network_capacity * 0.58},
            {"port": "PORT_HOU", "capacity": network_capacity * 0.22},
            {"port": "PORT_NY_NJ", "capacity": network_capacity * 0.20},
        ]
    )

    origins = pd.DataFrame(
        [
            {"origin": "ORIGIN_ASIA", "capacity": network_capacity * 0.70},
            {"origin": "ORIGIN_EU", "capacity": network_capacity * 0.20},
            {"origin": "ORIGIN_AMERICAS", "capacity": network_capacity * 0.15},
        ]
    )

    origin_port_shares = {
        ("ORIGIN_ASIA", "PORT_LA_LB"): 0.65,
        ("ORIGIN_ASIA", "PORT_HOU"): 0.20,
        ("ORIGIN_ASIA", "PORT_NY_NJ"): 0.15,
        ("ORIGIN_EU", "PORT_LA_LB"): 0.10,
        ("ORIGIN_EU", "PORT_HOU"): 0.25,
        ("ORIGIN_EU", "PORT_NY_NJ"): 0.65,
        ("ORIGIN_AMERICAS", "PORT_LA_LB"): 0.20,
        ("ORIGIN_AMERICAS", "PORT_HOU"): 0.55,
        ("ORIGIN_AMERICAS", "PORT_NY_NJ"): 0.25,
    }
    mode_specs_leg0 = {
        "ocean": {"cap_ratio": 0.92, "unit_cost": 0.014, "lead_time": 18.0},
        "air": {"cap_ratio": 0.08, "unit_cost": 0.095, "lead_time": 2.0},
    }

    intl_port_rows: List[Dict[str, object]] = []
    for (origin, port), pair_share in origin_port_shares.items():
        origin_cap = float(origins.loc[origins["origin"] == origin, "capacity"].iloc[0])
        base_pair_cap = origin_cap * pair_share
        for mode, spec in mode_specs_leg0.items():
            intl_port_rows.append(
                {
                    "from_origin": origin,
                    "to_port": port,
                    "mode": mode,
                    "capacity": base_pair_cap * spec["cap_ratio"],
                    "unit_cost": spec["unit_cost"],
                    "lead_time": spec["lead_time"],
                    "is_air": mode == "air",
                }
            )
    intl_port = pd.DataFrame(intl_port_rows)

    dcs = pd.DataFrame(
        [
            {"dc": "DC_WEST", "capacity": network_capacity * 0.46},
            {"dc": "DC_CENTRAL", "capacity": network_capacity * 0.36},
            {"dc": "DC_EAST", "capacity": network_capacity * 0.34},
        ]
    )

    pair_shares = {
        ("PORT_LA_LB", "DC_WEST"): 0.55,
        ("PORT_LA_LB", "DC_CENTRAL"): 0.35,
        ("PORT_LA_LB", "DC_EAST"): 0.10,
        ("PORT_HOU", "DC_WEST"): 0.15,
        ("PORT_HOU", "DC_CENTRAL"): 0.45,
        ("PORT_HOU", "DC_EAST"): 0.40,
        ("PORT_NY_NJ", "DC_WEST"): 0.10,
        ("PORT_NY_NJ", "DC_CENTRAL"): 0.30,
        ("PORT_NY_NJ", "DC_EAST"): 0.60,
    }

    mode_specs_leg1 = {
        "rail": {"cap_ratio": 0.55, "unit_cost": 0.010, "lead_time": 5.0},
        "truck": {"cap_ratio": 0.35, "unit_cost": 0.018, "lead_time": 3.0},
        "air": {"cap_ratio": 0.10, "unit_cost": 0.070, "lead_time": 1.0},
    }

    port_dc_rows: List[Dict[str, object]] = []
    for (port, dc), pair_share in pair_shares.items():
        port_cap = float(ports.loc[ports["port"] == port, "capacity"].iloc[0])
        base_pair_cap = port_cap * pair_share
        for mode, spec in mode_specs_leg1.items():
            port_dc_rows.append(
                {
                    "from_node": port,
                    "to_node": dc,
                    "mode": mode,
                    "capacity": base_pair_cap * spec["cap_ratio"],
                    "unit_cost": spec["unit_cost"],
                    "lead_time": spec["lead_time"],
                    "is_air": mode == "air",
                }
            )
    port_dc = pd.DataFrame(port_dc_rows)

    mode_specs_leg2 = {
        "rail": {"unit_cost": 0.012, "lead_time": 4.0},
        "truck": {"unit_cost": 0.020, "lead_time": 2.0},
        "air": {"unit_cost": 0.085, "lead_time": 1.0},
    }
    preferred_dc = {"WEST": "DC_WEST", "CENTRAL": "DC_CENTRAL", "EAST": "DC_EAST"}
    primary_cap = {"rail": 0.45, "truck": 0.45, "air": 0.10}
    secondary_cap = {"rail": 0.12, "truck": 0.18, "air": 0.08}

    dc_state_rows: List[Dict[str, object]] = []
    for row in state_demand.itertuples(index=False):
        state = str(row.state_abbr)
        region = str(row.region)
        demand_usd = float(row.demand_usd)
        primary = preferred_dc[region]
        for dc in ["DC_WEST", "DC_CENTRAL", "DC_EAST"]:
            is_primary = dc == primary
            for mode, spec in mode_specs_leg2.items():
                cap_ratio = primary_cap[mode] if is_primary else secondary_cap[mode]
                unit_cost = spec["unit_cost"] + (0.0 if is_primary else 0.006)
                lead_time = spec["lead_time"] + (0.0 if is_primary else 1.0)
                dc_state_rows.append(
                    {
                        "from_node": dc,
                        "state": state,
                        "region": region,
                        "mode": mode,
                        "capacity": demand_usd * cap_ratio,
                        "unit_cost": unit_cost,
                        "lead_time": lead_time,
                        "is_air": mode == "air",
                        "is_primary": is_primary,
                    }
                )
    dc_state = pd.DataFrame(dc_state_rows)

    return origins, intl_port, ports, dcs, port_dc, dc_state


def _strategy_configs() -> List[StrategyConfig]:
    return [
        StrategyConfig(
            key="cost_min",
            name="Cost-Minimizing",
            stress=True,
            delay_penalty=0.0009,
            unmet_penalty=1.40,
            core_unmet_multiplier=1.00,
            noncore_unmet_multiplier=1.00,
            air_cap_multiplier=0.80,
            reroute_discount=0.000,
            min_core_air_share=0.000,
        ),
        StrategyConfig(
            key="resilience_first",
            name="Resilience-First",
            stress=True,
            delay_penalty=0.0022,
            unmet_penalty=3.50,
            core_unmet_multiplier=1.40,
            noncore_unmet_multiplier=1.15,
            air_cap_multiplier=1.80,
            reroute_discount=0.004,
            min_core_air_share=0.035,
        ),
        StrategyConfig(
            key="targeted",
            name="Targeted Allocation",
            stress=True,
            delay_penalty=0.0016,
            unmet_penalty=2.30,
            core_unmet_multiplier=3.00,
            noncore_unmet_multiplier=0.90,
            air_cap_multiplier=1.45,
            reroute_discount=0.003,
            min_core_air_share=0.020,
        ),
    ]


def _strategy_by_key(key: str) -> StrategyConfig:
    if key == "baseline":
        return _scenario_config_baseline()
    for cfg in _strategy_configs():
        if cfg.key == key:
            return cfg
    raise ValueError(f"Unknown strategy key: {key}")


def _scenario_config_baseline() -> StrategyConfig:
    return StrategyConfig(
        key="baseline",
        name="Baseline",
        stress=False,
        delay_penalty=0.0009,
        unmet_penalty=1.35,
        core_unmet_multiplier=1.00,
        noncore_unmet_multiplier=1.00,
        air_cap_multiplier=0.75,
        reroute_discount=0.000,
        min_core_air_share=0.000,
    )


def _prepare_scenario_network(
    origins: pd.DataFrame,
    intl_port: pd.DataFrame,
    ports: pd.DataFrame,
    dcs: pd.DataFrame,
    port_dc: pd.DataFrame,
    dc_state: pd.DataFrame,
    strategy: StrategyConfig,
    weather_port_multipliers: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    o = origins.copy()
    l0 = intl_port.copy()
    p = ports.copy()
    d = dcs.copy()
    l1 = port_dc.copy()
    l2 = dc_state.copy()

    if strategy.stress:
        if weather_port_multipliers:
            # Real, live weather-driven disruption (NHC active storms) takes
            # precedence over the synthetic West-Coast shock assumption.
            for port, multiplier in weather_port_multipliers.items():
                if multiplier >= 1.0:
                    continue
                p.loc[p["port"] == port, "capacity"] *= multiplier
                l1.loc[l1["from_node"] == port, "capacity"] *= multiplier
                l0.loc[l0["to_port"] == port, "capacity"] *= multiplier
                if port == "PORT_LA_LB":
                    d.loc[d["dc"] == "DC_WEST", "capacity"] *= multiplier
            unaffected_ports = [p_name for p_name in p["port"] if weather_port_multipliers.get(p_name, 1.0) >= 1.0]
            if unaffected_ports:
                p.loc[p["port"].isin(unaffected_ports), "capacity"] *= 1.10
                l0.loc[l0["to_port"].isin(unaffected_ports), "capacity"] *= 1.10
        else:
            p.loc[p["port"] == "PORT_LA_LB", "capacity"] *= 0.35
            p.loc[p["port"].isin(["PORT_HOU", "PORT_NY_NJ"]), "capacity"] *= 1.12
            d.loc[d["dc"] == "DC_WEST", "capacity"] *= 0.82

            la_mask = l1["from_node"] == "PORT_LA_LB"
            l1.loc[la_mask, "capacity"] *= 0.45

            la_intl = l0["to_port"] == "PORT_LA_LB"
            l0.loc[la_intl, "capacity"] *= 0.50
            l0.loc[l0["to_port"].isin(["PORT_HOU", "PORT_NY_NJ"]), "capacity"] *= 1.10

    air_mask_l0 = l0["mode"] == "air"
    air_mask_l1 = l1["mode"] == "air"
    air_mask_l2 = l2["mode"] == "air"
    l0.loc[air_mask_l0, "capacity"] *= strategy.air_cap_multiplier
    l1.loc[air_mask_l1, "capacity"] *= strategy.air_cap_multiplier
    l2.loc[air_mask_l2, "capacity"] *= strategy.air_cap_multiplier

    if strategy.reroute_discount > 0:
        reroute_mask = ~l2["is_primary"]
        l2.loc[reroute_mask, "unit_cost"] = (l2.loc[reroute_mask, "unit_cost"] - strategy.reroute_discount).clip(lower=0.005)

    if strategy.stress and strategy.key in {"resilience_first", "targeted"}:
        core_air_mask = air_mask_l2 & l2["state"].isin(CORE_STATES)
        l2.loc[core_air_mask, "capacity"] *= 1.25

    return o, l0, p, d, l1, l2


def _solve_network(
    state_demand: pd.DataFrame,
    origins: pd.DataFrame,
    intl_port: pd.DataFrame,
    ports: pd.DataFrame,
    dcs: pd.DataFrame,
    port_dc: pd.DataFrame,
    dc_state: pd.DataFrame,
    strategy: StrategyConfig,
    preference: Optional[OptimizationPreference] = None,
) -> Dict[str, object]:
    preference = preference or OptimizationPreference()

    states = state_demand[["state_abbr", "region", "demand_usd"]].copy().reset_index(drop=True)

    n0 = len(intl_port)
    n1 = len(port_dc)
    n2 = len(dc_state)
    ns = len(states)
    n_vars = n0 + n1 + n2 + ns

    c = np.zeros(n_vars, dtype=float)

    intl_port_cost = (
        intl_port["unit_cost"].to_numpy(dtype=float)
        + strategy.delay_penalty * intl_port["lead_time"].to_numpy(dtype=float)
        + np.where(intl_port["is_air"].to_numpy(dtype=bool), 0.050, 0.0)
    )
    port_dc_cost = (
        port_dc["unit_cost"].to_numpy(dtype=float)
        + strategy.delay_penalty * port_dc["lead_time"].to_numpy(dtype=float)
        + np.where(port_dc["is_air"].to_numpy(dtype=bool), 0.020, 0.0)
    )
    dc_state_cost = (
        dc_state["unit_cost"].to_numpy(dtype=float)
        + strategy.delay_penalty * dc_state["lead_time"].to_numpy(dtype=float)
        + np.where(dc_state["is_air"].to_numpy(dtype=bool), 0.030, 0.0)
    )

    transport_cost = np.zeros(n_vars, dtype=float)
    transport_cost[:n0] = intl_port_cost
    transport_cost[n0 : n0 + n1] = port_dc_cost
    transport_cost[n0 + n1 : n0 + n1 + n2] = dc_state_cost

    unmet_penalty = np.full(ns, strategy.unmet_penalty, dtype=float)
    core_mask = states["state_abbr"].isin(CORE_STATES).to_numpy(dtype=bool)
    unmet_penalty[core_mask] *= strategy.core_unmet_multiplier
    unmet_penalty[~core_mask] *= strategy.noncore_unmet_multiplier
    service_loss = np.zeros(n_vars, dtype=float)
    service_loss[n0 + n1 + n2 :] = unmet_penalty

    if preference.mode == "strategy":
        c = transport_cost + service_loss
    elif preference.mode == "mixed":
        weight = float(np.clip(preference.service_weight, 0.0, 1.0))
        transport_scale = max(float(np.quantile(transport_cost[: n1 + n2], 0.75)), 1e-6)
        service_scale = max(float(np.quantile(unmet_penalty, 0.75)), 1e-6)
        c = (1.0 - weight) * (transport_cost / transport_scale) + weight * (service_loss / service_scale)
    elif preference.mode == "min_cost_with_fill":
        c = transport_cost
    else:
        raise ValueError(f"Unknown optimization mode: {preference.mode}")

    bounds: List[Tuple[float, float]] = []
    bounds.extend([(0.0, float(v)) for v in intl_port["capacity"].to_numpy(dtype=float)])
    bounds.extend([(0.0, float(v)) for v in port_dc["capacity"].to_numpy(dtype=float)])
    bounds.extend([(0.0, float(v)) for v in dc_state["capacity"].to_numpy(dtype=float)])
    bounds.extend([(0.0, float(v)) for v in states["demand_usd"].to_numpy(dtype=float)])

    ineq_rows: List[np.ndarray] = []
    ineq_rhs: List[float] = []
    ineq_labels: List[str] = []

    for origin_row in origins.itertuples(index=False):
        row = np.zeros(n_vars, dtype=float)
        mask = intl_port["from_origin"] == origin_row.origin
        row[np.where(mask.to_numpy())[0]] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(float(origin_row.capacity))
        ineq_labels.append(f"origin_capacity::{origin_row.origin}")

    for port_row in ports.itertuples(index=False):
        row_in = np.zeros(n_vars, dtype=float)
        in_mask = intl_port["to_port"] == port_row.port
        row_in[np.where(in_mask.to_numpy())[0]] = 1.0
        ineq_rows.append(row_in)
        ineq_rhs.append(float(port_row.capacity))
        ineq_labels.append(f"port_in_capacity::{port_row.port}")

        row = np.zeros(n_vars, dtype=float)
        mask = port_dc["from_node"] == port_row.port
        row[np.where(mask.to_numpy())[0] + n0] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(float(port_row.capacity))
        ineq_labels.append(f"port_out_capacity::{port_row.port}")

    for dc_row in dcs.itertuples(index=False):
        row_in = np.zeros(n_vars, dtype=float)
        mask_in = port_dc["to_node"] == dc_row.dc
        row_in[np.where(mask_in.to_numpy())[0] + n0] = 1.0
        ineq_rows.append(row_in)
        ineq_rhs.append(float(dc_row.capacity))
        ineq_labels.append(f"dc_in_capacity::{dc_row.dc}")

        row_out = np.zeros(n_vars, dtype=float)
        mask_out = dc_state["from_node"] == dc_row.dc
        out_idx = np.where(mask_out.to_numpy())[0] + n0 + n1
        row_out[out_idx] = 1.0
        ineq_rows.append(row_out)
        ineq_rhs.append(float(dc_row.capacity))
        ineq_labels.append(f"dc_out_capacity::{dc_row.dc}")

    A_ub = np.vstack(ineq_rows) if ineq_rows else None
    b_ub = np.array(ineq_rhs, dtype=float) if ineq_rhs else None

    if preference.min_fill_rate is not None:
        min_fill_rate = float(np.clip(preference.min_fill_rate, 0.0, 1.0))
        total_demand = float(states["demand_usd"].sum())
        max_unmet = (1.0 - min_fill_rate) * total_demand
        row = np.zeros(n_vars, dtype=float)
        row[n0 + n1 + n2 :] = 1.0
        if A_ub is None:
            A_ub = row.reshape(1, -1)
            b_ub = np.array([max_unmet], dtype=float)
        else:
            A_ub = np.vstack([A_ub, row])
            b_ub = np.concatenate([b_ub, np.array([max_unmet], dtype=float)])
        ineq_labels.append("service_level::min_fill_rate")

    # Ensure stress strategies can activate emergency air channel for core states.
    if strategy.min_core_air_share > 0:
        core_demand = float(states.loc[states["state_abbr"].isin(CORE_STATES), "demand_usd"].sum())
        if core_demand > 0:
            row = np.zeros(n_vars, dtype=float)
            core_air_mask = dc_state["state"].isin(CORE_STATES) & (dc_state["mode"] == "air")
            row[np.where(core_air_mask.to_numpy())[0] + n0 + n1] = -1.0
            min_air_flow = strategy.min_core_air_share * core_demand
            if A_ub is None:
                A_ub = row.reshape(1, -1)
                b_ub = np.array([-min_air_flow], dtype=float)
            else:
                A_ub = np.vstack([A_ub, row])
                b_ub = np.concatenate([b_ub, np.array([-min_air_flow], dtype=float)])
            ineq_labels.append("min_core_air_flow")

    eq_rows: List[np.ndarray] = []
    eq_rhs: List[float] = []

    for port in ports["port"].tolist():
        row = np.zeros(n_vars, dtype=float)
        in_mask = intl_port["to_port"] == port
        out_mask = port_dc["from_node"] == port
        row[np.where(in_mask.to_numpy())[0]] = 1.0
        row[np.where(out_mask.to_numpy())[0] + n0] = -1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)

    for dc in dcs["dc"].tolist():
        row = np.zeros(n_vars, dtype=float)
        in_mask = port_dc["to_node"] == dc
        out_mask = dc_state["from_node"] == dc
        row[np.where(in_mask.to_numpy())[0] + n0] = 1.0
        row[np.where(out_mask.to_numpy())[0] + n0 + n1] = -1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)

    for s_idx, s_row in enumerate(states.itertuples(index=False)):
        row = np.zeros(n_vars, dtype=float)
        to_mask = dc_state["state"] == s_row.state_abbr
        row[np.where(to_mask.to_numpy())[0] + n0 + n1] = 1.0
        row[n0 + n1 + n2 + s_idx] = 1.0
        eq_rows.append(row)
        eq_rhs.append(float(s_row.demand_usd))

    A_eq = np.vstack(eq_rows)
    b_eq = np.array(eq_rhs, dtype=float)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP solver failed for {strategy.key}: {result.message}")

    x0 = result.x[:n0]
    x1 = result.x[n0 : n0 + n1]
    x2 = result.x[n0 + n1 : n0 + n1 + n2]
    unmet = result.x[n0 + n1 + n2 :]

    flow_intl_port = intl_port.copy()
    flow_intl_port["flow_usd"] = x0
    flow_intl_port["strategy"] = strategy.name
    flow_intl_port["scenario"] = strategy.key

    flow_port_dc = port_dc.copy()
    flow_port_dc["flow_usd"] = x1
    flow_port_dc["strategy"] = strategy.name
    flow_port_dc["scenario"] = strategy.key

    flow_dc_state = dc_state.copy()
    flow_dc_state["flow_usd"] = x2
    flow_dc_state["strategy"] = strategy.name
    flow_dc_state["scenario"] = strategy.key

    state_result = states.copy()
    state_result["delivered_usd"] = 0.0
    state_result["unmet_usd"] = unmet
    for s_idx, s_row in enumerate(states.itertuples(index=False)):
        mask = flow_dc_state["state"] == s_row.state_abbr
        state_result.loc[s_idx, "delivered_usd"] = float(flow_dc_state.loc[mask, "flow_usd"].sum())
    state_result["fill_rate"] = np.where(
        state_result["demand_usd"] > 0,
        state_result["delivered_usd"] / state_result["demand_usd"],
        0.0,
    )
    state_result["shortfall_usd"] = state_result["unmet_usd"]
    state_result["strategy"] = strategy.name
    state_result["scenario"] = "stress" if strategy.stress else "baseline"

    delivered = float(state_result["delivered_usd"].sum())
    total_demand = float(state_result["demand_usd"].sum())
    total_unmet = float(state_result["unmet_usd"].sum())

    logistics_cost = float(
        (flow_intl_port["flow_usd"] * flow_intl_port["unit_cost"]).sum()
        + (flow_port_dc["flow_usd"] * flow_port_dc["unit_cost"]).sum()
        + (flow_dc_state["flow_usd"] * flow_dc_state["unit_cost"]).sum()
    )
    lead_time_total = float(
        (flow_intl_port["flow_usd"] * flow_intl_port["lead_time"]).sum()
        + (flow_port_dc["flow_usd"] * flow_port_dc["lead_time"]).sum()
        + (flow_dc_state["flow_usd"] * flow_dc_state["lead_time"]).sum()
    )
    air_flow = float(flow_dc_state.loc[flow_dc_state["mode"] == "air", "flow_usd"].sum())

    summary = {
        "strategy_key": strategy.key,
        "strategy": strategy.name,
        "scenario": "stress" if strategy.stress else "baseline",
        "total_demand_usd": total_demand,
        "delivered_usd": delivered,
        "unmet_usd": total_unmet,
        "fill_rate": (delivered / total_demand) if total_demand > 0 else 0.0,
        "total_logistics_cost_usd": logistics_cost,
        "avg_lead_time_days": (lead_time_total / delivered) if delivered > 0 else 0.0,
        "air_express_share": (air_flow / delivered) if delivered > 0 else 0.0,
        "objective_value": float(result.fun),
        "objective_mode": preference.mode,
        "service_weight": preference.service_weight,
        "min_fill_rate_constraint": preference.min_fill_rate,
    }

    bottlenecks = []
    if hasattr(result, "ineqlin") and result.ineqlin is not None:
        slacks = np.asarray(result.ineqlin.residual, dtype=float)
        marginals = np.asarray(result.ineqlin.marginals, dtype=float)
        for idx, label in enumerate(ineq_labels):
            shadow = max(0.0, -float(marginals[idx]))
            active = bool(slacks[idx] <= 1e-6)
            if active or shadow > 1e-6:
                bottlenecks.append(
                    {
                        "strategy": strategy.name,
                        "scenario": summary["scenario"],
                        "constraint": label,
                        "constraint_type": "node_capacity",
                        "slack": float(slacks[idx]),
                        "shadow_price": shadow,
                    }
                )

    if hasattr(result, "upper") and result.upper is not None:
        upper_residual = np.asarray(result.upper.residual, dtype=float)
        upper_marginals = np.asarray(result.upper.marginals, dtype=float)
        variable_labels = []
        for row in flow_intl_port.itertuples(index=False):
            variable_labels.append(f"lane_capacity::INTL_TO_PORT::{row.from_origin}->{row.to_port}::{row.mode}")
        for row in flow_port_dc.itertuples(index=False):
            variable_labels.append(f"lane_capacity::PORT_TO_DC::{row.from_node}->{row.to_node}::{row.mode}")
        for row in flow_dc_state.itertuples(index=False):
            variable_labels.append(f"lane_capacity::DC_TO_STATE::{row.from_node}->{row.state}::{row.mode}")
        for row in state_result.itertuples(index=False):
            variable_labels.append(f"unmet_upper_bound::{row.state_abbr}")

        for idx, label in enumerate(variable_labels):
            shadow = max(0.0, -float(upper_marginals[idx]))
            at_upper = bool(upper_residual[idx] <= 1e-6)
            if ("lane_capacity" in label) and (at_upper or shadow > 1e-6):
                bottlenecks.append(
                    {
                        "strategy": strategy.name,
                        "scenario": summary["scenario"],
                        "constraint": label,
                        "constraint_type": "lane_capacity",
                        "slack": float(upper_residual[idx]),
                        "shadow_price": shadow,
                    }
                )

    bottleneck_df = pd.DataFrame(bottlenecks)
    if not bottleneck_df.empty:
        bottleneck_df = bottleneck_df.sort_values(["shadow_price", "slack"], ascending=[False, True]).reset_index(drop=True)

    return {
        "summary": summary,
        "flow_intl_port": flow_intl_port,
        "flow_port_dc": flow_port_dc,
        "flow_dc_state": flow_dc_state,
        "state_result": state_result,
        "bottlenecks": bottleneck_df,
    }


def _port_usage(flow_port_dc: pd.DataFrame) -> pd.DataFrame:
    usage = (
        flow_port_dc.groupby(["scenario", "strategy", "from_node"], as_index=False)
        .agg(flow_usd=("flow_usd", "sum"))
        .sort_values("flow_usd", ascending=False)
    )
    totals = usage.groupby(["scenario", "strategy"]) ["flow_usd"].transform("sum")
    usage["share"] = np.where(totals > 0, usage["flow_usd"] / totals, 0.0)
    usage["share_pct"] = usage["share"] * 100.0
    usage = usage.rename(columns={"from_node": "port"})
    return usage


def _modal_usage(flow_port_dc: pd.DataFrame, flow_dc_state: pd.DataFrame, strategy: str, scenario: str) -> pd.DataFrame:
    upstream = flow_port_dc.groupby("mode", as_index=False).agg(flow_usd=("flow_usd", "sum"))
    upstream["leg"] = "port_to_dc"
    downstream = flow_dc_state.groupby("mode", as_index=False).agg(flow_usd=("flow_usd", "sum"))
    downstream["leg"] = "dc_to_state"
    out = pd.concat([upstream, downstream], ignore_index=True)
    out["strategy"] = strategy
    out["scenario"] = scenario
    totals = out.groupby(["strategy", "scenario", "leg"]) ["flow_usd"].transform("sum")
    out["share"] = np.where(totals > 0, out["flow_usd"] / totals, 0.0)
    out["share_pct"] = out["share"] * 100.0
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Module2 network optimization with optional blended objective controls."
    )
    parser.add_argument(
        "--objective-mode",
        choices=["strategy", "mixed", "min_cost_with_fill"],
        default="strategy",
        help="strategy: keep current strategy objective; mixed: blend cost and service; min_cost_with_fill: minimize cost under min fill-rate.",
    )
    parser.add_argument(
        "--service-weight",
        type=float,
        default=0.5,
        help="Service priority weight in [0,1] for mixed mode. 0 = pure cost, 1 = pure service.",
    )
    parser.add_argument(
        "--min-fill-rate",
        type=float,
        default=None,
        help="Optional fill-rate lower bound in [0,1], e.g. 0.78 for 78%%.",
    )
    parser.add_argument(
        "--custom-strategy",
        choices=["cost_min", "resilience_first", "targeted"],
        default=None,
        help="Run one additional custom scenario using the selected base strategy parameters.",
    )
    parser.add_argument(
        "--custom-label",
        type=str,
        default="Custom-Blended",
        help="Display label for the optional custom scenario.",
    )
    parser.add_argument(
        "--weather-stress",
        action="store_true",
        help="Query NOAA NHC for active tropical cyclones and, if any are near a modeled port, "
        "drive the stress-scenario port capacity shock from real storm proximity/intensity "
        "instead of the synthetic West-Coast assumption.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.service_weight <= 1.0:
        raise ValueError("--service-weight must be within [0, 1]")
    if args.min_fill_rate is not None and not 0.0 <= args.min_fill_rate <= 1.0:
        raise ValueError("--min-fill-rate must be within [0, 1]")
    return args


def main() -> None:
    args = _parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = _read_csv("domestic_nodes.csv")
    lanes = _read_csv("domestic_lanes.csv")
    observed_demand = _read_csv("domestic_demand.csv")
    state_demand_raw = _read_csv("domestic_demand_state_assumption.csv")

    state_demand = _build_state_demand_panel(state_demand_raw)

    origins, intl_port, ports, dcs, port_dc, dc_state = _build_network(state_demand, lanes)

    weather_port_multipliers: Optional[Dict[str, float]] = None
    weather_report: Dict[str, object] = {"enabled": args.weather_stress, "active_storms": [], "port_impacts": []}
    if args.weather_stress:
        detector = WeatherDisruptionDetector()
        storms = detector.fetch_active_storms()
        impacts_df = detector.get_port_impacts(storms)
        weather_port_multipliers = detector.get_port_capacity_multipliers(storms)
        weather_report["active_storms"] = storms
        weather_report["port_impacts"] = impacts_df.to_dict(orient="records")
        if impacts_df.empty:
            print("[weather-stress] No active NHC storms within impact radius of modeled ports; "
                  "falling back to synthetic West-Coast stress assumption.")
        else:
            print(f"[weather-stress] Detected {len(impacts_df)} port(s) with active weather impact:")
            print(impacts_df.to_string(index=False))
        with open(REPORT_DIR / "module2_weather_disruption.json", "w", encoding="utf-8") as f:
            json.dump(weather_report, f, indent=2, ensure_ascii=False)

    baseline_cfg = _scenario_config_baseline()
    baseline_pref = OptimizationPreference(mode="strategy", service_weight=0.5, min_fill_rate=None)
    baseline_origins, baseline_l0, baseline_ports, baseline_dcs, baseline_l1, baseline_l2 = _prepare_scenario_network(
        origins,
        intl_port,
        ports,
        dcs,
        port_dc,
        dc_state,
        baseline_cfg,
    )
    baseline_result = _solve_network(
        state_demand,
        baseline_origins,
        baseline_l0,
        baseline_ports,
        baseline_dcs,
        baseline_l1,
        baseline_l2,
        baseline_cfg,
        preference=baseline_pref,
    )

    strategy_outputs: List[Dict[str, object]] = []
    for cfg in _strategy_configs():
        o, l0, p, d, l1, l2 = _prepare_scenario_network(
            origins, intl_port, ports, dcs, port_dc, dc_state, cfg,
            weather_port_multipliers=weather_port_multipliers,
        )
        strategy_outputs.append(_solve_network(state_demand, o, l0, p, d, l1, l2, cfg, preference=baseline_pref))

    custom_result: Optional[Dict[str, object]] = None
    custom_error: Optional[str] = None
    if args.custom_strategy is not None:
        custom_cfg = _strategy_by_key(args.custom_strategy)
        custom_pref = OptimizationPreference(
            mode=args.objective_mode,
            service_weight=args.service_weight,
            min_fill_rate=args.min_fill_rate,
        )
        custom_cfg = StrategyConfig(
            key=f"custom_{custom_cfg.key}",
            name=args.custom_label,
            stress=custom_cfg.stress,
            delay_penalty=custom_cfg.delay_penalty,
            unmet_penalty=custom_cfg.unmet_penalty,
            core_unmet_multiplier=custom_cfg.core_unmet_multiplier,
            noncore_unmet_multiplier=custom_cfg.noncore_unmet_multiplier,
            air_cap_multiplier=custom_cfg.air_cap_multiplier,
            reroute_discount=custom_cfg.reroute_discount,
            min_core_air_share=custom_cfg.min_core_air_share,
        )
        o, l0, p, d, l1, l2 = _prepare_scenario_network(
            origins, intl_port, ports, dcs, port_dc, dc_state, custom_cfg,
            weather_port_multipliers=weather_port_multipliers,
        )
        try:
            custom_result = _solve_network(state_demand, o, l0, p, d, l1, l2, custom_cfg, preference=custom_pref)
            strategy_outputs.append(custom_result)
        except RuntimeError as exc:
            custom_error = str(exc)

    summaries = [baseline_result["summary"]] + [s["summary"] for s in strategy_outputs]
    scenario_comparison = pd.DataFrame(summaries)

    tradeoff = pd.DataFrame([s["summary"] for s in strategy_outputs])[
        [
            "strategy",
            "total_logistics_cost_usd",
            "avg_lead_time_days",
            "fill_rate",
            "air_express_share",
            "delivered_usd",
            "unmet_usd",
        ]
    ].copy()
    tradeoff["fill_rate_pct"] = tradeoff["fill_rate"] * 100.0
    tradeoff["air_express_share_pct"] = tradeoff["air_express_share"] * 100.0

    state_fulfillment = pd.concat([baseline_result["state_result"]] + [s["state_result"] for s in strategy_outputs], ignore_index=True)

    bottlenecks = pd.concat([s["bottlenecks"] for s in strategy_outputs if not s["bottlenecks"].empty], ignore_index=True)

    port_usage = pd.concat(
        [
            _port_usage(baseline_result["flow_port_dc"]),
            *[_port_usage(s["flow_port_dc"]) for s in strategy_outputs],
        ],
        ignore_index=True,
    )

    modal_usage = pd.concat(
        [
            _modal_usage(
                baseline_result["flow_port_dc"],
                baseline_result["flow_dc_state"],
                baseline_result["summary"]["strategy"],
                baseline_result["summary"]["scenario"],
            ),
            *[
                _modal_usage(s["flow_port_dc"], s["flow_dc_state"], s["summary"]["strategy"], s["summary"]["scenario"])
                for s in strategy_outputs
            ],
        ],
        ignore_index=True,
    )

    demand_zone_summary = (
        state_demand[["demand_zone", "state_abbr", "region", "demand_usd"]]
        .rename(columns={"demand_usd": "demand_value_usd"})
        .sort_values("demand_value_usd", ascending=False)
        .reset_index(drop=True)
    )
    total_demand = float(demand_zone_summary["demand_value_usd"].sum())
    demand_zone_summary["demand_share"] = np.where(total_demand > 0, demand_zone_summary["demand_value_usd"] / total_demand, 0.0)
    demand_zone_summary["demand_share_pct"] = demand_zone_summary["demand_share"] * 100.0

    node_capacity_summary = _node_capacity_summary(nodes)
    lane_geo_summary = _lane_geography_summary(lanes)

    state_cap = node_capacity_summary.rename(columns={"state": "state_abbr"})
    state_context = demand_zone_summary.merge(
        state_cap[["state_abbr", "handling_capacity", "storage_capacity", "node_count", "handling_share", "handling_share_pct"]],
        on="state_abbr",
        how="left",
    )
    state_context[["handling_capacity", "storage_capacity", "node_count", "handling_share", "handling_share_pct"]] = (
        state_context[["handling_capacity", "storage_capacity", "node_count", "handling_share", "handling_share_pct"]].fillna(0.0)
    )
    state_context["supply_to_demand_ratio"] = np.where(
        state_context["demand_value_usd"] > 0,
        state_context["handling_capacity"] / state_context["demand_value_usd"],
        0.0,
    )

    baseline_result["flow_intl_port"].to_csv(REPORT_DIR / "module2_baseline_intl_flow_allocation.csv", index=False)
    strategy_outputs[0]["flow_intl_port"].to_csv(REPORT_DIR / "module2_stress_cost_intl_flow_allocation.csv", index=False)
    strategy_outputs[1]["flow_intl_port"].to_csv(REPORT_DIR / "module2_stress_resilience_intl_flow_allocation.csv", index=False)
    strategy_outputs[2]["flow_intl_port"].to_csv(REPORT_DIR / "module2_stress_targeted_intl_flow_allocation.csv", index=False)

    baseline_result["flow_port_dc"].to_csv(REPORT_DIR / "module2_baseline_flow_allocation.csv", index=False)
    strategy_outputs[0]["flow_port_dc"].to_csv(REPORT_DIR / "module2_stress_cost_flow_allocation.csv", index=False)
    strategy_outputs[1]["flow_port_dc"].to_csv(REPORT_DIR / "module2_stress_resilience_flow_allocation.csv", index=False)
    strategy_outputs[2]["flow_port_dc"].to_csv(REPORT_DIR / "module2_stress_targeted_flow_allocation.csv", index=False)

    scenario_comparison.to_csv(REPORT_DIR / "module2_scenario_comparison.csv", index=False)
    tradeoff.to_csv(REPORT_DIR / "module2_strategy_tradeoff.csv", index=False)
    state_fulfillment.to_csv(REPORT_DIR / "module2_state_fulfillment_by_strategy.csv", index=False)
    bottlenecks.to_csv(REPORT_DIR / "module2_bottleneck_shadow_prices.csv", index=False)
    port_usage.to_csv(REPORT_DIR / "module2_port_rerouting_summary.csv", index=False)
    modal_usage.to_csv(REPORT_DIR / "module2_modal_shift_summary.csv", index=False)

    demand_zone_summary.to_csv(REPORT_DIR / "module2_demand_zone_summary.csv", index=False)
    node_capacity_summary.to_csv(REPORT_DIR / "module2_node_state_capacity_summary.csv", index=False)
    lane_geo_summary.to_csv(REPORT_DIR / "module2_lane_geography_summary.csv", index=False)
    state_context.to_csv(REPORT_DIR / "module2_state_context_summary.csv", index=False)

    model_metrics = pd.DataFrame(
        {
            "metric": [
                "observed_total_demand_usd",
                "state_panel_total_demand_usd",
                "state_count",
                "lane_rows",
                "node_rows",
            ],
            "value": [
                float(observed_demand["demand_value_usd"].sum()),
                total_demand,
                float(state_demand["state_abbr"].nunique()),
                float(len(lanes)),
                float(len(nodes)),
            ],
        }
    )
    model_metrics.to_csv(REPORT_DIR / "module2_model_metrics.csv", index=False)

    report = {
        "model": {
            "type": "three-leg min-cost flow with multimodal rerouting",
            "flow_conservation": "overseas->port, port->dc, and dc->state equality constraints",
            "stress_design": "PORT_LA_LB shock plus overseas reroute and optional air scaling",
        },
        "inputs": {
            "state_rows": int(len(state_demand)),
            "state_total_demand_usd": total_demand,
            "lane_rows": int(len(lanes)),
            "node_rows": int(len(nodes)),
        },
        "baseline": baseline_result["summary"],
        "stress_cost_min": strategy_outputs[0]["summary"],
        "stress_resilience_first": strategy_outputs[1]["summary"],
        "stress_targeted": strategy_outputs[2]["summary"],
        "custom_scenario": custom_result["summary"] if custom_result is not None else None,
        "custom_scenario_error": custom_error,
        "top_state_shortfalls_resilience": strategy_outputs[1]["state_result"].sort_values("shortfall_usd", ascending=False).head(10).to_dict(orient="records"),
        "top_bottlenecks_resilience": bottlenecks[bottlenecks["strategy"] == "Resilience-First"].head(15).to_dict(orient="records"),
    }

    with open(REPORT_DIR / "module2_model_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Module2 optimization complete")
    print(
        json.dumps(
            {
                "baseline_fill_rate": baseline_result["summary"]["fill_rate"],
                "stress_cost_fill_rate": strategy_outputs[0]["summary"]["fill_rate"],
                "stress_resilience_fill_rate": strategy_outputs[1]["summary"]["fill_rate"],
                "stress_targeted_fill_rate": strategy_outputs[2]["summary"]["fill_rate"],
                "tradeoff_rows": int(len(tradeoff)),
                "state_fulfillment_rows": int(len(state_fulfillment)),
                "custom_scenario_status": "ok" if custom_result is not None else (custom_error or "not_requested"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
