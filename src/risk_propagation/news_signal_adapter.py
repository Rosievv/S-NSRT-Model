"""Convert time-available news signals into Module 1 stress scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .propagation_engine import HS_ELASTICITY_MAP, DisruptionResult, PropagationEngine


EVENT_SEVERITY_PRIORS: Dict[str, float] = {
    "pandemic": 0.15,
    "earthquake": 0.55,
    "flood": 0.50,
    "export_control": 0.20,
    "drought": 0.25,
    "power_shortage": 0.45,
    "logistics_disruption": 0.15,
    "other": 0.10,
}

EVENT_TYPE_MULTIPLIERS: Dict[str, float] = {
    "pandemic": 1.0,
    "earthquake": 1.4,
    "flood": 1.2,
    "export_control": 1.1,
    "drought": 1.3,
    "power_shortage": 1.8,
    "logistics_disruption": 1.0,
    "other": 1.0,
}

EVENT_SUBSTITUTION_MULTIPLIERS: Dict[str, float] = {
    "pandemic": 0.90,
    "earthquake": 0.35,
    "flood": 0.35,
    "export_control": 0.50,
    "drought": 0.50,
    "power_shortage": 0.25,
    "logistics_disruption": 0.75,
    "other": 0.60,
}

INITIAL_SUBSTITUTION_SCALES: Dict[str, float] = {
    "pandemic": 0.75,
    "earthquake": 0.20,
    "flood": 0.20,
    "export_control": 0.45,
    "drought": 0.45,
    "power_shortage": 0.20,
    "logistics_disruption": 0.50,
    "other": 0.45,
}

OBJECT_HS_MAP: Dict[str, List[str]] = {
    "semiconductor_fabrication": ["854231", "854232", "854239", "848620"],
    "semiconductor_materials": ["381800", "280461", "280469"],
    "electronics": ["854231", "854232", "854239"],
    "memory_components": ["854232"],
    "storage_devices": ["854232"],
    "electronics_assembly": ["854231", "854232", "854239"],
    "electronics_clusters": ["854231", "854232", "854239"],
    "wafer_fabrication": ["854231", "854232", "854239", "848620"],
    "assembly_and_test": ["854231", "854232", "854239"],
    "semiconductor_packaging": ["854231", "854232", "854239"],
}


@dataclass(frozen=True)
class NewsRiskSignal:
    """A news event as known at publication time."""

    event_key: str
    signal_date: pd.Timestamp
    event_type: str
    countries: List[str]
    affected_objects: List[str]
    alarm_score: float
    headline: str = ""
    corroborating_sources: int = 0


@dataclass(frozen=True)
class NewsTriggeredResult:
    """A news-triggered stress result with its data-vintage metadata."""

    signal: NewsRiskSignal
    dynamic_severity: float
    hs_codes: List[str]
    government_data_available_through: pd.Timestamp
    government_data_start: pd.Timestamp
    propagation: DisruptionResult


def infer_hs_codes(affected_objects: List[str]) -> List[str]:
    """Map extracted supply-chain objects to the HS slice used for propagation."""
    codes: List[str] = []
    for affected_object in affected_objects:
        for hs_code in OBJECT_HS_MAP.get(affected_object, []):
            if hs_code not in codes:
                codes.append(hs_code)
    return codes


def news_dynamic_severity(signal: NewsRiskSignal) -> float:
    """Estimate scenario severity without treating confidence as realized loss."""
    prior = EVENT_SEVERITY_PRIORS.get(signal.event_type, EVENT_SEVERITY_PRIORS["other"])
    event_multiplier = EVENT_TYPE_MULTIPLIERS.get(signal.event_type, 1.0)
    normalized_alarm = min(max((float(signal.alarm_score) - 0.65) / 0.35, 0.0), 1.0)
    alarm_multiplier = 0.75 + 0.25 * normalized_alarm

    headline = signal.headline.lower()
    if "total lockdown" in headline or "complete shutdown" in headline:
        scope_multiplier = 1.60
    elif "lockdown" in headline or "capacity" in headline or "water supplies cut" in headline:
        scope_multiplier = 1.20
    else:
        scope_multiplier = 1.0

    corroboration_multiplier = 1.0 + 0.05 * min(max(signal.corroborating_sources, 0), 2)
    severity = prior * event_multiplier * alarm_multiplier * scope_multiplier * corroboration_multiplier
    return round(min(max(severity, 0.01), 1.0), 3)


def run_news_triggered_scenario(
    trade_df: pd.DataFrame,
    signal: NewsRiskSignal,
    government_data_lag_months: int = 1,
    lookback_months: int = 24,
) -> NewsTriggeredResult:
    """Run Module 1 using only government observations available before the signal."""
    if not signal.countries:
        raise ValueError("A news-triggered scenario requires at least one affected country")
    if not 0.0 <= signal.alarm_score <= 1.0:
        raise ValueError("alarm_score must be between 0 and 1")
    if government_data_lag_months < 0:
        raise ValueError("government_data_lag_months cannot be negative")
    if lookback_months < 1:
        raise ValueError("lookback_months must be positive")

    signal_date = pd.Timestamp(signal.signal_date)
    signal_month = signal_date.to_period("M").to_timestamp()
    available_through = signal_month - pd.DateOffset(months=government_data_lag_months)
    data_start = available_through - pd.DateOffset(months=lookback_months - 1)

    panel = trade_df.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["hs_code"] = panel["hs_code"].astype(str)
    panel = panel.loc[panel["date"].between(data_start, available_through)].copy()

    hs_codes = infer_hs_codes(signal.affected_objects)
    if hs_codes:
        panel = panel.loc[panel["hs_code"].isin(hs_codes)].copy()
    if panel.empty:
        raise ValueError(f"No lagged trade observations available for {signal.event_key}")

    severity = news_dynamic_severity(signal)
    substitution_scale = (
        EVENT_SUBSTITUTION_MULTIPLIERS.get(signal.event_type, 0.60)
        * INITIAL_SUBSTITUTION_SCALES.get(signal.event_type, 0.45)
    )
    scaled_hs_elasticity = {
        hs_code: elasticity * substitution_scale
        for hs_code, elasticity in HS_ELASTICITY_MAP.items()
    }
    engine = PropagationEngine(
        trade_df=panel,
        substitution_elasticity=0.3 * substitution_scale,
        hs_elasticity_map=scaled_hs_elasticity,
        use_weighted_substitution=True,
        concentration_penalty_lambda=0.5,
        geo_penalty_factor=0.6,
    )
    propagation = engine.simulate_node_shock(
        countries=signal.countries,
        severity=severity,
        scenario_name=f"news_triggered_{signal.event_key}",
    )
    return NewsTriggeredResult(
        signal=signal,
        dynamic_severity=severity,
        hs_codes=hs_codes,
        government_data_available_through=available_through,
        government_data_start=data_start,
        propagation=propagation,
    )