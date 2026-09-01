import pandas as pd

from src.demand_forecasting import (
    add_shortage_metrics,
    apply_gap_elasticity,
    apply_risk_gating,
    adjust_supply_quantiles,
    attach_module1_risk,
    build_inventory_scenarios,
)


def _forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2024-01-01"]),
            "forecast_date": pd.to_datetime(["2024-04-01"]),
            "hs_code": ["854231"],
            "q10": [80.0],
            "q50": [100.0],
            "q90": [120.0],
            "actual_supply": [70.0],
            "demand_proxy": [100.0],
        }
    )


def test_risk_uses_origin_quarter_and_preserves_quantile_order() -> None:
    risk = pd.DataFrame(
        {
            "quarter_end": ["2024-03-31", "2024-06-30"],
            "scenario": ["top_1_supplier_failure"] * 2,
            "predicted_gap_pct": [10.0, 90.0],
            "lower_gap_pct": [5.0, 80.0],
            "upper_gap_pct": [20.0, 95.0],
        }
    )

    adjusted = adjust_supply_quantiles(attach_module1_risk(_forecast(), risk))

    assert adjusted.loc[0, "risk_q10"] == 64.0
    assert adjusted.loc[0, "risk_q50"] == 90.0
    assert adjusted.loc[0, "risk_q90"] == 114.0


def test_shortage_and_inventory_scenarios_use_risk_adjusted_supply() -> None:
    frame = _forecast().assign(
        predicted_gap_pct=10.0,
        lower_gap_pct=5.0,
        upper_gap_pct=20.0,
    )
    scored = add_shortage_metrics(adjust_supply_quantiles(frame), demand_col="demand_proxy")
    inventory = build_inventory_scenarios(scored, initial_months=(0.05, 1.0))

    assert scored.loc[0, "risk_shortfall_usd"] == 10.0
    assert inventory.loc[0, "risk_stockout_flag"]
    assert not inventory.loc[1, "risk_stockout_flag"]


def test_inventory_stress_accumulates_over_forecast_horizon() -> None:
    frame = _forecast().assign(
        horizon_months=3,
        predicted_gap_pct=10.0,
        lower_gap_pct=5.0,
        upper_gap_pct=20.0,
    )
    scored = add_shortage_metrics(adjust_supply_quantiles(frame), demand_col="demand_proxy")

    inventory = build_inventory_scenarios(scored, initial_months=(0.2,))

    assert inventory.loc[0, "risk_cumulative_shortfall_usd"] == 10.0
    assert inventory.loc[0, "risk_stockout_step"] == 3
    assert inventory.loc[0, "risk_stockout_flag"]


def test_risk_gate_activates_only_above_threshold() -> None:
    frame = pd.DataFrame(
        {
            "predicted_gap_pct": [1.5, 2.0, 4.0],
            "lower_gap_pct": [1.0, 1.5, 3.0],
            "upper_gap_pct": [2.0, 2.5, 5.0],
        }
    )

    gated = apply_risk_gating(frame, threshold_pct=2.0, mode="hard_gate")

    assert gated["trigger_active"].tolist() == [False, True, True]
    assert gated["effective_predicted_gap_pct"].tolist() == [0.0, 2.0, 4.0]


def test_gap_elasticity_reduces_long_horizon_discount_strength() -> None:
    frame = pd.DataFrame(
        {
            "q10": [80.0, 80.0],
            "q50": [100.0, 100.0],
            "q90": [120.0, 120.0],
            "horizon_months": [1, 6],
            "effective_predicted_gap_pct": [10.0, 10.0],
            "effective_lower_gap_pct": [5.0, 5.0],
            "effective_upper_gap_pct": [20.0, 20.0],
        }
    )

    elastic = apply_gap_elasticity(frame, elasticity_by_horizon={1: 0.1, 6: 0.5})
    adjusted = adjust_supply_quantiles(elastic)

    assert adjusted.loc[0, "risk_q50"] == 91.0
    assert adjusted.loc[1, "risk_q50"] == 95.0