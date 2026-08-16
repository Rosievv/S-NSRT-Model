from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from risk_propagation.news_signal_adapter import (
    NewsRiskSignal,
    news_dynamic_severity,
    run_news_triggered_scenario,
)


def _signal(event_type: str, alarm_score: float = 0.9) -> NewsRiskSignal:
    return NewsRiskSignal(
        event_key=f"test_{event_type}",
        signal_date=pd.Timestamp("2021-03-15"),
        event_type=event_type,
        countries=["Taiwan"],
        affected_objects=["wafer_fabrication"],
        alarm_score=alarm_score,
        headline="Taiwan water supplies cut for semiconductor capacity",
        corroborating_sources=1,
    )


def test_event_types_produce_different_dynamic_severity() -> None:
    earthquake = news_dynamic_severity(_signal("earthquake"))
    drought = news_dynamic_severity(_signal("drought"))

    assert earthquake > drought
    assert 0 < drought < 1


def test_propagation_uses_only_lagged_government_data() -> None:
    dates = pd.to_datetime(["2020-12-01", "2021-01-01", "2021-02-01", "2021-03-01"])
    trade_df = pd.DataFrame(
        [
            {"date": date, "hs_code": hs_code, "country": country, "value_usd": value}
            for date in dates
            for hs_code in ["854231", "854232", "854239"]
            for country, value in [("Taiwan", 80.0), ("Japan", 20.0)]
        ]
    )

    result = run_news_triggered_scenario(
        trade_df,
        _signal("drought"),
        government_data_lag_months=1,
        lookback_months=12,
    )

    assert result.government_data_available_through == pd.Timestamp("2021-02-01")
    assert result.government_data_available_through < result.signal.signal_date
    assert result.propagation.supply_gap_pct > 0
    assert result.propagation.scenario_name == "news_triggered_test_drought"
    assert result.propagation.details["substitution_elasticity"] < 0.3