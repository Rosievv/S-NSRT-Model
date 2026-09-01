from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "analysis"))

from build_event_listener_v2 import _available_support, extract_event


def _event(title: str, source: str = "Reuters") -> pd.Series:
    return pd.Series(
        {
            "title": title,
            "source": source,
            "url": "https://example.com/event",
        }
    )


def test_panama_canal_drought_routes_to_transport() -> None:
    result = extract_event(_event("Panama canal says it will slash booking slots due to drought"))

    assert "drought" in result["predicted_causes"]
    assert "capacity_restriction" in result["predicted_events"]
    assert result["v2_impact_target"] == "transport_channel"
    assert result["predicted_object_top3"] == "canal"


def test_benign_investment_news_remains_non_event() -> None:
    result = extract_event(_event("Malaysia: A soaring semiconductor opportunity"))

    assert result["predicted_events"] == "unknown"
    assert result["v2_event_type"] == "other"
    assert result["v2_impact_target"] == "supply_source"


def test_support_after_confirmation_is_excluded() -> None:
    row = pd.Series(
        {
            "event_date": "2021-03-24",
            "confirmation_date": "2021-03-23",
            "official_support_sources": (
                '[{"date": "2021-03-22", "institution": "A"}, '
                '{"date": "2021-03-29", "institution": "B"}]'
            ),
        }
    )

    support = _available_support(row)

    assert [item["institution"] for item in support] == ["A"]