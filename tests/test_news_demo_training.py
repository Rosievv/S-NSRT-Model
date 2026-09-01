from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from risk_propagation.news_demo_training import ALERT_FEATURE_COLUMNS, build_alert_features


def test_alert_features_exclude_event_identity_and_dates() -> None:
    rows = pd.DataFrame(
        [
            {
                "event_key": "known_future_event",
                "event_date": "2025-01-01",
                "v2_confidence": 0.8,
                "independent_source_count": 2,
                "predicted_events": "capacity_restriction",
                "v2_location": "Taiwan",
                "predicted_object_top3": "wafer_fabrication",
                "source_tier": "wire",
            }
        ]
    )

    features = build_alert_features(rows)

    assert list(features.columns) == ALERT_FEATURE_COLUMNS
    assert "event_key" not in features
    assert "event_date" not in features
    assert features.notna().all().all()