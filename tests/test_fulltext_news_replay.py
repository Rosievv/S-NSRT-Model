from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "analysis"))

from build_fulltext_news_replay import _publication_date, build_fulltext_replay


def test_publication_date_prefers_parsed_article_date() -> None:
    row = pd.Series(
        {
            "parsed_publish_date": "2021-08-24T07:24:03+08:00",
            "published_at": "20210830T091500Z",
        }
    )

    assert _publication_date(row) == pd.Timestamp("2021-08-24")


def test_publication_date_falls_back_to_discovery_timestamp() -> None:
    row = pd.Series({"parsed_publish_date": None, "published_at": "20210830T091500Z"})

    assert _publication_date(row) == pd.Timestamp("2021-08-30")


def test_event_summary_uses_fulltext_match_columns() -> None:
    _, event_result, metrics = build_fulltext_replay()
    covered = event_result.loc[event_result["has_model_eligible_body"]]

    assert metrics["earliest_body_location_match_rate_pct"] == round(
        float(covered["fulltext_location_match"].mean() * 100), 2
    )