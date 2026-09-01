import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "data_collection"))

from fetch_disruption_news import (
    _deduplicate,
    _domain_tier,
    _sha256,
    _source_from_title,
    _validate_body,
    load_cached_articles,
)


def test_source_is_split_from_google_news_title() -> None:
    title, source = _source_from_title("Malaysia chip output disrupted - Reuters")

    assert title == "Malaysia chip output disrupted"
    assert source == "Reuters"


def test_short_navigation_page_is_not_full_text() -> None:
    valid, reason = _validate_body(
        "What To Read Next",
        {"required_terms": ["malaysia"], "impact_terms": ["chip"]},
    )

    assert not valid
    assert reason == "body_too_short:17"


def test_deduplicate_uses_normalized_title_and_source() -> None:
    records = [
        {"title": "Malaysia Chip Shock", "source": "Reuters", "url": "https://example.com/1"},
        {"title": "Malaysia: Chip Shock", "source": "REUTERS", "url": "https://example.com/2"},
    ]

    assert _deduplicate(records) == [records[0]]


def test_source_quality_is_based_on_resolved_domain() -> None:
    assert _domain_tier("https://www.straitstimes.com/asia/example") == "mainstream"
    assert _domain_tier("https://www.bbc.co.uk/news/example") == "mainstream"
    assert _domain_tier("https://www.scmp.com/news/example") == "mainstream"
    assert _domain_tier("https://www.tomshardware.com/news/example") == "industry"
    assert _domain_tier("https://unknown.example/article") == "other"


def test_cached_article_requires_matching_body_hash(tmp_path: Path) -> None:
    body = ("Malaysia semiconductor production lockdown disruption. " * 10).encode()
    text_path = tmp_path / "article.txt"
    text_path.write_bytes(body)
    metadata = {
        "status": "saved",
        "url": "https://www.straitstimes.com/asia/example",
        "resolved_url": "https://www.straitstimes.com/asia/example",
        "full_text_path": str(text_path.relative_to(ROOT_DIR)) if text_path.is_relative_to(ROOT_DIR) else str(text_path),
        "full_text_sha256": _sha256(body),
    }
    (tmp_path / "article.json").write_text(json.dumps(metadata), encoding="utf-8")

    cached = load_cached_articles(
        tmp_path,
        {"required_terms": ["malaysia"], "impact_terms": ["semiconductor"]},
    )

    assert len(cached) == 1
    assert cached[0]["eligible_for_model"] is True

    text_path.write_text("tampered body", encoding="utf-8")

    assert load_cached_articles(
        tmp_path,
        {"required_terms": ["malaysia"], "impact_terms": ["semiconductor"]},
    ) == []