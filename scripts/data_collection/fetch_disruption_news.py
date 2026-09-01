#!/usr/bin/env python3
"""Discover and locally archive full-text disruption news articles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree as ET

import newspaper
import pandas as pd
import requests
from gdeltdoc import Filters, GdeltDoc, near


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "disruption_news"
USER_AGENT = "S-NSRT-Model/1.0 full-text-news-evidence-collector"
MIN_BODY_CHARS = 300
REQUEST_INTERVAL_SECONDS = 5.5

MAINSTREAM_DOMAINS = {
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "bloomberg.com",
    "cnbc.com",
    "cnn.com",
    "ft.com",
    "japantimes.co.jp",
    "nikkei.com",
    "reuters.com",
    "scmp.com",
    "straitstimes.com",
    "theguardian.com",
    "wsj.com",
}
INDUSTRY_DOMAINS = {
    "supplychaindive.com",
    "tomshardware.com",
}
VERIFIED_LOCAL_DOMAINS = {
    "malaymail.com",
    "nst.com.my",
    "soyacincau.com",
}

EVENT_QUERIES = {
    "japan_earthquake_2011": {
        "start_date": "2011-03-01",
        "end_date": "2011-05-01",
        "google_queries": ["Japan earthquake semiconductor supply chain"],
        "gdelt_near": [(20, "Japan", "earthquake")],
        "required_terms": ["japan"],
        "impact_terms": ["earthquake", "tsunami", "semiconductor", "factory", "supply"],
        "seed_urls": [{"url": "https://www.bbc.com/news/world-asia-pacific-12709598", "title": "Japan earthquake: Tsunami hits north-east", "source": "BBC News", "published_at": "2011-03-11"}],
    },
    "thai_flood_2011": {
        "start_date": "2011-07-01",
        "end_date": "2011-12-01",
        "google_queries": ["Thailand floods electronics factories supply chain"],
        "gdelt_near": [(20, "Thailand", "flood")],
        "required_terms": ["thailand"],
        "impact_terms": ["flood", "factory", "electronics", "semiconductor", "supply"],
        "seed_urls": [{"url": "http://www.bangkokpost.com/news/local/249670/north-northeast-inundated-by-effects-of-nock-ten", "title": "North, Northeast inundated by effects of Nock-ten", "source": "Bangkok Post", "published_at": "2011-08-01"}],
    },
    "japan_export_controls_2019": {
        "start_date": "2019-06-15",
        "end_date": "2019-10-01",
        "google_queries": ["Japan export controls South Korea semiconductor materials"],
        "gdelt_near": [(20, "Japan", "export controls")],
        "required_terms": ["japan"],
        "impact_terms": ["korea", "export", "semiconductor", "materials", "license"],
        "seed_urls": [{"url": "https://www.meti.go.jp/english/press/2019/0701_001.html", "title": "Update of METI licensing policies and procedures on exports of controlled items to the Republic of Korea", "source": "METI Japan", "published_at": "2019-07-01"}],
    },
    "covid_q1_2020": {
        "start_date": "2020-01-01",
        "end_date": "2020-05-01",
        "google_queries": ["China coronavirus lockdown electronics supply chain"],
        "gdelt_near": [(20, "China", "lockdown")],
        "required_terms": ["china"],
        "impact_terms": ["coronavirus", "covid", "lockdown", "factory", "supply"],
        "seed_urls": [{"url": "https://www.bbc.com/news/world-asia-china-51217455", "title": "China coronavirus: Lockdown measures rise across Hubei province", "source": "BBC News", "published_at": "2020-01-23"}],
    },
    "taiwan_drought_2021": {
        "start_date": "2021-02-15",
        "end_date": "2021-06-01",
        "google_queries": ["Taiwan drought chip output water restrictions"],
        "gdelt_near": [(20, "Taiwan", "drought")],
        "required_terms": ["taiwan"],
        "impact_terms": ["drought", "water", "chip", "semiconductor", "production"],
        "seed_urls": [{"url": "https://www.reuters.com/article/us-taiwan-drought/taiwan-tightens-water-use-as-drought-threatens-chip-output-idUSKBN2BG1K4", "title": "Taiwan tightens water use as drought threatens chip output", "source": "Reuters", "published_at": "2021-03-24"}],
    },
    "malaysia_asia_shock_2021": {
        "start_date": "2021-05-01",
        "end_date": "2021-10-01",
        "google_queries": [
            "Malaysia semiconductor lockdown",
            "Malaysia chip factories lockdown",
            "Malaysia semiconductor shortage Covid",
        ],
        "gdelt_near": [(15, "Malaysia", "semiconductor"), (15, "Malaysia", "chip")],
        "required_terms": ["malaysia"],
        "impact_terms": [
            "semiconductor",
            "chip",
            "factory",
            "production",
            "lockdown",
            "covid",
        ],
        "seed_urls": [
            {
                "url": "https://www.nst.com.my/news/nation/2021/05/694006/malaysia-goes-under-full-lockdown-again-tuesday",
                "title": "Malaysia goes under full lockdown again from Tuesday",
                "source": "New Straits Times",
                "published_at": "2021-05-28T20:16:54+08:00",
            },
        ],
    },
    "beirut_port_explosion_2020": {
        "start_date": "2020-08-01",
        "end_date": "2020-09-01",
        "google_queries": ["Beirut port explosion supply chain disruption"],
        "gdelt_near": [(15, "Beirut", "port explosion")],
        "required_terms": ["beirut"],
        "impact_terms": ["port", "explosion", "destroyed", "shipping", "supply"],
        "seed_urls": [{"url": "https://news.un.org/en/story/2020/08/1069542", "title": "Lebanon: UN actively assisting in response to huge explosions at Beirut port", "source": "UN News", "published_at": "2020-08-04"}],
    },
    "suez_ever_given_2021": {
        "start_date": "2021-03-20",
        "end_date": "2021-04-15",
        "google_queries": ["Ever Given Suez Canal blocked shipping"],
        "gdelt_near": [(15, "Suez Canal", "blocked")],
        "required_terms": ["suez"],
        "impact_terms": ["canal", "blocked", "ship", "shipping", "ever given"],
        "seed_urls": [{"url": "https://www.bbc.com/news/world-middle-east-56505413", "title": "Egypt Suez Canal blocked by huge container ship", "source": "BBC News", "published_at": "2021-03-24"}],
    },
    "yantian_port_covid_2021": {
        "start_date": "2021-05-20",
        "end_date": "2021-07-15",
        "google_queries": ["Yantian port Covid congestion container shipping"],
        "gdelt_near": [(15, "Yantian", "port")],
        "required_terms": ["yantian"],
        "impact_terms": ["port", "congestion", "container", "shipping", "covid"],
        "seed_urls": [{"url": "https://www.reuters.com/world/asia-pacific/major-shipping-firms-warn-worsening-congestion-chinas-yantian-port-2021-06-03/", "title": "Major shipping firms warn of worsening congestion at China Yantian port", "source": "Reuters", "published_at": "2021-06-03"}],
    },
    "panama_canal_drought_2023": {
        "start_date": "2023-08-01",
        "end_date": "2023-12-01",
        "google_queries": ["Panama Canal drought booking slots restrictions"],
        "gdelt_near": [(15, "Panama Canal", "drought")],
        "required_terms": ["panama"],
        "impact_terms": ["canal", "drought", "transit", "booking", "shipping"],
        "seed_urls": [{"url": "https://www.reuters.com/business/panama-canal-says-will-slash-booking-slots-due-drought-2023-10-31/", "title": "Panama canal says will slash booking slots due to drought", "source": "Reuters", "published_at": "2023-10-31"}],
    },
    "red_sea_rerouting_2023": {
        "start_date": "2023-11-15",
        "end_date": "2024-01-15",
        "google_queries": ["Red Sea shipping attacks Maersk pause rerouting"],
        "gdelt_near": [(15, "Red Sea", "shipping")],
        "required_terms": ["red sea"],
        "impact_terms": ["ship", "shipping", "maersk", "pause", "reroute", "attack"],
        "seed_urls": [{"url": "https://www.reuters.com/world/maersk-pause-all-container-shipments-through-red-sea-2023-12-15/", "title": "Maersk to pause all container ship traffic through the Red Sea", "source": "Reuters", "published_at": "2023-12-15"}],
    },
    "baltimore_bridge_collapse_2024": {
        "start_date": "2024-03-20",
        "end_date": "2024-05-01",
        "google_queries": ["Baltimore bridge collapse port shipping channel"],
        "gdelt_near": [(15, "Baltimore", "bridge collapse")],
        "required_terms": ["baltimore"],
        "impact_terms": ["bridge", "collapse", "port", "shipping", "channel"],
        "seed_urls": [{"url": "https://www.bbc.com/news/world-us-canada-68663318", "title": "Baltimore bridge collapse: Six presumed dead after ship collides with bridge", "source": "BBC News", "published_at": "2024-03-26"}],
    },
}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:90] or "article"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_from_title(title: str) -> tuple[str, str]:
    parts = title.rsplit(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return title.strip(), urlparse(title).netloc or "Unknown"


def discover_google_news(event_key: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for query in config["google_queries"]:
        dated_query = f'{query} after:{config["start_date"]} before:{config["end_date"]}'
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(dated_query)}&hl=en-US&gl=US&ceid=US:en"
        )
        attempt: dict[str, Any] = {
            "provider": "google_news_rss",
            "query": dated_query,
            "url": url,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall(".//item")
            attempt.update({"status": "success", "records": len(items)})
            for item in items:
                raw_title = item.findtext("title") or "Unknown"
                title, source = _source_from_title(raw_title)
                published = item.findtext("pubDate")
                records.append(
                    {
                        "event_key": event_key,
                        "discovery_provider": "google_news_rss",
                        "discovery_query": dated_query,
                        "title": title,
                        "source": source,
                        "published_at": parsedate_to_datetime(published).isoformat() if published else None,
                        "url": item.findtext("link"),
                    }
                )
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            attempt.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        attempts.append(attempt)
    return records, attempts


def discover_gdelt(event_key: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    client = GdeltDoc()
    for distance, first, second in config["gdelt_near"]:
        query = near(distance, first, second)
        attempt: dict[str, Any] = {
            "provider": "gdelt_doc_api",
            "query": query,
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            frame = client.article_search(
                Filters(
                    near=query,
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    num_records=250,
                )
            )
            attempt.update({"status": "success", "records": len(frame)})
            for row in frame.to_dict("records"):
                records.append(
                    {
                        "event_key": event_key,
                        "discovery_provider": "gdelt_doc_api",
                        "discovery_query": query,
                        "title": row.get("title"),
                        "source": row.get("domain"),
                        "published_at": row.get("seendate"),
                        "url": row.get("url"),
                    }
                )
        except (ValueError, requests.RequestException) as exc:
            attempt.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        attempts.append(attempt)
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return records, attempts


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (
            re.sub(r"\W+", "", str(record.get("title", "")).lower()),
            str(record.get("source", "")).lower(),
        )
        if key not in seen and record.get("url"):
            seen.add(key)
            deduplicated.append(record)
    return deduplicated


def _candidate_score(record: dict[str, Any], config: dict[str, Any]) -> int:
    text = f'{record.get("title", "")} {record.get("source", "")}'.lower()
    score = 0
    score += 4 * sum(term in text for term in config["required_terms"])
    score += 2 * sum(term in text for term in config["impact_terms"])
    if any(source in text for source in ["reuters", "bbc", "cnbc", "ap news", "financial times", "nikkei", "bloomberg"]):
        score += 3
    return score


def _domain_tier(url: str) -> str:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if any(domain == value or domain.endswith(f".{value}") for value in MAINSTREAM_DOMAINS):
        return "mainstream"
    if any(domain == value or domain.endswith(f".{value}") for value in INDUSTRY_DOMAINS):
        return "industry"
    if any(domain == value or domain.endswith(f".{value}") for value in VERIFIED_LOCAL_DOMAINS):
        return "verified_local"
    return "other"


def _validate_body(text: str, config: dict[str, Any]) -> tuple[bool, str]:
    normalized = " ".join(text.split()).lower()
    if len(normalized) < MIN_BODY_CHARS:
        return False, f"body_too_short:{len(normalized)}"
    if not all(term in normalized for term in config["required_terms"]):
        return False, "missing_required_event_terms"
    if not any(term in normalized for term in config["impact_terms"]):
        return False, "missing_disruption_impact_terms"
    blocked_markers = [
        "enable javascript and cookies to continue",
        "subscribe to continue reading",
        "what to read next",
        "access denied",
    ]
    if any(marker in normalized for marker in blocked_markers) and len(normalized) < 1000:
        return False, "blocked_or_navigation_page"
    return True, "validated_full_text"


def scrape_article(record: dict[str, Any], config: dict[str, Any], event_dir: Path) -> dict[str, Any]:
    result = {
        **record,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    try:
        article = newspaper.article(str(record["url"]), request_timeout=30)
        article.download()
        article.parse()
        body = article.text or ""
        valid, validation = _validate_body(body, config)
        result.update(
            {
                "resolved_url": article.url,
                "parsed_title": article.title,
                "parsed_publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "body_char_count": len(body),
                "validation": validation,
            }
        )
        if not valid:
            return result

        url_suffix = hashlib.sha256(article.url.encode("utf-8")).hexdigest()[:10]
        filename_base = f'{_slug(article.title or str(record.get("title", "article")))}-{url_suffix}'
        html_bytes = (article.html or "").encode("utf-8")
        text_bytes = body.encode("utf-8")
        html_path = event_dir / f"{filename_base}.html"
        text_path = event_dir / f"{filename_base}.txt"
        metadata_path = event_dir / f"{filename_base}.json"
        html_path.write_bytes(html_bytes)
        text_path.write_bytes(text_bytes)
        source_tier = _domain_tier(article.url)
        result.update(
            {
                "status": "saved",
            "source_tier": source_tier,
            "eligible_for_model": source_tier in {"mainstream", "industry", "verified_local"},
                "raw_html_path": str(html_path.relative_to(ROOT_DIR)),
                "full_text_path": str(text_path.relative_to(ROOT_DIR)),
                "metadata_path": str(metadata_path.relative_to(ROOT_DIR)),
                "raw_html_sha256": _sha256(html_bytes),
                "full_text_sha256": _sha256(text_bytes),
            }
        )
        metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:
        result.update({"validation": "parser_or_download_failure", "error": f"{type(exc).__name__}: {exc}"})
    return result


def load_cached_articles(event_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    cached: list[dict[str, Any]] = []
    for metadata_path in event_dir.glob("*.json"):
        if metadata_path.name == "manifest.json":
            continue
        try:
            result = json.loads(metadata_path.read_text(encoding="utf-8"))
            text_path_value = result.get("full_text_path")
            expected_hash = result.get("full_text_sha256")
            if result.get("status") != "saved" or not text_path_value or not expected_hash:
                continue
            text_path = ROOT_DIR / text_path_value
            text_bytes = text_path.read_bytes()
            if _sha256(text_bytes) != expected_hash:
                continue
            valid, validation = _validate_body(text_bytes.decode("utf-8"), config)
            if not valid:
                continue
            resolved_url = str(result.get("resolved_url") or result.get("url") or "")
            source_tier = _domain_tier(resolved_url)
            result.update(
                {
                    "validation": validation,
                    "source_tier": source_tier,
                    "eligible_for_model": source_tier in {"mainstream", "industry", "verified_local"},
                    "collection_source": "verified_local_cache",
                }
            )
            cached.append(result)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return cached


def collect_event(event_key: str, max_articles: int) -> dict[str, Any]:
    config = EVENT_QUERIES[event_key]
    event_dir = OUTPUT_DIR / event_key
    event_dir.mkdir(parents=True, exist_ok=True)

    cached_results = load_cached_articles(event_dir, config)
    gdelt_records, gdelt_attempts = discover_gdelt(event_key, config)
    google_records, google_attempts = discover_google_news(event_key, config)
    seeds = [
        {
            "event_key": event_key,
            "discovery_provider": "verified_seed",
            "discovery_query": "existing_replay_evidence",
            **seed,
        }
        for seed in config["seed_urls"]
    ]
    cached_candidates = [
        {
            key: result.get(key)
            for key in [
                "event_key",
                "discovery_provider",
                "discovery_query",
                "title",
                "source",
                "published_at",
                "url",
            ]
        }
        for result in cached_results
    ]
    candidates = _deduplicate(gdelt_records + google_records + seeds + cached_candidates)
    candidates.sort(key=lambda record: _candidate_score(record, config), reverse=True)
    candidate_path = event_dir / "discovered_candidates.csv"
    pd.DataFrame(candidates).to_csv(candidate_path, index=False)

    cached_urls = {str(result.get("url")) for result in cached_results}
    new_candidates = [record for record in candidates if str(record.get("url")) not in cached_urls]
    remaining_attempts = max(max_articles - len(cached_results), 0)
    results = cached_results + [
        scrape_article(record, config, event_dir)
        for record in new_candidates[:remaining_attempts]
    ]
    body_hashes: set[str] = set()
    for result in results:
        body_hash = result.get("full_text_sha256")
        if result["status"] != "saved" or not body_hash:
            continue
        result["duplicate_body"] = body_hash in body_hashes
        if result["duplicate_body"]:
            result["eligible_for_model"] = False
        body_hashes.add(str(body_hash))

    full_text_index = pd.DataFrame(
        [
            {
                key: result.get(key)
                for key in [
                    "event_key",
                    "source",
                    "source_tier",
                    "eligible_for_model",
                    "title",
                    "parsed_title",
                    "published_at",
                    "parsed_publish_date",
                    "resolved_url",
                    "body_char_count",
                    "full_text_path",
                    "full_text_sha256",
                    "duplicate_body",
                ]
            }
            for result in results
            if result["status"] == "saved"
        ]
    )
    index_path = event_dir / "full_text_index.csv"
    full_text_index.to_csv(index_path, index=False)
    manifest = {
        "event_key": event_key,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": "Only validated article bodies are saved as evidence; metadata-only and blocked pages remain failed records.",
        "minimum_body_chars": MIN_BODY_CHARS,
        "discovery_attempts": gdelt_attempts + google_attempts,
        "candidate_count": len(candidates),
        "attempted_article_count": len(results),
        "cached_article_count": len(cached_results),
        "saved_article_count": sum(result["status"] == "saved" for result in results),
        "model_eligible_article_count": sum(bool(result.get("eligible_for_model")) for result in results),
        "candidate_index_path": str(candidate_path.relative_to(ROOT_DIR)),
        "full_text_index_path": str(index_path.relative_to(ROOT_DIR)),
        "articles": results,
    }
    manifest_path = event_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=["all", *sorted(EVENT_QUERIES)], default="malaysia_asia_shock_2021")
    parser.add_argument("--max-articles", type=int, default=15)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_keys = sorted(EVENT_QUERIES) if args.event == "all" else [args.event]
    manifests = []
    for event_key in event_keys:
        manifest = collect_event(event_key, max(args.max_articles, 1))
        manifests.append(manifest)
        print(
            f'{event_key}: discovered {manifest["candidate_count"]}; '
            f'attempted {manifest["attempted_article_count"]}; '
            f'saved {manifest["saved_article_count"]}; '
            f'model eligible {manifest["model_eligible_article_count"]}.'
        )

    indexes = []
    for manifest in manifests:
        index_path = ROOT_DIR / manifest["full_text_index_path"]
        if index_path.exists() and index_path.stat().st_size > 0:
            try:
                indexes.append(pd.read_csv(index_path))
            except pd.errors.EmptyDataError:
                continue
    corpus = pd.concat(indexes, ignore_index=True) if indexes else pd.DataFrame()
    corpus_path = OUTPUT_DIR / "full_text_corpus_index.csv"
    corpus.to_csv(corpus_path, index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "events_collected": len(manifests),
        "saved_articles": sum(item["saved_article_count"] for item in manifests),
        "model_eligible_articles": sum(item["model_eligible_article_count"] for item in manifests),
        "corpus_index_path": str(corpus_path.relative_to(ROOT_DIR)),
        "event_manifests": [
            str((OUTPUT_DIR / item["event_key"] / "manifest.json").relative_to(ROOT_DIR))
            for item in manifests
        ],
    }
    (OUTPUT_DIR / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()