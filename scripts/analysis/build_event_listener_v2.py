#!/usr/bin/env python3
"""Build a leakage-controlled multi-label event-listening replay."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from build_event_monitor_v1 import (
    FIGURE3_OFFICIAL_SUPPORT,
    build_figure3_news_replay_events,
    build_transport_replay_events,
    evaluate_figure3_news_replay,
    evaluate_operational_replay,
    _load_deduplicated_customs_data,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"

CAUSE_RULES = {
    "pandemic": ["covid", "coronavirus", "pandemic", "lockdown", "outbreak"],
    "earthquake": ["earthquake", "tsunami", "seismic"],
    "flood": ["flood", "flooding", "inundated", "inundation"],
    "drought": ["drought", "water shortage", "low water"],
    "security": ["attack", "security situation", "collision", "explosion"],
    "policy": ["export control", "controlled items", "licensing", "sanctions"],
}

EVENT_RULES = {
    "export_control": [
        "export control",
        "exports of controlled items",
        "licensing policies",
        "license requirement",
    ],
    "capacity_restriction": [
        "booking slots",
        "daily transits",
        "water use",
        "lockdown",
        "stopped accepting",
    ],
    "transport_disruption": [
        "blocked",
        "congestion",
        "pause",
        "rerouting",
        "bridge collapse",
        "ship collides",
        "explosions at beirut port",
    ],
    "physical_disaster": ["earthquake", "tsunami", "flood", "inundated", "explosion"],
}

ENTITY_RULES = {
    "Japan": ["japan", "meti"],
    "Thailand": ["thailand", "bangkok post", "bangkok"],
    "Korea, South": ["republic of korea", "south korea", "korea"],
    "China": ["china", "hubei", "wuhan", "shenzhen", "yantian"],
    "Taiwan": ["taiwan", "taipei"],
    "Malaysia": ["malaysia", "kuala lumpur"],
    "Lebanon": ["lebanon", "beirut"],
    "Egypt": ["egypt", "suez"],
    "Panama": ["panama"],
    "Red Sea": ["red sea", "gulf of aden", "bab el-mandeb", "suez"],
    "United States": ["united states", "baltimore"],
}

TRANSPORT_OBJECT_RULES = {
    "port_terminal": ["port", "terminal", "yantian", "beirut"],
    "canal": ["canal", "suez", "panama canal"],
    "maritime_lane": ["red sea", "ship traffic", "shipping lane", "gulf of aden"],
    "bridge_port_access": ["bridge collapse", "ship collides", "baltimore bridge"],
}

SUPPLY_GRAPH = {
    "Japan": ["semiconductor_fabrication", "semiconductor_materials", "electronics"],
    "Thailand": ["memory_components", "storage_devices", "electronics_assembly"],
    "Korea, South": ["semiconductor_materials", "memory_components", "displays"],
    "China": ["electronics_clusters", "semiconductor_fabrication", "electronics_assembly"],
    "Taiwan": ["wafer_fabrication", "semiconductor_fabrication", "electronics"],
    "Malaysia": ["assembly_and_test", "semiconductor_packaging", "electronics_assembly"],
}

GROUND_TRUTH_OBJECT = {
    "japan_earthquake_2011": "semiconductor_fabrication",
    "thai_flood_2011": "memory_components",
    "japan_export_controls_2019": "semiconductor_materials",
    "covid_q1_2020": "electronics_clusters",
    "taiwan_drought_2021": "wafer_fabrication",
    "malaysia_asia_shock_2021": "assembly_and_test",
    "beirut_port_explosion_2020": "port_terminal",
    "suez_ever_given_2021": "canal",
    "yantian_port_covid_2021": "port_terminal",
    "panama_canal_drought_2023": "canal",
    "red_sea_rerouting_2023": "maritime_lane",
    "baltimore_bridge_collapse_2024": "bridge_port_access",
}

SOURCE_RELIABILITY = {
    "official": 1.0,
    "operator": 0.95,
    "wire": 0.9,
    "major_news": 0.85,
    "local_news": 0.75,
}

NEGATIVE_CONTROL_EVENTS = [
    {
        "event_key": "control_manufacturing_tax_rules_2024",
        "event_date": "2024-10-23",
        "source": "Federal Register",
        "title": "Advanced Manufacturing Investment Credit Rules Under Sections 48D and 50",
        "url": "https://www.federalregister.gov/documents/2024/10/23/2024-23857/advanced-manufacturing-investment-credit-rules-under-sections-48d-and-50",
    },
    {
        "event_key": "control_manufacturing_tax_correction_2024",
        "event_date": "2024-11-25",
        "source": "Federal Register",
        "title": "Advanced Manufacturing Investment Credit Rules Under Sections 48D and 50; Correction",
        "url": "https://www.federalregister.gov/documents/2024/11/25/2024-27427/advanced-manufacturing-investment-credit-rules-under-sections-48d-and-50-correction",
    },
    {
        "event_key": "control_semiconductor_research_2024",
        "event_date": "2024-10-02",
        "source": "Federal Register",
        "title": "Artificial Intelligence-Powered Autonomous Experimentation (AI/AE) for Sustainable Semiconductor Materials",
        "url": "https://www.federalregister.gov/documents/2024/10/02/2024-22664/artificial-intelligence-powered-autonomous-experimentation-aiae-for-sustainable-semiconductor",
    },
    {
        "event_key": "control_h1b_modernization_2024",
        "event_date": "2024-12-18",
        "source": "Federal Register",
        "title": "Modernizing H-1B Requirements, Providing Flexibility in the F-1 Program, and Program Improvements Affecting Other Nonimmigrant Workers",
        "url": "https://www.federalregister.gov/documents/2024/12/18/2024-29354/modernizing-h-1b-requirements-providing-flexibility-in-the-f-1-program-and-program-improvements",
    },
    {
        "event_key": "control_environmental_procedures_2024",
        "event_date": "2024-12-11",
        "source": "Federal Register",
        "title": "National Environmental Policy Act; Proposed Implementing Procedures and Categorical Exclusions",
        "url": "https://www.federalregister.gov/documents/2024/12/11/2024-29088/national-environmental-policy-act-proposed-implementing-procedures-and-categorical-exclusions",
    },
    {
        "event_key": "control_records_schedule_2024",
        "event_date": "2024-10-17",
        "source": "Federal Register",
        "title": "Records Schedules; Availability and Request for Comments",
        "url": "https://www.federalregister.gov/documents/2024/10/17/2024-23927/records-schedules-availability-and-request-for-comments",
    },
]

SUPPLY_TYPE_MAP = {
    "pandemic": "pandemic",
    "earthquake": "earthquake",
    "flood": "flood",
    "drought": "drought",
    "export_control": "export_control",
}


def _matches(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()) is not None


def _all_matches(text: str, rules: dict[str, list[str]]) -> list[str]:
    return [label for label, phrases in rules.items() if any(_matches(text, phrase) for phrase in phrases)]


def _source_tier(source: str) -> str:
    low = source.lower()
    if any(value in low for value in ["government", "authority", "organization", "usgs", "who", "wto", "meti"]):
        return "official"
    if any(value in low for value in ["maersk", "msc", "hapag", "cma cgm"]):
        return "operator"
    if "reuters" in low:
        return "wire"
    if any(value in low for value in ["bbc", "un news"]):
        return "major_news"
    return "local_news"


def _detect_entities(text: str) -> list[str]:
    return [entity for entity, aliases in ENTITY_RULES.items() if any(_matches(text, alias) for alias in aliases)]


def _detect_transport_objects(text: str) -> list[str]:
    return _all_matches(text, TRANSPORT_OBJECT_RULES)


def _impact_channel(events: list[str], objects: list[str]) -> str:
    if "transport_disruption" in events or any(
        value in objects for value in ["canal", "maritime_lane", "bridge_port_access"]
    ):
        return "transport_channel"
    if "port_terminal" in objects and "physical_disaster" in events:
        return "transport_channel"
    return "supply_source"


def _primary_event_type(causes: list[str], events: list[str], channel: str) -> str:
    if channel == "transport_channel":
        return "logistics_disruption"
    if "export_control" in events:
        return "export_control"
    for cause in ["pandemic", "earthquake", "flood", "drought"]:
        if cause in causes:
            return SUPPLY_TYPE_MAP[cause]
    return "other"


def _top_impacted_objects(channel: str, entities: list[str], transport_objects: list[str]) -> list[str]:
    if channel == "transport_channel":
        ordered = ["bridge_port_access", "canal", "maritime_lane", "port_terminal"]
        return [value for value in ordered if value in transport_objects][:3]

    candidates: list[str] = []
    for entity in entities:
        for value in SUPPLY_GRAPH.get(entity, []):
            if value not in candidates:
                candidates.append(value)
    return candidates[:3]


def extract_event(row: pd.Series) -> dict:
    text = " ".join([str(row["title"]), str(row["source"]), str(row["url"])])
    causes = _all_matches(text, CAUSE_RULES)
    events = _all_matches(text, EVENT_RULES)
    entities = _detect_entities(text)
    transport_objects = _detect_transport_objects(text)
    channel = _impact_channel(events, transport_objects)
    predicted_type = _primary_event_type(causes, events, channel)
    top_objects = _top_impacted_objects(channel, entities, transport_objects)
    specificity = sum(bool(values) for values in [causes, events, entities, top_objects]) / 4
    source_tier = _source_tier(str(row["source"]))
    confidence = min(0.99, 0.45 + 0.3 * specificity + 0.2 * SOURCE_RELIABILITY[source_tier])
    return {
        "predicted_causes": "; ".join(causes) or "unknown",
        "predicted_events": "; ".join(events) or "unknown",
        "v2_location": "; ".join(entities) or "Unknown",
        "v2_event_type": predicted_type,
        "v2_impact_target": channel,
        "predicted_object_top3": "; ".join(top_objects) or "unknown",
        "source_tier": source_tier,
        "v2_confidence": round(confidence, 3),
    }


def _location_match(predicted: str, ground_truth: str) -> bool:
    if predicted == "Unknown":
        return False
    truth = ground_truth.lower()
    return any(value.strip().lower() in truth for value in predicted.split(";") if value.strip())


def _available_support(row: pd.Series) -> list[dict]:
    support = json.loads(row["official_support_sources"])
    cutoff = pd.to_datetime(row["confirmation_date"], errors="coerce")
    if pd.isna(cutoff):
        cutoff = pd.to_datetime(row["event_date"], errors="coerce")
    return [item for item in support if pd.to_datetime(item["date"], errors="coerce") <= cutoff]


def _signal_score(
    source_tier: str,
    confidence: float,
    independent_sources: int,
    has_event_evidence: bool,
) -> float:
    corroboration = min(independent_sources, 3) / 3
    return round(
        0.30 * SOURCE_RELIABILITY[source_tier]
        + 0.35 * confidence
        + 0.25 * float(has_event_evidence)
        + 0.10 * corroboration,
        3,
    )


def build_v2_replay() -> tuple[pd.DataFrame, dict, dict]:
    supply_replay = build_figure3_news_replay_events()
    trade_df = _load_deduplicated_customs_data()
    supply_eval, _ = evaluate_figure3_news_replay(supply_replay, trade_df)
    transport_replay = build_transport_replay_events()
    baseline, baseline_metrics = evaluate_operational_replay(supply_eval, transport_replay)

    extracted = pd.DataFrame([extract_event(row) for _, row in baseline.iterrows()])
    result = pd.concat([baseline.reset_index(drop=True), extracted], axis=1)
    result["v2_event_type_match"] = result["v2_event_type"] == result["ground_truth_event_type"]
    result["v2_impact_target_match"] = result["v2_impact_target"] == result["ground_truth_impact_target"]
    result["v2_location_match"] = result.apply(
        lambda row: _location_match(row["v2_location"], row["ground_truth_location"]), axis=1
    )
    result["ground_truth_object_type"] = result["event_key"].map(GROUND_TRUTH_OBJECT)
    result["v2_object_top3_hit"] = result.apply(
        lambda row: row["ground_truth_object_type"] in row["predicted_object_top3"].split("; "),
        axis=1,
    )

    support_lists = result.apply(_available_support, axis=1)
    result["time_available_support_count"] = support_lists.map(len)
    result["independent_source_count"] = result["time_available_support_count"] + 1
    result["risk_score"] = result.apply(
        lambda row: _signal_score(
            row["source_tier"],
            row["v2_confidence"],
            row["independent_source_count"],
            row["predicted_events"] != "unknown",
        ),
        axis=1,
    )
    result["alert_level"] = pd.cut(
        result["risk_score"],
        bins=[-float("inf"), 0.65, 0.8, float("inf")],
        labels=["watch", "warning", "critical"],
        right=False,
    ).astype(str)

    result["listening_detection_date"] = result["event_date"]
    for index, support in support_lists.items():
        available_dates = [pd.Timestamp(result.at[index, "event_date"])]
        available_dates.extend(pd.Timestamp(item["date"]) for item in support)
        result.at[index, "listening_detection_date"] = min(available_dates).strftime("%Y-%m-%d")
    result["detection_advance_days"] = (
        pd.to_datetime(result["event_date"])
        - pd.to_datetime(result["listening_detection_date"])
    ).dt.days
    result["listening_lead_days"] = (
        pd.to_datetime(result["confirmation_date"], errors="coerce")
        - pd.to_datetime(result["listening_detection_date"])
    ).dt.days
    result["listening_early_warning_success"] = result.apply(
        lambda row: bool(row["listening_lead_days"] >= 14)
        if row["channel"] == "supply_source" and pd.notna(row["listening_lead_days"])
        else bool(row["listening_lead_days"] > 0)
        if row["channel"] == "transport_channel" and pd.notna(row["listening_lead_days"])
        else False,
        axis=1,
    )

    controls = pd.DataFrame(NEGATIVE_CONTROL_EVENTS)
    control_extracted = pd.DataFrame([extract_event(row) for _, row in controls.iterrows()])
    controls = pd.concat([controls, control_extracted], axis=1)
    controls["independent_source_count"] = 1
    controls["risk_score"] = controls.apply(
        lambda row: _signal_score(
            row["source_tier"],
            row["v2_confidence"],
            1,
            row["predicted_events"] != "unknown",
        ),
        axis=1,
    )
    controls["alert_level"] = pd.cut(
        controls["risk_score"],
        bins=[-float("inf"), 0.65, 0.8, float("inf")],
        labels=["watch", "warning", "critical"],
        right=False,
    ).astype(str)
    control_false_alerts = controls["alert_level"].isin(["warning", "critical"])

    metrics = {
        "events_evaluated": int(len(result)),
        "method": "development replay; title, source, URL metadata, and only time-available corroboration",
        "event_type_match_rate_pct": round(float(result["v2_event_type_match"].mean() * 100), 2),
        "impact_target_match_rate_pct": round(float(result["v2_impact_target_match"].mean() * 100), 2),
        "location_match_rate_pct": round(float(result["v2_location_match"].mean() * 100), 2),
        "object_top3_hit_rate_pct": round(float(result["v2_object_top3_hit"].mean() * 100), 2),
        "warning_or_critical_rate_pct": round(float(result["alert_level"].isin(["warning", "critical"]).mean() * 100), 2),
        "negative_controls": int(len(controls)),
        "negative_control_false_alert_rate_pct": round(float(control_false_alerts.mean() * 100), 2),
        "confirmed_events": int(result["confirmation_status"].eq("confirmed").sum()),
        "baseline_early_warning_success_rate_pct": baseline_metrics["early_warning_success_rate_pct"],
        "listening_early_warning_success_rate_pct": round(
            float(result["listening_early_warning_success"].mean() * 100), 2
        ),
        "events_with_earlier_detection": int(result["detection_advance_days"].gt(0).sum()),
        "total_detection_days_gained": int(result["detection_advance_days"].sum()),
        "leakage_control": {
            "post_confirmation_support_excluded": True,
            "historical_web_body_not_used": True,
            "ground_truth_not_used_as_classifier_input": True,
        },
    }
    return result, controls, metrics, baseline_metrics


def plot_metric_comparison(metrics: dict, baseline_metrics: dict) -> None:
    labels = ["Event type", "Impact channel", "Location"]
    baseline = [
        baseline_metrics["event_type_match_rate_pct"],
        baseline_metrics["impact_target_match_rate_pct"],
        baseline_metrics["location_match_rate_pct"],
    ]
    v2 = [
        metrics["event_type_match_rate_pct"],
        metrics["impact_target_match_rate_pct"],
        metrics["location_match_rate_pct"],
    ]
    positions = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    baseline_bars = ax.bar([value - width / 2 for value in positions], baseline, width, label="V1", color="#7b8a8b")
    v2_bars = ax.bar([value + width / 2 for value in positions], v2, width, label="V2", color="#167d83")
    ax.axhline(80, color="#c55a11", linestyle="--", linewidth=1.4, label="80% target")
    ax.bar_label(baseline_bars, fmt="%.1f%%", padding=3)
    ax.bar_label(v2_bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(list(positions), labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy")
    ax.set_title("Event Listener Development Replay: V1 vs V2")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_listener_v2_metric_comparison.png", dpi=170)
    plt.close(fig)


def plot_risk_scores(result: pd.DataFrame, controls: pd.DataFrame) -> None:
    positives = result[["event_key", "risk_score", "alert_level"]].copy()
    positives["sample"] = "Historical event"
    negatives = controls[["event_key", "risk_score", "alert_level"]].copy()
    negatives["sample"] = "Negative control"
    plot_df = pd.concat([positives, negatives], ignore_index=True).sort_values("risk_score")
    colors = plot_df["alert_level"].map(
        {"watch": "#d1a54b", "warning": "#d66b2c", "critical": "#b5362f"}
    )
    colors = colors.where(plot_df["sample"] == "Historical event", "#607d8b")
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    bars = ax.barh(plot_df["event_key"], plot_df["risk_score"], color=colors)
    ax.axvline(0.65, color="#6d6d6d", linestyle="--", linewidth=1.2)
    ax.axvline(0.8, color="#383838", linestyle="--", linewidth=1.2)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Risk score")
    ax.set_title("Multi-Source Risk Score Using Time-Available Evidence")
    ax.legend(
        handles=[
            Patch(facecolor="#d1a54b", label="Watch"),
            Patch(facecolor="#d66b2c", label="Warning"),
            Patch(facecolor="#b5362f", label="Critical"),
            Patch(facecolor="#607d8b", label="Negative control"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_listener_v2_risk_scores.png", dpi=170)
    plt.close(fig)


def plot_capability_summary(metrics: dict) -> None:
    labels = ["Type", "Channel", "Location", "Object Top-3", "Early warning"]
    values = [
        metrics["event_type_match_rate_pct"],
        metrics["impact_target_match_rate_pct"],
        metrics["location_match_rate_pct"],
        metrics["object_top3_hit_rate_pct"],
        metrics["listening_early_warning_success_rate_pct"],
    ]
    colors = ["#167d83" if value >= 80 else "#d66b2c" for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(80, color="#2f3e46", linestyle="--", linewidth=1.4, label="80% target")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Rate")
    ax.set_title("What Improved, and What Still Requires Leading Indicators")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_listener_v2_capability_summary.png", dpi=170)
    plt.close(fig)


def plot_detection_gains(result: pd.DataFrame) -> None:
    plot_df = result.loc[result["detection_advance_days"] > 0].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("detection_advance_days")
    colors = plot_df["confirmation_status"].map(
        {"confirmed": "#167d83", "not_confirmed": "#9e9e9e"}
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(plot_df["event_key"], plot_df["detection_advance_days"], color=colors)
    ax.bar_label(bars, labels=[f"{value} days" for value in plot_df["detection_advance_days"]], padding=3)
    ax.set_xlim(0, max(plot_df["detection_advance_days"].max() + 3, 14))
    ax.set_xlabel("Earlier than the primary news record")
    ax.set_title("Detection Time Gained from Earlier Time-Available Sources")
    ax.legend(
        handles=[
            Patch(facecolor="#167d83", label="Outcome confirmed"),
            Patch(facecolor="#9e9e9e", label="Outcome not confirmed"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_listener_v2_detection_gains.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result, controls, metrics, baseline_metrics = build_v2_replay()
    result.to_csv(OUT_DIR / "event_listener_v2_evaluation.csv", index=False)
    controls.to_csv(OUT_DIR / "event_listener_v2_negative_controls.csv", index=False)
    (OUT_DIR / "event_listener_v2_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_metric_comparison(metrics, baseline_metrics)
    plot_risk_scores(result, controls)
    plot_capability_summary(metrics)
    plot_detection_gains(result)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()