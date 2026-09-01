#!/usr/bin/env python3
"""Evaluate locally archived news bodies in the 12-event historical replay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from build_event_listener_v2 import (
    ROOT_DIR,
    _location_match,
    _signal_score,
    build_v2_replay,
    extract_event,
)


OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"
CORPUS_INDEX = ROOT_DIR / "data" / "raw" / "disruption_news" / "full_text_corpus_index.csv"

EVENT_LABELS = {
    "japan_earthquake_2011": "Japan earthquake",
    "thai_flood_2011": "Thailand floods",
    "japan_export_controls_2019": "Japan-Korea controls",
    "covid_q1_2020": "China COVID lockdown",
    "taiwan_drought_2021": "Taiwan drought",
    "malaysia_asia_shock_2021": "Malaysia lockdown",
    "beirut_port_explosion_2020": "Beirut port explosion",
    "suez_ever_given_2021": "Suez blockage",
    "yantian_port_covid_2021": "Yantian congestion",
    "panama_canal_drought_2023": "Panama restrictions",
    "red_sea_rerouting_2023": "Red Sea carrier pause",
    "baltimore_bridge_collapse_2024": "Baltimore collapse",
}

FIGURE3_EVENT_KEYS = [
    "japan_earthquake_2011",
    "thai_flood_2011",
    "japan_export_controls_2019",
    "covid_q1_2020",
    "taiwan_drought_2021",
    "malaysia_asia_shock_2021",
]

FIGURE3_OPERATIONAL_EVENTS = {
    "japan_earthquake_2011": ("2011-03-11", "Earthquake and tsunami"),
    "thai_flood_2011": ("2011-08-01", "Flood disruption reported"),
    "japan_export_controls_2019": ("2019-07-01", "Export controls announced"),
    "covid_q1_2020": ("2020-01-23", "Hubei lockdown"),
    "taiwan_drought_2021": ("2021-03-25", "Science-park water cut"),
    "malaysia_asia_shock_2021": ("2021-06-01", "Total lockdown began"),
}

COLORS = {
    "ink": "#253238",
    "muted": "#60747b",
    "saved": "#9ba8aa",
    "eligible": "#287271",
    "early": "#23856d",
    "same_day": "#d6a534",
    "late": "#bd4b42",
    "unconfirmed": "#8b9699",
    "paper": "#f4f1e9",
    "line": "#d3d7d4",
}


def _publication_date(row: pd.Series) -> pd.Timestamp:
    for field in ["parsed_publish_date", "published_at"]:
        raw_value = row.get(field)
        if pd.isna(raw_value):
            continue
        text = str(raw_value)
        date_token = text[:8] if len(text) >= 8 and text[:8].isdigit() else text[:10]
        value = pd.to_datetime(date_token, errors="coerce")
        if pd.notna(value):
            return value.normalize()
    return pd.NaT


def _extract_body(row: pd.Series) -> dict:
    body = (ROOT_DIR / str(row["full_text_path"])).read_text(encoding="utf-8")
    extraction_input = pd.Series(
        {
            "title": f'{row.get("parsed_title", "")}\n{body}',
            "source": row.get("source", "Unknown"),
            "url": row.get("resolved_url", ""),
        }
    )
    return {**extract_event(extraction_input), "full_text_chars_used": len(body)}


def build_fulltext_replay() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline, _, listener_metrics, _ = build_v2_replay()
    corpus = pd.read_csv(CORPUS_INDEX)
    corpus["eligible_for_model"] = corpus["eligible_for_model"].fillna(False).astype(bool)
    corpus["article_date"] = corpus.apply(_publication_date, axis=1)

    eligible = corpus.loc[corpus["eligible_for_model"] & corpus["article_date"].notna()].copy()
    extracted = pd.DataFrame([_extract_body(row) for _, row in eligible.iterrows()], index=eligible.index)
    articles = pd.concat([eligible, extracted], axis=1)

    truth_columns = [
        "event_key",
        "ground_truth_event_type",
        "ground_truth_impact_target",
        "ground_truth_location",
        "ground_truth_object_type",
        "channel",
        "confirmation_status",
        "confirmation_date",
        "listening_detection_date",
    ]
    articles = articles.merge(baseline[truth_columns], on="event_key", how="left")
    articles["event_type_match"] = articles["v2_event_type"] == articles["ground_truth_event_type"]
    articles["impact_target_match"] = articles["v2_impact_target"] == articles["ground_truth_impact_target"]
    articles["location_match"] = articles.apply(
        lambda row: _location_match(row["v2_location"], row["ground_truth_location"]), axis=1
    )
    articles["object_top3_hit"] = articles.apply(
        lambda row: row["ground_truth_object_type"] in row["predicted_object_top3"].split("; "), axis=1
    )
    articles["confirmation_date"] = pd.to_datetime(articles["confirmation_date"], errors="coerce")
    articles["lead_days"] = (articles["confirmation_date"] - articles["article_date"]).dt.days
    articles["available_by_confirmation"] = (
        articles["confirmation_date"].notna()
        & articles["article_date"].le(articles["confirmation_date"])
    )

    earliest = (
        articles.sort_values(["event_key", "article_date"])
        .groupby("event_key", as_index=False)
        .first()
    )
    earliest["strict_early"] = earliest.apply(
        lambda row: bool(row["lead_days"] >= 14)
        if row["channel"] == "supply_source" and pd.notna(row["lead_days"])
        else bool(row["lead_days"] > 0)
        if row["channel"] == "transport_channel" and pd.notna(row["lead_days"])
        else False,
        axis=1,
    )
    earliest = earliest.rename(
        columns={
            "event_type_match": "fulltext_event_type_match",
            "impact_target_match": "fulltext_impact_target_match",
            "location_match": "fulltext_location_match",
            "object_top3_hit": "fulltext_object_top3_hit",
            "lead_days": "fulltext_lead_days",
        }
    )

    event_result = baseline.merge(
        earliest[
            [
                "event_key",
                "article_date",
                "source",
                "parsed_title",
                "full_text_chars_used",
                "fulltext_event_type_match",
                "fulltext_impact_target_match",
                "fulltext_location_match",
                "fulltext_object_top3_hit",
                "fulltext_lead_days",
                "strict_early",
            ]
        ],
        on="event_key",
        how="left",
        suffixes=("", "_fulltext"),
    )
    event_result["has_model_eligible_body"] = event_result["article_date"].notna()
    event_result["fulltext_status"] = event_result.apply(
        lambda row: "not_covered"
        if not row["has_model_eligible_body"]
        else "unconfirmed"
        if row["confirmation_status"] != "confirmed"
        else "early"
        if row["fulltext_lead_days"] > 0
        else "same_day"
        if row["fulltext_lead_days"] == 0
        else "late",
        axis=1,
    )

    combined_dates = pd.to_datetime(event_result["listening_detection_date"])
    has_body = event_result["article_date"].notna()
    combined_dates.loc[has_body] = pd.concat(
        [combined_dates.loc[has_body], event_result.loc[has_body, "article_date"]], axis=1
    ).min(axis=1)
    event_result["combined_detection_date"] = combined_dates
    event_result["combined_lead_days"] = (
        pd.to_datetime(event_result["confirmation_date"], errors="coerce")
        - event_result["combined_detection_date"]
    ).dt.days
    event_result["combined_strict_early"] = event_result.apply(
        lambda row: bool(row["combined_lead_days"] >= 14)
        if row["channel"] == "supply_source" and pd.notna(row["combined_lead_days"])
        else bool(row["combined_lead_days"] > 0)
        if row["channel"] == "transport_channel" and pd.notna(row["combined_lead_days"])
        else False,
        axis=1,
    )

    covered = event_result.loc[event_result["has_model_eligible_body"]]
    malaysia = articles.loc[articles["event_key"] == "malaysia_asia_shock_2021"]
    metrics = {
        "events_in_replay": int(len(event_result)),
        "saved_full_text_articles": int(len(corpus)),
        "model_eligible_full_text_articles": int(len(articles)),
        "events_with_model_eligible_full_text": int(articles["event_key"].nunique()),
        "event_coverage_rate_pct": round(float(articles["event_key"].nunique() / len(event_result) * 100), 2),
        "eligible_full_text_total_characters": int(articles["full_text_chars_used"].sum()),
        "earliest_body_event_type_match_rate_pct": round(float(covered["fulltext_event_type_match"].mean() * 100), 2),
        "earliest_body_impact_target_match_rate_pct": round(float(covered["fulltext_impact_target_match"].mean() * 100), 2),
        "earliest_body_location_match_rate_pct": round(float(covered["fulltext_location_match"].mean() * 100), 2),
        "earliest_body_object_top3_hit_rate_pct": round(float(covered["fulltext_object_top3_hit"].mean() * 100), 2),
        "covered_events_strict_early_count": int(covered["strict_early"].fillna(False).sum()),
        "full_replay_strict_early_rate_from_body_only_pct": round(
            float(event_result["strict_early"].fillna(False).mean() * 100), 2
        ),
        "existing_listener_strict_early_rate_pct": listener_metrics["listening_early_warning_success_rate_pct"],
        "combined_listener_and_fulltext_strict_early_rate_pct": round(
            float(event_result["combined_strict_early"].mean() * 100), 2
        ),
        "events_where_fulltext_improved_detection_date": int(
            (event_result["combined_detection_date"] < pd.to_datetime(event_result["listening_detection_date"])).sum()
        ),
        "malaysia_model_eligible_articles": int(len(malaysia)),
        "malaysia_first_article_date": malaysia["article_date"].min().strftime("%Y-%m-%d") if not malaysia.empty else None,
        "malaysia_last_article_date": malaysia["article_date"].max().strftime("%Y-%m-%d") if not malaysia.empty else None,
        "interpretation": "Full text is evaluated only from its publication date; post-consequence articles cannot improve historical warning time.",
    }
    return articles, event_result, metrics


def plot_corpus_coverage(corpus: pd.DataFrame, event_result: pd.DataFrame) -> None:
    counts = corpus.groupby("event_key").agg(
        saved=("event_key", "size"),
        eligible=("eligible_for_model", "sum"),
    )
    counts = event_result[["event_key"]].drop_duplicates().set_index("event_key").join(counts).fillna(0)
    counts["label"] = counts.index.map(EVENT_LABELS)
    counts = counts.sort_values(["eligible", "saved"])
    positions = range(len(counts))

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.barh(positions, counts["saved"], color=COLORS["saved"], label="Validated body saved")
    ax.barh(positions, counts["eligible"], color=COLORS["eligible"], label="Eligible for model")
    ax.set_yticks(list(positions), counts["label"])
    ax.set_xlabel("Number of full-text articles")
    fig.suptitle("Real Full-Text News Coverage Across the 12-Event Replay", x=0.125, y=0.97, ha="left", fontsize=18, weight="bold")
    fig.text(0.125, 0.925, "Source quality and body-content validation are required for model eligibility", color=COLORS["muted"])
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "fulltext_news_corpus_coverage.png", dpi=180)
    plt.close(fig)


def plot_understanding(metrics: dict) -> None:
    labels = ["Event type", "Location", "Impact channel", "Affected object Top-3"]
    values = [
        metrics["earliest_body_event_type_match_rate_pct"],
        metrics["earliest_body_location_match_rate_pct"],
        metrics["earliest_body_impact_target_match_rate_pct"],
        metrics["earliest_body_object_top3_hit_rate_pct"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=COLORS["paper"])
    bars = ax.bar(labels, values, color=["#287271", "#3d7e9a", "#d97732", "#8a5d3b"])
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Match rate")
    fig.suptitle("What the Earliest Available Article Body Could Identify", x=0.125, y=0.97, ha="left", fontsize=18, weight="bold")
    fig.text(0.125, 0.92, "Evaluated on the 6 events with model-eligible full text; development replay, not a blind test", color=COLORS["muted"])
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIG_DIR / "fulltext_news_event_understanding.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_timing(event_result: pd.DataFrame, metrics: dict) -> None:
    covered = event_result.loc[event_result["has_model_eligible_body"]].copy()
    covered["label"] = covered["event_key"].map(EVENT_LABELS)
    covered["plot_lead"] = covered["fulltext_lead_days"].fillna(0)
    covered = covered.sort_values(["confirmation_status", "plot_lead"])
    colors = covered["fulltext_status"].map(
        {
            "early": COLORS["early"],
            "same_day": COLORS["same_day"],
            "late": COLORS["late"],
            "unconfirmed": COLORS["unconfirmed"],
        }
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.barh(covered["label"], covered["plot_lead"], color=colors)
    ax.axvline(0, color=COLORS["ink"], linewidth=1.1)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Days from earliest eligible article body to consequence confirmation")
    fig.suptitle("Full-Text Evidence Timing: Semantic Value Does Not Guarantee Early Warning", x=0.08, y=0.97, ha="left", fontsize=17, weight="bold")
    fig.text(0.08, 0.91, "Only publication-time-available bodies are used; gray events have no confirmed consequence date", color=COLORS["muted"])
    for bar, (_, row) in zip(bars, covered.iterrows()):
        if row["fulltext_status"] == "unconfirmed":
            text = f'No consequence confirmation; first body {row["article_date"].date()}'
            x = 1
        else:
            text = "Same day" if row["plot_lead"] == 0 else f'{int(row["plot_lead"]):+d} days'
            x = row["plot_lead"] + (2 if row["plot_lead"] >= 0 else -1)
        ax.text(x, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=9.5)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["early"], label="Before consequence"),
            Patch(facecolor=COLORS["same_day"], label="Same day"),
            Patch(facecolor=COLORS["late"], label="After consequence"),
            Patch(facecolor=COLORS["unconfirmed"], label="Outcome unconfirmed"),
        ],
        loc="lower right",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.08,
        0.015,
        f'Combined listener + full text strict early-alert rate: {metrics["combined_listener_and_fulltext_strict_early_rate_pct"]:.2f}%. '
        f'Full text improved the earliest detection date for {metrics["events_where_fulltext_improved_detection_date"]} events.',
        color=COLORS["muted"],
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.88))
    fig.savefig(FIG_DIR / "fulltext_news_timing_evidence.png", dpi=180)
    plt.close(fig)


def plot_figure3_event_validation(corpus: pd.DataFrame, event_result: pd.DataFrame) -> None:
    corpus = corpus.loc[corpus["event_key"].isin(FIGURE3_EVENT_KEYS)].copy()
    event_result = event_result.loc[event_result["event_key"].isin(FIGURE3_EVENT_KEYS)].copy()
    counts = corpus.groupby("event_key").agg(
        saved=("event_key", "size"),
        eligible=("eligible_for_model", "sum"),
    )
    dashboard = event_result.set_index("event_key").join(counts).fillna({"saved": 0, "eligible": 0})
    dashboard["label"] = dashboard.index.map(EVENT_LABELS)
    dashboard = dashboard.loc[FIGURE3_EVENT_KEYS]
    positions = list(range(len(dashboard)))

    covered = dashboard.loc[dashboard["has_model_eligible_body"]]
    eligible_articles = corpus["eligible_for_model"].fillna(False).astype(bool)
    subset_metrics = {
        "source_validated": int(dashboard["official_support_count"].gt(0).sum()),
        "eligible_events": int(dashboard["has_model_eligible_body"].sum()),
        "eligible_bodies": int(eligible_articles.sum()),
        "confirmed_events": int(dashboard["confirmation_status"].eq("confirmed").sum()),
        "event_type_match": float(covered["fulltext_event_type_match"].mean() * 100),
    }

    semantic_columns = [
        "fulltext_event_type_match",
        "fulltext_location_match",
        "fulltext_impact_target_match",
        "fulltext_object_top3_hit",
    ]
    semantic = dashboard[semantic_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    semantic_display = semantic.copy()
    semantic_display[pd.isna(semantic_display)] = -1

    fig = plt.figure(figsize=(18, 10.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 7],
        width_ratios=[1.25, 2.15, 2.1],
        left=0.16,
        right=0.97,
        top=0.86,
        bottom=0.15,
        wspace=0.24,
        hspace=0.16,
    )

    summary_ax = fig.add_subplot(grid[0, :])
    summary_ax.axis("off")
    summary_items = [
        (f'{subset_metrics["source_validated"]}/6', "Events with official corroboration"),
        (f'{subset_metrics["eligible_events"]}/6', "Events with eligible full text"),
        (f'{subset_metrics["eligible_bodies"]}', "Model-eligible article bodies"),
        (f'{subset_metrics["confirmed_events"]}/6', "Census consequence confirmed"),
    ]
    for index, (value, label) in enumerate(summary_items):
        x = 0.02 + index * 0.245
        summary_ax.text(x, 0.63, value, fontsize=20, weight="bold", color=COLORS["ink"], transform=summary_ax.transAxes)
        summary_ax.text(x, 0.24, label, fontsize=9.5, color=COLORS["muted"], transform=summary_ax.transAxes)
        if index < len(summary_items) - 1:
            summary_ax.plot([x + 0.215, x + 0.215], [0.18, 0.84], color=COLORS["line"], transform=summary_ax.transAxes)

    coverage_ax = fig.add_subplot(grid[1, 0])
    coverage_ax.barh(
        [position - 0.18 for position in positions],
        dashboard["official_support_count"],
        color="#3d7e9a",
        height=0.32,
        label="Official sources",
    )
    coverage_ax.barh(
        [position + 0.18 for position in positions],
        dashboard["eligible"],
        color=COLORS["eligible"],
        height=0.32,
        label="Eligible news bodies",
    )
    coverage_ax.set_yticks(positions, dashboard["label"])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlabel("Number of evidence items")
    coverage_ax.set_title("1  Traceable evidence", loc="left", fontsize=12, weight="bold", pad=14)
    coverage_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    coverage_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    coverage_ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8.5,
    )

    semantic_ax = fig.add_subplot(grid[1, 1])
    semantic_ax.imshow(
        semantic_display,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(["#e3e5e2", "#c95d52", "#287271"]),
        vmin=-1,
        vmax=1,
    )
    semantic_ax.set_xticks(
        range(4),
        ["Event\ntype", "Location", "Impact\nchannel", "Affected object\nTop-3"],
        fontsize=9,
    )
    semantic_ax.set_yticks(positions, [""] * len(positions))
    semantic_ax.set_title("2  What the earliest eligible body identified", loc="left", fontsize=12, weight="bold", pad=14)
    semantic_ax.tick_params(length=0)
    semantic_ax.set_xticks([value - 0.5 for value in range(1, 4)], minor=True)
    semantic_ax.set_yticks([value - 0.5 for value in range(1, len(positions))], minor=True)
    semantic_ax.grid(which="minor", color="white", linewidth=2)
    for row_index in positions:
        for column_index in range(4):
            value = semantic_display[row_index, column_index]
            label = "No body" if value == -1 and column_index == 1 else "✓" if value == 1 else "×" if value == 0 else ""
            semantic_ax.text(column_index, row_index, label, ha="center", va="center", fontsize=10, color="white" if value >= 0 else COLORS["muted"], weight="bold")

    timing_ax = fig.add_subplot(grid[1, 2])
    timing_ax.axvline(0, color=COLORS["ink"], linewidth=1.1)
    timing_ax.axvline(14, color=COLORS["line"], linewidth=1.1, linestyle="--")
    timing_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    for position, (_, row) in zip(positions, dashboard.iterrows()):
        if row["confirmation_status"] != "confirmed":
            timing_ax.scatter(0, position, marker="D", s=55, color=COLORS["unconfirmed"], zorder=3)
            timing_ax.text(4, position, "Sources validated; Census outcome unconfirmed", va="center", fontsize=8.5, color=COLORS["muted"])
            continue
        lead = float(row["combined_lead_days"])
        color = COLORS["early"] if lead > 0 else COLORS["same_day"] if lead == 0 else COLORS["late"]
        timing_ax.hlines(position, min(0, lead), max(0, lead), color=color, linewidth=6)
        timing_ax.scatter(lead, position, s=48, color=color, zorder=3)
        timing_ax.text(lead + (2 if lead >= 0 else -2), position, f"{int(lead):+d}d", ha="left" if lead >= 0 else "right", va="center", fontsize=8.5)
    timing_ax.set_yticks(positions, [""] * len(positions))
    timing_ax.invert_yaxis()
    timing_ax.set_xlabel("Days before (+) or after (-) Census consequence")
    timing_ax.set_title("3  Earliest evidence vs. Census consequence", loc="left", fontsize=12, weight="bold", pad=14)
    timing_ax.tick_params(axis="y", length=0)
    timing_ax.legend(
        handles=[
            Patch(facecolor=COLORS["early"], label="Before consequence"),
            Patch(facecolor=COLORS["same_day"], label="Same day"),
            Patch(facecolor=COLORS["late"], label="After consequence"),
            Patch(facecolor=COLORS["unconfirmed"], label="Outcome unconfirmed"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8.5,
    )

    for ax in [coverage_ax, semantic_ax, timing_ax]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Figure 3 Event Validation: Real News, Official Evidence, and Observed Impact", x=0.06, y=0.965, ha="left", fontsize=21, weight="bold")
    fig.text(
        0.06,
        0.922,
        "The six highlighted supply events only; evidence dates are preserved and later reports are never backdated",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    fig.text(
        0.16,
        0.025,
        f'Full-text event-type match: {subset_metrics["event_type_match"]:.1f}% on the {subset_metrics["eligible_events"]} covered events. '
        "Taiwan and Malaysia have source-validated operational evidence but do not cross the current Census confirmation threshold.",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    fig.savefig(FIG_DIR / "figure3_event_validation_dashboard.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_figure3_alarm_timeline(event_result: pd.DataFrame) -> pd.DataFrame:
    timeline = event_result.loc[event_result["event_key"].isin(FIGURE3_EVENT_KEYS)].copy()
    timeline = timeline.set_index("event_key").loc[FIGURE3_EVENT_KEYS].reset_index()
    timeline["event_label"] = timeline["event_key"].map(EVENT_LABELS)
    timeline["news_identified_date"] = pd.to_datetime(timeline["event_date"])
    timeline["operational_event_date"] = pd.to_datetime(
        timeline["event_key"].map(lambda key: FIGURE3_OPERATIONAL_EVENTS[key][0])
    )
    timeline["operational_event"] = timeline["event_key"].map(
        lambda key: FIGURE3_OPERATIONAL_EVENTS[key][1]
    )
    timeline["census_consequence_date"] = pd.to_datetime(
        timeline["confirmation_date"], errors="coerce"
    )
    timeline["news_to_event_days"] = (
        timeline["operational_event_date"] - timeline["news_identified_date"]
    ).dt.days
    timeline["news_to_census_consequence_days"] = (
        timeline["census_consequence_date"] - timeline["news_identified_date"]
    ).dt.days
    timeline["support_available_at_news"] = timeline.apply(
        lambda row: sum(
            pd.Timestamp(item["date"]) <= row["news_identified_date"]
            for item in json.loads(row["official_support_sources"])
        ),
        axis=1,
    )
    timeline["risk_score_at_news"] = timeline.apply(
        lambda row: _signal_score(
            row["source_tier"],
            row["v2_confidence"],
            int(row["support_available_at_news"]) + 1,
            row["predicted_events"] != "unknown",
        ),
        axis=1,
    )
    timeline["alert_level_at_news"] = pd.cut(
        timeline["risk_score_at_news"],
        bins=[-float("inf"), 0.65, 0.8, float("inf")],
        labels=["watch", "warning", "critical"],
        right=False,
    ).astype(str)
    return timeline


def plot_figure3_news_alarm_timeline(
    timeline: pd.DataFrame,
    output_path: Path | None = None,
    integrated_event_df: pd.DataFrame | None = None,
    integrated_model_label: str = "Integrated",
) -> None:
    if integrated_event_df is not None and not integrated_event_df.empty:
        integrated_values = integrated_event_df[
            ["event_key", "news_triggered_supply_gap_pct", "observed_supply_gap_pct"]
        ].copy()
        timeline = timeline.merge(integrated_values, on="event_key", how="left")

    positions = list(range(len(timeline)))
    confirmed_leads = timeline["news_to_census_consequence_days"].dropna()
    max_days = max(35, int(confirmed_leads.max()) + 45)

    fig, (timeline_ax, risk_ax) = plt.subplots(
        1,
        2,
        figsize=(17, 8.2),
        gridspec_kw={"width_ratios": [3.3, 1], "wspace": 0.16},
    )
    for position, (_, row) in zip(positions, timeline.iterrows()):
        event_days = int(row["news_to_event_days"])
        consequence_days = row["news_to_census_consequence_days"]
        window_end = int(consequence_days) if pd.notna(consequence_days) else max(event_days, 4)

        timeline_ax.hlines(position, 0, window_end, color="#e6e8e5", linewidth=15, zorder=1)
        if event_days > 0:
            timeline_ax.hlines(position, 0, event_days, color="#287271", linewidth=8, zorder=2)
        if pd.notna(consequence_days) and consequence_days > event_days:
            timeline_ax.hlines(
                position,
                event_days,
                int(consequence_days),
                color="#d97732",
                linewidth=8,
                zorder=2,
            )

        timeline_ax.scatter(0, position, s=90, color="#236a96", edgecolor="white", linewidth=1.5, zorder=5)
        timeline_ax.scatter(event_days, position, s=92, marker="D", color="#d97732", edgecolor="white", linewidth=1.5, zorder=4)
        if pd.notna(consequence_days):
            timeline_ax.scatter(
                int(consequence_days),
                position,
                s=125,
                marker="*",
                color="#b84940",
                edgecolor="white",
                linewidth=1,
                zorder=5,
            )

        event_text = "same day" if event_days == 0 else f"event +{event_days}d"
        consequence_text = (
            f"Census impact +{int(consequence_days)}d"
            if pd.notna(consequence_days)
            else "no Census threshold crossing"
        )
        timeline_ax.text(
            window_end + 2,
            position,
            f"{event_text}  |  {consequence_text}",
            va="center",
            fontsize=9,
            color=COLORS["muted"],
        )

    timeline_ax.set_yticks(positions, timeline["event_label"])
    timeline_ax.invert_yaxis()
    timeline_ax.set_xlim(-4, max_days)
    timeline_ax.set_xlabel("Days after relevant news was identified")
    timeline_ax.set_title("News-to-event and news-to-impact lead time", loc="left", fontsize=13, weight="bold")
    timeline_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    timeline_ax.set_axisbelow(True)
    timeline_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#236a96", markersize=9, label="Relevant news identified"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#d97732", markersize=8, label="Operational event"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#b84940", markersize=12, label="Census consequence"),
            Patch(facecolor="#287271", label="Pre-event alarm window"),
            Patch(facecolor="#d97732", label="Supply-impact warning window"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    risk_colors = timeline["alert_level_at_news"].map(
        {"watch": "#d1a54b", "warning": "#d66b2c", "critical": "#b5362f"}
    )
    bars = risk_ax.barh(positions, timeline["risk_score_at_news"], color=risk_colors, height=0.56)
    risk_ax.axvline(0.65, color="#a6abad", linestyle="--", linewidth=1.1)
    risk_ax.axvline(0.8, color=COLORS["ink"], linestyle="--", linewidth=1.1)
    risk_ax.set_yticks(positions, [""] * len(positions))
    risk_ax.invert_yaxis()
    risk_ax.set_xlim(0, 1.08)
    risk_ax.set_xlabel("Multi-source risk score")
    risk_ax.set_title("Alarm state", loc="left", fontsize=13, weight="bold")
    risk_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    risk_ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, timeline.iterrows()):
        if "news_triggered_supply_gap_pct" in timeline.columns:
            expected_gap = float(row["news_triggered_supply_gap_pct"])
            observed_gap = float(row["observed_supply_gap_pct"])
            label = (
                f'{float(row["risk_score_at_news"]):.2f} {str(row["alert_level_at_news"]).title()}\n'
                f"Gap {expected_gap:.2f}% -> {observed_gap:.2f}%"
            )
            label_x = float(row["risk_score_at_news"]) - 0.02
            label_color = "white"
            label_size = 7.4
            label_alignment = "right"
        else:
            label = f'{float(row["risk_score_at_news"]):.2f} {str(row["alert_level_at_news"]).title()}'
            label_x = float(row["risk_score_at_news"]) + 0.02
            label_color = "black"
            label_size = 8.5
            label_alignment = "left"
        risk_ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=label_alignment,
            color=label_color,
            fontsize=label_size,
        )

    for ax in [timeline_ax, risk_ax]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    title = "Figure 3: News-Triggered Early Alarm Before Supply Consequences"
    if integrated_event_df is not None:
        title += f" + {integrated_model_label} Gap"
    fig.suptitle(
        title,
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "Day 0 is the relevant report's publication date; the alarm score uses only news and official evidence available by that date",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    footer = (
        "Same-day detection does not predict the physical event, but it can still warn of downstream supply impact. "
        "Taiwan and Malaysia show pre-event alarm lead without a Census threshold crossing."
    )
    if integrated_event_df is not None:
        footer = (
            "Alarm score measures urgency; expected gap is the Day-0 propagation estimate; observed gap is evaluated later and never backdated."
        )
    fig.text(
        0.075,
        0.025,
        footer,
        color=COLORS["muted"],
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.19, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(output_path or FIG_DIR / "figure3_news_early_alarm_timeline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    articles, event_result, metrics = build_fulltext_replay()
    corpus = pd.read_csv(CORPUS_INDEX)
    articles.to_csv(OUT_DIR / "fulltext_news_article_evaluation.csv", index=False)
    event_result.to_csv(OUT_DIR / "fulltext_news_event_replay.csv", index=False)
    (OUT_DIR / "fulltext_news_replay_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    plot_corpus_coverage(corpus, event_result)
    plot_understanding(metrics)
    plot_timing(event_result, metrics)
    plot_figure3_event_validation(corpus, event_result)
    alarm_timeline = build_figure3_alarm_timeline(event_result)
    alarm_timeline.to_csv(
        OUT_DIR / "figure3_news_early_alarm_timeline.csv",
        columns=[
            "event_key",
            "event_label",
            "source",
            "title",
            "news_identified_date",
            "operational_event_date",
            "operational_event",
            "census_consequence_date",
            "news_to_event_days",
            "news_to_census_consequence_days",
            "support_available_at_news",
            "risk_score_at_news",
            "alert_level_at_news",
            "confirmation_status",
        ],
        index=False,
    )
    plot_figure3_news_alarm_timeline(alarm_timeline)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()