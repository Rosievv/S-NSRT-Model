#!/usr/bin/env python3
"""Create operations-focused visuals for the real-event monitoring proposal."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from build_event_listener_v2 import build_v2_replay


ROOT_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT_DIR / "reports" / "module1" / "figures"

COLORS = {
    "ink": "#253238",
    "muted": "#60747b",
    "paper": "#f4f1e9",
    "news": "#287271",
    "official": "#d97732",
    "supply": "#347c78",
    "transport": "#d06b36",
    "early": "#23856d",
    "same_day": "#d6a534",
    "late": "#bd4b42",
    "unconfirmed": "#8b9699",
    "line": "#c9cec9",
}

EVENT_LABELS = {
    "japan_earthquake_2011": "2011 Japan earthquake",
    "thai_flood_2011": "2011 Thailand floods",
    "japan_export_controls_2019": "2019 Japan-Korea export controls",
    "covid_q1_2020": "2020 China COVID lockdown",
    "taiwan_drought_2021": "2021 Taiwan drought",
    "malaysia_asia_shock_2021": "2021 Malaysia lockdown",
    "beirut_port_explosion_2020": "2020 Beirut port explosion",
    "suez_ever_given_2021": "2021 Suez Canal blockage",
    "yantian_port_covid_2021": "2021 Yantian port congestion",
    "panama_canal_drought_2023": "2023 Panama Canal restrictions",
    "red_sea_rerouting_2023": "2023 Red Sea carrier pause",
    "baltimore_bridge_collapse_2024": "2024 Baltimore bridge collapse",
}


def _configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ["PingFang SC", "Arial Unicode MS", "Heiti TC", "Songti SC"]:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    textcolor: str = "white",
    fontsize: float = 12,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        weight="semibold",
        linespacing=1.35,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["muted"],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def plot_monitoring_workflow() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2), facecolor=COLORS["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["paper"])

    ax.text(0.05, 0.92, "Real-Event Monitoring and Risk Analysis Workflow", fontsize=24, weight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.865,
        "Use timestamped public news and authoritative notices to identify events and route them to the appropriate risk module",
        fontsize=12.5,
        color=COLORS["muted"],
    )

    _box(ax, (0.05, 0.60), 0.17, 0.16, "Public news sources\nGoogle News / Reuters / BBC", COLORS["news"], fontsize=11.5)
    _box(ax, (0.05, 0.34), 0.17, 0.16, "Authoritative notices\nGovernment / port / canal / operator", COLORS["official"], fontsize=11.5)

    _box(ax, (0.31, 0.46), 0.20, 0.20, "Event extraction and aggregation\nTime · location · type\nSource reliability · corroboration", COLORS["ink"], fontsize=11.5)
    _arrow(ax, (0.22, 0.68), (0.31, 0.59), COLORS["news"])
    _arrow(ax, (0.22, 0.42), (0.31, 0.53), COLORS["official"])

    _box(ax, (0.59, 0.62), 0.16, 0.14, "Supply-source impact\nCountry / cluster / material", COLORS["supply"], fontsize=11.5)
    _box(ax, (0.59, 0.34), 0.16, 0.14, "Transport-channel impact\nPort / canal / shipping lane", COLORS["transport"], fontsize=11.5)
    _arrow(ax, (0.51, 0.57), (0.59, 0.69), COLORS["supply"])
    _arrow(ax, (0.51, 0.55), (0.59, 0.41), COLORS["transport"])

    _box(ax, (0.82, 0.62), 0.14, 0.14, "Trigger supply-risk analysis\nExposure · alternatives · stress test", COLORS["supply"], fontsize=10.2)
    _box(ax, (0.82, 0.34), 0.14, 0.14, "Trigger transport-risk analysis\nRerouting · capacity · delay propagation", COLORS["transport"], fontsize=10.2)
    _arrow(ax, (0.75, 0.69), (0.82, 0.69), COLORS["supply"])
    _arrow(ax, (0.75, 0.41), (0.82, 0.41), COLORS["transport"])

    ax.text(0.05, 0.16, "Historical replay questions", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.105,
        "Was the event detected in time?  Were its location, type, and affected object identified?  Was the alert earlier than Customs or operational consequences?",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.savefig(FIG_DIR / "real_event_monitor_workflow.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _lead_status(row: pd.Series) -> str:
    if row["confirmation_status"] != "confirmed":
        return "unconfirmed"
    if row["listening_lead_days"] > 0:
        return "early"
    if row["listening_lead_days"] == 0:
        return "same_day"
    return "late"


def _plot_channel_leads(ax: plt.Axes, data: pd.DataFrame, channel: str, title: str, rule: str) -> None:
    subset = data.loc[data["channel"] == channel].copy()
    subset["label"] = subset["event_key"].map(EVENT_LABELS)
    subset["status"] = subset.apply(_lead_status, axis=1)
    subset["plot_lead"] = subset["listening_lead_days"].fillna(0)
    subset = subset.sort_values(["confirmation_status", "plot_lead"])
    colors = subset["status"].map({key: COLORS[key] for key in ["early", "same_day", "late", "unconfirmed"]})

    bars = ax.barh(subset["label"], subset["plot_lead"], color=colors, height=0.62)
    ax.axvline(0, color=COLORS["ink"], linewidth=1.1)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=15, weight="bold", color=COLORS["ink"], pad=12)
    ax.text(0, 1.01, rule, transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom")
    ax.set_xlabel("Lead days relative to confirmation (negative means late detection)", color=COLORS["muted"])
    ax.tick_params(axis="y", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for bar, (_, row) in zip(bars, subset.iterrows()):
        if row["status"] == "unconfirmed":
            text = "Outcome threshold not reached"
            x = 1.2
            ha = "left"
        else:
            lead = int(row["plot_lead"])
            text = "Same day" if lead == 0 else f"{lead:+d} days"
            x = lead + (2 if lead >= 0 else -0.5)
            ha = "left" if lead >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, text, va="center", ha=ha, fontsize=9.5, color=COLORS["ink"])


def plot_historical_replay(result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), facecolor="white", gridspec_kw={"wspace": 0.43})
    _plot_channel_leads(
        axes[0],
        result,
        "supply_source",
        "Supply-source events (6)",
        "Confirmation: Census import value declined at least 20% YoY; early warning requires 14+ days",
    )
    _plot_channel_leads(
        axes[1],
        result,
        "transport_channel",
        "Transport-channel events (6)",
        "Confirmation: first documented closure, restriction, carrier pause, or container stop; detection must precede the consequence",
    )
    fig.suptitle("Replay of 12 Real Historical Events: Signal Lead Time Before Consequence Confirmation", x=0.06, ha="left", fontsize=21, weight="bold", color=COLORS["ink"])
    fig.text(
        0.06,
        0.025,
        "Green = early; yellow = same day; red = late; gray = consequence threshold not reached. Development replay, not a blind generalization test.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    fig.savefig(FIG_DIR / "real_event_monitor_historical_replay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_operational_evidence(result: pd.DataFrame, controls: pd.DataFrame, metrics: dict) -> None:
    result = result.copy()
    result["status"] = result.apply(_lead_status, axis=1)
    status_counts = result["status"].value_counts()
    recognition = [
        metrics["event_type_match_rate_pct"],
        metrics["location_match_rate_pct"],
        metrics["impact_target_match_rate_pct"],
        metrics["object_top3_hit_rate_pct"],
    ]

    fig = plt.figure(figsize=(15, 8.5), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.28], hspace=0.35, wspace=0.32)
    fig.suptitle("Real-Event Monitoring: Operational Evidence from Historical Replay", x=0.055, ha="left", fontsize=23, weight="bold", color=COLORS["ink"])

    cards = [
        ("12", "Real historical events", "6 supply source + 6 transport channel"),
        ("10", "Consequences confirmed", "Customs anomaly or operational consequence"),
        ("5", "Strict early alerts", "4 supply + 1 transport"),
    ]
    for index, (value, title, subtitle) in enumerate(cards):
        ax = fig.add_subplot(grid[0, index])
        ax.axis("off")
        ax.text(0.02, 0.65, value, fontsize=42, weight="bold", color=[COLORS["news"], COLORS["official"], COLORS["early"]][index])
        ax.text(0.02, 0.36, title, fontsize=14, weight="bold", color=COLORS["ink"])
        ax.text(0.02, 0.16, subtitle, fontsize=10.5, color=COLORS["muted"])
        ax.plot([0.02, 0.96], [0.04, 0.04], color=COLORS["line"], linewidth=1.2)

    ax_recognition = fig.add_subplot(grid[1, :2])
    labels = ["Event type", "Location", "Impact channel", "Affected object Top-3"]
    bars = ax_recognition.barh(labels[::-1], recognition[::-1], color=[COLORS["news"], COLORS["supply"], COLORS["official"], COLORS["transport"]])
    ax_recognition.bar_label(bars, fmt="%.0f%%", padding=5, fontsize=11, weight="bold")
    ax_recognition.set_xlim(0, 110)
    ax_recognition.set_title("Event Understanding and Routing", loc="left", fontsize=15, weight="bold", color=COLORS["ink"])
    ax_recognition.text(0, 1.03, "Hit rate within the 12-event development replay", transform=ax_recognition.transAxes, color=COLORS["muted"], fontsize=10)
    ax_recognition.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    ax_recognition.set_axisbelow(True)
    for spine in ax_recognition.spines.values():
        spine.set_visible(False)

    ax_outcomes = fig.add_subplot(grid[1, 2])
    outcome_values = [
        int(status_counts.get("early", 0)),
        int(status_counts.get("same_day", 0)),
        int(status_counts.get("late", 0)),
        int(status_counts.get("unconfirmed", 0)),
    ]
    outcome_labels = ["Early", "Same day", "Late", "Unconfirmed"]
    outcome_colors = [COLORS["early"], COLORS["same_day"], COLORS["late"], COLORS["unconfirmed"]]
    wedges, _ = ax_outcomes.pie(
        outcome_values,
        colors=outcome_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": COLORS["paper"], "linewidth": 2},
    )
    ax_outcomes.text(0, 0.08, "41.7%", ha="center", va="center", fontsize=24, weight="bold", color=COLORS["ink"])
    ax_outcomes.text(0, -0.13, "strict early-alert rate", ha="center", va="center", fontsize=10, color=COLORS["muted"])
    ax_outcomes.set_title("Alert Timing", fontsize=15, weight="bold", color=COLORS["ink"], pad=12)
    ax_outcomes.legend(wedges, [f"{label} {value}" for label, value in zip(outcome_labels, outcome_values)], loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9.5)

    fig.text(
        0.055,
        0.025,
        f"Additional check: {len(controls)} real official non-disruption records triggered no elevated alerts; earlier time-available sources gained 18 total days across 3 events. Development replay, not a production blind test.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    fig.savefig(FIG_DIR / "real_event_monitor_operational_evidence.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_fonts()
    result, controls, metrics, _ = build_v2_replay()
    plot_monitoring_workflow()
    plot_historical_replay(result)
    plot_operational_evidence(result, controls, metrics)
    print("Created operations-focused event-monitor visuals:")
    for name in [
        "real_event_monitor_workflow.png",
        "real_event_monitor_historical_replay.png",
        "real_event_monitor_operational_evidence.png",
    ]:
        print(FIG_DIR / name)


if __name__ == "__main__":
    main()