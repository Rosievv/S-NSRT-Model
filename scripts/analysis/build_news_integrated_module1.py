#!/usr/bin/env python3
"""Run an experimental news-triggered integration with Module 1 propagation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from build_event_listener_v2 import _signal_score, build_v2_replay
from build_fulltext_news_replay import plot_figure3_news_alarm_timeline
from module1_backtest import generate_figure3
from module1_data_loader import load_module1_trade_data
from risk_propagation import NewsRiskSignal, StressTestRunner, run_news_triggered_scenario
from risk_propagation.stress_testing import HISTORICAL_EVENTS


OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"
BACKTEST_PATH = OUT_DIR / "module1_backtest.csv"
UNIFIED_VALIDATION_PATH = OUT_DIR / "module1_2020_2026_unified_validation.csv"
ALARM_TIMELINE_PATH = OUT_DIR / "figure3_news_early_alarm_timeline.csv"
FIGURE3_EVENT_KEYS = [
    "japan_earthquake_2011",
    "thai_flood_2011",
    "japan_export_controls_2019",
    "covid_q1_2020",
    "taiwan_drought_2021",
    "malaysia_asia_shock_2021",
]
EVENT_LABELS = {
    "japan_earthquake_2011": "Japan earthquake",
    "thai_flood_2011": "Thailand floods",
    "japan_export_controls_2019": "Japan-Korea controls",
    "covid_q1_2020": "China COVID lockdown",
    "taiwan_drought_2021": "Taiwan drought",
    "malaysia_asia_shock_2021": "Malaysia lockdown",
}
MALAYSIA_OUTCOME_EVENT = {
    "name": "malaysia_asia_shock_2021",
    "description": "Malaysia total lockdown and semiconductor assembly/test disruption",
    "date_range": ("2021-06", "2021-08"),
    "affected_countries": ["Malaysia"],
    "affected_hs_codes": ["854231", "854232", "854239"],
    "event_type": "pandemic",
    "estimated_severity": 0.20,
}
COLORS = {
    "ink": "#253238",
    "muted": "#60747b",
    "static": "#9ba8aa",
    "news": "#287271",
    "observed": "#d97732",
    "critical": "#b5362f",
    "line": "#d3d7d4",
}


def _split_values(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip() and item.strip() != "unknown"]


def _day0_support(row: pd.Series) -> int:
    signal_date = pd.Timestamp(row["event_date"])
    return sum(
        pd.Timestamp(item["date"]) <= signal_date
        for item in json.loads(row["official_support_sources"])
    )


def build_integrated_replay() -> tuple[pd.DataFrame, dict]:
    trade_df = load_module1_trade_data()
    listener, _, _, _ = build_v2_replay()
    listener = listener.loc[listener["event_key"].isin(FIGURE3_EVENT_KEYS)].copy()
    known_countries = set(trade_df["country"].astype(str))

    stress_runner = StressTestRunner(trade_df)
    static_backtest = stress_runner.backtest_all().set_index("event")
    event_definitions = {event["name"]: event for event in HISTORICAL_EVENTS}
    event_definitions[MALAYSIA_OUTCOME_EVENT["name"]] = MALAYSIA_OUTCOME_EVENT

    records = []
    for _, row in listener.iterrows():
        support_count = _day0_support(row)
        day0_score = _signal_score(
            row["source_tier"],
            row["v2_confidence"],
            support_count + 1,
            row["predicted_events"] != "unknown",
        )
        countries = [
            country for country in _split_values(row["v2_location"])
            if country in known_countries
        ]
        signal = NewsRiskSignal(
            event_key=row["event_key"],
            signal_date=pd.Timestamp(row["event_date"]),
            event_type=row["v2_event_type"],
            countries=countries,
            affected_objects=_split_values(row["predicted_object_top3"]),
            alarm_score=day0_score,
            headline=row["title"],
            corroborating_sources=support_count,
        )
        triggered = run_news_triggered_scenario(trade_df, signal)
        outcome = stress_runner._build_observed_supply_gap(event_definitions[row["event_key"]])
        static_row = static_backtest.loc[row["event_key"]] if row["event_key"] in static_backtest.index else None
        static_gap = float(static_row["predicted_supply_gap_pct"]) if static_row is not None else np.nan
        observed_gap = float(outcome["observed_supply_gap_pct"])
        records.append(
            {
                "event_key": row["event_key"],
                "event_label": EVENT_LABELS[row["event_key"]],
                "news_date": pd.Timestamp(row["event_date"]).strftime("%Y-%m-%d"),
                "news_source": row["source"],
                "news_title": row["title"],
                "detected_event_type": row["v2_event_type"],
                "detected_countries": "; ".join(countries),
                "detected_objects": "; ".join(signal.affected_objects),
                "day0_alarm_score": day0_score,
                "day0_alarm_level": "critical" if day0_score >= 0.8 else "warning" if day0_score >= 0.65 else "watch",
                "dynamic_severity": triggered.dynamic_severity,
                "government_data_available_through": triggered.government_data_available_through.strftime("%Y-%m-%d"),
                "government_data_start": triggered.government_data_start.strftime("%Y-%m-%d"),
                "news_triggered_supply_gap_pct": triggered.propagation.supply_gap_pct,
                "static_backtest_supply_gap_pct": static_gap,
                "observed_supply_gap_pct": observed_gap,
                "news_triggered_abs_error_pct": abs(triggered.propagation.supply_gap_pct - observed_gap),
                "static_abs_error_pct": abs(static_gap - observed_gap) if pd.notna(static_gap) else np.nan,
                "substitution_absorbed_pct": triggered.propagation.substitution_absorbed_pct,
                "hs_codes": "; ".join(triggered.hs_codes),
            }
        )

    result = pd.DataFrame(records).set_index("event_key").loc[FIGURE3_EVENT_KEYS].reset_index()
    comparable = result.dropna(subset=["static_backtest_supply_gap_pct"])
    metrics = {
        "events": int(len(result)),
        "day0_critical_alarms": int(result["day0_alarm_level"].eq("critical").sum()),
        "government_data_lag_months": 1,
        "government_data_lookback_months": 24,
        "static_mae_pct_points_common_events": round(float(comparable["static_abs_error_pct"].mean()), 2),
        "news_triggered_mae_pct_points_common_events": round(float(comparable["news_triggered_abs_error_pct"].mean()), 2),
        "news_triggered_mae_pct_points_all_events": round(float(result["news_triggered_abs_error_pct"].mean()), 2),
        "magnitude_accuracy_improved_on_common_events": bool(
            comparable["news_triggered_abs_error_pct"].mean() < comparable["static_abs_error_pct"].mean()
        ),
        "interpretation": (
            "News integration provides a time-available dynamic stress scenario. "
            "Magnitude accuracy remains experimental and must be calibrated on a larger out-of-sample event set."
        ),
    }
    return result, metrics


def plot_integrated_replay(result: pd.DataFrame, metrics: dict) -> None:
    positions = np.arange(len(result))
    fig, (severity_ax, impact_ax) = plt.subplots(
        1,
        2,
        figsize=(16, 8.2),
        gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.28},
    )

    severity_bars = severity_ax.barh(
        positions,
        result["dynamic_severity"] * 100,
        color=COLORS["critical"],
        height=0.58,
    )
    severity_ax.set_yticks(positions, result["event_label"])
    severity_ax.invert_yaxis()
    severity_ax.set_xlabel("News-derived dynamic severity (%)")
    severity_ax.set_title("Day-0 scenario update", loc="left", fontsize=13, weight="bold")
    severity_ax.bar_label(severity_bars, fmt="%.1f%%", padding=3, fontsize=9)
    severity_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)

    width = 0.24
    static_values = result["static_backtest_supply_gap_pct"].fillna(0)
    impact_ax.barh(positions - width, static_values, height=width, color=COLORS["static"], label="Original static backtest")
    impact_ax.barh(positions, result["news_triggered_supply_gap_pct"], height=width, color=COLORS["news"], label="News-triggered propagation")
    impact_ax.barh(positions + width, result["observed_supply_gap_pct"], height=width, color=COLORS["observed"], label="Observed event-window gap")
    impact_ax.set_yticks(positions, [""] * len(positions))
    impact_ax.invert_yaxis()
    impact_ax.set_xlabel("Supply gap (%)")
    impact_ax.set_title("Propagation output vs. later observed outcome", loc="left", fontsize=13, weight="bold")
    impact_ax.grid(axis="x", color=COLORS["line"], linewidth=0.8)
    impact_ax.legend(frameon=False, loc="lower right")
    impact_ax.text(
        0.99,
        0.99,
        f'Common-event MAE\nStatic: {metrics["static_mae_pct_points_common_events"]:.2f} pp\n'
        f'News-triggered: {metrics["news_triggered_mae_pct_points_common_events"]:.2f} pp',
        transform=impact_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": COLORS["line"], "boxstyle": "round,pad=0.4"},
    )

    for ax in [severity_ax, impact_ax]:
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Experimental Integration: News Signals Update Module 1 Before Government Outcomes Arrive",
        x=0.065,
        y=0.97,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.065,
        0.925,
        "News determines the event scenario at publication time; lagged Census data determines exposure, substitution, and propagated supply gap",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    fig.text(
        0.065,
        0.03,
        "This replay validates timeliness and technical integration. It does not yet demonstrate improved magnitude accuracy; severity calibration requires more out-of-sample events.",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.19, right=0.97, top=0.86, bottom=0.12)
    fig.savefig(FIG_DIR / "figure3_news_integrated_module1.png", dpi=180)
    plt.close(fig)


def plot_integrated_legacy_views(result: pd.DataFrame) -> None:
    """Regenerate the two legacy layouts with integrated event values."""
    required_paths = [BACKTEST_PATH, UNIFIED_VALIDATION_PATH, ALARM_TIMELINE_PATH]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing legacy chart inputs: {missing}")

    generate_figure3(
        backtest_df=pd.read_csv(BACKTEST_PATH),
        projection_df=pd.DataFrame(),
        validation_df=pd.read_csv(UNIFIED_VALIDATION_PATH),
        output_path=FIG_DIR / "figure3_backtest_and_2026_forecast_integrated.png",
        integrated_event_df=result,
        integrated_event_position="actual_line",
    )
    plot_figure3_news_alarm_timeline(
        timeline=pd.read_csv(ALARM_TIMELINE_PATH),
        output_path=FIG_DIR / "figure3_news_early_alarm_timeline_integrated.png",
        integrated_event_df=result,
    )


def plot_module1_workflow() -> None:
    """Visualize the current news-to-propagation decision flow."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_facecolor("#f7f8f5")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        detail: str,
        facecolor: str,
        edgecolor: str,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.6,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        ax.add_patch(patch)
        ax.text(x + 0.22, y + height - 0.35, title, fontsize=11.5, weight="bold", va="top", color=COLORS["ink"])
        ax.text(x + 0.22, y + height - 0.78, detail, fontsize=8.6, va="top", color=COLORS["muted"], linespacing=1.35)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#748488") -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.7,
                color=color,
                shrinkA=3,
                shrinkB=3,
                connectionstyle="arc3,rad=0",
            )
        )

    ax.text(0.55, 8.45, "Module 1: From Timely News to an Actionable Supply-Risk Warning", fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(
        0.55,
        8.02,
        "News supplies speed and event context; lagged government trade data supplies exposure and network structure",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    ax.text(0.65, 7.28, "TIMELY SIGNAL LANE", fontsize=10, weight="bold", color="#a93630")
    box(0.65, 5.55, 2.65, 1.45, "1  Listen", "Real news + official sources\nPublication-time evidence only", "#fff1ed", "#c95a45")
    box(3.75, 5.55, 2.65, 1.45, "2  Understand", "Event type + country\nAffected object + cause\nReliability + corroboration", "#fff1ed", "#c95a45")
    box(6.85, 5.55, 2.65, 1.45, "3  Set Day-0 scenario", "Alarm score = confidence\nSeverity = disruption scope", "#fff1ed", "#c95a45")
    arrow((3.30, 6.28), (3.75, 6.28), "#c95a45")
    arrow((6.40, 6.28), (6.85, 6.28), "#c95a45")

    ax.text(0.65, 4.55, "STRUCTURAL BASELINE LANE", fontsize=10, weight="bold", color="#1f6968")
    box(0.65, 2.82, 2.65, 1.45, "A  Load Census trade", "Country x HS x month values\nDeduplicate latest snapshots\nRemove aggregate geographies", "#e8f3f1", "#287271")
    box(3.75, 2.82, 2.65, 1.45, "B  Prevent leakage", "Data through signal month - 1\n24-month lookback only", "#e8f3f1", "#287271")
    box(6.85, 2.82, 2.65, 1.45, "C  Build exposure", "Supplier share + HHI\nReliability + activity\nHS substitution elasticity", "#e8f3f1", "#287271")
    arrow((3.30, 3.55), (3.75, 3.55), "#287271")
    arrow((6.40, 3.55), (6.85, 3.55), "#287271")

    box(10.35, 4.12, 2.70, 2.05, "4  Propagate shock", "Direct supplier loss\n- weighted substitution capacity\n= net supply gap by HS", "#edf1f2", "#53686e")
    arrow((9.50, 6.28), (10.35, 5.56), "#c95a45")
    arrow((9.50, 3.55), (10.35, 4.72), "#287271")

    box(13.60, 4.12, 1.85, 2.05, "5  Warn firms", "Alarm level\nExpected gap\nAffected HS\nData vintage", "#fff5e9", "#d97732")
    arrow((13.05, 5.15), (13.60, 5.15), "#d97732")

    ax.text(10.35, 3.40, "Later outcome feedback", fontsize=10, weight="bold", color=COLORS["ink"])
    ax.text(10.35, 3.03, "Observed Census event-window gap evaluates calibration; it never backdates the Day-0 warning.", fontsize=9.5, color=COLORS["muted"])
    arrow((14.50, 4.10), (12.75, 3.38), "#9ba8aa")

    ax.text(
        0.65,
        1.25,
        "Operational interpretation",
        fontsize=11,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.65,
        0.82,
        "A high alarm score means investigate now. A high propagated supply gap means prepare inventory, sourcing, or supplier actions.",
        fontsize=11,
        color=COLORS["muted"],
    )
    ax.text(
        0.65,
        0.42,
        "Current status: technically integrated and leakage-controlled; impact-severity calibration remains experimental.",
        fontsize=10.5,
        color="#a93630",
        weight="bold",
    )

    fig.savefig(FIG_DIR / "module1_news_integration_flow.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result, metrics = build_integrated_replay()
    result.to_csv(OUT_DIR / "figure3_news_integrated_module1.csv", index=False)
    (OUT_DIR / "figure3_news_integrated_module1_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    plot_integrated_replay(result, metrics)
    plot_integrated_legacy_views(result)
    plot_module1_workflow()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()