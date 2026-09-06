#!/usr/bin/env python3
"""
Visualize the Module 1 HS 9403 (furniture) vs Panama Canal drought validation.

Reads the outputs of module1_panama_furniture_validation.py
(reports/module1/panama_furniture_hs9403_monthly.csv and
panama_furniture_hs9403_validation.json) and renders a 3-panel dashboard:

  A) YoY% time series for the Asia source-country group vs the Canada/Mexico
     control group, with the event date, -20% confirmation threshold, and
     each group's confirmation month marked.
  B) Model-predicted vs trend-counterfactual-observed supply gap (%).
  C) Early-warning timeline: ACP official advisory / news wire date vs the
     approximate date customs/trade data would become publicly available,
     showing the lead time news-based monitoring provides.

Output: reports/module1/figures/panama_furniture_hs9403_dashboard.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"

COLORS = {
    "ink": "#253238",
    "muted": "#60747b",
    "asia": "#bd4b42",
    "control": "#287271",
    "threshold": "#8b9699",
    "event": "#253238",
    "predicted": "#bd4b42",
    "observed": "#287271",
    "advisory": "#d6a534",
    "news": "#bd4b42",
    "model": "#287271",
    "customs": "#8b9699",
    "paper": "#f4f1e9",
    "line": "#d3d7d4",
}

EVENT_DATE = pd.Timestamp("2023-10-31")
PLOT_START = pd.Timestamp("2022-06-01")
PLOT_END = pd.Timestamp("2024-12-01")


def _load() -> tuple[pd.DataFrame, dict]:
    monthly = pd.read_csv(OUT_DIR / "panama_furniture_hs9403_monthly.csv", parse_dates=["date"])
    metrics = json.loads((OUT_DIR / "panama_furniture_hs9403_validation.json").read_text(encoding="utf-8"))
    return monthly, metrics


def _panel_yoy_timeline(ax, monthly: pd.DataFrame, metrics: dict) -> None:
    asia = monthly[(monthly["series"] == "asia_selected_group") & monthly["date"].between(PLOT_START, PLOT_END)]
    control = monthly[(monthly["series"] == "control_canada_mexico") & monthly["date"].between(PLOT_START, PLOT_END)]

    ax.plot(asia["date"], asia["yoy_pct"], color=COLORS["asia"], lw=2.2, label="Asia source-country group (auto-selected)")
    ax.plot(control["date"], control["yoy_pct"], color=COLORS["control"], lw=2.0, ls="--", label="Control: Canada + Mexico (overland)")

    ax.axhline(-20.0, color=COLORS["threshold"], lw=1.2, ls=":")
    ax.text(PLOT_START, -20.0, " -20% confirmation threshold", color=COLORS["muted"], fontsize=8.5, va="bottom")

    ax.axvline(EVENT_DATE, color=COLORS["event"], lw=1.4)
    ax.text(EVENT_DATE, ax.get_ylim()[1] if ax.get_ylim()[1] else 10, "  ACP booking-slot\n  cut (2023-10-31)",
            color=COLORS["ink"], fontsize=8.5, va="top")

    earliest_asia = metrics.get("earliest_asia_confirmation_month")
    if earliest_asia:
        d = pd.Timestamp(earliest_asia)
        row = asia.loc[asia["date"] == d]
        if not row.empty:
            ax.scatter([d], [row["yoy_pct"].iloc[0]], color=COLORS["asia"], zorder=5, s=60, edgecolor="white", lw=1)
            ax.annotate("Earliest Asia\nconfirmation", (d, row["yoy_pct"].iloc[0]), textcoords="offset points",
                        xytext=(8, -28), fontsize=8, color=COLORS["asia"])

    control_month = metrics.get("control_group_result_canada_mexico", {}).get("confirmation_month")
    if control_month:
        d = pd.Timestamp(control_month)
        row = control.loc[control["date"] == d]
        if not row.empty:
            ax.scatter([d], [row["yoy_pct"].iloc[0]], color=COLORS["control"], zorder=5, s=60, edgecolor="white", lw=1)
            ax.annotate(
                f"Control confirms\n(lag {metrics.get('control_lag_months_vs_earliest_asia')} mo)",
                (d, row["yoy_pct"].iloc[0]), textcoords="offset points", xytext=(8, 10), fontsize=8, color=COLORS["control"],
            )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("YoY % change in import value")
    ax.set_title("A) HS 9403 import YoY%: Asia source countries vs overland control group", loc="left", fontsize=11, weight="bold", color=COLORS["ink"])
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _panel_supply_gap(ax, metrics: dict) -> None:
    gap = metrics["module1_supply_gap_prediction"]
    sm = metrics["existing_data_validation_checks"]["severity_match_via_yoy_method"]

    labels = [
        "Model-predicted\n(same-day, PropagationEngine)",
        "Trend-counterfactual observed\n(insensitive metric)",
        "YoY-implied severity\n(trade-data-confirmed)",
    ]
    values = [gap["predicted_supply_gap_pct"], gap["observed_supply_gap_pct"], sm["yoy_implied_severity_pct"]]
    severities = [gap["severity_bin_predicted"], gap["severity_bin_observed"], sm["yoy_implied_severity_bin"]]
    colors = [COLORS["predicted"], COLORS["muted"], COLORS["observed"]]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, val, sev in zip(bars, values, severities):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val}%\n({sev})",
                ha="center", va="bottom", fontsize=9, color=COLORS["ink"])
    ax.set_ylim(0, max(values) * 1.5 + 3)
    ax.set_ylabel("Supply gap / severity (%)")
    ax.set_title("B) Model prediction vs two observed-severity metrics", loc="left", fontsize=11, weight="bold", color=COLORS["ink"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.2, linespacing=1.4)
    match_text = (
        "Severity MATCH vs trade-confirmed YoY metric: HIGH = HIGH"
        if sm.get("severity_bins_match")
        else "Severity mismatch vs trade-confirmed YoY metric"
    )
    ax.text(
        0.02, 0.98,
        match_text + "\n(trend-counterfactual metric is 0% in every month -- see report note)",
        transform=ax.transAxes, fontsize=8, va="top", color=COLORS["asia"] if sm.get("severity_bins_match") else COLORS["muted"],
        weight="bold",
    )


def _panel_early_warning(ax, metrics: dict) -> None:
    ew = metrics["news_early_warning"]
    advisory_date = pd.Timestamp(ew["official_advisory_date"])
    news_date = pd.Timestamp(ew["news_wire_date"])
    customs_date = pd.Timestamp(ew["census_public_availability_date_approx"])

    events = [
        (advisory_date, "ACP official advisory", COLORS["advisory"], 0.0, (-45, 20)),
        (news_date, "Reuters news wire", COLORS["news"], 0.0, (55, -22)),
        (news_date, "Model prediction\navailable (same day)", COLORS["model"], 0.35, (0, 16)),
        (customs_date, "Customs data publicly\navailable (~est.)", COLORS["customs"], 0.0, (0, 20)),
    ]
    for date, label, color, y, offset in events:
        ax.scatter([date], [y], color=color, s=110, zorder=5, edgecolor="white", lw=1.2)
        ax.annotate(f"{label}\n{date.strftime('%Y-%m-%d')}", (date, y), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=8, color=COLORS["ink"])

    ax.annotate(
        "", xy=(customs_date, -0.42), xytext=(news_date, -0.42),
        arrowprops=dict(arrowstyle="<->", color=COLORS["muted"], lw=1.2),
    )
    ax.text(
        news_date + (customs_date - news_date) / 2, -0.55,
        f"~{ew['lead_time_days_news_wire_vs_customs_data']} days early-warning lead time",
        ha="center", fontsize=9.5, color=COLORS["ink"], weight="bold",
    )

    pad = pd.Timedelta(days=4)
    ax.set_xlim(advisory_date - pad, customs_date + pad)
    ax.set_ylim(-0.75, 1.05)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("C) News/official advisory vs customs-data early-warning lead time", loc="left", fontsize=11, weight="bold", color=COLORS["ink"])
    ax.tick_params(labelsize=8.2)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    monthly, metrics = _load()

    fig = plt.figure(figsize=(12, 12), facecolor=COLORS["paper"])
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.6, wspace=0.3, top=0.9, bottom=0.06, left=0.08, right=0.96)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_facecolor(COLORS["paper"])

    _panel_yoy_timeline(ax_a, monthly, metrics)
    _panel_supply_gap(ax_b, metrics)

    # Recommendation + rationale text panel
    ax_c.axis("off")
    ax_c.set_title("Recommendation", loc="left", fontsize=11, weight="bold", color=COLORS["ink"])
    ax_c.text(
        0, 0.9, metrics["recommendation"].replace("_", " ").upper(),
        fontsize=12, weight="bold", color=COLORS["asia"], va="top", wrap=True,
    )
    breadth = metrics["breadth_pct_of_selected_asia_confirmed"]
    sm = metrics["existing_data_validation_checks"]["severity_match_via_yoy_method"]
    ax_c.text(
        0, 0.65,
        f"Breadth: {breadth}% of selected Asia countries confirmed\n"
        f"Asia median trough YoY: {metrics['asia_median_trough_yoy_pct']}%\n"
        f"Control lag vs Asia: {metrics['control_lag_months_vs_earliest_asia']} months\n"
        f"Severity match (predicted vs YoY-confirmed): "
        f"{sm['predicted_severity_bin']} = {sm['yoy_implied_severity_bin']} "
        f"({'MATCH' if sm['severity_bins_match'] else 'no match'})",
        fontsize=9, color=COLORS["ink"], va="top",
    )

    _panel_early_warning(ax_d, metrics)

    fig.suptitle(
        "Module 1 Validation: HS 9403 Furniture vs Panama Canal Drought 2023-2024",
        x=0.08, y=0.985, ha="left", fontsize=16, weight="bold", color=COLORS["ink"],
    )
    fig.text(
        0.08, 0.955,
        "Auto-selected Asia sources: " + ", ".join(metrics["auto_selected_main_asia_source_countries"]),
        fontsize=9.5, color=COLORS["muted"],
    )

    out_path = FIG_DIR / "panama_furniture_hs9403_dashboard.png"
    fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
