#!/usr/bin/env python3
"""
Run Module 1 historical backtests plus 2025-2026 forward projections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
)


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from module1_data_loader import load_module1_trade_data
from risk_propagation import PropagationEngine, SupplyChainNetwork, StressTestRunner
from risk_propagation.stress_testing import HISTORICAL_EVENTS


OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "module1_stress_summary.csv"
BACKTEST_PATH = OUT_DIR / "module1_backtest.csv"
RUNTIME_BACKTEST_PATH = OUT_DIR / "module1_backtest_runtime.csv"
PROJECTION_JSON_PATH = OUT_DIR / "module1_projections_2025_2026.json"
FIGURE3_PATH = FIG_DIR / "figure3_backtest_and_2026_forecast.png"


def _event_start_map() -> dict[str, pd.Timestamp]:
    return {
        event["name"]: pd.Timestamp(f"{event['date_range'][0]}-01")
        for event in HISTORICAL_EVENTS
    }


def _historical_summary_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    historical = summary_df.copy()
    historical["phase"] = "historical"
    historical["quarter"] = ""
    historical["quarter_start"] = ""
    historical["quarter_end"] = ""
    historical["description"] = ""
    historical["scenario_type"] = ""
    historical["lower_gap_pct"] = pd.NA
    historical["expected_gap_pct"] = pd.NA
    historical["upper_gap_pct"] = pd.NA
    historical["projection_band_width_pct"] = pd.NA
    historical["scenario_status"] = "historical_backtest"
    return historical


def _forecast_summary_frame(projection_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        return projection_df.copy()

    forecast = projection_df.copy()
    forecast["phase"] = "forecast"
    forecast["supply_gap_pct"] = forecast["expected_gap_pct"]
    forecast["scenario"] = forecast["scenario"].astype(str)
    forecast["shocked_nodes"] = forecast["shocked_nodes"].astype(str)
    forecast["severity"] = forecast["severity"].astype(float)
    return forecast


def _align_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "phase",
        "scenario",
        "shocked_nodes",
        "severity",
        "original_supply_B",
        "disrupted_supply_B",
        "supply_gap_pct",
        "substitution_absorbed_pct",
        "most_affected_hs",
        "description",
        "scenario_type",
        "quarter",
        "quarter_start",
        "quarter_end",
        "lower_gap_pct",
        "expected_gap_pct",
        "upper_gap_pct",
        "projection_band_width_pct",
        "scenario_status",
    ]

    aligned = frame.copy()
    for column in ordered_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[ordered_columns]


def save_projection_payload(projection_df: pd.DataFrame) -> None:
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "projection_window": {
            "start": "2025-01-01",
            "end": "2026-12-31",
        },
        "records": projection_df.to_dict(orient="records"),
    }
    PROJECTION_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str))


def generate_figure3(
    backtest_df: pd.DataFrame,
    projection_df: pd.DataFrame,
    validation_df: pd.DataFrame | None = None,
) -> None:
    if backtest_df.empty:
        raise ValueError("Backtest dataframe is empty; cannot build Figure 3.")

    event_dates = _event_start_map()
    hist = backtest_df.copy()
    hist["event_date"] = hist["event"].map(event_dates)
    hist = hist.dropna(subset=["event_date"]).sort_values("event_date")

    forecast = projection_df.copy()
    if not forecast.empty:
        forecast["quarter_end"] = pd.to_datetime(forecast["quarter_end"])
        forecast = forecast.sort_values(["quarter_end", "scenario"])

    validation = pd.DataFrame()
    if validation_df is not None and not validation_df.empty:
        validation = validation_df.copy()
        if "quarter_end" in validation.columns:
            validation["quarter_end"] = pd.to_datetime(validation["quarter_end"])
            validation = validation.sort_values(["quarter_end", "scenario"])

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.bar(
        hist["event_date"],
        hist["observed_supply_gap_pct"],
        width=18,
        alpha=0.45,
        color="#4c78a8",
        label="[Historical Actual] Observed Historical Gap",
    )
    ax.plot(
        hist["event_date"],
        hist["predicted_supply_gap_pct"],
        linestyle="--",
        marker="o",
        color="#1f2d3d",
        label="[Historical Model Fit] Backtest Fit",
    )

    for event_name, label in [
        ("japan_earthquake_2011", "2011 Japan Earthquake"),
        ("taiwan_drought_2021", "2021 Taiwan Drought"),
        ("china_power_shortage_2021", "2021 Malaysia/Asia Shock"),
    ]:
        subset = hist.loc[hist["event"] == event_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        ax.annotate(
            label,
            xy=(row["event_date"], row["observed_supply_gap_pct"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555"},
        )

    ax.axvline(pd.Timestamp("2024-12-31"), linestyle="--", color="#666666", linewidth=1.3)
    ax.text(
        pd.Timestamp("2024-12-31"),
        ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 1.0,
        "2024 (Data End)",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        color="#555555",
    )

    if not forecast.empty:
        baseline = forecast.loc[forecast["scenario"] == "baseline_moderate"].copy()
        top1 = forecast.loc[forecast["scenario"] == "top_1_supplier_failure"].copy()
        baseline_2025 = baseline.loc[baseline["quarter_end"] < pd.Timestamp("2026-01-01")].copy()
        baseline_2026 = baseline.loc[baseline["quarter_end"] >= pd.Timestamp("2026-01-01")].copy()

        if not baseline_2025.empty:
            ax.plot(
                baseline_2025["quarter_end"],
                baseline_2025["expected_gap_pct"],
                color="#f58518",
                marker="o",
                linewidth=2.8,
                label="[Forecast] 2025 Baseline",
            )

        if not baseline_2026.empty:
            ax.plot(
                baseline_2026["quarter_end"],
                baseline_2026["expected_gap_pct"],
                color="#f58518",
                marker="o",
                linewidth=1.9,
                linestyle="--",
                alpha=0.8,
                label="[Forecast] 2026 Baseline",
            )

        if not top1.empty:
            ax.plot(
                top1["quarter_end"],
                top1["expected_gap_pct"],
                color="#54a24b",
                marker="o",
                linewidth=2.2,
                label="[Forecast] 2025-2026 Top-1 Supplier Failure",
            )

        if not validation.empty:
            baseline_actual = validation.loc[validation["scenario"] == "baseline_moderate"].copy()
            top1_actual = validation.loc[validation["scenario"] == "top_1_supplier_failure"].copy()

            if not baseline_actual.empty:
                ax.plot(
                    baseline_actual["quarter_end"],
                    baseline_actual["actual_expected_gap_pct"],
                    color="#e45756",
                    marker="x",
                    markersize=7,
                    linewidth=1.8,
                    linestyle=":",
                    label="[Actual 2025] Baseline Realized",
                )

            if not top1_actual.empty:
                ax.plot(
                    top1_actual["quarter_end"],
                    top1_actual["actual_expected_gap_pct"],
                    color="#2e7d32",
                    marker="x",
                    markersize=7,
                    linewidth=1.8,
                    linestyle=":",
                    label="[Actual 2025] Top-1 Realized",
                )

        combined_bounds = forecast.groupby("quarter_end").agg(
            lower_gap_pct=("lower_gap_pct", "min"),
            upper_gap_pct=("upper_gap_pct", "max"),
        )
        ax.fill_between(
            combined_bounds.index,
            combined_bounds["lower_gap_pct"],
            combined_bounds["upper_gap_pct"],
            color="#76b7b2",
            alpha=0.18,
            label="[Forecast] Projected Risk Bounds",
        )

        ax.annotate(
            "2025-2026 Out-of-Sample Predictive Simulation Zone",
            xy=(pd.Timestamp("2025-06-30"), max(combined_bounds["upper_gap_pct"].max(), 5.0)),
            xytext=(20, 18),
            textcoords="offset points",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#76b7b2", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#76b7b2"},
        )

        if not validation.empty:
            top1_2025 = validation.loc[validation["scenario"] == "top_1_supplier_failure"].copy()
            if not top1_2025.empty:
                top1_2025["abs_error_pct"] = (
                    top1_2025["predicted_gap_pct"] - top1_2025["actual_expected_gap_pct"]
                ).abs()
                mae = float(top1_2025["abs_error_pct"].mean())
                rmse = float((((top1_2025["predicted_gap_pct"] - top1_2025["actual_expected_gap_pct"]) ** 2).mean()) ** 0.5)
                ax.text(
                    pd.Timestamp("2025-02-15"),
                    46.0,
                    f"2025 Top-1 Validation\\nMAE={mae:.2f}% | RMSE={rmse:.2f}%",
                    fontsize=9,
                    va="top",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#2e7d32", "alpha": 0.92},
                )

    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), color="#90caf9", alpha=0.12)
    ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"), color="#ffcc80", alpha=0.12)
    ax.text(
        pd.Timestamp("2025-06-30"),
        1.0,
        "2025: Forecast + Actual Validation",
        fontsize=9,
        ha="center",
        va="bottom",
        color="#0d47a1",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#90caf9", "alpha": 0.9},
    )
    ax.text(
        pd.Timestamp("2026-06-30"),
        1.0,
        "2026: Forecast Only",
        fontsize=9,
        ha="center",
        va="bottom",
        color="#e65100",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#ffcc80", "alpha": 0.9},
    )
    ax.set_title("Module 1 Figure 3: Historical Backtest and 2025-2026 Forward Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Supply Gap (%) / Disruption Index")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE3_PATH, dpi=300)
    plt.close(fig)


def run_module1_backtest() -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    trade_df = load_module1_trade_data()

    network = SupplyChainNetwork(trade_df)
    graph = network.build_network()
    centrality_df = network.compute_centrality(graph)
    critical_nodes = network.identify_critical_nodes(graph)

    runner = StressTestRunner(trade_df)
    scenario_results = runner.run_all_standard()
    summary_df = PropagationEngine.results_to_dataframe(scenario_results)
    computed_backtest_df = runner.backtest_all()
    projection_df = runner.generate_forward_projections_2025_2026()

    if BACKTEST_PATH.exists():
        historical_backtest_df = pd.read_csv(BACKTEST_PATH)
    else:
        historical_backtest_df = computed_backtest_df.copy()
        historical_backtest_df.to_csv(BACKTEST_PATH, index=False)

    computed_backtest_df.to_csv(RUNTIME_BACKTEST_PATH, index=False)

    stress_json_path = OUT_DIR / "module1_stress_results.json"
    with stress_json_path.open("w") as file_obj:
        json.dump(
            [
                {
                    "scenario_name": result.scenario_name,
                    "shocked_nodes": result.shocked_nodes,
                    "severity": result.severity,
                    "original_supply": result.original_supply,
                    "disrupted_supply": result.disrupted_supply,
                    "supply_gap_pct": result.supply_gap_pct,
                    "substitution_absorbed_pct": result.substitution_absorbed_pct,
                    "most_affected_hs": result.most_affected_hs,
                    "details": result.details,
                }
                for result in scenario_results
            ],
            file_obj,
            indent=2,
        )

    summary_df.to_csv(OUT_DIR / "module1_stress_summary.csv", index=False)
    centrality_df.to_csv(OUT_DIR / "module1_centrality.csv", index=False)

    if not projection_df.empty:
        save_projection_payload(projection_df)
        combined_summary = pd.concat(
            [
                _align_summary_columns(_historical_summary_frame(summary_df)),
                _align_summary_columns(_forecast_summary_frame(projection_df)),
            ],
            ignore_index=True,
            sort=False,
        )
        combined_summary.to_csv(SUMMARY_PATH, index=False)
    else:
        summary_df.to_csv(SUMMARY_PATH, index=False)

    if not historical_backtest_df.empty:
        generate_figure3(historical_backtest_df, projection_df)

    return {
        "trade_df": trade_df,
        "summary_df": summary_df,
        "backtest_df": historical_backtest_df,
        "runtime_backtest_df": computed_backtest_df,
        "projection_df": projection_df,
        "centrality_df": centrality_df,
        "critical_nodes": critical_nodes,
    }


def main() -> None:
    outputs = run_module1_backtest()
    trade_df = outputs["trade_df"]
    summary_df = outputs["summary_df"]
    backtest_df = outputs["backtest_df"]
    projection_df = outputs["projection_df"]

    print("Module 1 completed.")
    print(f"Total trade records: {len(trade_df):,}")
    print(f"Critical nodes (>5% share): {outputs['critical_nodes']}")

    if not summary_df.empty:
        print("\nScenario summary:")
        print(
            summary_df[["scenario", "supply_gap_pct", "substitution_absorbed_pct", "most_affected_hs"]]
            .to_string(index=False)
        )

    if not backtest_df.empty and {"directional_accuracy_pct", "pairwise_ranking_accuracy_pct"}.issubset(backtest_df.columns):
        print("\nHistorical backtest metrics (2010-2024):")
        print(
            "Directional accuracy: "
            f"{backtest_df['directional_accuracy_pct'].iloc[0]:.2f}% | "
            "Pairwise ranking accuracy: "
            f"{backtest_df['pairwise_ranking_accuracy_pct'].iloc[0]:.2f}% | "
            "MAE: "
            f"{backtest_df['abs_error_pct'].mean():.2f}%"
        )

    if not projection_df.empty:
        print("\nForward projection averages (2025-2026):")
        for scenario_name in ["baseline_moderate", "top_1_supplier_failure"]:
            subset = projection_df.loc[projection_df["scenario"] == scenario_name]
            if subset.empty:
                continue
            print(
                f"{scenario_name}: mean={subset['expected_gap_pct'].mean():.2f}% | "
                f"lower={subset['lower_gap_pct'].mean():.2f}% | "
                f"upper={subset['upper_gap_pct'].mean():.2f}%"
            )

    print(f"\nSaved outputs to: {OUT_DIR}")
    print(f"Saved charts to: {FIG_DIR}")
    print(f"Figure 3: {FIGURE3_PATH}")


if __name__ == "__main__":
    main()