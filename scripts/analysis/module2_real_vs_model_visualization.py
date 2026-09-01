"""Create visualization and audit explanation for real-vs-model-only comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"
FIG_DIR = REPORT_DIR / "figures"


def _load_inputs() -> tuple[dict, pd.DataFrame]:
    summary_path = REPORT_DIR / "module2_real_vs_model_only.json"
    annual_path = REPORT_DIR / "annual_actual_transportation_costs.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")
    if not annual_path.exists():
        raise FileNotFoundError(f"Missing file: {annual_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    annual = pd.read_csv(annual_path)
    return summary, annual


def _build_calc_table(summary: dict) -> pd.DataFrame:
    selected = summary["selected_model"]
    real = summary["real_baseline"]
    gaps = summary["gaps"]

    rows = [
        {
            "metric": "demand_real_usd",
            "formula": "state_panel_total_demand_usd",
            "value": float(real["state_panel_total_demand_usd"]),
            "unit": "USD",
            "source": "module2_model_metrics.csv::state_panel_total_demand_usd",
        },
        {
            "metric": "demand_model_usd",
            "formula": "selected_model.total_demand_usd",
            "value": float(selected["total_demand_usd"]),
            "unit": "USD",
            "source": "module2_scenario_comparison.csv::total_demand_usd",
        },
        {
            "metric": "demand_gap_usd",
            "formula": "demand_model_usd - demand_real_usd",
            "value": float(gaps["demand_gap_usd"]),
            "unit": "USD",
            "source": "computed",
        },
        {
            "metric": "cost_real_annual_usd_m",
            "formula": "Actual_Total_Transportation_Cost_USD_M (Year=2022)",
            "value": float(real["transport_cost_annual_2022_usd_m"]),
            "unit": "USD million",
            "source": "annual_actual_transportation_costs.csv::Actual_Total_Transportation_Cost_USD_M",
        },
        {
            "metric": "cost_real_monthly_proxy_usd",
            "formula": "cost_real_annual_usd_m * 1e6 / 12",
            "value": float(real["transport_cost_monthly_proxy_usd"]),
            "unit": "USD",
            "source": "computed",
        },
        {
            "metric": "cost_model_monthly_usd",
            "formula": "selected_model.total_logistics_cost_usd",
            "value": float(selected["total_logistics_cost_usd"]),
            "unit": "USD",
            "source": "module2_scenario_comparison.csv::total_logistics_cost_usd",
        },
        {
            "metric": "cost_gap_usd",
            "formula": "cost_model_monthly_usd - cost_real_monthly_proxy_usd",
            "value": float(gaps["monthly_cost_gap_usd"]),
            "unit": "USD",
            "source": "computed",
        },
        {
            "metric": "cost_gap_pct",
            "formula": "cost_gap_usd / cost_real_monthly_proxy_usd * 100",
            "value": float(gaps["monthly_cost_gap_pct"]),
            "unit": "%",
            "source": "computed",
        },
        {
            "metric": "fill_rate_model",
            "formula": "selected_model.fill_rate",
            "value": float(selected["fill_rate"]),
            "unit": "ratio",
            "source": "module2_scenario_comparison.csv::fill_rate",
        },
        {
            "metric": "delivered_model_usd",
            "formula": "selected_model.delivered_usd",
            "value": float(selected["delivered_usd"]),
            "unit": "USD",
            "source": "module2_scenario_comparison.csv::delivered_usd",
        },
        {
            "metric": "unmet_model_usd",
            "formula": "selected_model.unmet_usd",
            "value": float(selected["unmet_usd"]),
            "unit": "USD",
            "source": "module2_scenario_comparison.csv::unmet_usd",
        },
    ]
    return pd.DataFrame(rows)


def _plot(summary: dict, annual: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected = summary["selected_model"]
    real = summary["real_baseline"]
    gaps = summary["gaps"]

    demand_real_b = float(real["state_panel_total_demand_usd"]) / 1e9
    demand_model_b = float(selected["total_demand_usd"]) / 1e9

    cost_real_m = float(real["transport_cost_monthly_proxy_usd"]) / 1e6
    cost_model_m = float(selected["total_logistics_cost_usd"]) / 1e6

    delivered_b = float(selected["delivered_usd"]) / 1e9
    unmet_b = float(selected["unmet_usd"]) / 1e9

    r2022 = annual[annual["Year"] == 2022]
    if r2022.empty:
        raise RuntimeError("Year 2022 not found in annual_actual_transportation_costs.csv")
    r2022 = r2022.iloc[0]

    mode_labels = ["Truck", "Air", "Rail"]
    mode_values = [
        float(r2022["Truck_Cost_USD_M"]),
        float(r2022["Air_Cost_USD_M"]),
        float(r2022["Rail_Cost_USD_M"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # A: Demand real vs model
    axes[0, 0].bar(["Real demand", "Model demand"], [demand_real_b, demand_model_b], color=["#334155", "#0ea5e9"])
    axes[0, 0].set_title("Demand: Real vs Model")
    axes[0, 0].set_ylabel("USD (billion)")
    for i, v in enumerate([demand_real_b, demand_model_b]):
        axes[0, 0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

    # B: Monthly transport cost real proxy vs model
    axes[0, 1].bar(["Real monthly proxy", "Model monthly cost"], [cost_real_m, cost_model_m], color=["#475569", "#10b981"])
    axes[0, 1].set_title("Monthly Transport Cost: Real vs Model")
    axes[0, 1].set_ylabel("USD (million)")
    for i, v in enumerate([cost_real_m, cost_model_m]):
        axes[0, 1].text(i, v + 6, f"{v:.1f}", ha="center", fontsize=9)
    axes[0, 1].text(
        0.5,
        max(cost_real_m, cost_model_m) * 0.25,
        f"Gap: {float(gaps['monthly_cost_gap_usd'])/1e6:,.1f} M ({float(gaps['monthly_cost_gap_pct']):.2f}%)",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#94a3b8"},
    )

    # C: Delivered vs unmet in selected model
    axes[1, 0].bar(["Selected model"], [delivered_b], color="#16a34a", label="Delivered")
    axes[1, 0].bar(["Selected model"], [unmet_b], bottom=[delivered_b], color="#ef4444", label="Unmet")
    axes[1, 0].set_title("Model Fulfillment Composition")
    axes[1, 0].set_ylabel("USD (billion)")
    axes[1, 0].legend()
    axes[1, 0].text(0, delivered_b / 2, f"Delivered\n{delivered_b:.2f}", ha="center", va="center", fontsize=9, color="white")
    axes[1, 0].text(0, delivered_b + unmet_b / 2, f"Unmet\n{unmet_b:.2f}", ha="center", va="center", fontsize=9, color="white")

    # D: Real annual mode split for traceability
    axes[1, 1].bar(mode_labels, mode_values, color=["#2563eb", "#f97316", "#9333ea"])
    axes[1, 1].set_title("Real 2022 Annual Transport Cost by Mode")
    axes[1, 1].set_ylabel("USD (million)")
    for i, v in enumerate(mode_values):
        axes[1, 1].text(i, v + 40, f"{v:,.0f}", ha="center", fontsize=9)

    fig.suptitle(
        f"Module2 Real vs Model ({selected['strategy']})",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / "module2_real_vs_model_only.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def _write_explanation(summary: dict, calc_table: pd.DataFrame, fig_path: Path) -> Path:
    selected = summary["selected_model"]
    real = summary["real_baseline"]
    gaps = summary["gaps"]

    lines: list[str] = []
    lines.append("# Module2 Real vs Model: Audit Explanation")
    lines.append("")
    lines.append("## 1) What this chart compares")
    lines.append("- This is a single-model comparison against real values only.")
    lines.append(f"- Selected model result: {selected['strategy']} ({selected['strategy_key']}).")
    lines.append("- No model-vs-model deltas are included.")
    lines.append("")
    lines.append("## 2) Data sources")
    lines.append("- Real demand baseline: reports/module2/module2_model_metrics.csv (metric=state_panel_total_demand_usd).")
    lines.append("- Model outputs: reports/module2/module2_scenario_comparison.csv.")
    lines.append("- Real annual transport cost proxy: reports/module2/annual_actual_transportation_costs.csv (Year=2022).")
    lines.append("- Comparison summary input: reports/module2/module2_real_vs_model_only.json.")
    lines.append("")
    lines.append("## 3) Calculation steps")
    lines.append("1. Demand gap")
    lines.append("- real_demand_usd = state_panel_total_demand_usd")
    lines.append("- model_demand_usd = selected_model.total_demand_usd")
    lines.append("- demand_gap_usd = model_demand_usd - real_demand_usd")
    lines.append("- demand_gap_pct = demand_gap_usd / real_demand_usd * 100")
    lines.append("")
    lines.append("2. Monthly cost gap")
    lines.append("- real_annual_cost_usd_m = Actual_Total_Transportation_Cost_USD_M (2022)")
    lines.append("- real_monthly_cost_proxy_usd = real_annual_cost_usd_m * 1,000,000 / 12")
    lines.append("- model_monthly_cost_usd = selected_model.total_logistics_cost_usd")
    lines.append("- monthly_cost_gap_usd = model_monthly_cost_usd - real_monthly_cost_proxy_usd")
    lines.append("- monthly_cost_gap_pct = monthly_cost_gap_usd / real_monthly_cost_proxy_usd * 100")
    lines.append("")
    lines.append("3. Model fulfillment composition")
    lines.append("- delivered_usd = selected_model.delivered_usd")
    lines.append("- unmet_usd = selected_model.unmet_usd")
    lines.append("- fill_rate = delivered_usd / (delivered_usd + unmet_usd)")
    lines.append("")
    lines.append("## 4) Numbers used in this run")
    lines.append(f"- real_demand_usd = {float(real['state_panel_total_demand_usd']):,.2f}")
    lines.append(f"- model_demand_usd = {float(selected['total_demand_usd']):,.2f}")
    lines.append(f"- demand_gap_usd = {float(gaps['demand_gap_usd']):,.2f} ({float(gaps['demand_gap_pct']):.4f}%)")
    lines.append(f"- real_monthly_cost_proxy_usd = {float(real['transport_cost_monthly_proxy_usd']):,.2f}")
    lines.append(f"- model_monthly_cost_usd = {float(selected['total_logistics_cost_usd']):,.2f}")
    lines.append(f"- monthly_cost_gap_usd = {float(gaps['monthly_cost_gap_usd']):,.2f} ({float(gaps['monthly_cost_gap_pct']):.4f}%)")
    lines.append(f"- delivered_usd = {float(selected['delivered_usd']):,.2f}")
    lines.append(f"- unmet_usd = {float(selected['unmet_usd']):,.2f}")
    lines.append(f"- fill_rate = {float(selected['fill_rate']):.6f}")
    lines.append("")
    lines.append("## 5) Output files")
    lines.append(f"- Figure: {fig_path.relative_to(ROOT).as_posix()}")
    lines.append("- Calculation table: reports/module2/module2_real_vs_model_only_calc_table.csv")
    lines.append("- This explanation: reports/module2/module2_real_vs_model_only_explained.md")

    out_md = REPORT_DIR / "module2_real_vs_model_only_explained.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    calc_out = REPORT_DIR / "module2_real_vs_model_only_calc_table.csv"
    calc_table.to_csv(calc_out, index=False)

    return out_md


def main() -> None:
    summary, annual = _load_inputs()
    calc_table = _build_calc_table(summary)
    fig_path = _plot(summary, annual)
    md_path = _write_explanation(summary, calc_table, fig_path)

    print(f"Wrote: {fig_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {REPORT_DIR / 'module2_real_vs_model_only_calc_table.csv'}")


if __name__ == "__main__":
    main()
