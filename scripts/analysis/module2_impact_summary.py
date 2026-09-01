"""Generate Module2 impact KPIs, visual dashboard, and management summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"
FIG_DIR = REPORT_DIR / "figures"

SCENARIO_FLOW_FILES = {
    "Baseline": "module2_baseline_flow_allocation.csv",
    "Cost-Minimizing": "module2_stress_cost_flow_allocation.csv",
    "Resilience-First": "module2_stress_resilience_flow_allocation.csv",
    "Targeted Allocation": "module2_stress_targeted_flow_allocation.csv",
}


def _active_routes(flow_path: Path) -> Dict[str, int]:
    df = pd.read_csv(flow_path)
    active = int((df["flow_usd"] > 1e-6).sum())
    total = int(len(df))
    return {"active": active, "total": total}


def _pct(value: float) -> float:
    return round(value * 100.0, 3)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    scenario = pd.read_csv(REPORT_DIR / "module2_scenario_comparison.csv")
    tradeoff = pd.read_csv(REPORT_DIR / "module2_strategy_tradeoff.csv")

    scenario = scenario[scenario["strategy"].isin(SCENARIO_FLOW_FILES.keys())].copy()
    tradeoff = tradeoff[tradeoff["strategy"].isin(["Cost-Minimizing", "Resilience-First", "Targeted Allocation"])].copy()

    route_rows = []
    for name, file_name in SCENARIO_FLOW_FILES.items():
        counts = _active_routes(REPORT_DIR / file_name)
        route_rows.append({"strategy": name, **counts})
    routes = pd.DataFrame(route_rows)

    baseline_active = int(routes.loc[routes["strategy"] == "Baseline", "active"].iloc[0])

    cost_row = tradeoff.loc[tradeoff["strategy"] == "Cost-Minimizing"].iloc[0]
    resilience_row = tradeoff.loc[tradeoff["strategy"] == "Resilience-First"].iloc[0]
    targeted_row = tradeoff.loc[tradeoff["strategy"] == "Targeted Allocation"].iloc[0]

    fill_gain_res_vs_cost = float(resilience_row["fill_rate"] - cost_row["fill_rate"])
    fill_gain_tgt_vs_cost = float(targeted_row["fill_rate"] - cost_row["fill_rate"])

    lead_improve_res_vs_cost = float((cost_row["avg_lead_time_days"] - resilience_row["avg_lead_time_days"]) / cost_row["avg_lead_time_days"])
    lead_improve_tgt_vs_cost = float((cost_row["avg_lead_time_days"] - targeted_row["avg_lead_time_days"]) / cost_row["avg_lead_time_days"])

    savings_cost_vs_res = float(resilience_row["total_logistics_cost_usd"] - cost_row["total_logistics_cost_usd"])
    savings_tgt_vs_res = float(resilience_row["total_logistics_cost_usd"] - targeted_row["total_logistics_cost_usd"])

    route_delta_res = int(routes.loc[routes["strategy"] == "Resilience-First", "active"].iloc[0] - baseline_active)
    route_delta_cost = int(routes.loc[routes["strategy"] == "Cost-Minimizing", "active"].iloc[0] - baseline_active)
    route_delta_tgt = int(routes.loc[routes["strategy"] == "Targeted Allocation", "active"].iloc[0] - baseline_active)

    summary = {
        "route_optimization": {
            "trunk_routes_total": int(routes["total"].max()),
            "baseline_active_routes": baseline_active,
            "cost_min_active_routes": int(routes.loc[routes["strategy"] == "Cost-Minimizing", "active"].iloc[0]),
            "resilience_active_routes": int(routes.loc[routes["strategy"] == "Resilience-First", "active"].iloc[0]),
            "targeted_active_routes": int(routes.loc[routes["strategy"] == "Targeted Allocation", "active"].iloc[0]),
            "route_delta_vs_baseline": {
                "cost_min": route_delta_cost,
                "resilience": route_delta_res,
                "targeted": route_delta_tgt,
            },
        },
        "efficiency_improvement": {
            "fill_rate_gain_resilience_vs_cost_pct_point": round(fill_gain_res_vs_cost * 100.0, 3),
            "fill_rate_gain_targeted_vs_cost_pct_point": round(fill_gain_tgt_vs_cost * 100.0, 3),
            "lead_time_improve_resilience_vs_cost_pct": round(lead_improve_res_vs_cost * 100.0, 3),
            "lead_time_improve_targeted_vs_cost_pct": round(lead_improve_tgt_vs_cost * 100.0, 3),
        },
        "cost_savings": {
            "save_cost_min_vs_resilience_usd": round(savings_cost_vs_res, 2),
            "save_targeted_vs_resilience_usd": round(savings_tgt_vs_res, 2),
            "save_targeted_vs_resilience_musd": round(savings_tgt_vs_res / 1e6, 3),
        },
    }

    with open(REPORT_DIR / "module2_impact_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # A: Active route count
    order_routes = ["Baseline", "Cost-Minimizing", "Resilience-First", "Targeted Allocation"]
    rplot = routes.set_index("strategy").loc[order_routes].reset_index()
    axes[0, 0].bar(rplot["strategy"], rplot["active"], color=["#9ca3af", "#60a5fa", "#f59e0b", "#34d399"])
    axes[0, 0].set_title("Active Trunk Routes (Port->DC)")
    axes[0, 0].set_ylabel("Active routes")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(rplot["active"]):
        axes[0, 0].text(i, v + 0.2, str(int(v)), ha="center", fontsize=9)

    # B: Fill rate
    tplot = tradeoff.set_index("strategy").loc[["Cost-Minimizing", "Resilience-First", "Targeted Allocation"]].reset_index()
    axes[0, 1].bar(tplot["strategy"], tplot["fill_rate"] * 100.0, color=["#60a5fa", "#f59e0b", "#34d399"])
    axes[0, 1].set_title("Service Level (Fill Rate)")
    axes[0, 1].set_ylabel("Fill rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(tplot["fill_rate"] * 100.0):
        axes[0, 1].text(i, v + 0.2, f"{v:.2f}%", ha="center", fontsize=9)

    # C: Cost comparison
    axes[1, 0].bar(tplot["strategy"], tplot["total_logistics_cost_usd"] / 1e6, color=["#60a5fa", "#f59e0b", "#34d399"])
    axes[1, 0].set_title("Logistics Cost")
    axes[1, 0].set_ylabel("Cost (Million USD)")
    axes[1, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(tplot["total_logistics_cost_usd"] / 1e6):
        axes[1, 0].text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=9)

    # D: Savings vs resilience-first
    savings_labels = ["Cost-Min vs Resilience", "Targeted vs Resilience"]
    savings_vals = [savings_cost_vs_res / 1e6, savings_tgt_vs_res / 1e6]
    axes[1, 1].bar(savings_labels, savings_vals, color=["#3b82f6", "#10b981"])
    axes[1, 1].set_title("Potential Savings vs Resilience-First")
    axes[1, 1].set_ylabel("Savings (Million USD)")
    axes[1, 1].tick_params(axis="x", rotation=10)
    for i, v in enumerate(savings_vals):
        axes[1, 1].text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "module2_impact_summary.png", dpi=220)
    plt.close(fig)

    md_lines = [
        "# Module2 Impact Summary",
        "",
        "## 1) We optimized how many routes",
        f"- Trunk-route search space (Port->DC multimodal): {summary['route_optimization']['trunk_routes_total']} routes",
        f"- Baseline active routes: {summary['route_optimization']['baseline_active_routes']}",
        f"- Cost-Minimizing active routes: {summary['route_optimization']['cost_min_active_routes']} ({route_delta_cost:+d} vs baseline)",
        f"- Resilience-First active routes: {summary['route_optimization']['resilience_active_routes']} ({route_delta_res:+d} vs baseline)",
        f"- Targeted Allocation active routes: {summary['route_optimization']['targeted_active_routes']} ({route_delta_tgt:+d} vs baseline)",
        "",
        "## 2) How much efficiency improved",
        f"- Fill-rate gain (Resilience vs Cost-Min): {fill_gain_res_vs_cost * 100.0:.3f} percentage points",
        f"- Fill-rate gain (Targeted vs Cost-Min): {fill_gain_tgt_vs_cost * 100.0:.3f} percentage points",
        f"- Lead-time improvement (Resilience vs Cost-Min): {lead_improve_res_vs_cost * 100.0:.3f}%",
        f"- Lead-time change (Targeted vs Cost-Min): {lead_improve_tgt_vs_cost * 100.0:.3f}%",
        "",
        "## 3) How much money can be saved",
        f"- Cost-Minimizing saves vs Resilience-First: ${savings_cost_vs_res:,.2f}",
        f"- Targeted Allocation saves vs Resilience-First: ${savings_tgt_vs_res:,.2f}",
        "",
        "## 4) What Module2 does and why",
        "- Built a two-leg network optimization model (Port->DC->State) with rail/truck/air modes.",
        "- Added stress-shock simulation and rerouting to evaluate resilience under disruptions.",
        "- Added state-level fulfillment and unmet-demand outputs to support service governance.",
        "- Added bottleneck shadow-price diagnostics to identify high-ROI capacity expansions.",
        "",
        "## 5) Current result interpretation",
        "- Cost-Minimizing is the lowest-cost option but yields lower fill rate.",
        "- Resilience-First achieves the best fill rate and faster lead time, with higher cost.",
        "- Targeted Allocation is the practical middle ground for service-cost balance.",
    ]
    (REPORT_DIR / "module2_impact_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
