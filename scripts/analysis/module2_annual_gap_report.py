"""Build annual real-vs-model transportation cost gap report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(REPORT_DIR / "annual_actual_transportation_costs.csv")
    scen = pd.read_csv(REPORT_DIR / "module2_scenario_comparison.csv")

    target_year = 2022
    real_row = real.loc[real["Year"] == target_year]
    if real_row.empty:
        raise RuntimeError(f"No real annual cost row for year {target_year}")
    real_row = real_row.iloc[0]

    strategies = ["Baseline", "Cost-Minimizing", "Resilience-First", "Targeted Allocation"]
    model = scen[scen["strategy"].isin(strategies)].copy()
    if model.empty:
        raise RuntimeError("No model rows found for expected strategies")

    # Assumption: current module2 run is monthly (2022-01). Annualized by x12 for yearly comparison.
    model["model_monthly_cost_usd"] = pd.to_numeric(model["total_logistics_cost_usd"], errors="coerce")
    model["model_annual_cost_usd"] = model["model_monthly_cost_usd"] * 12.0
    model["model_annual_cost_usd_m"] = model["model_annual_cost_usd"] / 1e6

    real_annual_cost_m = float(real_row["Actual_Total_Transportation_Cost_USD_M"])

    # Gap to real annual cost
    model["gap_vs_real_usd_m"] = model["model_annual_cost_usd_m"] - real_annual_cost_m
    model["gap_vs_real_pct"] = model["gap_vs_real_usd_m"] / real_annual_cost_m * 100.0

    # Calibrate scale: use baseline as anchor so calibrated baseline matches real annual level.
    base = model[model["strategy"] == "Baseline"].iloc[0]
    scale = real_annual_cost_m / float(base["model_annual_cost_usd_m"])
    model["calibrated_annual_cost_usd_m"] = model["model_annual_cost_usd_m"] * scale

    # Keep the same relative spread after calibration.
    row_cost = model[model["strategy"] == "Cost-Minimizing"].iloc[0]
    row_res = model[model["strategy"] == "Resilience-First"].iloc[0]
    row_tgt = model[model["strategy"] == "Targeted Allocation"].iloc[0]

    savings_cost_vs_res_m = float(row_res["calibrated_annual_cost_usd_m"] - row_cost["calibrated_annual_cost_usd_m"])
    savings_tgt_vs_res_m = float(row_res["calibrated_annual_cost_usd_m"] - row_tgt["calibrated_annual_cost_usd_m"])

    fill_cost = float(row_cost["fill_rate"])
    fill_res = float(row_res["fill_rate"])
    fill_tgt = float(row_tgt["fill_rate"])

    summary = {
        "target_year": target_year,
        "real_annual_transport_cost_usd_m": real_annual_cost_m,
        "real_breakdown_usd_m": {
            "truck": float(real_row["Truck_Cost_USD_M"]),
            "air": float(real_row["Air_Cost_USD_M"]),
            "rail": float(real_row["Rail_Cost_USD_M"]),
        },
        "assumption": "Model monthly costs (2022-01) annualized by x12 for yearly comparison.",
        "scale_factor_baseline_to_real": scale,
        "model_rows": model[
            [
                "strategy",
                "scenario",
                "fill_rate",
                "model_monthly_cost_usd",
                "model_annual_cost_usd_m",
                "gap_vs_real_usd_m",
                "gap_vs_real_pct",
                "calibrated_annual_cost_usd_m",
            ]
        ].to_dict(orient="records"),
        "calibrated_tradeoff": {
            "costmin_vs_resilience_saving_usd_m": savings_cost_vs_res_m,
            "targeted_vs_resilience_saving_usd_m": savings_tgt_vs_res_m,
            "fill_rate_costmin": fill_cost,
            "fill_rate_resilience": fill_res,
            "fill_rate_targeted": fill_tgt,
            "fill_rate_gap_res_minus_cost_pct_point": (fill_res - fill_cost) * 100.0,
            "fill_rate_gap_res_minus_targeted_pct_point": (fill_res - fill_tgt) * 100.0,
        },
    }

    out_json = REPORT_DIR / "module2_annual_real_vs_model_gap.json"
    out_md = REPORT_DIR / "module2_annual_real_vs_model_gap.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Module2 年度真实成本 vs 模型成本差距")
    lines.append("")
    lines.append(f"对比年份: {target_year}")
    lines.append(f"真实年度总运输成本 (USD M): {real_annual_cost_m:,.2f}")
    lines.append(
        f"真实方式拆分 (USD M): truck={float(real_row['Truck_Cost_USD_M']):,.2f}, air={float(real_row['Air_Cost_USD_M']):,.2f}, rail={float(real_row['Rail_Cost_USD_M']):,.2f}"
    )
    lines.append("")
    lines.append("## 1) 可比口径说明")
    lines.append("- 真实年度成本来自 annual_actual_transportation_costs.csv。")
    lines.append("- 模型当前是月度结果 (2022-01)，此处按 x12 年化后再比较。")
    lines.append("")
    lines.append("## 2) 年化后模型与真实的差距")
    for r in summary["model_rows"]:
        lines.append(
            f"- {r['strategy']}: model_annual={r['model_annual_cost_usd_m']:,.2f} USD M, gap={r['gap_vs_real_usd_m']:,.2f} USD M ({r['gap_vs_real_pct']:.2f}%)"
        )
    lines.append("")
    lines.append("## 3) 基线校准后的策略结论")
    lines.append(f"- baseline->real 缩放系数: {scale:.6f}")
    lines.append(
        f"- Cost-Min 相对 Resilience 节约: {savings_cost_vs_res_m:,.2f} USD M"
    )
    lines.append(
        f"- Targeted 相对 Resilience 节约: {savings_tgt_vs_res_m:,.2f} USD M"
    )
    lines.append(
        f"- 履约率差 (Resilience - Cost-Min): {(fill_res - fill_cost) * 100.0:.3f} 个百分点"
    )
    lines.append(
        f"- 履约率差 (Resilience - Targeted): {(fill_res - fill_tgt) * 100.0:.3f} 个百分点"
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
