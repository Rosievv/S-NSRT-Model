"""Compare real values vs a single selected model result (no model-vs-model deltas)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_MODULE2_DIR = ROOT / "data" / "raw" / "Module2"
REPORT_DIR = ROOT / "reports" / "module2"
PROCESSED_DIR = ROOT / "data" / "processed"

HS_CODE = "854231"


def _pick_model_row(df: pd.DataFrame) -> pd.Series:
    # Prefer a business-balanced strategy if present; otherwise fallback deterministically.
    preferred_keys = ["targeted", "baseline", "resilience_first", "cost_min"]
    for key in preferred_keys:
        hit = df[df["strategy_key"] == key]
        if not hit.empty:
            return hit.iloc[0]
    return df.iloc[0]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(REPORT_DIR / "module2_scenario_comparison.csv")
    metrics_df = pd.read_csv(REPORT_DIR / "module2_model_metrics.csv")
    raw_demand_df = pd.read_csv(RAW_MODULE2_DIR / "domestic_demand.csv")
    state_demand_df = pd.read_csv(PROCESSED_DIR / "domestic_demand_state_assumption.csv")
    annual_real_df = pd.read_csv(REPORT_DIR / "annual_actual_transportation_costs.csv")

    if model_df.empty:
        raise RuntimeError("module2_scenario_comparison.csv is empty")

    model_row = _pick_model_row(model_df)

    # Real demand from the raw 2022 HS 854231 Module2 input.
    raw_demand_df["hs_code"] = raw_demand_df["hs_code"].astype(str)
    raw_2022_854231 = raw_demand_df[(raw_demand_df["year"] == 2022) & (raw_demand_df["hs_code"] == HS_CODE)].copy()
    if raw_2022_854231.empty:
        raise RuntimeError("No 2022 rows found for HS 854231 in data/raw/Module2/domestic_demand.csv")
    real_2022_854231_usd = float(raw_2022_854231["demand_value_usd"].sum())

    # Model-side demand for the same HS code.
    state_demand_df["hs_code"] = state_demand_df["hs_code"].astype(str)
    model_854231_df = state_demand_df[state_demand_df["hs_code"] == HS_CODE].copy()
    if model_854231_df.empty:
        raise RuntimeError("No 854231 rows found in data/processed/domestic_demand_state_assumption.csv")
    model_854231_usd = float(model_854231_df["demand_value_usd"].sum())

    # Real transportation cost: annual 2022 estimate converted to monthly for same model granularity.
    real_2022 = annual_real_df[annual_real_df["Year"] == 2022]
    if real_2022.empty:
        raise RuntimeError("Year 2022 not found in annual_actual_transportation_costs.csv")
    real_2022 = real_2022.iloc[0]
    real_annual_transport_cost_m = float(real_2022["Actual_Total_Transportation_Cost_USD_M"])
    real_monthly_transport_cost_usd = real_annual_transport_cost_m * 1e6 / 12.0

    model_total_logistics_cost_usd = float(model_row["total_logistics_cost_usd"])

    demand_gap_usd = model_854231_usd - real_2022_854231_usd
    demand_gap_pct = demand_gap_usd / real_2022_854231_usd * 100.0 if real_2022_854231_usd else 0.0

    cost_gap_usd = model_total_logistics_cost_usd - real_monthly_transport_cost_usd
    cost_gap_pct = cost_gap_usd / real_monthly_transport_cost_usd * 100.0 if real_monthly_transport_cost_usd else 0.0

    summary = {
        "comparison_mode": "module2_hs854231_real_vs_model",
        "hs_code": HS_CODE,
        "raw_2022_total_demand_usd": real_2022_854231_usd,
        "model_2022_total_demand_usd": model_854231_usd,
        "selected_model": {
            "strategy_key": str(model_row["strategy_key"]),
            "strategy": str(model_row["strategy"]),
            "scenario": str(model_row["scenario"]),
            "fill_rate": float(model_row["fill_rate"]),
            "total_demand_usd": model_854231_usd,
            "delivered_usd": float(model_row["delivered_usd"]),
            "unmet_usd": float(model_row["unmet_usd"]),
            "total_logistics_cost_usd": model_total_logistics_cost_usd,
            "avg_lead_time_days": float(model_row["avg_lead_time_days"]),
            "air_express_share": float(model_row["air_express_share"]),
        },
        "real_baseline": {
            "raw_2022_hs854231_total_demand_usd": real_2022_854231_usd,
            "raw_2022_row_count": int(len(raw_2022_854231)),
            "processed_2022_hs854231_total_demand_usd": model_854231_usd,
            "processed_2022_row_count": int(len(model_854231_df)),
            "transport_cost_annual_2022_usd_m": real_annual_transport_cost_m,
            "transport_cost_monthly_proxy_usd": real_monthly_transport_cost_usd,
            "transport_mode_cost_share_2022": {
                "truck": float(real_2022["Truck_Cost_USD_M"]) / real_annual_transport_cost_m,
                "air": float(real_2022["Air_Cost_USD_M"]) / real_annual_transport_cost_m,
                "rail": float(real_2022["Rail_Cost_USD_M"]) / real_annual_transport_cost_m,
            },
        },
        "gaps": {
            "hs854231_demand_gap_usd": demand_gap_usd,
            "hs854231_demand_gap_pct": demand_gap_pct,
            "monthly_cost_gap_usd": cost_gap_usd,
            "monthly_cost_gap_pct": cost_gap_pct,
        },
        "notes": [
            "This report focuses only on HS 854231.",
            "The real baseline uses the raw 2022 Module2 HS 854231 demand total.",
            "The model-side demand uses the processed 2022 HS 854231 total feeding the optimizer.",
            "Monthly model cost is compared against the monthlyized 2022 real transportation cost proxy.",
        ],
    }

    out_json = REPORT_DIR / "module2_real_vs_model_only.json"
    out_md = REPORT_DIR / "module2_real_vs_model_only.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Module2 HS 854231 真实值 vs 模型结果")
    lines.append("")
    lines.append("## 对比对象")
    lines.append(f"- HS 码: {HS_CODE}")
    lines.append(f"- 选定模型策略: {summary['selected_model']['strategy']} ({summary['selected_model']['strategy_key']})")
    lines.append(f"- 模型场景标签: {summary['selected_model']['scenario']}")
    lines.append("")
    lines.append("## 真实值")
    lines.append(f"- 2022 年 raw HS 854231 总额 (USD): {real_2022_854231_usd:,.2f}")
    lines.append(f"- 2022 年 raw 行数: {len(raw_2022_854231):,}")
    lines.append(f"- 2022 年运输成本总额 (USD M): {real_annual_transport_cost_m:,.2f}")
    lines.append(f"- 月化运输成本代理 (USD): {real_monthly_transport_cost_usd:,.2f}")
    lines.append("")
    lines.append("## 模型结果")
    lines.append(f"- 2022 年 processed HS 854231 总额 (USD): {model_854231_usd:,.2f}")
    lines.append(f"- 2022 年 processed 行数: {len(model_854231_df):,}")
    lines.append(f"- total_logistics_cost_usd: {model_total_logistics_cost_usd:,.2f}")
    lines.append(f"- fill_rate: {float(model_row['fill_rate']):.6f}")
    lines.append(f"- delivered_usd: {float(model_row['delivered_usd']):,.2f}")
    lines.append(f"- unmet_usd: {float(model_row['unmet_usd']):,.2f}")
    lines.append("")
    lines.append("## 真实-模型偏差")
    lines.append(f"- 854231 真实值 vs 模型值: {demand_gap_usd:,.2f} USD ({demand_gap_pct:.4f}%)")
    lines.append(f"- 月化成本偏差: {cost_gap_usd:,.2f} USD ({cost_gap_pct:.4f}%)")
    lines.append("")
    lines.append("## 说明")
    for note in summary["notes"]:
        lines.append(f"- {note}")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
