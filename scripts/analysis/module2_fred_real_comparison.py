"""Fetch FRED macro series and compare them with Module2 scenario outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"
PROCESSED_DIR = ROOT / "data" / "processed"

FRED_URLS = {
    "CAPUTLG334S": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CAPUTLG334S",
    "RETAILIRSA": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RETAILIRSA",
}


def _load_fred_series(series_id: str) -> pd.DataFrame:
    df = pd.read_csv(FRED_URLS[series_id])
    df = df.rename(columns={"observation_date": "month", series_id: "value"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["month", "value"]).sort_values("month").reset_index(drop=True)
    df["series_id"] = series_id
    return df


def _zscore_from_history(df: pd.DataFrame, month: pd.Timestamp) -> float:
    hist = df[df["month"] <= month]["value"]
    if len(hist) < 24:
        return float("nan")
    std = hist.std(ddof=0)
    if std == 0 or pd.isna(std):
        return float("nan")
    current = float(df.loc[df["month"] == month, "value"].iloc[0])
    return (current - float(hist.mean())) / float(std)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Real/public-demand month used by current module2 run
    demand = pd.read_csv(PROCESSED_DIR / "domestic_demand_state_assumption.csv")
    demand["month"] = pd.to_datetime(demand["month"], errors="coerce")
    target_month = demand["month"].dropna().min()

    scen = pd.read_csv(REPORT_DIR / "module2_scenario_comparison.csv")
    scen = scen[scen["strategy"].isin(["Baseline", "Cost-Minimizing", "Resilience-First", "Targeted Allocation"])].copy()

    cap = _load_fred_series("CAPUTLG334S")
    inv = _load_fred_series("RETAILIRSA")

    # Align to nearest previous observation to avoid missing month edge cases
    cap_ref = cap[cap["month"] <= target_month].tail(1)
    inv_ref = inv[inv["month"] <= target_month].tail(1)
    if cap_ref.empty or inv_ref.empty:
        raise RuntimeError("FRED series does not cover target month")

    cap_month = pd.Timestamp(cap_ref["month"].iloc[0])
    inv_month = pd.Timestamp(inv_ref["month"].iloc[0])
    cap_val = float(cap_ref["value"].iloc[0])
    inv_val = float(inv_ref["value"].iloc[0])

    cap_z = _zscore_from_history(cap, cap_month)
    inv_z = _zscore_from_history(inv, inv_month)

    # Tightness proxy: higher cap util + lower inv/sales => tighter system.
    # We use +z(cap) - z(inv)
    tightness = cap_z - inv_z if pd.notna(cap_z) and pd.notna(inv_z) else float("nan")

    # Model tradeoff rows
    row_base = scen[scen["strategy"] == "Baseline"].iloc[0]
    row_cost = scen[scen["strategy"] == "Cost-Minimizing"].iloc[0]
    row_res = scen[scen["strategy"] == "Resilience-First"].iloc[0]
    row_tgt = scen[scen["strategy"] == "Targeted Allocation"].iloc[0]

    save_cost_vs_res = float(row_res["total_logistics_cost_usd"] - row_cost["total_logistics_cost_usd"])
    fill_delta_cost_vs_res = float(row_res["fill_rate"] - row_cost["fill_rate"])

    out = {
        "target_month": str(target_month.date()),
        "fred": {
            "CAPUTLG334S": {
                "month": str(cap_month.date()),
                "value": cap_val,
                "zscore_to_history": cap_z,
                "source": "FRED official series CAPUTLG334S",
            },
            "RETAILIRSA": {
                "month": str(inv_month.date()),
                "value": inv_val,
                "zscore_to_history": inv_z,
                "source": "FRED official series RETAILIRSA",
            },
            "tightness_proxy": tightness,
            "tightness_formula": "z(CAPUTLG334S) - z(RETAILIRSA)",
        },
        "model": {
            "baseline": row_base.to_dict(),
            "cost_minimizing": row_cost.to_dict(),
            "resilience_first": row_res.to_dict(),
            "targeted_allocation": row_tgt.to_dict(),
            "costmin_vs_resilience": {
                "cost_saving_usd": save_cost_vs_res,
                "fill_rate_delta_pct_point": fill_delta_cost_vs_res * 100.0,
            },
        },
        "interpretation": {
            "type": "real-macro-benchmark-vs-model-scenarios",
            "note": "FRED series are real US macro observations. Model rows are optimization outputs under your calibrated network. This comparison anchors model decisions to real macro conditions, but is not post-deployment realized savings accounting.",
        },
    }

    json_path = REPORT_DIR / "module2_fred_real_comparison.json"
    md_path = REPORT_DIR / "module2_fred_real_comparison.md"

    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# Module2 与 FRED 真实宏观数据对比")
    md.append("")
    md.append("## 1) 真实数据来源")
    md.append("- CAPUTLG334S: FRED 官方序列（计算机与电子制造业产能利用率）")
    md.append("- RETAILIRSA: FRED 官方序列（美国零售库存销售比）")
    md.append("- 获取方式：FRED 官方 CSV 接口")
    md.append("")
    md.append("## 2) 本次对齐月份")
    md.append(f"- 模型需求基准月份: {target_month.date()}")
    md.append(f"- CAPUTLG334S 使用月份: {cap_month.date()}，值: {cap_val:.4f}，历史 zscore: {cap_z:.4f}")
    md.append(f"- RETAILIRSA 使用月份: {inv_month.date()}，值: {inv_val:.4f}，历史 zscore: {inv_z:.4f}")
    md.append(f"- 宏观紧张度代理 tightness = z(cap) - z(inv) = {tightness:.4f}")
    md.append("")
    md.append("## 3) 模型结果（同一轮）")
    md.append(f"- Baseline: fill_rate={float(row_base['fill_rate']):.6f}, cost={float(row_base['total_logistics_cost_usd']):,.2f} USD")
    md.append(f"- Cost-Minimizing: fill_rate={float(row_cost['fill_rate']):.6f}, cost={float(row_cost['total_logistics_cost_usd']):,.2f} USD")
    md.append(f"- Resilience-First: fill_rate={float(row_res['fill_rate']):.6f}, cost={float(row_res['total_logistics_cost_usd']):,.2f} USD")
    md.append(f"- Targeted Allocation: fill_rate={float(row_tgt['fill_rate']):.6f}, cost={float(row_tgt['total_logistics_cost_usd']):,.2f} USD")
    md.append("")
    md.append("## 4) 真实宏观条件下的模型权衡")
    md.append(f"- Cost-Min 相对 Resilience 的模型估计节约: {save_cost_vs_res:,.2f} USD")
    md.append(f"- 同时履约率差异: {-fill_delta_cost_vs_res * 100.0:.3f} 个百分点（Cost-Min 更低）")
    md.append("")
    md.append("## 5) 结论边界")
    md.append("- 以上 FRED 数据是真实美国宏观观测值。")
    md.append("- 对比结果是‘真实宏观数据校准下的模型估计结果’。")
    md.append("- 若要声称‘真实已实现节约’，仍需企业执行后订单级实绩对账。")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
