"""Fetch valid FRED orders/backlog series and compare with Module2 outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"

# User-provided IDs were checked and returned 404.
REQUESTED_IDS = ["NO34SVS", "UO34SVS"]

# Valid FRED alternatives for computers/electronic products (NAICS 334)
# A-series are seasonally adjusted, U-series are not seasonally adjusted.
VALID_IDS: Dict[str, str] = {
    "A34SNO": "Manufacturers' New Orders: Computers and Electronic Products (SA)",
    "A34SUO": "Manufacturers' Unfilled Orders: Computers and Electronic Products (SA)",
    "A34SUS": "Manufacturers' Unfilled Orders to Shipments Ratio: Computers and Electronic Products (SA)",
}


def fetch_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df = df.rename(columns={"observation_date": "month", series_id: "value"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["month", "value"]).sort_values("month").reset_index(drop=True)
    df["series_id"] = series_id
    return df


def zscore_upto(df: pd.DataFrame, month: pd.Timestamp) -> float:
    hist = df[df["month"] <= month]["value"]
    if len(hist) < 24:
        return float("nan")
    std = hist.std(ddof=0)
    if std == 0 or pd.isna(std):
        return float("nan")
    cur = float(df.loc[df["month"] == month, "value"].iloc[0])
    return (cur - float(hist.mean())) / float(std)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scen = pd.read_csv(REPORT_DIR / "module2_scenario_comparison.csv")
    scen = scen[scen["strategy"].isin(["Baseline", "Cost-Minimizing", "Resilience-First", "Targeted Allocation"])].copy()

    # Use the common month from current run context (2022-01 in this workspace)
    target_month = pd.Timestamp("2022-01-01")

    requested_status = {}
    for sid in REQUESTED_IDS:
        try:
            _ = fetch_series(sid)
            requested_status[sid] = "available"
        except Exception:
            requested_status[sid] = "not_found_404"

    series_data = {}
    for sid, label in VALID_IDS.items():
        df = fetch_series(sid)
        ref = df[df["month"] <= target_month].tail(1)
        if ref.empty:
            raise RuntimeError(f"No data at or before {target_month.date()} for {sid}")
        m = pd.Timestamp(ref["month"].iloc[0])
        v = float(ref["value"].iloc[0])
        z = zscore_upto(df, m)
        series_data[sid] = {
            "label": label,
            "month": str(m.date()),
            "value": v,
            "zscore_to_history": z,
        }

    row_cost = scen[scen["strategy"] == "Cost-Minimizing"].iloc[0]
    row_res = scen[scen["strategy"] == "Resilience-First"].iloc[0]
    row_tgt = scen[scen["strategy"] == "Targeted Allocation"].iloc[0]

    save_cost_vs_res = float(row_res["total_logistics_cost_usd"] - row_cost["total_logistics_cost_usd"])
    fill_delta_cost_vs_res = float(row_res["fill_rate"] - row_cost["fill_rate"])

    out = {
        "target_month": str(target_month.date()),
        "requested_ids_status": requested_status,
        "fred_valid_series": series_data,
        "model_tradeoff": {
            "costmin_vs_resilience_cost_saving_usd": save_cost_vs_res,
            "costmin_vs_resilience_fill_rate_delta_pct_point": -fill_delta_cost_vs_res * 100.0,
            "targeted_fill_rate": float(row_tgt["fill_rate"]),
        },
        "note": "FRED macro series are real observations; model metrics remain optimization outputs. This gives real-macro-anchored comparison, not post-deployment realized accounting.",
    }

    json_path = REPORT_DIR / "module2_fred_orders_backlog_comparison.json"
    md_path = REPORT_DIR / "module2_fred_orders_backlog_comparison.md"

    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Module2 与 FRED 新订单/积压订单对比")
    lines.append("")
    lines.append("## 1) 你提供的序列ID可用性")
    for sid in REQUESTED_IDS:
        lines.append(f"- {sid}: {requested_status[sid]}")
    lines.append("- 说明：`NO34SVS`/`UO34SVS` 在 FRED 页面返回 404，本报告使用同主题有效序列替代。")
    lines.append("")
    lines.append("## 2) 使用的有效 FRED 真实序列（NAICS 334）")
    for sid, meta in series_data.items():
        lines.append(f"- {sid} ({meta['label']}) -> month={meta['month']}, value={meta['value']:.4f}, zscore={meta['zscore_to_history']:.4f}")
    lines.append("")
    lines.append("## 3) 与模型结果的对比结论")
    lines.append(f"- Cost-Min 相对 Resilience 的模型估计节约: {save_cost_vs_res:,.2f} USD")
    lines.append(f"- Cost-Min 相对 Resilience 的履约率损失: {fill_delta_cost_vs_res * 100.0:.3f} 个百分点")
    lines.append("- 解释：上述节约/履约差异是模型输出；FRED 序列用于锚定美国真实宏观供需状态。")
    lines.append("")
    lines.append("## 4) Cass Freight Index 说明")
    lines.append("- Cass 页面未暴露稳定可抓取的历史文件直链，建议你手动从 Historical Data Archive 下载 Excel。")
    lines.append("- 下载后放入 data/raw，我们可以立刻并入同一份对比框架。")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
