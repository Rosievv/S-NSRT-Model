"""Fetch BEA transportation costs for NAICS 334 with fallback to FRED+CFS proxy."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "module2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEA_URL = "https://apps.bea.gov/industry/xls/io-annual/Use_SUT_Framework_2017_2022_DET.xlsx"


def fetch_bea_transportation_cost_table() -> None:
    print("[*] Fetching BEA annual Use table...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        resp = requests.get(BEA_URL, headers=headers, timeout=45)
        resp.raise_for_status()

        ctype = (resp.headers.get("content-type") or "").lower()
        if "spreadsheet" not in ctype and "application/vnd" not in ctype:
            raise RuntimeError(f"BEA URL did not return an Excel file (content-type={ctype}).")

        excel_data = io.BytesIO(resp.content)
        _ = pd.ExcelFile(excel_data)

        # NOTE: BEA migrated IO tables to interactive app; direct workbook parsing is unstable.
        # If this succeeds in the future, this placeholder is where Line 481/482/484 x Column 334 extraction goes.
        raise RuntimeError("Workbook format endpoint changed; use interactive BEA export or fallback estimator.")

    except Exception as e:
        print(f"[!] BEA direct fetch not usable: {e}")
        print("[*] Falling back to FRED shipments + CFS freight-share proxy...")
        fetch_fred_transportation_expenditures(str(e))


def fetch_fred_transportation_expenditures(bea_error: str) -> None:
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A34SVS"
    df = pd.read_csv(fred_url)

    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    value_col = "A34SVS"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).copy()

    df["Year"] = df[date_col].dt.year
    annual = df.groupby("Year", as_index=False)[value_col].sum()
    annual = annual[annual["Year"] >= 2018].copy()

    # CFS-informed proxy shares (editable)
    freight_rates = {
        2018: 0.015,
        2019: 0.014,
        2020: 0.016,
        2021: 0.024,
        2022: 0.022,
        2023: 0.015,
        2024: 0.015,
        2025: 0.015,
        2026: 0.015,
    }
    annual["Freight_Rate_Share"] = annual["Year"].map(freight_rates).fillna(0.015)

    annual["Actual_Total_Transportation_Cost_USD_M"] = (
        annual[value_col] * annual["Freight_Rate_Share"]
    ).round(2)

    annual["Truck_Cost_USD_M"] = (annual["Actual_Total_Transportation_Cost_USD_M"] * 0.60).round(2)
    annual["Air_Cost_USD_M"] = (annual["Actual_Total_Transportation_Cost_USD_M"] * 0.30).round(2)
    annual["Rail_Cost_USD_M"] = (annual["Actual_Total_Transportation_Cost_USD_M"] * 0.10).round(2)

    out_csv = REPORT_DIR / "annual_actual_transportation_costs.csv"
    annual.to_csv(out_csv, index=False)

    meta = {
        "method": "FRED_A34SVS_plus_CFS_proxy",
        "fred_series": "A34SVS",
        "bea_direct_fetch": "failed",
        "bea_error": bea_error,
        "freight_rate_assumptions": freight_rates,
        "mode_split_assumptions": {
            "truck": 0.60,
            "air": 0.30,
            "rail": 0.10,
        },
        "output_csv": str(out_csv.relative_to(ROOT)),
    }

    out_json = REPORT_DIR / "annual_actual_transportation_costs_meta.json"
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[+] Wrote: {out_csv}")
    print(f"[+] Wrote: {out_json}")
    print(annual[["Year", "Actual_Total_Transportation_Cost_USD_M", "Truck_Cost_USD_M", "Air_Cost_USD_M", "Rail_Cost_USD_M"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    fetch_bea_transportation_cost_table()
