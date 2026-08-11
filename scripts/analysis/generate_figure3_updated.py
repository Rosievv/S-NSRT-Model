#!/usr/bin/env python3
"""
Regenerate Module 1 Figure 3 from saved backtest and forward-forecast outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from module1_backtest import FIGURE3_PATH, BACKTEST_PATH, PROJECTION_JSON_PATH, generate_figure3


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "reports" / "module1"
SUMMARY_PATH = OUT_DIR / "module1_stress_summary.csv"


def _load_projection_frame() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    summary_df = pd.read_csv(SUMMARY_PATH)
    forecast_df = summary_df.loc[summary_df["phase"] == "forecast"].copy()
    if forecast_df.empty and PROJECTION_JSON_PATH.exists():
        with PROJECTION_JSON_PATH.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        records = payload.get("records", [])
        return pd.DataFrame(records)
    return forecast_df


def main() -> None:
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Missing backtest file: {BACKTEST_PATH}")

    backtest_df = pd.read_csv(BACKTEST_PATH)
    projection_df = _load_projection_frame()
    generate_figure3(backtest_df, projection_df)
    print(f"Saved Figure 3 to {FIGURE3_PATH}")


if __name__ == "__main__":
    main()