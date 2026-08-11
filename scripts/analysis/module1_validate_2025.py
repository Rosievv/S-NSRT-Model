#!/usr/bin/env python3
"""
Validate 2025 forward projections with newly collected 2025 actual trade data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from module1_backtest import BACKTEST_PATH, FIGURE3_PATH, PROJECTION_JSON_PATH, generate_figure3
from module1_data_loader import load_module1_trade_data
from risk_propagation import StressTestRunner
from risk_propagation.stress_testing import FORWARD_SCENARIO_LIBRARY


OUT_DIR = ROOT_DIR / "reports" / "module1"
HIST_RAW_DIR = ROOT_DIR / "data" / "raw"
ACTUAL_2025_RAW_DIR = ROOT_DIR / "data" / "raw" / "2025 data"
VALIDATION_CSV_PATH = OUT_DIR / "module1_2025_validation.csv"
VALIDATION_JSON_PATH = OUT_DIR / "module1_2025_validation_metrics.json"


def _compute_rmse(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(math.sqrt(float((series**2).mean())))


def _build_actual_2025_scenario_frame(actual_2025_df: pd.DataFrame) -> pd.DataFrame:
    runner = StressTestRunner(actual_2025_df)
    working = actual_2025_df.copy()
    working["quarter"] = working["date"].dt.to_period("Q")

    rows: list[dict] = []
    for quarter in pd.period_range("2025Q1", "2025Q4", freq="Q"):
        quarter_df = working.loc[working["quarter"] == quarter].copy()
        if quarter_df.empty:
            continue

        for scenario in FORWARD_SCENARIO_LIBRARY:
            result = runner._run_projection_case(
                trade_df=quarter_df,
                scenario=scenario,
                severity_multiplier=1.0,
                elasticity_multiplier=1.0,
            )
            rows.append(
                {
                    "quarter": str(quarter),
                    "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
                    "scenario": scenario["name"],
                    "actual_expected_gap_pct": round(float(result.supply_gap_pct), 2),
                    "actual_substitution_absorbed_pct": round(
                        float(result.substitution_absorbed_pct), 2
                    ),
                    "actual_original_supply_B": round(float(result.original_supply / 1e9), 2),
                    "actual_disrupted_supply_B": round(float(result.disrupted_supply / 1e9), 2),
                }
            )

    return pd.DataFrame(rows)


def run_validation_2025() -> tuple[pd.DataFrame, dict]:
    if not ACTUAL_2025_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing 2025 raw folder: {ACTUAL_2025_RAW_DIR}")

    historical_df = load_module1_trade_data(raw_dir=HIST_RAW_DIR)
    historical_df = historical_df.loc[historical_df["date"] <= pd.Timestamp("2024-12-31")].copy()
    projection_df = StressTestRunner(historical_df).generate_forward_projections_2025_2026()

    actual_2025_df = load_module1_trade_data(raw_dir=ACTUAL_2025_RAW_DIR)
    actual_2025_df = actual_2025_df.loc[
        (actual_2025_df["date"] >= pd.Timestamp("2025-01-01"))
        & (actual_2025_df["date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    actual_frame = _build_actual_2025_scenario_frame(actual_2025_df)
    predicted_2025 = projection_df.loc[
        projection_df["quarter"].astype(str).str.startswith("2025")
    ][["quarter", "quarter_end", "scenario", "expected_gap_pct"]].copy()
    predicted_2025 = predicted_2025.rename(columns={"expected_gap_pct": "predicted_gap_pct"})

    validation = predicted_2025.merge(
        actual_frame,
        on=["quarter", "quarter_end", "scenario"],
        how="inner",
    )
    validation["error_pct"] = validation["predicted_gap_pct"] - validation["actual_expected_gap_pct"]
    validation["abs_error_pct"] = validation["error_pct"].abs()

    scenario_metrics = (
        validation.groupby("scenario", as_index=False)
        .agg(
            mae_pct=("abs_error_pct", "mean"),
            rmse_pct=("error_pct", _compute_rmse),
            mean_predicted_gap_pct=("predicted_gap_pct", "mean"),
            mean_actual_gap_pct=("actual_expected_gap_pct", "mean"),
        )
        .sort_values("scenario")
    )

    overall_metrics = {
        "overall_mae_pct": round(float(validation["abs_error_pct"].mean()), 4),
        "overall_rmse_pct": round(_compute_rmse(validation["error_pct"]), 4),
        "records": int(len(validation)),
    }

    payload = {
        "validation_window": {"start": "2025-01-01", "end": "2025-12-31"},
        "metrics": {
            "overall": overall_metrics,
            "by_scenario": scenario_metrics.to_dict(orient="records"),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(VALIDATION_CSV_PATH, index=False)
    VALIDATION_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return validation, payload


def main() -> None:
    validation_df, metrics_payload = run_validation_2025()

    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Missing backtest baseline: {BACKTEST_PATH}")
    if not PROJECTION_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing forward projection payload: {PROJECTION_JSON_PATH}")

    backtest_df = pd.read_csv(BACKTEST_PATH)
    with PROJECTION_JSON_PATH.open("r", encoding="utf-8") as file_obj:
        projection_records = json.load(file_obj).get("records", [])
    projection_df = pd.DataFrame(projection_records)

    generate_figure3(backtest_df, projection_df, validation_df=validation_df)

    print("2025 validation completed.")
    print(f"Validation CSV: {VALIDATION_CSV_PATH}")
    print(f"Validation metrics JSON: {VALIDATION_JSON_PATH}")
    print(f"Updated figure: {FIGURE3_PATH}")

    overall = metrics_payload["metrics"]["overall"]
    print(
        f"Overall 2025 accuracy -> MAE={overall['overall_mae_pct']:.2f}% | "
        f"RMSE={overall['overall_rmse_pct']:.2f}% | "
        f"N={overall['records']}"
    )

    by_scenario = pd.DataFrame(metrics_payload["metrics"]["by_scenario"])
    if not by_scenario.empty:
        print("\nBy scenario:")
        print(by_scenario.to_string(index=False))


if __name__ == "__main__":
    main()
