#!/usr/bin/env python3
"""
Build unified 2010-2026 quarterly predicted-vs-actual series for Module 1.

- 2010-2024 predictions are generated with rolling out-of-sample training.
- 2025-2026 predictions are generated out-of-sample from <=2024 data.
- Actuals are computed from realized quarterly trade panels (2010-2025 available).
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
VALIDATION_CSV_PATH = OUT_DIR / "module1_2020_2026_unified_validation.csv"
VALIDATION_JSON_PATH = OUT_DIR / "module1_2020_2026_unified_validation_metrics.json"

TARGET_SCENARIOS = [
    "baseline_moderate",
    "top_1_supplier_failure",
]


def _scenario_override_map(top1_severity: float | None = None) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    if top1_severity is not None:
        overrides["top_1_supplier_failure"] = {"severity": float(top1_severity)}
    return overrides


def _compute_rmse(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(math.sqrt(float((series**2).mean())))


def _selected_scenarios() -> list[dict]:
    return [scenario for scenario in FORWARD_SCENARIO_LIBRARY if scenario["name"] in TARGET_SCENARIOS]


def _selected_scenarios_with_overrides(
    scenario_overrides: dict[str, dict] | None = None,
) -> list[dict]:
    scenarios = []
    for scenario in _selected_scenarios():
        merged = dict(scenario)
        if scenario_overrides and scenario["name"] in scenario_overrides:
            merged.update(scenario_overrides[scenario["name"]])
        scenarios.append(merged)
    return scenarios


def _build_actual_scenario_frame(
    actual_df: pd.DataFrame,
    start_quarter: str,
    end_quarter: str,
    scenario_overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    runner = StressTestRunner(actual_df)
    working = actual_df.copy()
    working["quarter"] = working["date"].dt.to_period("Q")

    rows: list[dict] = []
    for quarter in pd.period_range(start_quarter, end_quarter, freq="Q"):
        quarter_df = working.loc[working["quarter"] == quarter].copy()
        if quarter_df.empty:
            continue

        for scenario in _selected_scenarios_with_overrides(scenario_overrides):
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


def _build_projected_scenario_frame(
    train_df: pd.DataFrame,
    start_month: str,
    end_month: str,
    start_quarter: str,
    end_quarter: str,
    source_tag: str,
    scenario_overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    runner = StressTestRunner(train_df)
    projected_panel = runner._project_monthly_trade_panel(start=start_month, end=end_month)
    if projected_panel.empty:
        return pd.DataFrame()

    projected_panel = projected_panel.copy()
    projected_panel["quarter"] = projected_panel["date"].dt.to_period("Q")

    rows: list[dict] = []
    bound_cases = {
        "lower": (0.85, 1.15),
        "expected": (1.0, 1.0),
        "upper": (1.15, 0.85),
    }
    for quarter in pd.period_range(start_quarter, end_quarter, freq="Q"):
        quarter_df = projected_panel.loc[projected_panel["quarter"] == quarter].copy()
        if quarter_df.empty:
            continue

        for scenario in _selected_scenarios_with_overrides(scenario_overrides):
            case_results = {}
            for bound_name, (severity_multiplier, elasticity_multiplier) in bound_cases.items():
                case_results[bound_name] = runner._run_projection_case(
                    trade_df=quarter_df,
                    scenario=scenario,
                    severity_multiplier=severity_multiplier,
                    elasticity_multiplier=elasticity_multiplier,
                )

            lower_gap = min(
                case_results["lower"].supply_gap_pct,
                case_results["expected"].supply_gap_pct,
                case_results["upper"].supply_gap_pct,
            )
            upper_gap = max(
                case_results["lower"].supply_gap_pct,
                case_results["expected"].supply_gap_pct,
                case_results["upper"].supply_gap_pct,
            )
            expected_result = case_results["expected"]

            rows.append(
                {
                    "quarter": str(quarter),
                    "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
                    "scenario": scenario["name"],
                    "predicted_gap_pct": round(float(expected_result.supply_gap_pct), 2),
                    "lower_gap_pct": round(float(lower_gap), 2),
                    "upper_gap_pct": round(float(upper_gap), 2),
                    "prediction_source": source_tag,
                }
            )

    return pd.DataFrame(rows)


def _build_projected_scenario_frame_rolling(
    full_df: pd.DataFrame,
    start_quarter: str,
    end_quarter: str,
    source_tag: str,
    scenario_overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    bound_cases = {
        "lower": (0.85, 1.15),
        "expected": (1.0, 1.0),
        "upper": (1.15, 0.85),
    }

    for quarter in pd.period_range(start_quarter, end_quarter, freq="Q"):
        train_end = pd.Timestamp(quarter.start_time) - pd.Timedelta(days=1)
        train_slice = full_df.loc[full_df["date"] <= train_end].copy()
        if train_slice.empty:
            continue

        runner = StressTestRunner(train_slice)
        projected_panel = runner._project_monthly_trade_panel(
            start=quarter.start_time.strftime("%Y-%m"),
            end=quarter.end_time.strftime("%Y-%m"),
        )
        if projected_panel.empty:
            continue

        quarter_df = projected_panel.copy()
        for scenario in _selected_scenarios_with_overrides(scenario_overrides):
            case_results = {}
            for bound_name, (severity_multiplier, elasticity_multiplier) in bound_cases.items():
                case_results[bound_name] = runner._run_projection_case(
                    trade_df=quarter_df,
                    scenario=scenario,
                    severity_multiplier=severity_multiplier,
                    elasticity_multiplier=elasticity_multiplier,
                )

            lower_gap = min(
                case_results["lower"].supply_gap_pct,
                case_results["expected"].supply_gap_pct,
                case_results["upper"].supply_gap_pct,
            )
            upper_gap = max(
                case_results["lower"].supply_gap_pct,
                case_results["expected"].supply_gap_pct,
                case_results["upper"].supply_gap_pct,
            )
            expected_result = case_results["expected"]

            rows.append(
                {
                    "quarter": str(quarter),
                    "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
                    "scenario": scenario["name"],
                    "predicted_gap_pct": round(float(expected_result.supply_gap_pct), 2),
                    "lower_gap_pct": round(float(lower_gap), 2),
                    "upper_gap_pct": round(float(upper_gap), 2),
                    "prediction_source": source_tag,
                }
            )

    return pd.DataFrame(rows)


def _attach_errors(validation: pd.DataFrame) -> pd.DataFrame:
    output = validation.copy()
    has_actual = output["actual_expected_gap_pct"].notna()
    output.loc[has_actual, "error_pct"] = (
        output.loc[has_actual, "predicted_gap_pct"]
        - output.loc[has_actual, "actual_expected_gap_pct"]
    )
    output.loc[has_actual, "abs_error_pct"] = output.loc[has_actual, "error_pct"].abs()
    return output


def _compute_metrics_payload(validation: pd.DataFrame) -> dict:
    has_actual = validation["actual_expected_gap_pct"].notna()
    scenario_metrics = (
        validation.loc[has_actual]
        .groupby("scenario", as_index=False)
        .agg(
            mae_pct=("abs_error_pct", "mean"),
            rmse_pct=("error_pct", _compute_rmse),
            mean_predicted_gap_pct=("predicted_gap_pct", "mean"),
            mean_actual_gap_pct=("actual_expected_gap_pct", "mean"),
        )
        .sort_values("scenario")
    )

    overall_metrics = {
        "overall_mae_pct": round(float(validation.loc[has_actual, "abs_error_pct"].mean()), 4),
        "overall_rmse_pct": round(_compute_rmse(validation.loc[has_actual, "error_pct"]), 4),
        "records": int(has_actual.sum()),
    }

    return {
        "overall": overall_metrics,
        "by_scenario": scenario_metrics.to_dict(orient="records"),
    }


def _calibrate_top1_severity(
    historical_df: pd.DataFrame,
    actual_all_df: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    candidate_severity = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_severity = 1.0
    best_mae = float("inf")
    best_validation = pd.DataFrame()

    for severity in candidate_severity:
        overrides = _scenario_override_map(top1_severity=severity)
        pred_2010_2024 = _build_projected_scenario_frame_rolling(
            full_df=historical_df,
            start_quarter="2010Q1",
            end_quarter="2024Q4",
            source_tag="trained_to_prev_quarter",
            scenario_overrides=overrides,
        )
        pred_2025_2026 = _build_projected_scenario_frame(
            train_df=historical_df.loc[historical_df["date"] <= pd.Timestamp("2024-12-31")].copy(),
            start_month="2025-01",
            end_month="2026-12",
            start_quarter="2025Q1",
            end_quarter="2026Q4",
            source_tag="trained_to_2024",
            scenario_overrides=overrides,
        )
        pred_all = pd.concat([pred_2010_2024, pred_2025_2026], ignore_index=True)

        actual_frame = _build_actual_scenario_frame(
            actual_df=actual_all_df,
            start_quarter="2010Q1",
            end_quarter="2025Q4",
            scenario_overrides=overrides,
        )

        validation = pred_all.merge(
            actual_frame,
            on=["quarter", "quarter_end", "scenario"],
            how="left",
        )
        validation = validation.sort_values(["quarter_end", "scenario"]).reset_index(drop=True)
        validation = _attach_errors(validation)

        top1 = validation.loc[
            (validation["scenario"] == "top_1_supplier_failure")
            & validation["actual_expected_gap_pct"].notna()
        ].copy()
        mae = float(top1["abs_error_pct"].mean()) if not top1.empty else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_severity = severity
            best_validation = validation

    return best_severity, best_validation


def run_validation_2020_2026() -> tuple[pd.DataFrame, dict]:
    if not ACTUAL_2025_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing 2025 raw folder: {ACTUAL_2025_RAW_DIR}")

    historical_df = load_module1_trade_data(raw_dir=HIST_RAW_DIR)
    historical_df["date"] = pd.to_datetime(historical_df["date"])

    train_2024_df = historical_df.loc[historical_df["date"] <= pd.Timestamp("2024-12-31")].copy()

    actual_2025_df = load_module1_trade_data(raw_dir=ACTUAL_2025_RAW_DIR)
    actual_2025_df["date"] = pd.to_datetime(actual_2025_df["date"])
    actual_2025_df = actual_2025_df.loc[
        (actual_2025_df["date"] >= pd.Timestamp("2025-01-01"))
        & (actual_2025_df["date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    predicted_2010_2024 = _build_projected_scenario_frame_rolling(
        full_df=historical_df,
        start_quarter="2010Q1",
        end_quarter="2024Q4",
        source_tag="trained_to_prev_quarter",
    )
    predicted_2025_2026 = _build_projected_scenario_frame(
        train_df=train_2024_df,
        start_month="2025-01",
        end_month="2026-12",
        start_quarter="2025Q1",
        end_quarter="2026Q4",
        source_tag="trained_to_2024",
    )
    predicted_all = pd.concat([predicted_2010_2024, predicted_2025_2026], ignore_index=True)

    actual_2010_2024_df = historical_df.loc[
        (historical_df["date"] >= pd.Timestamp("2010-01-01"))
        & (historical_df["date"] <= pd.Timestamp("2024-12-31"))
    ].copy()
    actual_all_df = pd.concat([actual_2010_2024_df, actual_2025_df], ignore_index=True)
    actual_frame = _build_actual_scenario_frame(
        actual_df=actual_all_df,
        start_quarter="2010Q1",
        end_quarter="2025Q4",
    )

    validation = predicted_all.merge(
        actual_frame,
        on=["quarter", "quarter_end", "scenario"],
        how="left",
    )
    validation = validation.sort_values(["quarter_end", "scenario"]).reset_index(drop=True)
    validation = _attach_errors(validation)

    # Calibrate only top-1 severity using 2020-2025 realized panel.
    best_top1_severity, calibrated_validation = _calibrate_top1_severity(
        historical_df=historical_df,
        actual_all_df=actual_all_df,
    )
    if not calibrated_validation.empty:
        validation = calibrated_validation

    metrics = _compute_metrics_payload(validation)

    payload = {
        "prediction_window": {"start": "2010-01-01", "end": "2026-12-31"},
        "actual_window": {"start": "2010-01-01", "end": "2025-12-31"},
        "calibration": {
            "top_1_supplier_failure_severity": best_top1_severity,
            "method": "grid_search_on_2010_2025_top1_mae",
            "candidates": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "metrics": metrics,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(VALIDATION_CSV_PATH, index=False)
    VALIDATION_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return validation, payload


def main() -> None:
    validation_df, metrics_payload = run_validation_2020_2026()

    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Missing backtest baseline: {BACKTEST_PATH}")
    if not PROJECTION_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing forward projection payload: {PROJECTION_JSON_PATH}")

    backtest_df = pd.read_csv(BACKTEST_PATH)
    with PROJECTION_JSON_PATH.open("r", encoding="utf-8") as file_obj:
        projection_records = json.load(file_obj).get("records", [])
    projection_df = pd.DataFrame(projection_records)

    generate_figure3(backtest_df, projection_df, validation_df=validation_df)

    print("Unified 2010-2026 validation dataset completed.")
    print(f"Validation CSV: {VALIDATION_CSV_PATH}")
    print(f"Validation metrics JSON: {VALIDATION_JSON_PATH}")
    print(f"Updated figure: {FIGURE3_PATH}")

    overall = metrics_payload["metrics"]["overall"]
    calibration = metrics_payload.get("calibration", {})
    print(
        f"Overall 2010-2025 accuracy -> MAE={overall['overall_mae_pct']:.2f}% | "
        f"RMSE={overall['overall_rmse_pct']:.2f}% | "
        f"N={overall['records']}"
    )
    if calibration:
        print(
            "Calibrated top-1 severity: "
            f"{calibration.get('top_1_supplier_failure_severity')} "
            f"({calibration.get('method')})"
        )

    by_scenario = pd.DataFrame(metrics_payload["metrics"]["by_scenario"])
    if not by_scenario.empty:
        print("\nBy scenario:")
        print(by_scenario.to_string(index=False))


if __name__ == "__main__":
    main()
