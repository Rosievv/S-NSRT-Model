#!/usr/bin/env python3
"""Compare real vs model values for HS 854231/854232/854233/854239 only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.time_series_model import TimeSeriesForecaster, mean_absolute_error, mean_squared_error, r2_score


ROOT = REPO_ROOT
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "trained" / "grouped"
REPORT_DIR = ROOT / "reports" / "module1"

HS_CODES = ["854231", "854232", "854233", "854239"]


def _load_model(hs_code: str) -> TimeSeriesForecaster:
    model = TimeSeriesForecaster(
        target_variable="log_value_usd",
        forecast_horizon=1,
        model_type="xgboost",
        config={
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "random_state": 42,
        },
    )
    model.load_model(MODEL_DIR / f"value_forecaster_hs{hs_code}")
    return model


def _predict_for_hs(df: pd.DataFrame, hs_code: str) -> pd.DataFrame:
    subset = df[df["hs_code"].astype(str) == hs_code].copy()
    if subset.empty:
        return pd.DataFrame()

    model = _load_model(hs_code)
    X, y = model.prepare_data(subset, scale_features=True)
    y_pred = model.predict(X)

    ordered = subset.sort_values("date").reset_index(drop=True)
    dates = ordered.iloc[:-1]["date"].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "date": dates,
            "hs_code": hs_code,
            "actual_value_usd": np.expm1(y),
            "predicted_value_usd": np.expm1(y_pred),
        }
    )
    out["error_usd"] = out["predicted_value_usd"] - out["actual_value_usd"]
    out["abs_error_usd"] = out["error_usd"].abs()
    out["abs_error_pct"] = np.where(
        out["actual_value_usd"] != 0,
        out["abs_error_usd"] / out["actual_value_usd"] * 100.0,
        0.0,
    )
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(PROCESSED_DIR / "features_test_log.parquet")
    test_df["hs_code"] = test_df["hs_code"].astype(str)

    all_rows = []
    summary_rows = []

    for hs_code in HS_CODES:
        hs_frame = _predict_for_hs(test_df, hs_code)
        if hs_frame.empty:
            continue

        all_rows.append(hs_frame)

        actual = hs_frame["actual_value_usd"].to_numpy()
        pred = hs_frame["predicted_value_usd"].to_numpy()
        summary_rows.append(
            {
                "hs_code": hs_code,
                "n_rows": int(len(hs_frame)),
                "actual_total_usd": float(hs_frame["actual_value_usd"].sum()),
                "predicted_total_usd": float(hs_frame["predicted_value_usd"].sum()),
                "mae_usd": float(mean_absolute_error(actual, pred)),
                "rmse_usd": float(np.sqrt(mean_squared_error(actual, pred))),
                "r2": float(r2_score(actual, pred)),
                "mape_pct": float(
                    np.mean(
                        np.abs((actual[actual != 0] - pred[actual != 0]) / actual[actual != 0])
                    )
                    * 100.0
                    if np.any(actual != 0)
                    else 0.0
                ),
            }
        )

    if not all_rows:
        raise RuntimeError("No rows found for the requested HS codes")

    detail_df = pd.concat(all_rows, ignore_index=True).sort_values(["hs_code", "date"])
    summary_df = pd.DataFrame(summary_rows).sort_values("hs_code")

    detail_path = REPORT_DIR / "module1_hs8542_real_vs_model_detail.csv"
    summary_path = REPORT_DIR / "module1_hs8542_real_vs_model_summary.csv"
    report_path = REPORT_DIR / "module1_hs8542_real_vs_model.json"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "hs_codes": HS_CODES,
        "detail_file": str(detail_path.relative_to(ROOT)),
        "summary_file": str(summary_path.relative_to(ROOT)),
        "summary": summary_rows,
        "notes": [
            "This uses the existing grouped HS models saved under models/trained/grouped/.",
            "Predictions are produced on features_test_log.parquet and aligned to the one-step-ahead target.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {detail_path.relative_to(ROOT)}")
    print(f"Wrote: {summary_path.relative_to(ROOT)}")
    print(f"Wrote: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()