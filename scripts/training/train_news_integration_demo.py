#!/usr/bin/env python3
"""Train a small-sample news integration demo and regenerate two legacy views."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "analysis"))

from build_fulltext_news_replay import plot_figure3_news_alarm_timeline
from build_news_integrated_module1 import build_integrated_replay
from module1_backtest import generate_figure3
from risk_propagation.news_demo_training import train_news_demo_models


REPORT_DIR = ROOT_DIR / "reports" / "module1"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT_DIR / "models" / "trained"
EVENT_ROWS_PATH = REPORT_DIR / "event_listener_v2_evaluation.csv"
NEGATIVE_CONTROLS_PATH = REPORT_DIR / "event_listener_v2_negative_controls.csv"
BACKTEST_PATH = REPORT_DIR / "module1_backtest.csv"
VALIDATION_PATH = REPORT_DIR / "module1_2020_2026_unified_validation.csv"
ALARM_TIMELINE_PATH = REPORT_DIR / "figure3_news_early_alarm_timeline.csv"
PREDICTIONS_PATH = REPORT_DIR / "news_integration_trained_demo_predictions.csv"
METRICS_PATH = REPORT_DIR / "news_integration_trained_demo_metrics.json"
MODELS_PATH = MODEL_DIR / "news_integration_trained_demo_models.json"


def main() -> None:
    required_paths = [
        EVENT_ROWS_PATH,
        NEGATIVE_CONTROLS_PATH,
        BACKTEST_PATH,
        VALIDATION_PATH,
        ALARM_TIMELINE_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing demo training inputs: {missing}")

    impact_rows, _ = build_integrated_replay()
    predictions, metrics, artifacts = train_news_demo_models(
        event_rows=pd.read_csv(EVENT_ROWS_PATH),
        negative_controls=pd.read_csv(NEGATIVE_CONTROLS_PATH),
        impact_rows=impact_rows,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    MODELS_PATH.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    plot_rows = predictions.copy()
    plot_rows["rule_news_triggered_supply_gap_pct"] = plot_rows["news_triggered_supply_gap_pct"]
    plot_rows["news_triggered_supply_gap_pct"] = plot_rows["demo_trained_supply_gap_pct"]
    plot_rows["day0_alarm_score"] = plot_rows["demo_alert_probability_oof"]
    plot_rows["day0_alarm_level"] = pd.cut(
        plot_rows["day0_alarm_score"],
        bins=[-float("inf"), 0.65, 0.8, float("inf")],
        labels=["watch", "warning", "critical"],
        right=False,
    ).astype(str)

    generate_figure3(
        backtest_df=pd.read_csv(BACKTEST_PATH),
        projection_df=pd.DataFrame(),
        validation_df=pd.read_csv(VALIDATION_PATH),
        output_path=FIGURE_DIR / "figure3_backtest_and_2026_forecast_trained_demo.png",
        integrated_event_df=plot_rows,
        integrated_model_label="Trained Demo",
    )
    plot_figure3_news_alarm_timeline(
        timeline=pd.read_csv(ALARM_TIMELINE_PATH),
        output_path=FIGURE_DIR / "figure3_news_early_alarm_timeline_trained_demo.png",
        integrated_event_df=plot_rows,
        integrated_model_label="Trained Demo",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Models: {MODELS_PATH}")


if __name__ == "__main__":
    main()