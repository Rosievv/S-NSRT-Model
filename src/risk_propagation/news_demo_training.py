"""Small-sample training utilities for the experimental news integration demo."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ALERT_FEATURE_COLUMNS = [
    "confidence",
    "source_count_log",
    "event_detected",
    "location_known",
    "object_known",
    "source_reliability",
]
IMPACT_FEATURE_COLUMNS = [
    "rule_gap_log",
    "dynamic_severity",
    "unabsorbed_share",
    "signal_probability",
]
SOURCE_RELIABILITY = {
    "official": 1.0,
    "operator": 0.95,
    "wire": 0.90,
    "major_news": 0.85,
    "local_news": 0.75,
}


def build_alert_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build publication-time alert features without event identity or dates."""
    features = pd.DataFrame(index=frame.index)
    features["confidence"] = pd.to_numeric(frame["v2_confidence"], errors="coerce").fillna(0.5)
    source_count = pd.to_numeric(frame["independent_source_count"], errors="coerce").fillna(1.0)
    features["source_count_log"] = np.log1p(source_count)
    features["event_detected"] = frame["predicted_events"].fillna("unknown").ne("unknown").astype(float)
    locations = frame["v2_location"].fillna("unknown").astype(str).str.lower()
    features["location_known"] = ~locations.isin(["unknown", ""])
    features["object_known"] = frame["predicted_object_top3"].fillna("unknown").ne("unknown").astype(float)
    features["source_reliability"] = frame["source_tier"].map(SOURCE_RELIABILITY).fillna(0.65)
    return features[ALERT_FEATURE_COLUMNS].astype(float)


def build_impact_features(frame: pd.DataFrame, signal_probability: pd.Series) -> pd.DataFrame:
    """Combine time-available news calibration with structural propagation output."""
    features = pd.DataFrame(index=frame.index)
    features["rule_gap_log"] = np.log1p(frame["news_triggered_supply_gap_pct"].astype(float))
    features["dynamic_severity"] = frame["dynamic_severity"].astype(float)
    features["unabsorbed_share"] = 1.0 - frame["substitution_absorbed_pct"].astype(float) / 100.0
    features["signal_probability"] = signal_probability.astype(float)
    return features[IMPACT_FEATURE_COLUMNS].astype(float)


def _pipeline_payload(pipeline: Pipeline, feature_columns: list[str], target_transform: str) -> Dict[str, Any]:
    scaler = pipeline.named_steps["standardscaler"]
    estimator = pipeline.steps[-1][1]
    return {
        "model_class": estimator.__class__.__name__,
        "feature_columns": feature_columns,
        "target_transform": target_transform,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": np.asarray(estimator.coef_).reshape(-1).tolist(),
        "intercept": np.asarray(estimator.intercept_).reshape(-1).tolist(),
        "parameters": estimator.get_params(),
    }


def train_news_demo_models(
    event_rows: pd.DataFrame,
    negative_controls: pd.DataFrame,
    impact_rows: pd.DataFrame,
    impact_alpha: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Train demo models and return leakage-controlled leave-one-out predictions."""
    positives = event_rows.copy()
    controls = negative_controls.copy()
    positives["demo_target"] = 1
    controls["demo_target"] = 0
    alert_training = pd.concat([positives, controls], ignore_index=True, sort=False)
    alert_features = build_alert_features(alert_training)
    alert_target = alert_training["demo_target"].astype(int).to_numpy()

    alert_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=42),
    )
    alert_oof = cross_val_predict(
        alert_model,
        alert_features,
        alert_target,
        cv=LeaveOneOut(),
        method="predict_proba",
    )[:, 1]
    alert_model.fit(alert_features, alert_target)
    alert_final = alert_model.predict_proba(alert_features)[:, 1]

    alert_oof_by_event = dict(zip(alert_training["event_key"], alert_oof))
    alert_final_by_event = dict(zip(alert_training["event_key"], alert_final))
    result = impact_rows.copy()
    result["demo_alert_probability_oof"] = result["event_key"].map(alert_oof_by_event).astype(float)
    result["demo_alert_probability_final"] = result["event_key"].map(alert_final_by_event).astype(float)

    impact_features_oof = build_impact_features(result, result["demo_alert_probability_oof"])
    impact_target = np.log1p(result["observed_supply_gap_pct"].astype(float).to_numpy())
    impact_model = make_pipeline(StandardScaler(), Ridge(alpha=impact_alpha))
    impact_oof_log = cross_val_predict(
        impact_model,
        impact_features_oof,
        impact_target,
        cv=LeaveOneOut(),
    )
    result["demo_trained_supply_gap_pct"] = np.maximum(0.0, np.expm1(impact_oof_log))

    impact_features_final = build_impact_features(result, result["demo_alert_probability_final"])
    impact_model.fit(impact_features_final, impact_target)
    result["demo_trained_abs_error_pct"] = (
        result["demo_trained_supply_gap_pct"] - result["observed_supply_gap_pct"]
    ).abs()
    common_events = result.dropna(subset=["static_backtest_supply_gap_pct"]).copy()

    metrics = {
        "status": "small_sample_demo_not_for_production",
        "prediction_protocol": "leave_one_record_out_alert_and_leave_one_event_out_impact",
        "event_identity_used_as_feature": False,
        "alert_training_rows": int(len(alert_training)),
        "alert_positive_rows": int(alert_target.sum()),
        "alert_negative_rows": int((alert_target == 0).sum()),
        "alert_loocv_roc_auc": round(float(roc_auc_score(alert_target, alert_oof)), 3),
        "alert_loocv_brier_score": round(float(brier_score_loss(alert_target, alert_oof)), 3),
        "impact_training_events": int(len(result)),
        "impact_ridge_alpha": impact_alpha,
        "rule_impact_mae_pct_points": round(
            float(mean_absolute_error(result["observed_supply_gap_pct"], result["news_triggered_supply_gap_pct"])), 2
        ),
        "demo_impact_loeo_mae_pct_points": round(
            float(mean_absolute_error(result["observed_supply_gap_pct"], result["demo_trained_supply_gap_pct"])), 2
        ),
        "common_event_count": int(len(common_events)),
        "static_common_event_mae_pct_points": round(
            float(mean_absolute_error(
                common_events["observed_supply_gap_pct"], common_events["static_backtest_supply_gap_pct"]
            )),
            2,
        ),
        "rule_common_event_mae_pct_points": round(
            float(mean_absolute_error(
                common_events["observed_supply_gap_pct"], common_events["news_triggered_supply_gap_pct"]
            )),
            2,
        ),
        "demo_common_event_loeo_mae_pct_points": round(
            float(mean_absolute_error(
                common_events["observed_supply_gap_pct"], common_events["demo_trained_supply_gap_pct"]
            )),
            2,
        ),
        "caveat": (
            "The alert classes are curated and easily separable, while impact calibration has only six events. "
            "Metrics demonstrate pipeline execution, not production generalization."
        ),
    }
    artifacts = {
        "alert_model": _pipeline_payload(alert_model, ALERT_FEATURE_COLUMNS, "none"),
        "impact_model": _pipeline_payload(impact_model, IMPACT_FEATURE_COLUMNS, "log1p"),
    }
    return result, metrics, artifacts