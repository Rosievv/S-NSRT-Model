"""Train and evaluate Module 3 with structural-cycle upgrades.

Key upgrades implemented in this script:
1) Expanding-window retraining by forecast year.
2) Macro-cycle features to reduce structural level mismatch.
3) Conformalized quantile calibration (CQR) for interval coverage.
4) Module 1 risk used as model input features (plus legacy multiplicative track for comparison).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demand_forecasting import (
    QuantileForecaster,
    add_shortage_metrics,
    apply_gap_elasticity,
    apply_risk_gating,
    adjust_supply_quantiles,
    attach_module1_risk,
    build_inventory_scenarios,
)

TRAIN_PATH = Path("data/processed/features_train_full.parquet")
TEST_PATH = Path("data/processed/features_test_full.parquet")
MODULE1_PATH = Path("reports/module1/module1_2020_2026_unified_validation.csv")
OUTPUT_DIR = Path("reports/module3")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = Path("models/trained/module3")

HORIZONS = (1, 3, 6)
QUANTILES = (0.10, 0.50, 0.90)
GATE_THRESHOLD_GRID = (2.0, 5.0, 8.0)
DEFAULT_THRESHOLD = 5.0
DEMAND_MODE = "seasonal"  # options: ma12, seasonal
CQR_RISK_STRATA_THRESHOLD_PCT = 3.0
MIN_COVERAGE_FLOOR = 0.70
MIN_RECALL_FLOOR = 0.40
TARGET_COVERAGE_BY_HORIZON = {1: 0.78, 3: 0.84, 6: 0.86}
ELASTICITY_BY_HORIZON = {1: 0.06, 3: 0.22, 6: 0.45}
MAE_OBJECTIVE_WEIGHTS = {1: 0.45, 3: 0.35, 6: 0.20}
MODULE1_BIAS_ROLLING_WINDOW = 12
MODULE1_BIAS_CLIP_MIN = 0.85
MODULE1_BIAS_CLIP_MAX = 1.25


def _pinball(y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _risk_stratum_labels(gap_pct: pd.Series, threshold_pct: float) -> pd.Series:
    gap = gap_pct.fillna(0.0)
    return np.where(gap >= threshold_pct, "trigger_active", "low_risk")


def build_monthly_hs_panel(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    use_cols = ["date", "hs_code", "value_usd"]
    optional = [
        "quantity",
        "hhi",
        "top1_share",
        "top3_share",
        "top5_share",
        "n_suppliers",
        "hhi_change_mom",
        "hhi_change_yoy",
        "value_growth_mom",
        "value_growth_yoy",
    ]
    for col in optional:
        if col in train_raw.columns and col in test_raw.columns:
            use_cols.append(col)

    train_df = train_raw[use_cols].copy()
    test_df = test_raw[use_cols].copy()

    for frame in (train_df, test_df):
        frame["date"] = pd.to_datetime(frame["date"])

    full = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)

    agg = {}
    for col in use_cols:
        if col in {"date", "hs_code"}:
            continue
        agg[col] = "sum" if col in {"value_usd", "quantity", "n_suppliers"} else "mean"

    panel = full.groupby(["date", "hs_code", "split"], as_index=False).agg(agg)

    hs_train = set(panel.loc[panel["split"].eq("train"), "hs_code"].unique())
    hs_test = set(panel.loc[panel["split"].eq("test"), "hs_code"].unique())
    common_hs = sorted(hs_train.intersection(hs_test))
    panel = panel[panel["hs_code"].isin(common_hs)].copy()

    panel = panel.sort_values(["hs_code", "date"]).reset_index(drop=True)
    g = panel.groupby("hs_code", group_keys=False)

    panel["lag1_value"] = g["value_usd"].shift(1)
    panel["lag3_value"] = g["value_usd"].shift(3)
    panel["lag6_value"] = g["value_usd"].shift(6)
    panel["lag12_value"] = g["value_usd"].shift(12)
    panel["ma3_value"] = g["value_usd"].transform(lambda s: s.shift(1).rolling(3).mean())
    panel["ma6_value"] = g["value_usd"].transform(lambda s: s.shift(1).rolling(6).mean())
    panel["ma12_value"] = g["value_usd"].transform(lambda s: s.shift(1).rolling(12).mean())
    panel["std6_value"] = g["value_usd"].transform(lambda s: s.shift(1).rolling(6).std())

    panel["month"] = panel["date"].dt.month
    panel["month_sin"] = np.sin(2.0 * np.pi * panel["month"] / 12.0)
    panel["month_cos"] = np.cos(2.0 * np.pi * panel["month"] / 12.0)

    train_panel = panel[panel["split"].eq("train")].drop(columns="split").copy()
    test_panel = panel[panel["split"].eq("test")].drop(columns="split").copy()

    train_panel = train_panel.dropna(subset=["lag12_value"]).reset_index(drop=True)
    test_panel = test_panel.dropna(subset=["lag12_value"]).reset_index(drop=True)
    return train_panel, test_panel


def add_macro_cycle_features(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_panel, test_panel], ignore_index=True)
    total = combined.groupby("date", as_index=False)["value_usd"].sum().sort_values("date")
    total = total.rename(columns={"value_usd": "macro_total_value"})

    total["macro_total_lag1"] = total["macro_total_value"].shift(1)
    total["macro_total_lag3"] = total["macro_total_value"].shift(3)
    total["macro_total_lag12"] = total["macro_total_value"].shift(12)
    total["macro_total_ma6"] = total["macro_total_lag1"].rolling(6).mean()

    total["macro_mom"] = _safe_series_div(total["macro_total_lag1"], total["macro_total_lag1"].shift(1)) - 1.0
    total["macro_yoy"] = _safe_series_div(total["macro_total_lag1"], total["macro_total_lag12"]) - 1.0
    total["macro_deviation_6m"] = _safe_series_div(total["macro_total_lag1"], total["macro_total_ma6"]) - 1.0

    keep = [
        "date",
        "macro_total_lag1",
        "macro_total_lag3",
        "macro_total_lag12",
        "macro_total_ma6",
        "macro_mom",
        "macro_yoy",
        "macro_deviation_6m",
    ]

    train_out = train_panel.merge(total[keep], on="date", how="left")
    test_out = test_panel.merge(total[keep], on="date", how="left")

    for frame in (train_out, test_out):
        frame["macro_mom"] = frame["macro_mom"].replace([np.inf, -np.inf], np.nan)
        frame["macro_yoy"] = frame["macro_yoy"].replace([np.inf, -np.inf], np.nan)
        frame["macro_deviation_6m"] = frame["macro_deviation_6m"].replace([np.inf, -np.inf], np.nan)

    return train_out, test_out


def _safe_series_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0.0, np.nan)
    return numerator / den


def attach_module1_feature_inputs(panel: pd.DataFrame, module1_validation: pd.DataFrame) -> pd.DataFrame:
    risk = module1_validation.loc[
        module1_validation["scenario"].eq("top_1_supplier_failure"),
        ["quarter_end", "predicted_gap_pct", "lower_gap_pct", "upper_gap_pct"],
    ].copy()
    risk["quarter"] = pd.to_datetime(risk["quarter_end"]).dt.to_period("Q")
    risk = risk.drop_duplicates("quarter", keep="last").drop(columns="quarter_end")

    out = panel.copy()
    out["quarter"] = pd.to_datetime(out["date"]).dt.to_period("Q")
    out = out.merge(risk, on="quarter", how="left")

    for col in ("predicted_gap_pct", "lower_gap_pct", "upper_gap_pct"):
        out[col] = out[col].fillna(0.0).clip(0.0, 100.0)

    out["risk_gap_spread"] = (out["upper_gap_pct"] - out["lower_gap_pct"]).clip(lower=0.0)
    out["risk_macro_interaction"] = out["predicted_gap_pct"] * out["macro_yoy"].fillna(0.0)
    # HS-level exposure feature adds cross-sectional variation to the same-quarter Module1 signal.
    date_lag12_total = out.groupby("date")["lag12_value"].transform("sum")
    out["hs_value_share_lag12"] = _safe_series_div(out["lag12_value"], date_lag12_total).clip(0.0, 1.0)
    out["hs_value_share_lag12"] = out["hs_value_share_lag12"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["risk_gap_hs_exposure"] = out["predicted_gap_pct"] * out["hs_value_share_lag12"]

    hhi_center = out["hhi"].median() if "hhi" in out.columns else 0.0
    hhi_term = out["hhi"].fillna(hhi_center) if "hhi" in out.columns else 0.0
    out["risk_gap_hhi_interaction"] = out["predicted_gap_pct"] * hhi_term
    return out.drop(columns="quarter")


def apply_module1_rolling_bias_correction(
    frame: pd.DataFrame,
    pred_cols: Tuple[str, str, str],
    window: int,
    clip_min: float,
    clip_max: float,
) -> pd.DataFrame:
    out = frame.copy()
    q10_col, q50_col, q90_col = pred_cols
    out["module1_bias_factor"] = 1.0

    for horizon, part in out.groupby("horizon_months"):
        by_date = (
            part.groupby("forecast_date", as_index=False)[["actual_supply", q50_col]]
            .sum()
            .sort_values("forecast_date")
        )
        ratio = _safe_series_div(by_date["actual_supply"], by_date[q50_col]).replace([np.inf, -np.inf], np.nan)
        bias = ratio.shift(1).rolling(window, min_periods=3).median()
        bias = bias.fillna(1.0).clip(lower=clip_min, upper=clip_max)
        bias_map = dict(zip(by_date["forecast_date"], bias))

        idx = out["horizon_months"].eq(horizon)
        factors = out.loc[idx, "forecast_date"].map(bias_map).astype(float).fillna(1.0)
        out.loc[idx, "module1_bias_factor"] = factors
        out.loc[idx, q10_col] = out.loc[idx, q10_col].to_numpy(dtype=float) * factors.to_numpy(dtype=float)
        out.loc[idx, q50_col] = out.loc[idx, q50_col].to_numpy(dtype=float) * factors.to_numpy(dtype=float)
        out.loc[idx, q90_col] = out.loc[idx, q90_col].to_numpy(dtype=float) * factors.to_numpy(dtype=float)

    out[q10_col] = out[q10_col].clip(lower=0.0)
    out[q50_col] = np.maximum(out[q50_col], out[q10_col])
    out[q90_col] = np.maximum(out[q90_col], out[q50_col])
    return out


def seasonal_index_from_train(train_panel: pd.DataFrame) -> pd.DataFrame:
    frame = train_panel.copy()
    frame["month"] = frame["date"].dt.month
    month_avg = frame.groupby(["hs_code", "month"], as_index=False)["value_usd"].mean().rename(columns={"value_usd": "month_avg"})
    overall = frame.groupby("hs_code", as_index=False)["value_usd"].mean().rename(columns={"value_usd": "overall_avg"})
    idx = month_avg.merge(overall, on="hs_code", how="left")
    idx["seasonal_index"] = np.where(idx["overall_avg"] > 0, idx["month_avg"] / idx["overall_avg"], 1.0)
    idx["seasonal_index"] = idx["seasonal_index"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return idx[["hs_code", "month", "seasonal_index"]]


def feature_columns(panel: pd.DataFrame) -> List[str]:
    blocked = {"date", "hs_code", "value_usd", "month"}
    cols = [c for c in panel.columns if c not in blocked]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(panel[c])]
    return sorted(numeric_cols)


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        return 0.0
    sorted_scores = np.sort(scores)
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(sorted_scores[k - 1])


def split_fit_calibration(panel: pd.DataFrame, calib_months: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_date = pd.to_datetime(panel["date"]).max()
    calib_start = max_date - pd.DateOffset(months=calib_months)

    fit_df = panel[pd.to_datetime(panel["date"]) <= calib_start].copy()
    calib_df = panel[pd.to_datetime(panel["date"]) > calib_start].copy()

    if fit_df.empty or calib_df.empty:
        return panel.copy(), panel.iloc[0:0].copy()
    return fit_df, calib_df


def apply_conformal_interval_expansion(
    model: QuantileForecaster,
    fit_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    feature_cols: List[str],
    alpha: float,
) -> float:
    if calib_df.empty:
        return 0.0

    calib_supervised = model.build_supervised_frame(calib_df)
    calib_supervised = calib_supervised.dropna(subset=feature_cols + ["forecast_target"])
    if calib_supervised.empty:
        return 0.0

    calib_pred = model.predict(calib_supervised)
    merged = calib_supervised[["date", "hs_code", "forecast_target"]].merge(calib_pred, on=["date", "hs_code"], how="inner")
    if merged.empty:
        return 0.0

    y = merged["forecast_target"].to_numpy(dtype=float)
    q10 = merged["q10"].to_numpy(dtype=float)
    q90 = merged["q90"].to_numpy(dtype=float)

    scores = np.maximum(np.maximum(q10 - y, y - q90), 0.0)
    return conformal_qhat(scores, alpha=alpha)


def apply_conformal_interval_expansion_stratified(
    model: QuantileForecaster,
    calib_df: pd.DataFrame,
    feature_cols: List[str],
    alpha: float,
    risk_threshold_pct: float,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if calib_df.empty:
        return 0.0, {}, {}

    calib_supervised = model.build_supervised_frame(calib_df)
    needed = feature_cols + ["forecast_target", "predicted_gap_pct"]
    calib_supervised = calib_supervised.dropna(subset=needed)
    if calib_supervised.empty:
        return 0.0, {}, {}

    calib_pred = model.predict(calib_supervised)
    merged = calib_supervised[["date", "hs_code", "forecast_target", "predicted_gap_pct"]].merge(
        calib_pred,
        on=["date", "hs_code"],
        how="inner",
    )
    if merged.empty:
        return 0.0, {}, {}

    y = merged["forecast_target"].to_numpy(dtype=float)
    q10 = merged["q10"].to_numpy(dtype=float)
    q90 = merged["q90"].to_numpy(dtype=float)
    scores = np.maximum(np.maximum(q10 - y, y - q90), 0.0)
    merged["conformal_score"] = scores
    merged["risk_stratum"] = _risk_stratum_labels(merged["predicted_gap_pct"], risk_threshold_pct)

    global_qhat = conformal_qhat(merged["conformal_score"].to_numpy(dtype=float), alpha=alpha)
    stratum_qhat: Dict[str, float] = {}
    stratum_counts: Dict[str, int] = {}
    for stratum, part in merged.groupby("risk_stratum"):
        stratum_counts[str(stratum)] = int(len(part))
        stratum_qhat[str(stratum)] = conformal_qhat(part["conformal_score"].to_numpy(dtype=float), alpha=alpha)
    return global_qhat, stratum_qhat, stratum_counts


def predict_with_actual_expanding(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    feat_cols: List[str],
    horizon: int,
    backend: str,
    target_coverage: float,
    cqr_risk_strata_threshold_pct: float,
) -> Tuple[pd.DataFrame, QuantileForecaster, Dict[str, object]]:
    alpha = 1.0 - target_coverage

    proto = QuantileForecaster(
        quantiles=list(QUANTILES),
        target_col="value_usd",
        forecast_horizon=horizon,
        backend=backend,
    )
    supervised_test = proto.build_supervised_frame(test_panel)
    supervised_test = supervised_test.dropna(subset=feat_cols + ["forecast_target"]).copy()

    yearly_parts = []
    diagnostics = {}
    last_model = None

    for year in sorted(supervised_test["date"].dt.year.unique()):
        block = supervised_test[supervised_test["date"].dt.year == year].copy()
        if block.empty:
            continue

        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        prior_test = test_panel[pd.to_datetime(test_panel["date"]) <= cutoff].copy()
        fit_pool = pd.concat([train_panel, prior_test], ignore_index=True)
        fit_pool = fit_pool.drop_duplicates(subset=["date", "hs_code"], keep="last")
        fit_pool = fit_pool.sort_values(["date", "hs_code"]).reset_index(drop=True)

        fit_df, calib_df = split_fit_calibration(fit_pool, calib_months=12)

        model = QuantileForecaster(
            quantiles=list(QUANTILES),
            target_col="value_usd",
            forecast_horizon=horizon,
            backend=backend,
        )
        model.fit(fit_df, feature_cols=feat_cols)

        global_qhat, stratum_qhat, stratum_counts = apply_conformal_interval_expansion_stratified(
            model=model,
            calib_df=calib_df,
            feature_cols=feat_cols,
            alpha=alpha,
            risk_threshold_pct=cqr_risk_strata_threshold_pct,
        )

        if global_qhat <= 0.0:
            global_qhat = apply_conformal_interval_expansion(
                model=model,
                fit_df=fit_df,
                calib_df=calib_df,
                feature_cols=feat_cols,
                alpha=alpha,
            )

        preds = model.predict(block)
        block_gap = block["predicted_gap_pct"] if "predicted_gap_pct" in block.columns else pd.Series(0.0, index=block.index)
        block_stratum = _risk_stratum_labels(block_gap, cqr_risk_strata_threshold_pct)
        row_qhat = np.array([stratum_qhat.get(str(s), global_qhat) for s in block_stratum], dtype=float)

        preds["q10"] = (preds["q10"] - row_qhat).clip(lower=0.0)
        preds["q90"] = preds["q90"] + row_qhat
        preds["q50"] = np.maximum(preds["q50"], preds["q10"])
        preds["q90"] = np.maximum(preds["q90"], preds["q50"])

        merged = block[["date", "hs_code", "forecast_target", "lag12_value", "ma12_value", "month"]].merge(
            preds,
            on=["date", "hs_code"],
            how="inner",
        )
        merged = merged.rename(
            columns={
                "date": "origin_date",
                "forecast_target": "actual_supply",
                "lag12_value": "demand_proxy",
                "ma12_value": "demand_proxy_ma12",
            }
        )
        merged["forecast_date"] = pd.to_datetime(merged["origin_date"]) + pd.DateOffset(months=horizon)
        merged["horizon_months"] = horizon
        merged["cqr_qhat"] = row_qhat
        merged["cqr_qhat_global"] = global_qhat
        yearly_parts.append(merged)

        diagnostics[str(year)] = {
            "train_rows": int(len(fit_df)),
            "calibration_rows": int(len(calib_df)),
            "cqr_qhat": float(global_qhat),
            "cqr_qhat_by_stratum": {k: float(v) for k, v in stratum_qhat.items()},
            "calibration_rows_by_stratum": {k: int(v) for k, v in stratum_counts.items()},
        }
        last_model = model

    if not yearly_parts:
        return pd.DataFrame(), proto, diagnostics

    return pd.concat(yearly_parts, ignore_index=True), last_model if last_model is not None else proto, diagnostics


def apply_demand_proxy(frame: pd.DataFrame, seasonality: pd.DataFrame, mode: str) -> pd.DataFrame:
    result = frame.copy()
    result["forecast_month"] = pd.to_datetime(result["forecast_date"]).dt.month
    result = result.merge(
        seasonality.rename(columns={"month": "forecast_month"}),
        on=["hs_code", "forecast_month"],
        how="left",
    )
    result["seasonal_index"] = result["seasonal_index"].fillna(1.0)
    result["demand_proxy_base"] = result["demand_proxy"].fillna(result["demand_proxy_ma12"]).fillna(result["q50"])
    result["demand_proxy_seasonal"] = result["demand_proxy_base"] * result["seasonal_index"]

    if mode == "seasonal":
        result["demand_proxy_selected"] = result["demand_proxy_seasonal"]
    elif mode == "ma12":
        result["demand_proxy_selected"] = result["demand_proxy_base"]
    else:
        raise ValueError("mode must be 'seasonal' or 'ma12'")
    return result


def _align_to_base(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_date", "forecast_date", "hs_code", "horizon_months"]
    aligned = base[keys].merge(other, on=keys, how="left")
    return aligned


def enrich_tracks(
    baseline_predictions: pd.DataFrame,
    risk_model_predictions: pd.DataFrame,
    module1: pd.DataFrame,
    gate_threshold_pct: float,
    demand_mode: str,
    seasonality: pd.DataFrame,
    elasticity_by_horizon: Dict[int, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = attach_module1_risk(baseline_predictions, module1)
    base = apply_demand_proxy(base, seasonality=seasonality, mode=demand_mode)

    baseline = base.assign(risk_q10=base["q10"], risk_q50=base["q50"], risk_q90=base["q90"])
    baseline = add_shortage_metrics(baseline, demand_col="demand_proxy_selected")
    baseline["track"] = "baseline"
    baseline["trigger_active"] = baseline["predicted_gap_pct"] >= gate_threshold_pct

    legacy = apply_risk_gating(base, threshold_pct=gate_threshold_pct, mode="none")
    legacy = apply_gap_elasticity(legacy, elasticity_by_horizon=elasticity_by_horizon)
    legacy = adjust_supply_quantiles(legacy)
    legacy = add_shortage_metrics(legacy, demand_col="demand_proxy_selected")
    legacy["track"] = "ungated_stress"

    risk_track = attach_module1_risk(risk_model_predictions, module1)
    risk_track = apply_demand_proxy(risk_track, seasonality=seasonality, mode=demand_mode)
    risk_track = _align_to_base(baseline, risk_track)

    # Keep baseline trigger segmentation for fair comparison.
    risk_track["trigger_active"] = baseline["trigger_active"].values
    risk_track["predicted_gap_pct"] = baseline["predicted_gap_pct"].values
    risk_track["lower_gap_pct"] = baseline["lower_gap_pct"].values
    risk_track["upper_gap_pct"] = baseline["upper_gap_pct"].values

    # Legacy breach line is retained as a warning threshold for procurement alerts.
    breach = apply_risk_gating(baseline, threshold_pct=gate_threshold_pct, mode="hard_gate")
    breach = apply_gap_elasticity(breach, elasticity_by_horizon=elasticity_by_horizon)
    breach = adjust_supply_quantiles(breach)

    risk_track["risk_q10"] = risk_track["q10"]
    risk_track["risk_q50"] = risk_track["q50"]
    risk_track["risk_q90"] = risk_track["q90"]
    risk_track["breach_line_q10"] = breach["risk_q10"].values

    risk_track = apply_module1_rolling_bias_correction(
        risk_track,
        pred_cols=("risk_q10", "risk_q50", "risk_q90"),
        window=MODULE1_BIAS_ROLLING_WINDOW,
        clip_min=MODULE1_BIAS_CLIP_MIN,
        clip_max=MODULE1_BIAS_CLIP_MAX,
    )

    risk_track = add_shortage_metrics(risk_track, demand_col="demand_proxy_selected")
    risk_track["track"] = "gated_stress"
    risk_track["effective_predicted_gap_pct"] = breach["effective_predicted_gap_pct"].values

    return baseline, legacy, risk_track


def evaluate_track(frame: pd.DataFrame, prediction_prefix: str) -> Dict[str, float]:
    pred_col = "q50" if prediction_prefix == "baseline" else "risk_q50"
    q10_col = "q10" if prediction_prefix == "baseline" else "risk_q10"
    q90_col = "q90" if prediction_prefix == "baseline" else "risk_q90"
    flag_col = "baseline_shortage_flag" if prediction_prefix == "baseline" else "risk_shortage_flag"

    y_true = frame["actual_supply"].astype(float)
    y_pred = frame[pred_col].astype(float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    wape = _safe_div(float(np.abs(y_true - y_pred).sum()), float(np.abs(y_true).sum()))
    coverage = float(((y_true >= frame[q10_col]) & (y_true <= frame[q90_col])).mean())

    precision_den = float(frame[flag_col].sum())
    recall_den = float(frame["actual_shortage_flag"].sum())
    tp = float((frame[flag_col] & frame["actual_shortage_flag"]).sum())

    return {
        "mae": mae,
        "wape": wape,
        "coverage_q10_q90": coverage,
        "pinball_q10": _pinball(y_true, frame[q10_col], 0.10),
        "pinball_q50": _pinball(y_true, frame[pred_col], 0.50),
        "pinball_q90": _pinball(y_true, frame[q90_col], 0.90),
        "shortage_precision": _safe_div(tp, precision_den),
        "shortage_recall": _safe_div(tp, recall_den),
    }


def build_tradeoff_matrix(baseline: pd.DataFrame, ungated: pd.DataFrame, gated: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon in sorted(baseline["horizon_months"].unique()):
        b = baseline[baseline["horizon_months"].eq(horizon)]
        u = ungated[ungated["horizon_months"].eq(horizon)]
        g = gated[gated["horizon_months"].eq(horizon)]

        segments = {
            "all": b.index,
            "low_risk": b.index[~b["trigger_active"]],
            "trigger_active": b.index[b["trigger_active"]],
        }
        for segment, idx in segments.items():
            if len(idx) == 0:
                continue
            b_seg = b.loc[idx]
            u_seg = u.loc[idx]
            g_seg = g.loc[idx]

            metrics = {
                "baseline": evaluate_track(b_seg, "baseline"),
                "ungated_stress": evaluate_track(u_seg, "risk"),
                "gated_stress": evaluate_track(g_seg, "risk"),
            }
            for track_name, m in metrics.items():
                rows.append(
                    {
                        "horizon_months": int(horizon),
                        "segment": segment,
                        "track": track_name,
                        **m,
                    }
                )

    matrix = pd.DataFrame(rows)
    pivot = matrix.pivot_table(
        index=["horizon_months", "segment"],
        columns="track",
        values=["mae", "shortage_recall"],
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    if "mae_gated_stress" in pivot.columns and "mae_baseline" in pivot.columns:
        pivot["gated_mae_delta_vs_baseline"] = pivot["mae_gated_stress"] - pivot["mae_baseline"]
    if "shortage_recall_gated_stress" in pivot.columns and "shortage_recall_baseline" in pivot.columns:
        pivot["gated_recall_delta_vs_baseline"] = (
            pivot["shortage_recall_gated_stress"] - pivot["shortage_recall_baseline"]
        )
    return matrix.merge(pivot, on=["horizon_months", "segment"], how="left")


def inventory_summary(inventory_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (track, horizon, months), part in inventory_frame.groupby(["track", "horizon_months", "initial_inventory_months"]):
        erosion_pct = np.where(
            part["initial_inventory_months"] > 0,
            (1.0 - (part["risk_ending_inventory_months"] / part["initial_inventory_months"])).clip(0.0, 1.0) * 100.0,
            np.nan,
        )
        rows.append(
            {
                "track": track,
                "horizon_months": int(horizon),
                "initial_inventory_months": float(months),
                "risk_stockout_rate": float(part["risk_stockout_flag"].mean()),
                "risk_median_stockout_step": float(part["risk_stockout_step"].dropna().median())
                if part["risk_stockout_step"].notna().any()
                else float("nan"),
                "risk_mean_cumulative_shortfall_usd": float(part["risk_cumulative_shortfall_usd"].mean()),
                "risk_mean_days_of_supply_remaining": float((part["risk_ending_inventory_months"] * 30.0).mean()),
                "risk_mean_buffer_erosion_pct": float(np.nanmean(erosion_pct)),
            }
        )
    return pd.DataFrame(rows)


def build_predictions_output(baseline: pd.DataFrame, ungated: pd.DataFrame, gated: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_date", "forecast_date", "hs_code", "horizon_months"]

    out = baseline[keys + [
        "actual_supply",
        "q10",
        "q50",
        "q90",
        "demand_proxy_base",
        "demand_proxy_seasonal",
        "demand_proxy_selected",
        "predicted_gap_pct",
        "lower_gap_pct",
        "upper_gap_pct",
        "trigger_active",
        "actual_shortage_flag",
        "baseline_shortage_flag",
    ]].rename(columns={"q10": "baseline_q10", "q50": "baseline_q50", "q90": "baseline_q90"})

    out = out.merge(
        ungated[keys + ["risk_q10", "risk_q50", "risk_q90", "risk_shortage_flag"]].rename(
            columns={
                "risk_q10": "ungated_q10",
                "risk_q50": "ungated_q50",
                "risk_q90": "ungated_q90",
                "risk_shortage_flag": "ungated_shortage_flag",
            }
        ),
        on=keys,
        how="left",
    )

    out = out.merge(
        gated[keys + ["risk_q10", "risk_q50", "risk_q90", "risk_shortage_flag", "effective_predicted_gap_pct", "breach_line_q10"]].rename(
            columns={
                "risk_q10": "gated_q10",
                "risk_q50": "gated_q50",
                "risk_q90": "gated_q90",
                "risk_shortage_flag": "gated_shortage_flag",
            }
        ),
        on=keys,
        how="left",
    )

    return out.sort_values(["forecast_date", "hs_code", "horizon_months"]).reset_index(drop=True)


def plot_supply(predictions: pd.DataFrame, output_path: Path) -> None:
    plot_df = predictions.groupby(["forecast_date", "horizon_months"], as_index=False).agg(
        {
            "actual_supply": "sum",
            "baseline_q10": "sum",
            "baseline_q50": "sum",
            "baseline_q90": "sum",
            "ungated_q50": "sum",
            "gated_q10": "sum",
            "gated_q50": "sum",
            "breach_line_q10": "sum",
        }
    )

    horizons = sorted(plot_df["horizon_months"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.4 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        part = plot_df[plot_df["horizon_months"].eq(h)]
        ax.fill_between(part["forecast_date"], part["baseline_q10"], part["baseline_q90"], color="#2a9d8f", alpha=0.12, label="Baseline q10-q90")
        ax.plot(part["forecast_date"], part["actual_supply"], color="#1d3557", lw=2.2, label="Actual")
        ax.plot(part["forecast_date"], part["baseline_q50"], color="#2a9d8f", lw=1.8, label="Baseline q50")
        ax.plot(part["forecast_date"], part["ungated_q50"], color="#e76f51", lw=1.5, alpha=0.75, label="Legacy multiplicative q50")
        ax.plot(part["forecast_date"], part["gated_q50"], color="#e63946", lw=2.0, label="Risk-feature q50")
        ax.plot(part["forecast_date"], part["gated_q10"], color="#e63946", lw=1.2, linestyle="--", label="Risk-feature q10")
        ax.plot(part["forecast_date"], part["breach_line_q10"], color="#6d597a", lw=1.1, linestyle=":", label="Breach line q10")
        ax.set_title(f"Horizon {h} month")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Forecast date")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tradeoffs(tradeoff: pd.DataFrame, inventory_stats: pd.DataFrame, output_path: Path) -> None:
    all_seg = tradeoff[tradeoff["segment"].eq("all")]
    mae = all_seg.pivot_table(index="horizon_months", columns="track", values="mae")
    rec = all_seg.pivot_table(index="horizon_months", columns="track", values="shortage_recall")

    inv = inventory_stats[
        inventory_stats["track"].eq("gated_stress") & inventory_stats["initial_inventory_months"].isin([1.0, 2.0, 3.0])
    ]
    erosion_plot = inv.pivot_table(index="horizon_months", columns="initial_inventory_months", values="risk_mean_buffer_erosion_pct")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    mae.plot(kind="bar", ax=axes[0], color=["#2a9d8f", "#e76f51", "#e63946"])
    axes[0].set_title("MAE by horizon")
    axes[0].set_xlabel("Horizon months")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.2)

    rec.plot(kind="bar", ax=axes[1], color=["#2a9d8f", "#e76f51", "#e63946"])
    axes[1].set_title("Shortage recall by horizon")
    axes[1].set_xlabel("Horizon months")
    axes[1].set_ylabel("Recall")
    axes[1].grid(axis="y", alpha=0.2)

    erosion_plot.plot(kind="bar", ax=axes[2], colormap="OrRd")
    axes[2].set_title("Risk-feature buffer erosion % by buffer")
    axes[2].set_xlabel("Horizon months")
    axes[2].set_ylabel("Buffer erosion %")
    axes[2].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def select_threshold(
    grid_df: pd.DataFrame,
    default_threshold: float,
    min_coverage_floor: float,
    min_recall_floor: float,
    mae_objective_weights: Dict[int, float],
) -> Tuple[float, str]:
    ranked = grid_df.copy()
    weighted_col = "weighted_mae_delta_pct"

    def _weighted_mae_row(row: pd.Series) -> float:
        score = 0.0
        total = 0.0
        for horizon, weight in mae_objective_weights.items():
            col = f"mae_delta_pct_h{horizon}"
            value = row.get(col)
            if pd.notna(value):
                score += float(weight) * float(value)
                total += float(weight)
        return float(score / total) if total > 0 else float(row.get("mean_mae_delta_pct", np.nan))

    ranked[weighted_col] = ranked.apply(_weighted_mae_row, axis=1)
    feasible = ranked[
        (ranked["mean_coverage"] >= min_coverage_floor)
        & (ranked["mean_recall"] >= min_recall_floor)
    ].copy()

    if not feasible.empty:
        feasible = feasible.sort_values(
            [weighted_col, "mean_coverage", "mean_recall", "threshold_pct"],
            ascending=[True, False, False, True],
        )
        return float(feasible.iloc[0]["threshold_pct"]), "feasible_multi_objective"

    ranked["coverage_shortfall"] = (min_coverage_floor - ranked["mean_coverage"]).clip(lower=0.0)
    ranked["recall_shortfall"] = (min_recall_floor - ranked["mean_recall"]).clip(lower=0.0)
    ranked["multi_objective_penalty"] = (
        ranked["coverage_shortfall"] * 4.0
        + ranked["recall_shortfall"] * 2.0
        + ranked[weighted_col].clip(lower=0.0)
    )
    ranked = ranked.sort_values(["multi_objective_penalty", weighted_col, "threshold_pct"])
    if not ranked.empty:
        return float(ranked.iloc[0]["threshold_pct"]), "best_effort_multi_objective"
    return default_threshold, "default_threshold"


def run(backend: str = "sklearn", demand_mode: str = DEMAND_MODE) -> Dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)
    module1 = pd.read_csv(MODULE1_PATH)

    train_panel_raw, test_panel_raw = build_monthly_hs_panel(train_raw, test_raw)
    train_panel_macro, test_panel_macro = add_macro_cycle_features(train_panel_raw, test_panel_raw)

    # Attach Module 1 risk context to both tracks for stratified CQR segmentation.
    train_panel_macro = attach_module1_feature_inputs(train_panel_macro, module1)
    test_panel_macro = attach_module1_feature_inputs(test_panel_macro, module1)

    train_panel_risk = train_panel_macro.copy()
    test_panel_risk = test_panel_macro.copy()

    base_excluded = {
        "predicted_gap_pct",
        "lower_gap_pct",
        "upper_gap_pct",
        "risk_gap_spread",
        "risk_macro_interaction",
        "hs_value_share_lag12",
        "risk_gap_hs_exposure",
        "risk_gap_hhi_interaction",
    }
    feat_cols_base = [c for c in feature_columns(train_panel_macro) if c not in base_excluded]
    feat_cols_risk = feature_columns(train_panel_risk)
    seasonality = seasonal_index_from_train(train_panel_macro)

    baseline_parts = []
    risk_parts = []
    cqr_diagnostics = {"baseline": {}, "risk_feature": {}}

    for horizon in HORIZONS:
        target_coverage = float(TARGET_COVERAGE_BY_HORIZON.get(horizon, 0.80))
        pred_base, model_base, diag_base = predict_with_actual_expanding(
            train_panel=train_panel_macro,
            test_panel=test_panel_macro,
            feat_cols=feat_cols_base,
            horizon=horizon,
            backend=backend,
            target_coverage=target_coverage,
            cqr_risk_strata_threshold_pct=CQR_RISK_STRATA_THRESHOLD_PCT,
        )
        pred_risk, model_risk, diag_risk = predict_with_actual_expanding(
            train_panel=train_panel_risk,
            test_panel=test_panel_risk,
            feat_cols=feat_cols_risk,
            horizon=horizon,
            backend=backend,
            target_coverage=target_coverage,
            cqr_risk_strata_threshold_pct=CQR_RISK_STRATA_THRESHOLD_PCT,
        )

        baseline_parts.append(pred_base)
        risk_parts.append(pred_risk)
        cqr_diagnostics["baseline"][str(horizon)] = diag_base
        cqr_diagnostics["risk_feature"][str(horizon)] = diag_risk

        model_base.save(str(MODEL_DIR / f"quantile_forecaster_baseline_h{horizon}.joblib"))
        model_risk.save(str(MODEL_DIR / f"quantile_forecaster_risk_feature_h{horizon}.joblib"))

    baseline_predictions = pd.concat(baseline_parts, ignore_index=True)
    risk_feature_predictions = pd.concat(risk_parts, ignore_index=True)

    grid_rows = []
    cached_results = {}
    for threshold in GATE_THRESHOLD_GRID:
        baseline, ungated, gated = enrich_tracks(
            baseline_predictions=baseline_predictions,
            risk_model_predictions=risk_feature_predictions,
            module1=module1,
            gate_threshold_pct=threshold,
            demand_mode=demand_mode,
            seasonality=seasonality,
            elasticity_by_horizon=ELASTICITY_BY_HORIZON,
        )
        tradeoff = build_tradeoff_matrix(baseline, ungated, gated)
        all_seg = tradeoff[(tradeoff["segment"] == "all") & (tradeoff["track"] == "gated_stress")].copy()
        all_seg["mae_delta_pct"] = all_seg["gated_mae_delta_vs_baseline"] / all_seg["mae_baseline"].replace(0.0, np.nan)

        trigger_rate = float(gated["trigger_active"].mean())
        mean_coverage = float(all_seg["coverage_q10_q90"].mean())
        mean_recall = float(all_seg["shortage_recall"].mean())
        mae_delta_by_h = {
            int(h): float(v)
            for h, v in all_seg.set_index("horizon_months")["mae_delta_pct"].to_dict().items()
        }
        grid_rows.append(
            {
                "threshold_pct": threshold,
                "trigger_rate": trigger_rate,
                "mean_mae_delta_pct": float(all_seg["mae_delta_pct"].mean()),
                "mean_recall_delta": float(all_seg["gated_recall_delta_vs_baseline"].mean()),
                "mean_coverage": mean_coverage,
                "mean_recall": mean_recall,
                **{f"mae_delta_pct_h{h}": mae_delta_by_h.get(h, np.nan) for h in HORIZONS},
            }
        )
        cached_results[threshold] = (baseline, ungated, gated, tradeoff)

    grid_df = pd.DataFrame(grid_rows).sort_values("threshold_pct")
    chosen_threshold, threshold_selection_mode = select_threshold(
        grid_df,
        default_threshold=DEFAULT_THRESHOLD,
        min_coverage_floor=MIN_COVERAGE_FLOOR,
        min_recall_floor=MIN_RECALL_FLOOR,
        mae_objective_weights=MAE_OBJECTIVE_WEIGHTS,
    )

    baseline, ungated, gated, tradeoff = cached_results[chosen_threshold]

    inventory_frames = []
    for track_name, track_frame in (("baseline", baseline), ("ungated_stress", ungated), ("gated_stress", gated)):
        inv = build_inventory_scenarios(track_frame, initial_months=(1.0, 2.0, 3.0))
        inv["track"] = track_name
        inventory_frames.append(inv)
    inventory_all = pd.concat(inventory_frames, ignore_index=True)

    inventory_stats = inventory_summary(inventory_all)
    predictions_out = build_predictions_output(baseline, ungated, gated)

    predictions_out.to_csv(OUTPUT_DIR / "module3_integrated_predictions.csv", index=False)
    inventory_all.to_csv(OUTPUT_DIR / "module3_inventory_scenarios.csv", index=False)
    tradeoff.to_csv(OUTPUT_DIR / "module3_tradeoff_matrix.csv", index=False)
    grid_df.to_csv(OUTPUT_DIR / "module3_threshold_grid.csv", index=False)

    metrics = {
        "backend": backend,
        "demand_mode": demand_mode,
        "selected_gate_threshold_pct": chosen_threshold,
        "threshold_selection_mode": threshold_selection_mode,
        "threshold_objective": {
            "min_coverage_floor": MIN_COVERAGE_FLOOR,
            "min_recall_floor": MIN_RECALL_FLOOR,
            "mae_horizon_weights": {str(k): float(v) for k, v in MAE_OBJECTIVE_WEIGHTS.items()},
        },
        "target_interval_coverage_by_horizon": {str(k): float(v) for k, v in TARGET_COVERAGE_BY_HORIZON.items()},
        "target_interval_coverage": float(np.mean(list(TARGET_COVERAGE_BY_HORIZON.values()))),
        "cqr_risk_strata_threshold_pct": CQR_RISK_STRATA_THRESHOLD_PCT,
        "threshold_grid": grid_df.to_dict(orient="records"),
        "elasticity_by_horizon": {str(k): float(v) for k, v in ELASTICITY_BY_HORIZON.items()},
        "cqr_diagnostics": cqr_diagnostics,
        "feature_sets": {
            "baseline": feat_cols_base,
            "risk_feature": feat_cols_risk,
            "module1_inputs": [
                "predicted_gap_pct",
                "lower_gap_pct",
                "upper_gap_pct",
                "risk_gap_spread",
                "risk_macro_interaction",
                "hs_value_share_lag12",
                "risk_gap_hs_exposure",
                "risk_gap_hhi_interaction",
            ],
            "macro_cycle_inputs": [
                "macro_total_lag1",
                "macro_total_lag3",
                "macro_total_lag12",
                "macro_total_ma6",
                "macro_mom",
                "macro_yoy",
                "macro_deviation_6m",
            ],
        },
        "track_metrics_all": {
            "baseline": {
                str(h): evaluate_track(baseline[baseline["horizon_months"].eq(h)], "baseline") for h in HORIZONS
            },
            "ungated_stress": {
                str(h): evaluate_track(ungated[ungated["horizon_months"].eq(h)], "risk") for h in HORIZONS
            },
            "gated_stress": {
                str(h): evaluate_track(gated[gated["horizon_months"].eq(h)], "risk") for h in HORIZONS
            },
        },
        "inventory_summary": inventory_stats.to_dict(orient="records"),
        "st_isp_staging": {
            "S_monitor": "Module 1 risk signal aligned to origin quarter",
            "T_trigger": f"trigger_active when predicted_gap_pct >= {chosen_threshold}%",
            "I_simulate": "recursive monthly inventory balance with seasonal demand proxy",
            "SP_policy": "baseline (expanding+CQR) / legacy multiplicative / risk-feature fusion",
        },
        "module1_bias_correction": {
            "rolling_window_months": MODULE1_BIAS_ROLLING_WINDOW,
            "clip_min": MODULE1_BIAS_CLIP_MIN,
            "clip_max": MODULE1_BIAS_CLIP_MAX,
            "method": "horizon-wise rolling median(actual/pred), shifted by one period",
        },
    }

    with open(OUTPUT_DIR / "module3_integrated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / "module3_tradeoff_matrix.json", "w", encoding="utf-8") as f:
        json.dump(tradeoff.to_dict(orient="records"), f, indent=2)

    plot_supply(predictions_out, FIGURE_DIR / "module3_supply_forecast_integration.png")
    plot_tradeoffs(tradeoff, inventory_stats, FIGURE_DIR / "module3_shortage_inventory_impact.png")

    return metrics


if __name__ == "__main__":
    summary = run()
    print(
        json.dumps(
            {
                "backend": summary["backend"],
                "demand_mode": summary["demand_mode"],
                "selected_gate_threshold_pct": summary["selected_gate_threshold_pct"],
                "target_interval_coverage": summary["target_interval_coverage"],
            },
            indent=2,
        )
    )
