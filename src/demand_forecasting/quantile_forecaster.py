"""
Quantile Forecaster

Trains XGBoost models with ``reg:quantileerror`` objective to produce
**prediction intervals** (e.g., 10th / 50th / 90th percentiles) rather
than a single point estimate.

This gives downstream modules (shortage detection, risk scoring) a
principled measure of forecast uncertainty.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.DemandForecasting.QuantileForecaster")


class QuantileForecaster:
    """
    Multi-quantile XGBoost forecaster for supply/demand variables.

    For each requested quantile a separate XGBoost model is trained so
    that the resulting prediction intervals are coherent (q10 ≤ q50 ≤ q90).
    """

    DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        target_col: str = "value_usd",
        forecast_horizon: int = 1,
        xgb_params: Optional[Dict] = None,
        backend: str = "auto",
    ):
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.backend = backend
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self.models: Dict[float, object] = {}
        self.feature_cols: List[str] = []
        self._fitted = False

    def build_supervised_frame(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Attach the target observed exactly ``forecast_horizon`` months later."""
        groups = group_cols or [
            col for col in ("hs_code", "country") if col in df.columns
        ]
        if "date" not in df.columns or not groups:
            raise ValueError("Future-target alignment requires date and grouping columns")

        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        future = frame[groups + ["date", self.target_col]].copy()
        future["date"] = future["date"] - pd.DateOffset(months=self.forecast_horizon)
        future = future.rename(columns={self.target_col: "forecast_target"})
        return frame.merge(future, on=groups + ["date"], how="inner", validate="one_to_one")

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[float, Dict]:
        """
        Train one XGBoost quantile model per requested quantile.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target column.
        feature_cols : list[str], optional
            Feature column names.  If ``None``, all numeric columns
            except the target are used.

        Returns
        -------
        dict
            ``{quantile: {"n_estimators": …, "best_iteration": …}}``
        """
        df = self.build_supervised_frame(train_df).dropna(subset=["forecast_target"]).copy()
        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in {self.target_col, "forecast_target"}
            ]
        self.feature_cols = feature_cols
        df = df.dropna(subset=feature_cols)

        X = df[feature_cols].values
        y = df["forecast_target"].values

        # Log-transform target for better quantile estimation on skewed data
        y_log = np.log1p(np.maximum(y, 0))

        info: Dict[float, Dict] = {}
        for q in self.quantiles:
            logger.info("Training quantile=%.2f model …", q)
            model, backend = self._build_model(q)
            model.fit(X, y_log)
            self.models[q] = model
            info[q] = {"n_estimators": model.n_estimators, "backend": backend}

        self._fitted = True
        logger.info("Quantile forecaster fitted with %d quantiles", len(self.quantiles))
        return info

    def _build_model(self, quantile: float) -> Tuple[object, str]:
        """Build the requested quantile estimator, with a portable fallback."""
        if self.backend in {"auto", "xgboost"}:
            try:
                from xgboost import XGBRegressor

                model = XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=quantile,
                    tree_method="hist",
                    random_state=42,
                    **self.xgb_params,
                )
                return model, "xgboost"
            except Exception as exc:
                if self.backend == "xgboost":
                    raise
                logger.warning("XGBoost unavailable; using sklearn quantile boosting: %s", exc)

        from sklearn.ensemble import GradientBoostingRegressor

        params = {
            "n_estimators": self.xgb_params.get("n_estimators", 300),
            "max_depth": self.xgb_params.get("max_depth", 3),
            "learning_rate": self.xgb_params.get("learning_rate", 0.05),
            "subsample": self.xgb_params.get("subsample", 0.8),
            "random_state": 42,
        }
        return GradientBoostingRegressor(loss="quantile", alpha=quantile, **params), "sklearn"

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles for new data.

        Returns a DataFrame with one column per quantile,
        plus a ``prediction_interval_width`` column.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[self.feature_cols].values
        result = df[["date", "hs_code"]].copy() if {"date", "hs_code"}.issubset(df.columns) else df.iloc[:, :0].copy()

        for q in sorted(self.quantiles):
            preds_log = self.models[q].predict(X)
            preds = np.expm1(preds_log)  # inverse log1p
            col_name = f"q{int(q * 100):02d}"
            result[col_name] = preds

        # Enforce monotonicity: q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
        q_cols = [f"q{int(q * 100):02d}" for q in sorted(self.quantiles)]
        for i in range(1, len(q_cols)):
            result[q_cols[i]] = np.maximum(
                result[q_cols[i]].values, result[q_cols[i - 1]].values
            )

        # Interval width (e.g., q90 - q10 if both exist)
        if "q90" in result.columns and "q10" in result.columns:
            result["prediction_interval_width"] = result["q90"] - result["q10"]

        return result

    def forecast_with_intervals(
        self,
        df: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: predict with custom quantile set.

        The ``quantiles`` param only selects which columns to return;
        the model must have been trained on those quantiles already.
        """
        preds = self.predict(df)
        if quantiles is not None:
            cols_to_keep = [
                c for c in preds.columns
                if not c.startswith("q")
                or float(f"0.{c[1:]}") in quantiles
            ]
            # Always keep non-q columns
            cols_to_keep += [c for c in preds.columns if not c.startswith("q")]
            preds = preds[list(dict.fromkeys(cols_to_keep))]
        return preds

    # ------------------------------------------------------------------ #
    #  Feature importance
    # ------------------------------------------------------------------ #

    def feature_importance(self, quantile: float = 0.50) -> pd.DataFrame:
        """Return feature importance for the median model."""
        model = self.models.get(quantile)
        if model is None:
            raise ValueError(f"No model for quantile {quantile}")
        imp = model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str = "models/trained/quantile_forecaster.pkl"):
        """Save all quantile models."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "quantiles": self.quantiles,
            "target_col": self.target_col,
            "forecast_horizon": self.forecast_horizon,
            "backend": self.backend,
            "xgb_params": self.xgb_params,
            "feature_cols": self.feature_cols,
            "models": self.models,
        }
        with open(p, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved quantile forecaster to %s", p)

    def load(self, path: str = "models/trained/quantile_forecaster.pkl"):
        """Load quantile models from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.quantiles = state["quantiles"]
        self.target_col = state["target_col"]
        self.forecast_horizon = state.get("forecast_horizon", 1)
        self.backend = state.get("backend", "auto")
        self.xgb_params = state.get("xgb_params", self.xgb_params)
        self.feature_cols = state["feature_cols"]
        self.models = state["models"]
        self._fitted = True
        logger.info("Loaded quantile forecaster with %d models", len(self.models))
