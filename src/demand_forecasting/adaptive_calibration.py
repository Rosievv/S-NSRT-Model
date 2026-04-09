"""
Adaptive Calibration

Evaluates and corrects quantile forecast calibration over time.

A well-calibrated q90 forecast should contain 90 % of actual outcomes.
When the model is miscalibrated (e.g., q90 only covers 75 % of outcomes),
this module detects the error and applies a correction.

This is the **adaptive feedback** component referenced in the framework
design: it enables the forecasting system to self-correct as new data
arrive, without requiring full retraining.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.DemandForecasting.AdaptiveCalibration")


class AdaptiveCalibrator:
    """
    Evaluate quantile forecast calibration and apply corrections.
    """

    def __init__(self, quantiles: Optional[List[float]] = None):
        """
        Parameters
        ----------
        quantiles : list[float]
            The quantile levels being evaluated (must match those produced
            by the QuantileForecaster).
        """
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]
        self.calibration_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Calibration error
    # ------------------------------------------------------------------ #

    def compute_calibration_error(
        self,
        actuals: pd.Series,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each quantile q, compute the **observed coverage** — the
        fraction of actuals that fall below the q-th percentile prediction.

        A perfectly calibrated q90 should have observed coverage ≈ 0.90.

        Parameters
        ----------
        actuals : pd.Series
            Ground-truth values aligned to predictions.
        predictions : pd.DataFrame
            Quantile predictions with columns ``q10, q25, …``.

        Returns
        -------
        pd.DataFrame
            One row per quantile with ``target_coverage``,
            ``observed_coverage``, ``calibration_error``.
        """
        results = []
        for q in self.quantiles:
            col = f"q{int(q * 100):02d}"
            if col not in predictions.columns:
                continue
            below = (actuals.values <= predictions[col].values).mean()
            results.append({
                "quantile": q,
                "target_coverage": q,
                "observed_coverage": round(float(below), 4),
                "calibration_error": round(float(below - q), 4),
            })

        df = pd.DataFrame(results)
        avg_error = df["calibration_error"].abs().mean()
        logger.info(
            "Average absolute calibration error: %.4f (%s)",
            avg_error,
            "well-calibrated" if avg_error < 0.05 else "needs correction",
        )

        # Store in history
        self.calibration_history.append({
            "avg_abs_error": float(avg_error),
            "details": results,
        })

        return df

    # ------------------------------------------------------------------ #
    #  Recalibration
    # ------------------------------------------------------------------ #

    def recalibrate_predictions(
        self,
        predictions: pd.DataFrame,
        calibration_df: pd.DataFrame,
        method: str = "shift",
    ) -> pd.DataFrame:
        """
        Adjust predictions to correct systematic calibration errors.

        Parameters
        ----------
        predictions : pd.DataFrame
            Original quantile predictions.
        calibration_df : pd.DataFrame
            Output of ``compute_calibration_error()``.
        method : str
            ``"shift"`` — multiplicative correction based on error sign.

        Returns
        -------
        pd.DataFrame
            Adjusted predictions.
        """
        adjusted = predictions.copy()

        for _, row in calibration_df.iterrows():
            col = f"q{int(row['quantile'] * 100):02d}"
            if col not in adjusted.columns:
                continue
            error = row["calibration_error"]
            if abs(error) < 0.02:
                # Already well-calibrated
                continue

            # Shift: if observed coverage is too low (error < 0),
            # the predictions are too tight → widen by scaling up
            if error < 0:
                # Under-coverage: increase this quantile prediction
                scale = 1.0 + abs(error)
            else:
                # Over-coverage: decrease this quantile prediction
                scale = 1.0 - min(abs(error), 0.3)

            adjusted[col] = adjusted[col] * scale
            logger.info(
                "Recalibrated %s: error=%.3f → scale=%.3f", col, error, scale
            )

        # Re-enforce monotonicity
        q_cols = sorted(
            [c for c in adjusted.columns if c.startswith("q") and len(c) == 3],
        )
        for i in range(1, len(q_cols)):
            adjusted[q_cols[i]] = np.maximum(
                adjusted[q_cols[i]].values,
                adjusted[q_cols[i - 1]].values,
            )

        return adjusted

    # ------------------------------------------------------------------ #
    #  Incremental update recommendation
    # ------------------------------------------------------------------ #

    def recommend_update(
        self,
        calibration_df: pd.DataFrame,
        error_threshold: float = 0.10,
    ) -> Dict:
        """
        Based on calibration results, recommend whether a full model
        retrain is needed or if recalibration alone suffices.

        Returns
        -------
        dict
            ``{"action": "retrain" | "recalibrate" | "none",
               "reason": str, "avg_error": float}``
        """
        avg_err = calibration_df["calibration_error"].abs().mean()
        max_err = calibration_df["calibration_error"].abs().max()

        if max_err > 0.20:
            return {
                "action": "retrain",
                "reason": f"Max quantile error ({max_err:.3f}) exceeds 0.20; "
                          "model may be fundamentally stale.",
                "avg_error": float(avg_err),
            }
        elif avg_err > error_threshold:
            return {
                "action": "recalibrate",
                "reason": f"Avg error ({avg_err:.3f}) exceeds threshold "
                          f"({error_threshold}); apply recalibration.",
                "avg_error": float(avg_err),
            }
        else:
            return {
                "action": "none",
                "reason": "Calibration is within acceptable tolerance.",
                "avg_error": float(avg_err),
            }

    # ------------------------------------------------------------------ #
    #  History
    # ------------------------------------------------------------------ #

    def get_calibration_trend(self) -> pd.DataFrame:
        """Return historical calibration errors as a DataFrame."""
        if not self.calibration_history:
            return pd.DataFrame()
        return pd.DataFrame([
            {"evaluation_idx": i, "avg_abs_error": h["avg_abs_error"]}
            for i, h in enumerate(self.calibration_history)
        ])
