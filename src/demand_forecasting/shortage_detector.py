"""
Shortage Detector

Flags HS codes / time periods where supply is predicted to fall below
a demand baseline, using the uncertainty bands from the
:class:`QuantileForecaster`.

Key concepts
------------
* **Supply-demand ratio** = predicted supply (q50) / demand baseline
* **Shortage risk score** = composite of gap size + trend direction +
  forecast uncertainty (interval width)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.DemandForecasting.ShortageDetector")


class ShortageDetector:
    """
    Identify potential shortage conditions from quantile forecast output.
    """

    def __init__(
        self,
        shortage_threshold: float = 0.80,
        critical_threshold: float = 0.60,
    ):
        """
        Parameters
        ----------
        shortage_threshold : float
            Flag when predicted supply falls below this fraction of
            the demand baseline.
        critical_threshold : float
            Severe shortage level.
        """
        self.shortage_threshold = shortage_threshold
        self.critical_threshold = critical_threshold

    # ------------------------------------------------------------------ #
    #  Demand baseline
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_demand_baseline(
        historical_df: pd.DataFrame,
        target_col: str = "value_usd",
        group_col: str = "hs_code",
        method: str = "rolling_mean",
        window: int = 12,
    ) -> pd.DataFrame:
        """
        Derive a demand baseline per HS code from historical data.

        Parameters
        ----------
        method : str
            ``"rolling_mean"`` or ``"median"``.

        Returns
        -------
        pd.DataFrame
            With ``hs_code`` and ``demand_baseline``.
        """
        if method == "rolling_mean":
            baseline = (
                historical_df.groupby(group_col)[target_col]
                .apply(lambda s: s.rolling(window, min_periods=1).mean().iloc[-1])
                .reset_index()
            )
        else:
            baseline = (
                historical_df.groupby(group_col)[target_col]
                .median()
                .reset_index()
            )
        baseline.columns = [group_col, "demand_baseline"]
        return baseline

    # ------------------------------------------------------------------ #
    #  Supply-demand ratio
    # ------------------------------------------------------------------ #

    def compute_supply_demand_ratio(
        self,
        forecast_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        median_col: str = "q50",
        group_col: str = "hs_code",
    ) -> pd.DataFrame:
        """
        Compute the ratio of predicted supply to demand baseline.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Output of ``QuantileForecaster.predict()`` — must include
            ``q50`` (median forecast) and optionally ``q10``, ``q90``.
        baseline_df : pd.DataFrame
            From ``compute_demand_baseline()``.

        Returns
        -------
        pd.DataFrame
            With ``supply_demand_ratio`` and ``shortage_flag``.
        """
        merged = forecast_df.merge(baseline_df, on=group_col, how="left")
        merged["supply_demand_ratio"] = np.where(
            merged["demand_baseline"] > 0,
            merged[median_col] / merged["demand_baseline"],
            np.nan,
        )
        merged["shortage_flag"] = merged["supply_demand_ratio"] < self.shortage_threshold
        merged["critical_flag"] = merged["supply_demand_ratio"] < self.critical_threshold
        return merged

    # ------------------------------------------------------------------ #
    #  Composite shortage-risk score
    # ------------------------------------------------------------------ #

    def shortage_severity_score(
        self,
        ratio_df: pd.DataFrame,
        interval_width_col: str = "prediction_interval_width",
    ) -> pd.DataFrame:
        """
        Compute a composite shortage severity score [0, 100]:

        score = gap_penalty + trend_penalty + uncertainty_penalty

        * **gap_penalty** (0-50): how far below the threshold the ratio is
        * **trend_penalty** (0-25): is the ratio declining?
        * **uncertainty_penalty** (0-25): wider prediction intervals → more risk
        """
        df = ratio_df.copy()

        # Gap penalty  (0–50)
        gap = np.clip(1 - df["supply_demand_ratio"].fillna(1), 0, 1)
        df["gap_penalty"] = gap * 50

        # Trend penalty  (0–25): compare current ratio to rolling avg
        if "supply_demand_ratio" in df.columns:
            df["ratio_change"] = df.groupby("hs_code")["supply_demand_ratio"].pct_change()
            df["trend_penalty"] = np.where(
                df["ratio_change"] < 0,
                np.clip(abs(df["ratio_change"]) * 50, 0, 25),
                0,
            )
        else:
            df["trend_penalty"] = 0

        # Uncertainty penalty  (0–25)
        if interval_width_col in df.columns:
            median_width = df[interval_width_col].median()
            if median_width > 0:
                df["uncertainty_penalty"] = np.clip(
                    df[interval_width_col] / median_width * 12.5, 0, 25
                )
            else:
                df["uncertainty_penalty"] = 0
        else:
            df["uncertainty_penalty"] = 0

        df["shortage_risk_score"] = (
            df["gap_penalty"] + df["trend_penalty"] + df["uncertainty_penalty"]
        ).clip(0, 100)

        df["risk_level"] = pd.cut(
            df["shortage_risk_score"],
            bins=[-1, 20, 40, 60, 80, 101],
            labels=["very_low", "low", "moderate", "high", "critical"],
        )

        return df

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def flag_shortage_risks(
        self,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        target_col: str = "value_usd",
        group_col: str = "hs_code",
    ) -> pd.DataFrame:
        """
        End-to-end pipeline: baseline → ratio → score → flags.
        """
        baseline = self.compute_demand_baseline(
            historical_df, target_col=target_col, group_col=group_col
        )
        ratio_df = self.compute_supply_demand_ratio(
            forecast_df, baseline, group_col=group_col
        )
        scored = self.shortage_severity_score(ratio_df)
        n_flagged = scored["shortage_flag"].sum()
        n_critical = scored["critical_flag"].sum()
        logger.info(
            "Shortage detection: %d flagged (%d critical) out of %d forecasts",
            n_flagged, n_critical, len(scored),
        )
        return scored
