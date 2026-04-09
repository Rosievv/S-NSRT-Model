"""
Cost Transmission Analyzer

Models how upstream cost shocks (commodity prices, freight, PPI)
propagate to downstream trade values with time lags.

Techniques
----------
* Granger causality tests — do lagged commodity prices help predict
  trade-value changes?
* VAR (Vector Autoregression) — estimate impulse-response functions
  showing how a 1-std-dev shock to upstream prices affects downstream.
* Variance decomposition — attribute trade-value changes to specific
  cost drivers (materials, freight, macro).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.CostMonitoring.CostTransmission")


class CostTransmissionAnalyzer:
    """
    Estimate the pass-through of upstream cost shocks to semiconductor
    import trade values.
    """

    def __init__(
        self,
        trade_monthly: pd.Series,
        cost_series: Dict[str, pd.Series],
        max_lag: int = 6,
    ):
        """
        Parameters
        ----------
        trade_monthly : pd.Series
            Monthly aggregate import value (date-indexed).
        cost_series : dict[str, pd.Series]
            Named cost/price series (date-indexed), e.g.::

                {"PPI_semiconductor": ppi_series, "freight_index": freight_series}
        max_lag : int
            Maximum lag (months) to test in Granger / VAR models.
        """
        self.trade = trade_monthly.copy()
        self.costs = {k: v.copy() for k, v in cost_series.items()}
        self.max_lag = max_lag

    # ------------------------------------------------------------------ #
    #  Granger causality
    # ------------------------------------------------------------------ #

    def granger_causality(self) -> pd.DataFrame:
        """
        Pairwise Granger causality tests: does each cost series
        Granger-cause the trade series?

        Returns
        -------
        pd.DataFrame
            With ``cost_driver, lag, f_statistic, p_value, significant``.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.warning("statsmodels not installed; skipping Granger tests")
            return pd.DataFrame()

        results = []
        for name, series in self.costs.items():
            combined = pd.DataFrame({
                "trade": self.trade,
                name: series,
            }).dropna()

            if len(combined) < self.max_lag + 10:
                logger.warning("Insufficient data for %s; skipping", name)
                continue

            try:
                gc = grangercausalitytests(
                    combined[["trade", name]],
                    maxlag=self.max_lag,
                    verbose=False,
                )
                for lag in range(1, self.max_lag + 1):
                    test_result = gc[lag][0]["ssr_ftest"]
                    results.append({
                        "cost_driver": name,
                        "lag": lag,
                        "f_statistic": round(float(test_result[0]), 4),
                        "p_value": round(float(test_result[1]), 6),
                        "significant": test_result[1] < 0.05,
                    })
            except Exception as e:
                logger.warning("Granger test failed for %s: %s", name, e)

        df = pd.DataFrame(results)
        if not df.empty:
            n_sig = df["significant"].sum()
            logger.info(
                "Granger causality: %d / %d lag-driver pairs significant",
                n_sig, len(df),
            )
        return df

    # ------------------------------------------------------------------ #
    #  Cost-pressure composite index
    # ------------------------------------------------------------------ #

    def compute_cost_pressure_index(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Weighted composite index of upstream cost pressures.

        Each component is z-score normalised and then averaged with
        the given weights.
        """
        if weights is None:
            weights = {k: 1.0 / len(self.costs) for k in self.costs}

        combined = pd.DataFrame()
        for name, series in self.costs.items():
            z = (series - series.mean()) / series.std() if series.std() > 0 else series * 0
            combined[name] = z

        combined = combined.dropna()
        if combined.empty:
            return pd.Series(dtype=float)

        total_w = sum(weights.get(c, 0) for c in combined.columns)
        if total_w == 0:
            total_w = 1.0
        index = sum(
            combined[c] * weights.get(c, 0) / total_w
            for c in combined.columns
        )
        index.name = "cost_pressure_index"
        logger.info(
            "Cost pressure index: mean=%.2f, std=%.2f, latest=%.2f",
            index.mean(), index.std(), index.iloc[-1] if len(index) > 0 else 0,
        )
        return index

    # ------------------------------------------------------------------ #
    #  Variance decomposition (simplified)
    # ------------------------------------------------------------------ #

    def decompose_cost_drivers(self) -> pd.DataFrame:
        """
        Estimate how much of the trade-value variance can be explained
        by each cost driver (using incremental R² from OLS).
        """
        from numpy.linalg import lstsq

        combined = pd.DataFrame({"trade": self.trade})
        for name, series in self.costs.items():
            combined[name] = series
        combined = combined.dropna()

        if len(combined) < 10:
            return pd.DataFrame()

        y = combined["trade"].values
        y_mean = y.mean()
        ss_total = ((y - y_mean) ** 2).sum()

        results = []
        for name in self.costs:
            X = combined[name].values.reshape(-1, 1)
            X = np.column_stack([np.ones(len(X)), X])
            coef, _, _, _ = lstsq(X, y, rcond=None)
            y_pred = X @ coef
            ss_res = ((y - y_pred) ** 2).sum()
            r2 = 1 - ss_res / ss_total if ss_total > 0 else 0
            results.append({
                "cost_driver": name,
                "r_squared": round(float(max(r2, 0)), 4),
                "contribution_pct": round(float(max(r2, 0) * 100), 2),
            })

        df = pd.DataFrame(results).sort_values("r_squared", ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Pass-through elasticity
    # ------------------------------------------------------------------ #

    def estimate_passthrough_elasticity(
        self, lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Estimate the price-pass-through elasticity for each cost driver
        at different lags via simple OLS on log-returns.

        Returns ``cost_driver, lag, elasticity, t_stat``.
        """
        if lags is None:
            lags = [1, 3, 6]

        trade_returns = self.trade.pct_change().dropna()
        results = []

        for name, series in self.costs.items():
            cost_returns = series.pct_change().dropna()
            for lag in lags:
                cost_lagged = cost_returns.shift(lag)
                combined = pd.DataFrame({
                    "trade_ret": trade_returns,
                    "cost_ret": cost_lagged,
                }).dropna()

                if len(combined) < 10:
                    continue

                x = combined["cost_ret"].values
                y = combined["trade_ret"].values
                x_with_const = np.column_stack([np.ones(len(x)), x])
                coef, residuals, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
                elasticity = coef[1]

                # t-statistic approximation
                n = len(y)
                if n > 2 and len(residuals) > 0:
                    mse = residuals[0] / (n - 2)
                    xtx_inv = np.linalg.inv(x_with_const.T @ x_with_const)
                    se = np.sqrt(mse * xtx_inv[1, 1])
                    t_stat = elasticity / se if se > 0 else 0
                else:
                    t_stat = 0

                results.append({
                    "cost_driver": name,
                    "lag_months": lag,
                    "elasticity": round(float(elasticity), 4),
                    "t_statistic": round(float(t_stat), 2),
                    "significant": abs(t_stat) > 1.96,
                })

        return pd.DataFrame(results)
