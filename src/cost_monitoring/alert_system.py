"""
Cost-Push Alert System

Monitors upstream cost pressures and generates alerts when indicators
exceed dynamic thresholds.  Uses a traffic-light scheme:

* **Green** — within normal range
* **Yellow** — elevated (> 1σ above rolling mean)
* **Orange** — high (> 1.5σ)
* **Red** — critical (> 2σ)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.CostMonitoring.AlertSystem")


@dataclass
class CostAlert:
    """A single cost-pressure alert."""
    date: str
    indicator: str
    level: str            # green | yellow | orange | red
    current_value: float
    threshold: float
    z_score: float
    description: str


class CostAlertSystem:
    """
    Generate and manage cost-pressure early-warning alerts.
    """

    # Alert levels and their z-score thresholds
    LEVELS = [
        ("red",    2.0),
        ("orange", 1.5),
        ("yellow", 1.0),
        ("green",  0.0),
    ]

    def __init__(self, rolling_window: int = 12):
        """
        Parameters
        ----------
        rolling_window : int
            Months of history for computing rolling mean/std.
        """
        self.window = rolling_window

    # ------------------------------------------------------------------ #
    #  Threshold computation
    # ------------------------------------------------------------------ #

    def compute_thresholds(
        self, indicator_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each column in ``indicator_df``, compute rolling mean, std,
        and upper thresholds at each alert level.

        Returns a "latest period" threshold summary.
        """
        rows = []
        for col in indicator_df.columns:
            series = indicator_df[col].dropna()
            if len(series) < self.window:
                continue
            rolling_mean = series.rolling(self.window, min_periods=3).mean()
            rolling_std = series.rolling(self.window, min_periods=3).std()

            latest_val = series.iloc[-1]
            latest_mean = rolling_mean.iloc[-1]
            latest_std = rolling_std.iloc[-1]
            z = (latest_val - latest_mean) / latest_std if latest_std > 0 else 0

            rows.append({
                "indicator": col,
                "latest_value": float(latest_val),
                "rolling_mean": round(float(latest_mean), 4),
                "rolling_std": round(float(latest_std), 4),
                "z_score": round(float(z), 2),
                "threshold_yellow": round(float(latest_mean + 1.0 * latest_std), 4),
                "threshold_orange": round(float(latest_mean + 1.5 * latest_std), 4),
                "threshold_red": round(float(latest_mean + 2.0 * latest_std), 4),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Alert generation
    # ------------------------------------------------------------------ #

    def check_alerts(
        self,
        indicator_df: pd.DataFrame,
    ) -> List[CostAlert]:
        """
        Check current indicator levels against dynamic thresholds
        and generate alerts.
        """
        alerts: List[CostAlert] = []
        thresholds = self.compute_thresholds(indicator_df)

        for _, row in thresholds.iterrows():
            z = row["z_score"]
            level = "green"
            for lvl, z_thresh in self.LEVELS:
                if z >= z_thresh:
                    level = lvl
                    break

            alert = CostAlert(
                date=str(indicator_df.index[-1].date())
                if hasattr(indicator_df.index[-1], "date")
                else str(indicator_df.index[-1]),
                indicator=row["indicator"],
                level=level,
                current_value=row["latest_value"],
                threshold=row[f"threshold_{level}"]
                if level != "green" and f"threshold_{level}" in row
                else row["rolling_mean"],
                z_score=z,
                description=self._level_description(row["indicator"], level, z),
            )
            alerts.append(alert)

        n_elevated = sum(1 for a in alerts if a.level != "green")
        logger.info(
            "Cost alerts: %d elevated out of %d indicators",
            n_elevated, len(alerts),
        )
        return alerts

    @staticmethod
    def _level_description(indicator: str, level: str, z: float) -> str:
        if level == "red":
            return f"{indicator} is at CRITICAL level ({z:.1f}σ above rolling mean)"
        elif level == "orange":
            return f"{indicator} is HIGH ({z:.1f}σ above rolling mean)"
        elif level == "yellow":
            return f"{indicator} is ELEVATED ({z:.1f}σ above rolling mean)"
        return f"{indicator} is within normal range"

    # ------------------------------------------------------------------ #
    #  Report
    # ------------------------------------------------------------------ #

    def generate_alert_report(
        self, alerts: Optional[List[CostAlert]] = None, indicator_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Generate a structured alert report suitable for JSON serialisation.
        """
        if alerts is None:
            if indicator_df is None:
                raise ValueError("Provide alerts or indicator_df")
            alerts = self.check_alerts(indicator_df)

        report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "n_indicators": len(alerts),
            "n_elevated": sum(1 for a in alerts if a.level != "green"),
            "summary": {
                "red": [a.indicator for a in alerts if a.level == "red"],
                "orange": [a.indicator for a in alerts if a.level == "orange"],
                "yellow": [a.indicator for a in alerts if a.level == "yellow"],
                "green": [a.indicator for a in alerts if a.level == "green"],
            },
            "details": [
                {
                    "date": a.date,
                    "indicator": a.indicator,
                    "level": a.level,
                    "current_value": a.current_value,
                    "threshold": a.threshold,
                    "z_score": a.z_score,
                    "description": a.description,
                }
                for a in alerts
            ],
        }
        return report

    def alerts_to_dataframe(
        self, alerts: List[CostAlert]
    ) -> pd.DataFrame:
        """Convenience: convert alerts to DataFrame."""
        return pd.DataFrame([
            {
                "date": a.date,
                "indicator": a.indicator,
                "level": a.level,
                "current_value": a.current_value,
                "z_score": a.z_score,
            }
            for a in alerts
        ])
