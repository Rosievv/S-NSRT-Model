"""
Disruption Detector

Identifies lane-level disruptions from trade-volume drops and
freight-cost spikes using statistical anomaly detection on
time-series data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.Transportation.DisruptionDetector")


@dataclass
class Disruption:
    """A detected disruption event on a specific lane."""
    date: str
    region: str
    signal_type: str          # "volume_drop" | "cost_spike"
    severity: str             # "low" | "medium" | "high" | "critical"
    magnitude: float          # e.g. -0.45 for a 45 % volume drop
    details: Dict


class DisruptionDetector:
    """
    Detect lane-level disruptions from time-series anomalies.

    Two signal types are monitored:
    1. **Volume drops** — MoM changes in trade value that exceed a threshold.
    2. **Cost spikes** — freight / cost indices that exceed N standard
       deviations from a rolling mean.
    """

    def __init__(
        self,
        trade_df: pd.DataFrame,
        freight_df: Optional[pd.DataFrame] = None,
        volume_drop_threshold: float = 0.30,
        cost_spike_std: float = 2.0,
        rolling_window: int = 6,
    ):
        """
        Parameters
        ----------
        trade_df : pd.DataFrame
            Trade data with ``date, country, value_usd``.
        freight_df : pd.DataFrame, optional
            Freight/cost index data with ``date, series_id, value``.
        volume_drop_threshold : float
            Fraction (0–1).  Flag when MoM volume drops by more than this.
        cost_spike_std : float
            Number of standard deviations above rolling mean to flag cost spike.
        rolling_window : int
            Months for rolling statistics.
        """
        self.trade_df = trade_df.copy()
        self.trade_df["date"] = pd.to_datetime(self.trade_df["date"])
        self.freight_df = freight_df
        self.volume_threshold = volume_drop_threshold
        self.cost_std_threshold = cost_spike_std
        self.window = rolling_window

        # Pre-compute region mapping
        from .lane_network import _country_to_region
        self.trade_df["region"] = self.trade_df["country"].apply(_country_to_region)

    # ------------------------------------------------------------------ #
    #  Volume-based detection
    # ------------------------------------------------------------------ #

    def detect_volume_drops(self) -> List[Disruption]:
        """
        Detect months where a region's total import value drops by more
        than ``volume_drop_threshold`` compared to the previous month.
        """
        monthly = (
            self.trade_df.groupby([pd.Grouper(key="date", freq="ME"), "region"])
            ["value_usd"]
            .sum()
            .reset_index()
        )
        monthly = monthly.sort_values(["region", "date"])
        monthly["prev"] = monthly.groupby("region")["value_usd"].shift(1)
        monthly["mom_change"] = (
            (monthly["value_usd"] - monthly["prev"]) / monthly["prev"]
        )

        disruptions: List[Disruption] = []
        for _, row in monthly.iterrows():
            if pd.isna(row["mom_change"]):
                continue
            if row["mom_change"] < -self.volume_threshold:
                sev = self._classify_severity(abs(row["mom_change"]))
                disruptions.append(
                    Disruption(
                        date=str(row["date"].date()),
                        region=row["region"],
                        signal_type="volume_drop",
                        severity=sev,
                        magnitude=round(float(row["mom_change"]), 4),
                        details={
                            "value_usd": float(row["value_usd"]),
                            "prev_value_usd": float(row["prev"]),
                        },
                    )
                )
        logger.info("Detected %d volume-drop disruptions", len(disruptions))
        return disruptions

    # ------------------------------------------------------------------ #
    #  Cost-spike detection
    # ------------------------------------------------------------------ #

    def detect_cost_spikes(self, series_id: str = "GSCPI") -> List[Disruption]:
        """
        Detect months where a freight / cost index exceeds its rolling
        mean + ``cost_spike_std`` × rolling std.
        """
        if self.freight_df is None or self.freight_df.empty:
            logger.warning("No freight data; skipping cost-spike detection")
            return []

        df = self.freight_df.copy()
        if "series_id" in df.columns:
            df = df[df["series_id"] == series_id]
        if df.empty:
            return []

        df = df.sort_values("date")
        df["rolling_mean"] = df["value"].rolling(self.window, min_periods=2).mean()
        df["rolling_std"] = df["value"].rolling(self.window, min_periods=2).std()
        df["upper_band"] = df["rolling_mean"] + self.cost_std_threshold * df["rolling_std"]

        disruptions: List[Disruption] = []
        for _, row in df.iterrows():
            if pd.isna(row["upper_band"]):
                continue
            if row["value"] > row["upper_band"]:
                z = (
                    (row["value"] - row["rolling_mean"]) / row["rolling_std"]
                    if row["rolling_std"] > 0
                    else 0
                )
                sev = self._classify_severity(z / 5)  # normalise z to 0-1 ish
                disruptions.append(
                    Disruption(
                        date=str(pd.Timestamp(row["date"]).date()),
                        region="global",
                        signal_type="cost_spike",
                        severity=sev,
                        magnitude=round(float(z), 2),
                        details={
                            "series_id": series_id,
                            "value": float(row["value"]),
                            "rolling_mean": float(row["rolling_mean"]),
                            "upper_band": float(row["upper_band"]),
                        },
                    )
                )
        logger.info("Detected %d cost-spike disruptions", len(disruptions))
        return disruptions

    # ------------------------------------------------------------------ #
    #  Combined
    # ------------------------------------------------------------------ #

    def detect_all(self) -> List[Disruption]:
        """Run all detectors and return combined list sorted by date."""
        all_d = self.detect_volume_drops() + self.detect_cost_spikes()
        all_d.sort(key=lambda d: d.date)
        return all_d

    def disruptions_to_dataframe(self, disruptions: Optional[List[Disruption]] = None) -> pd.DataFrame:
        """Convert disruption list to DataFrame."""
        if disruptions is None:
            disruptions = self.detect_all()
        rows = [
            {
                "date": d.date,
                "region": d.region,
                "signal_type": d.signal_type,
                "severity": d.severity,
                "magnitude": d.magnitude,
            }
            for d in disruptions
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_severity(magnitude: float) -> str:
        """Map an absolute magnitude (0-1+) to severity level."""
        if magnitude >= 0.7:
            return "critical"
        elif magnitude >= 0.5:
            return "high"
        elif magnitude >= 0.3:
            return "medium"
        else:
            return "low"
