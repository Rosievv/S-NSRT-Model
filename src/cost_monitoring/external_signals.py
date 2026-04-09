"""
External Signal Loader

Domain-specific wrapper around the FRED collector that provides
ready-to-use, normalised economic indicator series for cost-monitoring
and risk-analysis modules.

Handles:
* Commodity prices (PPI, copper, silicon proxies)
* Freight indices (GSCPI / supply-chain pressure)
* Macro indicators (CPI, INDPRO, PMI)

All series are resampled to monthly frequency and optionally normalised
to a base-period index (period average = 100).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.CostMonitoring.ExternalSignals")


# Groupings of FRED series by domain
SIGNAL_GROUPS = {
    "commodity_prices": [
        "PCU33443344",   # PPI semiconductor manufacturing
        "PPIACO",        # PPI all commodities
        "WPU10170642",   # PPI copper wire
    ],
    "freight_indices": [
        "GSCPI",         # Global Supply Chain Pressure Index
    ],
    "macro_indicators": [
        "CPIAUCSL",      # CPI
        "INDPRO",        # Industrial Production
        "UNRATE",        # Unemployment
        "NAPM",          # ISM PMI
    ],
    "trade_indicators": [
        "DTWEXBGS",      # Trade-weighted USD
        "BOPGSTB",       # Trade balance
    ],
}


class ExternalSignalLoader:
    """
    Load, cache, and transform external economic signals for
    cost-monitoring analysis.
    """

    def __init__(self, fred_wide_df: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        fred_wide_df : pd.DataFrame, optional
            Wide-format FRED data (columns = series_id, index = date).
            If not provided, the loader will attempt to fetch from FRED
            at runtime.
        """
        self._wide: Optional[pd.DataFrame] = fred_wide_df

    def _ensure_data(self) -> pd.DataFrame:
        if self._wide is not None and not self._wide.empty:
            return self._wide
        # Attempt live fetch
        try:
            from ..collectors.fred_collector import FREDCollector
            collector = FREDCollector()
            raw = collector.fetch()
            self._wide = collector.pivot_series(raw)
            return self._wide
        except Exception as e:
            logger.error("Could not load FRED data: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Domain-specific getters
    # ------------------------------------------------------------------ #

    def get_commodity_prices(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Returns commodity/PPI series (monthly, raw values)."""
        return self._get_group("commodity_prices", start, end)

    def get_freight_indices(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        return self._get_group("freight_indices", start, end)

    def get_macro_indicators(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        return self._get_group("macro_indicators", start, end)

    def get_trade_indicators(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        return self._get_group("trade_indicators", start, end)

    def _get_group(
        self, group: str, start: Optional[str], end: Optional[str]
    ) -> pd.DataFrame:
        df = self._ensure_data()
        if df.empty:
            return df
        cols = [c for c in SIGNAL_GROUPS.get(group, []) if c in df.columns]
        out = df[cols].copy()
        if start:
            out = out[out.index >= pd.Timestamp(start)]
        if end:
            out = out[out.index <= pd.Timestamp(end)]
        return out

    # ------------------------------------------------------------------ #
    #  Normalisation
    # ------------------------------------------------------------------ #

    def normalise_to_index(
        self,
        df: pd.DataFrame,
        base_start: str = "2015-01-01",
        base_end: str = "2015-12-31",
    ) -> pd.DataFrame:
        """
        Normalise each column so that its average over the base period = 100.
        """
        base = df[
            (df.index >= pd.Timestamp(base_start))
            & (df.index <= pd.Timestamp(base_end))
        ]
        base_means = base.mean()
        normalised = df.copy()
        for col in normalised.columns:
            bm = base_means.get(col, 0)
            if bm and bm != 0:
                normalised[col] = normalised[col] / bm * 100
        return normalised

    # ------------------------------------------------------------------ #
    #  Convenience: get all signals as a dict of pd.Series
    # ------------------------------------------------------------------ #

    def as_series_dict(self) -> Dict[str, pd.Series]:
        """Return all available signals as ``{series_id: pd.Series}``."""
        df = self._ensure_data()
        return {col: df[col].dropna() for col in df.columns}

    def get_series(self, series_id: str) -> pd.Series:
        """Get a single series by FRED ID."""
        df = self._ensure_data()
        if series_id in df.columns:
            return df[series_id].dropna()
        raise KeyError(f"Series '{series_id}' not found in loaded data")
