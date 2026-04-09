"""
FRED (Federal Reserve Economic Data) Collector

Collects economic and commodity data from the FRED API for use as
external signals in supply chain risk analysis. Supports freight indices,
commodity prices, producer price indices, and macroeconomic indicators.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import requests

from .base_collector import BaseCollector
from ..utils import retry_with_backoff


# Default FRED series for supply chain analysis
DEFAULT_SERIES = {
    # Freight and logistics
    "GSCPI": "Global Supply Chain Pressure Index (NY Fed proxy)",
    # Commodity and producer prices
    "PCU33443344": "PPI: Semiconductor and electronic component manufacturing",
    "PPIACO": "PPI: All commodities",
    "WPU10170642": "PPI: Copper and copper-base alloy wire",
    # Macro indicators
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "UNRATE": "Unemployment Rate",
    "INDPRO": "Industrial Production Index",
    "NAPM": "ISM Manufacturing PMI",
    "NAPMSI": "ISM Manufacturing Supplier Deliveries Index",
    "NAPMNOI": "ISM Manufacturing New Orders Index",
    # Trade and dollar
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad, Goods and Services",
    "BOPGSTB": "Trade Balance: Goods and Services",
    # Interest rates (affect investment/capex in semiconductor fabs)
    "FEDFUNDS": "Federal Funds Effective Rate",
}


class FREDCollector(BaseCollector):
    """
    Collector for Federal Reserve Economic Data (FRED)
    
    Provides batch retrieval of multiple economic time series 
    relevant to supply chain risk monitoring.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        series_ids: Optional[List[str]] = None,
    ):
        super().__init__(
            collector_name="fred_collector",
            data_source="fred",
            start_date=start_date,
            end_date=end_date,
        )
        
        # API key from param, env, or config
        if api_key:
            self._api_key = api_key
        else:
            import os
            self._api_key = os.environ.get("FRED_API_KEY", "")
        
        self.series_ids = series_ids or list(DEFAULT_SERIES.keys())
        self.series_metadata = DEFAULT_SERIES
        self.logger.info(
            f"FREDCollector initialized with {len(self.series_ids)} series"
        )
    
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch all configured FRED series and combine into a single DataFrame.
        
        Returns:
            DataFrame with columns: date, series_id, value, description, source
        """
        date_range = self.get_date_range()
        all_frames: List[pd.DataFrame] = []
        
        for series_id in self.series_ids:
            try:
                df = self._fetch_series(
                    series_id,
                    date_range["start"],
                    date_range["end"],
                )
                if df is not None and not df.empty:
                    all_frames.append(df)
                    self.logger.info(
                        f"  {series_id}: {len(df)} observations"
                    )
                # Respect rate limit (~120 req/min for FRED)
                time.sleep(0.6)
            except Exception as e:
                self.logger.warning(f"Failed to fetch {series_id}: {e}")
        
        if not all_frames:
            self.logger.warning("No FRED series could be fetched")
            return pd.DataFrame(
                columns=["date", "series_id", "value", "description", "source"]
            )
        
        combined = pd.concat(all_frames, ignore_index=True)
        self.logger.info(
            f"Total FRED observations collected: {len(combined)} "
            f"across {combined['series_id'].nunique()} series"
        )
        return combined
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "asc",
        }
        
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        observations = data.get("observations", [])
        if not observations:
            return None
        
        rows = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == ".":  # FRED uses "." for missing
                continue
            rows.append({
                "date": pd.Timestamp(obs["date"]),
                "series_id": series_id,
                "value": float(val),
                "description": self.series_metadata.get(series_id, ""),
                "source": "FRED",
            })
        
        return pd.DataFrame(rows) if rows else None
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate fetched FRED data."""
        required = ["date", "series_id", "value"]
        missing = set(required) - set(df.columns)
        if missing:
            self.logger.error(f"Missing columns: {missing}")
            return False
        if df.empty:
            self.logger.warning("Empty FRED DataFrame")
            return False
        return True
    
    # ---- convenience helpers used by downstream modules ----
    
    def pivot_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Pivot the long-format FRED data into wide format
        (one column per series_id, indexed by date).
        
        Useful for correlation analysis and cost-transmission modeling.
        """
        if df is None:
            df = self.fetch()
        
        wide = df.pivot_table(
            index="date", columns="series_id", values="value", aggfunc="last"
        )
        wide = wide.sort_index().ffill()
        return wide
    
    def get_series_as_monthly(
        self, df: pd.DataFrame, series_id: str
    ) -> pd.Series:
        """Extract a single series resampled to month-end frequency."""
        subset = df[df["series_id"] == series_id].set_index("date")["value"]
        monthly = subset.resample("ME").last().ffill()
        return monthly
