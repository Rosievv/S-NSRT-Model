"""
Mixed-Data Loader

Smart data loader that implements the framework's **mixed-data logic**:

1. Check what data sources are available.
2. Load from the highest-reliability source first.
3. Fall back to proxy / public sources when primary data is missing.
4. Log provenance for every data element so downstream analyses know
   which source contributed each signal.

This design reflects a realistic constraint: in decentralised supply
environments, complete cross-party proprietary data are often
unavailable.  The framework works with whatever is available and
improves precision as richer data become accessible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .data_registry import DataRegistry

logger = logging.getLogger("SCRAM.DataIntegration.MixedDataLoader")


class MixedDataLoader:
    """
    Load data with automatic fallback and provenance tracking.
    """

    def __init__(
        self,
        registry: Optional[DataRegistry] = None,
        data_dir: str = "data",
    ):
        self.registry = registry or DataRegistry()
        self.data_dir = Path(data_dir)
        self._provenance_log: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Core loading
    # ------------------------------------------------------------------ #

    def load(
        self,
        source_name: str,
        fallback: bool = True,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Attempt to load data from ``source_name``.  If it fails and
        ``fallback=True``, try each source in the fallback chain.

        Returns
        -------
        (DataFrame, provenance_dict)
        """
        chain = (
            self.registry.get_fallback_chain(source_name)
            if fallback
            else [source_name]
        )

        for name in chain:
            try:
                df = self._try_load(name, **kwargs)
                if df is not None and not df.empty:
                    prov = {
                        "requested": source_name,
                        "loaded_from": name,
                        "is_fallback": name != source_name,
                        "rows": len(df),
                        "columns": list(df.columns),
                    }
                    self._provenance_log.append(prov)
                    logger.info(
                        "Loaded %s (%d rows) [via %s]",
                        source_name, len(df), name,
                    )
                    return df, prov
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)

        # All sources failed
        prov = {
            "requested": source_name,
            "loaded_from": None,
            "is_fallback": False,
            "rows": 0,
            "columns": [],
            "error": "All sources in fallback chain failed",
        }
        self._provenance_log.append(prov)
        return pd.DataFrame(), prov

    def _try_load(self, source_name: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Try loading data for a given source name.
        Checks local Parquet files first, then attempts live fetch.
        """
        # 1. Check for local files
        local = self._find_local_files(source_name)
        if local:
            dfs = [pd.read_parquet(f) for f in local]
            return pd.concat(dfs, ignore_index=True)

        # 2. Attempt collector-based fetch
        meta = self.registry.get(source_name)
        collector = self._get_collector(source_name)
        if collector is not None:
            return collector.fetch(**kwargs)

        return None

    def _find_local_files(self, source_name: str) -> List[Path]:
        """Search data/raw/ for Parquet files matching the source name."""
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            return []
        # Heuristic file matching
        mapping = {
            "us_census_trade": "us_census_*",
            "nyfed_gscpi": "macro_*",
            "fred_ppi_semiconductor": "fred_*",
            "fred_ppi_all_commodities": "fred_*",
            "fred_cpi": "fred_*",
            "fred_industrial_production": "fred_*",
            "fred_ism_pmi": "fred_*",
        }
        pattern = mapping.get(source_name, f"{source_name}_*")
        return sorted(raw_dir.glob(f"{pattern}.parquet"))

    def _get_collector(self, source_name: str):
        """Return an appropriate collector instance, or None."""
        try:
            if source_name == "us_census_trade":
                from ..collectors.us_census_collector import USCensusCollector
                return USCensusCollector()
            elif source_name.startswith("fred_") or source_name == "nyfed_gscpi":
                from ..collectors.fred_collector import FREDCollector
                return FREDCollector()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  Convenience: combined load
    # ------------------------------------------------------------------ #

    def load_all_available(self) -> Dict[str, pd.DataFrame]:
        """
        Attempt to load every registered source.
        Returns {source_name: DataFrame} for successful loads.
        """
        result = {}
        for meta in self.registry.list_sources():
            df, prov = self.load(meta.name)
            if not df.empty:
                result[meta.name] = df
        logger.info(
            "Loaded %d / %d registered sources",
            len(result),
            len(self.registry.list_sources()),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Provenance
    # ------------------------------------------------------------------ #

    def get_provenance_log(self) -> pd.DataFrame:
        """Return a DataFrame of all load attempts and their provenance."""
        return pd.DataFrame(self._provenance_log)

    def clear_provenance(self) -> None:
        self._provenance_log.clear()
