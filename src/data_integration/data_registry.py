"""
Data Source Registry

Central catalogue of every data source used by the framework.
Each source carries metadata describing its type, reliability,
update cadence, and coverage — enabling the mixed-data loader
to make informed fallback decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("SCRAM.DataIntegration.Registry")


@dataclass
class DataSourceMeta:
    """Metadata for a registered data source."""
    name: str
    source_type: str          # "internal" | "public" | "proxy"
    provider: str             # e.g. "US Census", "FRED", "NY Fed"
    description: str
    update_frequency: str     # "monthly" | "quarterly" | "annual" | "daily"
    reliability_score: float  # 0.0 – 1.0  (subjective assessment)
    coverage_start: Optional[str] = None  # "YYYY-MM"
    coverage_end: Optional[str] = None
    requires_api_key: bool = False
    fallback_sources: List[str] = field(default_factory=list)
    notes: str = ""


# Pre-built registry of sources used in SCRAM
_DEFAULT_SOURCES: List[DataSourceMeta] = [
    # ── Primary trade data ──
    DataSourceMeta(
        name="us_census_trade",
        source_type="public",
        provider="US Census Bureau",
        description="Monthly HS-6 level import/export data by country",
        update_frequency="monthly",
        reliability_score=0.95,
        coverage_start="2010-01",
        requires_api_key=True,
        notes="Primary data source for trade values and quantities",
    ),
    # ── Commodity / Price data ──
    DataSourceMeta(
        name="fred_ppi_semiconductor",
        source_type="public",
        provider="FRED (BLS via FRED)",
        description="PPI for semiconductor and electronic component manufacturing (PCU33443344)",
        update_frequency="monthly",
        reliability_score=0.90,
        requires_api_key=True,
        fallback_sources=["fred_ppi_all_commodities"],
    ),
    DataSourceMeta(
        name="fred_ppi_all_commodities",
        source_type="public",
        provider="FRED (BLS via FRED)",
        description="PPI for all commodities (PPIACO)",
        update_frequency="monthly",
        reliability_score=0.90,
        requires_api_key=True,
    ),
    # ── Freight / Supply-chain pressure ──
    DataSourceMeta(
        name="nyfed_gscpi",
        source_type="public",
        provider="NY Federal Reserve",
        description="Global Supply Chain Pressure Index",
        update_frequency="monthly",
        reliability_score=0.85,
        coverage_start="1997-01",
    ),
    # ── Macro ──
    DataSourceMeta(
        name="fred_cpi",
        source_type="public",
        provider="FRED (BLS via FRED)",
        description="Consumer Price Index for All Urban Consumers (CPIAUCSL)",
        update_frequency="monthly",
        reliability_score=0.95,
        requires_api_key=True,
    ),
    DataSourceMeta(
        name="fred_industrial_production",
        source_type="public",
        provider="FRED",
        description="Industrial Production Index (INDPRO)",
        update_frequency="monthly",
        reliability_score=0.90,
        requires_api_key=True,
    ),
    DataSourceMeta(
        name="fred_ism_pmi",
        source_type="public",
        provider="FRED (ISM via FRED)",
        description="ISM Manufacturing PMI (NAPM)",
        update_frequency="monthly",
        reliability_score=0.85,
        requires_api_key=True,
    ),
    # ── Mineral / USGS ──
    DataSourceMeta(
        name="usgs_minerals",
        source_type="public",
        provider="USGS",
        description="Annual mineral commodity summaries (gallium, germanium, silicon)",
        update_frequency="annual",
        reliability_score=0.70,
        notes="Annual data; requires interpolation for monthly models",
        fallback_sources=["fred_ppi_all_commodities"],
    ),
    # ── Internal / operational (placeholder for enterprise use) ──
    DataSourceMeta(
        name="internal_procurement",
        source_type="internal",
        provider="Enterprise ERP",
        description="Internal procurement and PO data (available in enterprise deployments)",
        update_frequency="daily",
        reliability_score=1.0,
        notes="Not available in open-source mode; framework degrades to public sources",
    ),
]


class DataRegistry:
    """
    Catalogue and query registered data sources.
    """

    def __init__(self, sources: Optional[List[DataSourceMeta]] = None):
        self._sources: Dict[str, DataSourceMeta] = {}
        for s in (sources or _DEFAULT_SOURCES):
            self.register(s)

    def register(self, meta: DataSourceMeta) -> None:
        self._sources[meta.name] = meta
        logger.debug("Registered data source: %s", meta.name)

    def get(self, name: str) -> DataSourceMeta:
        if name not in self._sources:
            raise KeyError(f"Data source '{name}' not registered")
        return self._sources[name]

    def list_sources(self, source_type: Optional[str] = None) -> List[DataSourceMeta]:
        """List sources, optionally filtered by type."""
        sources = list(self._sources.values())
        if source_type:
            sources = [s for s in sources if s.source_type == source_type]
        return sources

    def get_fallback_chain(self, name: str) -> List[str]:
        """Return the ordered fallback chain for a source."""
        meta = self.get(name)
        chain = [name] + meta.fallback_sources
        return chain

    def summary(self) -> Dict:
        """Summary statistics of the registry."""
        sources = list(self._sources.values())
        return {
            "total_sources": len(sources),
            "by_type": {
                t: sum(1 for s in sources if s.source_type == t)
                for t in {"public", "internal", "proxy"}
            },
            "avg_reliability": (
                sum(s.reliability_score for s in sources) / len(sources)
                if sources else 0
            ),
        }
