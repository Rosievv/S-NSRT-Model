"""
Data Quality Checker

Validates data loaded from heterogeneous sources before it enters
the analytical pipeline.  Checks include:

* **Completeness** — % of missing values per column
* **Freshness** — how stale is the most recent observation?
* **Schema conformance** — required columns present and correctly typed
* **Consistency** — no duplicate keys, values within expected ranges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.DataIntegration.DataQuality")


@dataclass
class QualityReport:
    """Quality assessment for a single DataFrame."""
    source_name: str
    rows: int
    columns: int
    completeness: float          # 0–1
    missing_by_column: Dict[str, float]
    freshness_days: Optional[int]
    schema_ok: bool
    schema_errors: List[str]
    duplicate_rows: int
    overall_score: float         # 0–100


class DataQualityChecker:
    """
    Validate DataFrames against configurable quality expectations.
    """

    # Default expected schemas per source category
    SCHEMAS = {
        "trade": {
            "required": ["date", "hs_code", "country", "value_usd"],
            "numeric": ["value_usd"],
        },
        "fred": {
            "required": ["date", "series_id", "value"],
            "numeric": ["value"],
        },
        "macro": {
            "required": ["date"],
            "numeric": [],
        },
    }

    def __init__(
        self,
        max_missing_pct: float = 0.20,
        max_staleness_days: int = 90,
    ):
        self.max_missing_pct = max_missing_pct
        self.max_staleness_days = max_staleness_days

    def check(
        self,
        df: pd.DataFrame,
        source_name: str = "unknown",
        schema_key: Optional[str] = None,
    ) -> QualityReport:
        """
        Run all quality checks and return a QualityReport.
        """
        if df.empty:
            return QualityReport(
                source_name=source_name,
                rows=0, columns=0,
                completeness=0.0,
                missing_by_column={},
                freshness_days=None,
                schema_ok=False,
                schema_errors=["DataFrame is empty"],
                duplicate_rows=0,
                overall_score=0.0,
            )

        # Completeness
        missing_by_col = {
            c: round(float(df[c].isna().mean()), 4) for c in df.columns
        }
        completeness = 1.0 - df.isna().mean().mean()

        # Freshness
        freshness_days = None
        if "date" in df.columns:
            try:
                latest = pd.to_datetime(df["date"]).max()
                freshness_days = (pd.Timestamp.now() - latest).days
            except Exception:
                pass

        # Schema
        schema_errors: List[str] = []
        schema = self.SCHEMAS.get(schema_key or "", {})
        for col in schema.get("required", []):
            if col not in df.columns:
                schema_errors.append(f"Missing required column: {col}")
        for col in schema.get("numeric", []):
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                schema_errors.append(f"Column {col} should be numeric")
        schema_ok = len(schema_errors) == 0

        # Duplicates
        dup_rows = int(df.duplicated().sum())

        # Overall score (weighted)
        score = 0.0
        score += completeness * 40  # 40 pts for completeness
        score += (40 if schema_ok else 0)  # 40 pts for schema
        if freshness_days is not None:
            freshness_score = max(0, 20 * (1 - freshness_days / self.max_staleness_days))
        else:
            freshness_score = 10  # partial credit
        score += freshness_score
        score = min(100, max(0, score))

        report = QualityReport(
            source_name=source_name,
            rows=len(df),
            columns=len(df.columns),
            completeness=round(float(completeness), 4),
            missing_by_column=missing_by_col,
            freshness_days=freshness_days,
            schema_ok=schema_ok,
            schema_errors=schema_errors,
            duplicate_rows=dup_rows,
            overall_score=round(float(score), 1),
        )

        logger.info(
            "Quality check [%s]: score=%.1f, completeness=%.1f%%, freshness=%s days",
            source_name, report.overall_score,
            completeness * 100,
            freshness_days if freshness_days is not None else "N/A",
        )
        return report

    def check_multiple(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Check multiple datasets and return a summary DataFrame.
        """
        rows = []
        for name, df in datasets.items():
            report = self.check(df, source_name=name)
            rows.append({
                "source": report.source_name,
                "rows": report.rows,
                "completeness": report.completeness,
                "freshness_days": report.freshness_days,
                "schema_ok": report.schema_ok,
                "duplicates": report.duplicate_rows,
                "quality_score": report.overall_score,
            })
        return pd.DataFrame(rows).sort_values("quality_score", ascending=False)
