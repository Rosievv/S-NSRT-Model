#!/usr/bin/env python3
"""
Shared data loading helpers for Module 1 analysis scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"


def load_trade_data(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load and merge raw US Census parquet files, keeping the latest snapshot."""
    source_dir = raw_dir or RAW_DIR
    files = sorted(source_dir.glob("us_census_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No us_census parquet files found in {source_dir}")

    dfs = [pd.read_parquet(file_path) for file_path in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df = (
        df.sort_values("collected_at", kind="stable")
        .drop_duplicates(["date", "hs_code", "country"], keep="last")
        .reset_index(drop=True)
    )
    return df


def filter_aggregate_country_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove aggregate country labels that are not country-level records."""
    aggregate_exact = {
        "Total For All Countries",
        "Apec",
        "Asean",
        "Asia",
        "Pacific Rim Countries",
        "Oecd",
        "North America",
        "Usmca (Nafta)",
        "Nato",
        "Europe",
        "European Union",
        "Euro Area",
        "Twenty Latin American Republics",
    }

    aggregate_patterns = [
        r"^total\b",
        r"all countries",
        r"\b(?:apec|asean|oecd|nato|nafta|usmca)\b",
        r"\b(?:pacific rim|euro area)\b",
        r"\b(?:asia|europe|north america)\b",
        r"\b(?:european union)\b",
        r"latin american republics",
    ]
    combined_pattern = re.compile("|".join(aggregate_patterns), re.IGNORECASE)

    country_series = df["country"].astype(str)
    exact_mask = country_series.isin(aggregate_exact)
    regex_mask = country_series.str.contains(combined_pattern, na=False)
    drop_mask = exact_mask | regex_mask

    filtered_df = df.loc[~drop_mask].copy()

    removed_labels = sorted(country_series.loc[drop_mask].dropna().unique().tolist())
    print(
        f"Filtered aggregate labels: removed {len(removed_labels)} labels and {drop_mask.sum():,} rows"
    )
    if removed_labels:
        print("Removed labels:", ", ".join(removed_labels))

    return filtered_df


def load_module1_trade_data(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load, clean, and return the trade panel used by Module 1."""
    trade_df = load_trade_data(raw_dir=raw_dir)
    return filter_aggregate_country_labels(trade_df)


def describe_trade_coverage(df: pd.DataFrame) -> Dict[str, object]:
    """Return a small coverage summary for sanity checks."""
    coverage = {
        "min_date": df["date"].min(),
        "max_date": df["date"].max(),
        "has_2025": bool((df["date"].dt.year == 2025).any()),
        "month_count": int(df["date"].dt.to_period("M").nunique()),
    }
    return coverage


if __name__ == "__main__":
    df = load_module1_trade_data()
    coverage = describe_trade_coverage(df)
    print(coverage)