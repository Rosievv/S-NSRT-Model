#!/usr/bin/env python3
"""
Run Module 1 (Risk Propagation) and generate visual outputs.

Outputs:
- reports/module1/module1_stress_results.json
- reports/module1/module1_stress_summary.csv
- reports/module1/module1_backtest.csv
- reports/module1/figures/module1_supply_gap_pct.png
- reports/module1/figures/module1_substitution_absorbed_pct.png
- reports/module1/figures/module1_backtest_comparison.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from risk_propagation import SupplyChainNetwork, PropagationEngine, StressTestRunner

RAW_DIR = ROOT_DIR / "data" / "raw"
OUT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = OUT_DIR / "figures"


def load_trade_data() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("us_census_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No us_census parquet files found in {RAW_DIR}")

    dfs = [pd.read_parquet(file_path) for file_path in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_aggregate_country_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-country aggregate labels in the `country` column."""
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
    print(f"Filtered aggregate labels: removed {len(removed_labels)} labels and {drop_mask.sum():,} rows")
    if removed_labels:
        print("Removed labels:", ", ".join(removed_labels))

    return filtered_df


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    trade_df = load_trade_data()
    trade_df = filter_aggregate_country_labels(trade_df)

    network = SupplyChainNetwork(trade_df)
    graph = network.build_network()
    centrality_df = network.compute_centrality(graph)
    critical_nodes = network.identify_critical_nodes(graph)

    runner = StressTestRunner(trade_df)
    scenario_results = runner.run_all_standard()
    summary_df = PropagationEngine.results_to_dataframe(scenario_results)
    backtest_df = runner.backtest_all()

    stress_json_path = OUT_DIR / "module1_stress_results.json"
    with stress_json_path.open("w") as file_obj:
        json.dump(
            [
                {
                    "scenario_name": result.scenario_name,
                    "shocked_nodes": result.shocked_nodes,
                    "severity": result.severity,
                    "original_supply": result.original_supply,
                    "disrupted_supply": result.disrupted_supply,
                    "supply_gap_pct": result.supply_gap_pct,
                    "substitution_absorbed_pct": result.substitution_absorbed_pct,
                    "most_affected_hs": result.most_affected_hs,
                    "details": result.details,
                }
                for result in scenario_results
            ],
            file_obj,
            indent=2,
        )

    summary_df.to_csv(OUT_DIR / "module1_stress_summary.csv", index=False)
    backtest_df.to_csv(OUT_DIR / "module1_backtest.csv", index=False)
    centrality_df.to_csv(OUT_DIR / "module1_centrality.csv", index=False)

    if not summary_df.empty:
        plot_bar(
            summary_df,
            x_col="scenario",
            y_col="supply_gap_pct",
            title="Module 1: Supply Gap by Scenario",
            ylabel="Supply Gap (%)",
            output_path=FIG_DIR / "module1_supply_gap_pct.png",
        )
        plot_bar(
            summary_df,
            x_col="scenario",
            y_col="substitution_absorbed_pct",
            title="Module 1: Substitution Absorption by Scenario",
            ylabel="Substitution Absorbed (%)",
            output_path=FIG_DIR / "module1_substitution_absorbed_pct.png",
        )

    if not backtest_df.empty and {"predicted_supply_gap_pct", "observed_supply_gap_pct"}.issubset(backtest_df.columns):
        plot_df = backtest_df.copy()

        plt.figure(figsize=(12, 5))
        x_positions = range(len(plot_df))
        plt.bar([x - 0.2 for x in x_positions], plot_df["predicted_supply_gap_pct"], width=0.4, label="Predicted Supply Gap %")
        plt.bar([x + 0.2 for x in x_positions], plot_df["observed_supply_gap_pct"], width=0.4, label="Observed Supply Gap %")
        plt.xticks(list(x_positions), plot_df["event"], rotation=30, ha="right")
        plt.ylabel("Percent (%)")
        plt.title("Module 1 Backtest: Predicted vs Observed Supply Gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "module1_backtest_comparison.png", dpi=150)
        plt.close()

    print("Module 1 completed.")
    print(f"Total trade records: {len(trade_df):,}")
    print(f"Critical nodes (>5% share): {critical_nodes}")
    if not summary_df.empty:
        print("\nScenario summary:")
        print(summary_df[["scenario", "supply_gap_pct", "substitution_absorbed_pct", "most_affected_hs"]].to_string(index=False))
    if not backtest_df.empty and {"directional_accuracy_pct", "pairwise_ranking_accuracy_pct"}.issubset(backtest_df.columns):
        print("\nBacktest directional metrics:")
        print(
            "Directional accuracy: "
            f"{backtest_df['directional_accuracy_pct'].iloc[0]:.2f}% | "
            "Pairwise ranking accuracy: "
            f"{backtest_df['pairwise_ranking_accuracy_pct'].iloc[0]:.2f}%"
        )
    print(f"\nSaved outputs to: {OUT_DIR}")
    print(f"Saved charts to: {FIG_DIR}")


if __name__ == "__main__":
    main()
