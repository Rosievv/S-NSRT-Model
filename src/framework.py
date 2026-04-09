"""
SCRAM Framework Orchestrator

Top-level entry point that ties all four analytical modules together
into a coordinated pipeline.  Modules can be run individually or as
a full integrated analysis.

Modules
-------
1. **Risk Propagation** — graph-based supply-network modelling and
   disruption simulation.
2. **Transportation Resilience** — logistics-lane disruption detection
   and re-routing optimisation.
3. **Demand Forecasting** — quantile-based supply forecasting with
   shortage detection and adaptive calibration.
4. **Cost Monitoring** — upstream cost-transmission analysis and
   early-warning alerts.

Usage
-----
    python -m src.framework --module all
    python -m src.framework --module risk_propagation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SCRAM.Framework")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)

# ====================================================================== #
#  Helpers
# ====================================================================== #

ROOT_DIR = Path(__file__).resolve().parent.parent  # scram/


def _load_trade_data() -> pd.DataFrame:
    """Load raw US Census trade data from Parquet files."""
    raw_dir = ROOT_DIR / "data" / "raw"
    files = sorted(raw_dir.glob("us_census_*.parquet"))
    if not files:
        logger.warning("No Census trade parquet files found in %s", raw_dir)
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d trade records from %d files", len(df), len(files))
    return df


def _load_feature_data(split: str = "train") -> pd.DataFrame:
    """Load engineered feature data."""
    p = ROOT_DIR / "data" / "processed" / f"features_{split}_full.parquet"
    if not p.exists():
        logger.warning("Feature file not found: %s", p)
        return pd.DataFrame()
    return pd.read_parquet(p)


def _load_fred_data() -> pd.DataFrame:
    """Attempt to load FRED data (local or live)."""
    raw_dir = ROOT_DIR / "data" / "raw"
    fred_files = sorted(raw_dir.glob("fred_*.parquet"))
    if fred_files:
        return pd.concat([pd.read_parquet(f) for f in fred_files], ignore_index=True)
    # Attempt live fetch
    try:
        from src.collectors.fred_collector import FREDCollector
        collector = FREDCollector()
        return collector.fetch()
    except Exception as e:
        logger.warning("Could not load FRED data: %s", e)
        return pd.DataFrame()


# ====================================================================== #
#  Module runners
# ====================================================================== #

def run_risk_propagation(trade_df: pd.DataFrame) -> Dict[str, Any]:
    """Module 1: Risk Propagation Simulation."""
    from src.risk_propagation import SupplyChainNetwork, PropagationEngine, StressTestRunner

    logger.info("=" * 60)
    logger.info("MODULE 1: Risk Propagation Simulation")
    logger.info("=" * 60)

    # Build supply-chain graph
    net = SupplyChainNetwork(trade_df)
    G = net.build_network()
    centrality = net.compute_centrality(G)
    critical = net.identify_critical_nodes(G)

    logger.info("Top 5 suppliers by market share:")
    if not centrality.empty:
        for _, row in centrality.head(5).iterrows():
            logger.info(
                "  %s: share=%.1f%%, pagerank=%.4f",
                row["country"], row["market_share"] * 100, row["pagerank"],
            )

    logger.info("Critical nodes (>5%% share): %s", critical)

    # Run stress tests
    runner = StressTestRunner(trade_df)
    results = runner.run_all_standard()
    summary = PropagationEngine.results_to_dataframe(results)
    logger.info("\nStress-test summary:\n%s", summary.to_string(index=False))

    # Backtest
    backtest = runner.backtest_all()
    if not backtest.empty:
        logger.info("\nHistorical back-test:\n%s", backtest.to_string(index=False))

    # Concentration tracking
    concentration = net.track_concentration_over_time()

    return {
        "centrality": centrality,
        "critical_nodes": critical,
        "stress_test_summary": summary,
        "backtest": backtest,
        "concentration_trend": concentration,
    }


def run_transportation_resilience(trade_df: pd.DataFrame, fred_df: pd.DataFrame) -> Dict[str, Any]:
    """Module 2: Transportation Resilience Optimisation."""
    from src.transportation import LogisticsNetwork, DisruptionDetector, ReRoutingOptimizer, LPReRouter

    logger.info("=" * 60)
    logger.info("MODULE 2: Transportation Resilience Optimisation")
    logger.info("=" * 60)

    # Build logistics network
    net = LogisticsNetwork(trade_df, freight_df=fred_df if not fred_df.empty else None)
    G = net.build()
    lane_summary = net.get_lane_summary(G)
    logger.info("Lane summary:\n%s", lane_summary.to_string(index=False))

    vulnerable = net.get_vulnerable_lanes(G)
    if not vulnerable.empty:
        logger.info("Vulnerable lanes (reliability < 0.5):\n%s", vulnerable.to_string(index=False))

    # Detect disruptions
    detector = DisruptionDetector(trade_df, freight_df=fred_df if not fred_df.empty else None)
    disruptions = detector.detect_all()
    disrupt_df = detector.disruptions_to_dataframe(disruptions)
    logger.info("Detected %d disruptions", len(disruptions))

    # Optimise routes
    total_demand = trade_df["value_usd"].sum() / trade_df["date"].dt.to_period("M").nunique()
    lp_router = LPReRouter(lane_summary)
    lp_result = lp_router.optimise(demand=total_demand)
    logger.info("LP optimal allocation:\n%s", lp_result.to_string(index=False))

    # RL optimiser
    rl_opt = ReRoutingOptimizer(lane_summary, demand=total_demand)
    train_info = rl_opt.train(n_episodes=3000)
    logger.info("RL training: %s", train_info)

    # Compare for a "East Asia disrupted" scenario
    n_lanes = len(lane_summary)
    disrupted_state = tuple([0] * n_lanes)
    # Set East Asia lane to disruption level 2 (out of 4)
    for i, row in lane_summary.iterrows():
        if row["region"] == "East Asia":
            temp = list(disrupted_state)
            temp[i] = 2
            disrupted_state = tuple(temp)
    rl_recommendation = rl_opt.recommend(disrupted_state)
    logger.info("RL recommendation (East Asia disrupted):\n%s", rl_recommendation.to_string(index=False))

    return {
        "lane_summary": lane_summary,
        "vulnerable_lanes": vulnerable,
        "disruptions": disrupt_df,
        "lp_allocation": lp_result,
        "rl_training": train_info,
        "rl_recommendation": rl_recommendation,
    }


def run_demand_forecasting(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Module 3: Demand & Out-of-Stock Forecasting."""
    from src.demand_forecasting import QuantileForecaster, ShortageDetector, AdaptiveCalibrator

    logger.info("=" * 60)
    logger.info("MODULE 3: Demand & Shortage Forecasting")
    logger.info("=" * 60)

    if train_df.empty or test_df.empty:
        logger.warning("Insufficient feature data for demand forecasting")
        return {}

    target = "value_usd"
    if target not in train_df.columns:
        logger.warning("Target column '%s' not in training data", target)
        return {}

    # Quantile forecaster
    qf = QuantileForecaster(target_col=target)
    feature_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c != target
    ]
    qf.fit(train_df, feature_cols=feature_cols)

    predictions = qf.predict(test_df)
    logger.info("Quantile predictions shape: %s", predictions.shape)
    logger.info("Sample predictions:\n%s", predictions.head().to_string())

    # Feature importance
    fi = qf.feature_importance()
    logger.info("Top 10 features:\n%s", fi.head(10).to_string(index=False))

    # Shortage detection
    detector = ShortageDetector()
    scored = detector.flag_shortage_risks(
        predictions, train_df, target_col=target
    )
    n_flagged = scored["shortage_flag"].sum() if "shortage_flag" in scored.columns else 0
    logger.info("Shortage flags: %d out of %d forecasts", n_flagged, len(scored))

    # Calibration
    if target in test_df.columns:
        calibrator = AdaptiveCalibrator(quantiles=qf.quantiles)
        cal_df = calibrator.compute_calibration_error(
            test_df[target], predictions,
        )
        logger.info("Calibration:\n%s", cal_df.to_string(index=False))
        rec = calibrator.recommend_update(cal_df)
        logger.info("Calibration recommendation: %s", rec)
    else:
        cal_df = pd.DataFrame()
        rec = {}

    return {
        "predictions": predictions,
        "feature_importance": fi,
        "shortage_scores": scored,
        "calibration": cal_df,
        "calibration_recommendation": rec,
    }


def run_cost_monitoring(trade_df: pd.DataFrame, fred_df: pd.DataFrame) -> Dict[str, Any]:
    """Module 4: Inflation & Cost-Push Monitoring."""
    from src.cost_monitoring import CostTransmissionAnalyzer, ExternalSignalLoader, CostAlertSystem

    logger.info("=" * 60)
    logger.info("MODULE 4: Cost-Push Monitoring")
    logger.info("=" * 60)

    if fred_df.empty:
        logger.warning("No FRED data available; cost monitoring limited")

    # External signals
    if not fred_df.empty and "series_id" in fred_df.columns:
        wide = fred_df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
        wide = wide.sort_index().ffill()
    else:
        wide = pd.DataFrame()

    signals = ExternalSignalLoader(fred_wide_df=wide)

    # Cost transmission analysis
    trade_monthly = trade_df.groupby(pd.Grouper(key="date", freq="ME"))["value_usd"].sum()
    cost_series = {}
    for col in wide.columns:
        s = wide[col].dropna()
        if len(s) > 12:
            cost_series[col] = s

    results_dict: Dict[str, Any] = {}

    if cost_series:
        analyzer = CostTransmissionAnalyzer(trade_monthly, cost_series)

        # Granger causality
        granger = analyzer.granger_causality()
        if not granger.empty:
            sig = granger[granger["significant"]]
            logger.info("Significant Granger causes:\n%s", sig.to_string(index=False))
            results_dict["granger_causality"] = granger

        # Cost pressure index
        cpi = analyzer.compute_cost_pressure_index()
        results_dict["cost_pressure_index"] = cpi

        # Passthrough elasticity
        elasticity = analyzer.estimate_passthrough_elasticity()
        if not elasticity.empty:
            logger.info("Price pass-through:\n%s", elasticity.to_string(index=False))
            results_dict["passthrough_elasticity"] = elasticity

        # Variance decomposition
        decomp = analyzer.decompose_cost_drivers()
        if not decomp.empty:
            logger.info("Cost-driver decomposition:\n%s", decomp.to_string(index=False))
            results_dict["cost_decomposition"] = decomp

    # Alert system
    if not wide.empty:
        alert_sys = CostAlertSystem()
        alerts = alert_sys.check_alerts(wide)
        report = alert_sys.generate_alert_report(alerts)
        logger.info(
            "Alert report: %d elevated, red=%s",
            report["n_elevated"],
            report["summary"]["red"],
        )
        results_dict["alert_report"] = report

    return results_dict


# ====================================================================== #
#  Main orchestrator
# ====================================================================== #

MODULES = {
    "risk_propagation": "Module 1: Risk Propagation Simulation",
    "transportation":   "Module 2: Transportation Resilience",
    "demand_forecast":  "Module 3: Demand & Shortage Forecasting",
    "cost_monitoring":  "Module 4: Cost-Push Monitoring",
}


def run(module: str = "all") -> Dict[str, Any]:
    """
    Run one or all modules.

    Parameters
    ----------
    module : str
        ``"all"`` or one of: ``risk_propagation``, ``transportation``,
        ``demand_forecast``, ``cost_monitoring``.
    """
    logger.info("SCRAM Framework — Supply Chain Risk Analysis Model")
    logger.info("=" * 60)

    # Load shared data
    trade_df = _load_trade_data()
    fred_df = _load_fred_data()
    train_df = _load_feature_data("train")
    test_df = _load_feature_data("test")

    all_results: Dict[str, Any] = {}
    modules_to_run = list(MODULES.keys()) if module == "all" else [module]

    for m in modules_to_run:
        if m not in MODULES:
            logger.error("Unknown module: %s", m)
            continue

        logger.info("\n>>> Starting %s", MODULES[m])
        try:
            if m == "risk_propagation":
                all_results[m] = run_risk_propagation(trade_df)
            elif m == "transportation":
                all_results[m] = run_transportation_resilience(trade_df, fred_df)
            elif m == "demand_forecast":
                all_results[m] = run_demand_forecasting(train_df, test_df)
            elif m == "cost_monitoring":
                all_results[m] = run_cost_monitoring(trade_df, fred_df)
        except Exception as e:
            logger.error("Module %s failed: %s", m, e, exc_info=True)
            all_results[m] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("SCRAM Framework run complete.  Modules executed: %s", list(all_results.keys()))
    return all_results


# ====================================================================== #
#  CLI
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="SCRAM Framework Orchestrator")
    parser.add_argument(
        "--module", "-m",
        default="all",
        choices=["all"] + list(MODULES.keys()),
        help="Which module to run (default: all)",
    )
    args = parser.parse_args()
    run(args.module)


if __name__ == "__main__":
    main()
