# SCRAM - Supply Chain Risk Analysis Model

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**SCRAM** is an open-source framework for analysing and predicting structural risks in complex supply chains. It translates recurring coordination challenges - concentration risk, disruption propagation, logistics fragility, cost-push transmission - into reusable predictive and decision-support modules.

The primary domain is the **U.S. semiconductor import supply chain**, using 15 years of public trade data (2010-2024) from the U.S. Census Bureau, FRED, and the NY Fed. The framework is designed to be **generalizable**: its modular components work across supply-chain verticals where similar coordination problems exist.

---

## Architecture

```
SCRAM Framework
|-- Data Layer
|   |-- Collectors          US Census, FRED, USGS, NY Fed GSCPI
|   |-- Data Integration    Mixed-data loader, Provenance tracking, Quality checks
|   |-- Feature Pipeline    49 engineered features (concentration, volatility, temporal, growth)
|
|-- Module 1: Risk Propagation Simulation
|   |-- Graph Network       Directed weighted supply-chain graph (NetworkX)
|   |-- Propagation Engine  Node/regional shocks with substitution elasticity
|   |-- Stress Testing      Scenario library + historical back-testing
|
|-- Module 2: Transportation Resilience Optimisation
|   |-- Lane Network        Bipartite logistics graph (region -> USA)
|   |-- Disruption Detector Volume-drop and cost-spike anomaly detection
|   |-- Re-Routing          Q-learning agent + LP solver for flow allocation
|
|-- Module 3: Demand and Shortage Forecasting
|   |-- Quantile Forecaster XGBoost with reg:quantileerror (prediction intervals)
|   |-- Shortage Detector   Supply-demand ratio scoring and risk flags
|   |-- Adaptive Calibration Coverage evaluation and self-correcting feedback
|
|-- Module 4: Cost-Push Monitoring
|   |-- Cost Transmission   Granger causality, pass-through elasticity, variance decomposition
|   |-- External Signals    Normalised commodity, freight, and macro indicators
|   |-- Alert System        Dynamic threshold alerts (green/yellow/orange/red)
|
|-- Framework Orchestrator   Run modules individually or as a coordinated pipeline
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Rosievv/S-NSRT-Model.git
cd S-NSRT-Model

# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) set API keys in .env
cp .env.example .env
# edit .env with US_CENSUS_API_KEY and FRED_API_KEY

# Run the full framework
python -m src.framework --module all

# Or run a single module
python -m src.framework --module risk_propagation
python -m src.framework --module transportation
python -m src.framework --module demand_forecast
python -m src.framework --module cost_monitoring
```

---

## Modules in Detail

### Module 1 - Risk Propagation Simulation

Models the semiconductor supply network as a **directed weighted graph** where nodes represent supplier countries and edges carry trade-flow values. Simulates how localised disruptions (e.g., a key supplier going offline) cascade through the Tier-N network.

| Component | Description |
|---|---|
| `SupplyChainNetwork` | Builds temporal graph snapshots from Census trade data; computes centrality (betweenness, eigenvector, PageRank), identifies critical nodes |
| `PropagationEngine` | Simulates node shocks and regional disruptions with configurable substitution elasticity; quantifies supply gaps per HS code |
| `StressTestRunner` | Executes predefined scenarios (China decoupling, Taiwan crisis, East Asia disruption) and back-tests against real historical events (COVID-19, 2011 Japan earthquake) |

```python
from src.risk_propagation import SupplyChainNetwork, StressTestRunner

net = SupplyChainNetwork(trade_df)
G = net.build_network(period="2019-06")
critical = net.identify_critical_nodes(G, threshold=0.05)

runner = StressTestRunner(trade_df)
results = runner.run_all_standard()
backtest = runner.backtest_all()
```

### Module 2 - Transportation Resilience Optimisation

Models logistics lanes between supplier regions and the US, detects disruptions from anomalous trade-volume drops and freight-cost spikes, and dynamically re-routes flows using two complementary approaches:

| Component | Description |
|---|---|
| `LogisticsNetwork` | Bipartite graph (supply regions to USA) with capacity, cost, transit time, and reliability attributes derived from trade variability and freight indices |
| `DisruptionDetector` | Statistical anomaly detection: MoM volume drops greater than 30% and freight cost greater than 2 sigma from rolling mean |
| `LPReRouter` | Linear-programming solver minimising weighted cost + time + reliability penalty subject to capacity and demand constraints |
| `ReRoutingOptimizer` | Tabular Q-learning agent trained over simulated disruption episodes; learns allocation policies that balance cost and resilience |

```python
from src.transportation import LogisticsNetwork, ReRoutingOptimizer

net = LogisticsNetwork(trade_df, freight_df=fred_df)
G = net.build()
lanes = net.get_lane_summary(G)

optimizer = ReRoutingOptimizer(lanes, demand=total_monthly_demand)
optimizer.train(n_episodes=5000)
recommendation = optimizer.recommend(disruption_state=(0, 2, 0, 0, 0))
```

### Module 3 - Demand and Shortage Forecasting

Produces **prediction intervals** (not just point estimates) using XGBoost quantile-error objective, enabling principled shortage-risk scoring.

| Component | Description |
|---|---|
| `QuantileForecaster` | Trains separate XGBoost models for each quantile (q10, q25, q50, q75, q90); enforces monotonicity across quantiles |
| `ShortageDetector` | Computes supply-demand ratio against historical baseline; generates composite shortage-risk scores (gap + trend + uncertainty) |
| `AdaptiveCalibrator` | Evaluates whether prediction intervals are well-calibrated (does q90 actually cover 90% of outcomes?) and applies corrections via the adaptive feedback loop |

```python
from src.demand_forecasting import QuantileForecaster, ShortageDetector

qf = QuantileForecaster(quantiles=[0.10, 0.50, 0.90])
qf.fit(train_df, feature_cols=feature_cols)
predictions = qf.predict(test_df)  # columns: q10, q50, q90, prediction_interval_width

detector = ShortageDetector(shortage_threshold=0.80)
scored = detector.flag_shortage_risks(predictions, train_df)
```

### Module 4 - Cost-Push Monitoring

Traces how upstream cost shocks (commodity prices, freight, PPI) propagate to downstream semiconductor trade values with time lags.

| Component | Description |
|---|---|
| `CostTransmissionAnalyzer` | Granger causality tests, pass-through elasticity estimation (OLS on log-returns at multiple lags), cost-pressure composite index |
| `ExternalSignalLoader` | Domain-specific FRED data wrapper providing normalised commodity, freight, and macro series |
| `CostAlertSystem` | Dynamic-threshold alerts using rolling z-scores; traffic-light levels (green to yellow to orange to red) |

```python
from src.cost_monitoring import CostTransmissionAnalyzer, CostAlertSystem

analyzer = CostTransmissionAnalyzer(trade_monthly, cost_series)
granger = analyzer.granger_causality()
elasticity = analyzer.estimate_passthrough_elasticity(lags=[1, 3, 6])

alerts = CostAlertSystem().check_alerts(indicator_df)
report = CostAlertSystem().generate_alert_report(alerts)
```

---

## Data Sources and Mixed-Data Logic

The framework uses a **mixed-data design**: where available, internal operational data can support more detailed modelling; where not, public and proxy signals provide usable inputs with graceful degradation.

| Source | Type | Update Frequency | Used By |
|---|---|---|---|
| US Census International Trade API | Public | Monthly | All modules |
| FRED (PPI, CPI, PMI, freight) | Public | Monthly | Cost monitoring, transportation |
| NY Fed GSCPI | Public | Monthly | Transportation, cost monitoring |
| USGS Mineral Commodity Summaries | Public | Annual | Risk propagation (planned) |
| Internal procurement / ERP data | Internal | Daily | All modules (enterprise mode) |

The `DataRegistry` catalogues each source with reliability scores and fallback chains. The `MixedDataLoader` automatically falls back to proxy sources when primary data is unavailable and logs provenance for every data element.

```python
from src.data_integration import MixedDataLoader

loader = MixedDataLoader()
trade_df, provenance = loader.load("us_census_trade", fallback=True)
all_data = loader.load_all_available()
print(loader.get_provenance_log())
```

---

## Feature Engineering

49 features across four extractors, all operating on the (date, hs_code, country) grain:

| Extractor | Features | Key Metrics |
|---|---|---|
| **Concentration** (11) | HHI, HHI rolling windows, top-N supplier share, Gini coefficient, supplier count | Market concentration risk |
| **Volatility** (12) | Rolling std/CoV (3/6/12m), volatility trend, stability score | Supply stability |
| **Temporal** (16) | Lags (1/3/6/12m), moving averages, momentum, seasonality index | Time-series patterns |
| **Growth** (10) | MoM/QoQ/YoY growth, CAGR, rolling growth averages, acceleration | Trend dynamics |

---

## Trained Models

| Model | Algorithm | Target | Test R-squared | Test MAPE | Status |
|---|---|---|---|---|---|
| HHI Forecaster | LightGBM | HHI concentration | 0.659 | 5.49% | Production-ready |
| Value Forecaster (log) | XGBoost | log(trade value) | -0.10 | 22.4% | Experimental |
| Value Forecaster (Huber) | XGBoost | log(trade value) | -0.22 | 22.4% | Experimental |
| Quantile Forecaster | XGBoost | trade value intervals | - | - | New module |
| Grouped Models (x4 HS) | XGBoost | per-HS trade value | - | - | Experimental |

---

## Project Structure

```
scram/
  src/
    framework.py                  Top-level orchestrator
    config_manager.py             YAML + env configuration
    collectors/                   Data collection
      base_collector.py
      us_census_collector.py      US Census trade API
      fred_collector.py           FRED economic data
      macro_collector.py          GSCPI + PMI
      usgs_collector.py           USGS minerals (scaffold)
    features/                     Feature engineering
      feature_pipeline.py
      concentration_features.py
      volatility_features.py
      temporal_features.py
      growth_features.py
    models/                       Base model abstractions
      base_model.py
      time_series_model.py
    risk_propagation/             Module 1
      graph_network.py
      propagation_engine.py
      stress_testing.py
    transportation/               Module 2
      lane_network.py
      disruption_detector.py
      rl_optimizer.py
    demand_forecasting/           Module 3
      quantile_forecaster.py
      shortage_detector.py
      adaptive_calibration.py
    cost_monitoring/              Module 4
      cost_transmission.py
      external_signals.py
      alert_system.py
    data_integration/             Mixed-data layer
      data_registry.py
      mixed_data_loader.py
      data_quality.py
    utils/
      common.py
  config/
    config.yaml
  data/
    raw/                          Collected Parquet files
    processed/                    Engineered features
  models/trained/                 Serialised models
  notebooks/                      Exploratory analysis
  reports/                        Generated reports
  tests/
  requirements.txt
  LICENSE                         Apache 2.0
  README.md
```

---

## Data Coverage

- **Time range**: 2010-01 to 2024-12 (15 years)
- **Training period**: 2010-2019 (37,756 records, 120 months)
- **Test period**: 2020-2024 (31,900 records, 60 months)
- **Countries**: 214 supplier countries/regions
- **HS codes**: 9 semiconductor-related categories (ICs, equipment, raw materials)
- **Total trade value**: USD 1.69 trillion (training period)

---

## Configuration

All settings are centralised in config/config.yaml. API keys are loaded from a .env file (see .env.example).

```yaml
# Required API keys (set in .env)
US_CENSUS_API_KEY: your_key_here
FRED_API_KEY: your_key_here
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type check
mypy src/

# Lint
flake8 src/ tests/
```

---

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
