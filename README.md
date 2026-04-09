# SCRAM: Supply Chain Risk Analysis Model

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SCRAM** (Supply Chain Risk Analysis Model) is an enterprise-grade data collection and modeling framework for semiconductor supply chain risk analysis, based on U.S. public data sources.

## 📋 Project Overview

This project implements a comprehensive data-driven approach to:
- Monitor structural vulnerabilities in semiconductor supply chains
- Predict disruption events using historical patterns
- Quantify concentration risks and dynamic anomalies
- Provide actionable intelligence for supply chain resilience

### Project Scale

- **Data Coverage**: 15 years (2010-2024)
- **Training Data**: 37,756 records across 120 months (2010-2019)
- **Test Data**: 31,900 records across 60 months (2020-2024)
- **Countries Covered**: 214 countries/regions
- **HS Codes**: 9 semiconductor-related categories
- **Total Trade Value**: $1,692B in training period
- **Features Engineered**: 49 features across 5 categories
- **Trained Models**: 6+ model variants (XGBoost, LightGBM)

### Key Features

- ✅ **Modular Architecture**: Extensible design for easy addition of new data sources and models
- ✅ **Enterprise Best Practices**: Configuration management, logging, error handling, and data validation
- ✅ **Multi-Source Integration**: US Census, USGS, GSCPI, ISM PMI
- ✅ **Automated Data Pipeline**: Batch collection with rate limiting and retry mechanisms
- ✅ **Advanced Feature Engineering**: 49 features across concentration, volatility, growth, and temporal dimensions
- ✅ **Machine Learning Models**: XGBoost and LightGBM implementations with multiple loss functions
- ✅ **Robust Predictions**: Huber loss and grouped modeling for improved accuracy
- ✅ **Production-Ready**: Type hints, documentation, comprehensive error handling, and model persistence
- ✅ **Comprehensive Analysis**: Diagnostic scripts, validation tools, and performance monitoring

## 🏗️ Architecture

```
scram/
├── src/                          # Source code
│   ├── collectors/               # Data collection modules
│   │   ├── base_collector.py    # Abstract base class
│   │   ├── us_census_collector.py
│   │   ├── usgs_collector.py
│   │   └── macro_collector.py
│   ├── features/                 # Feature engineering
│   │   ├── base_feature.py      # Abstract feature class
│   │   ├── concentration_features.py  # HHI, Gini, etc.
│   │   ├── volatility_features.py     # CoV, std dev
│   │   ├── temporal_features.py       # Time-series
│   │   ├── growth_features.py         # Growth rates
│   │   └── feature_pipeline.py        # Complete pipeline
│   ├── models/                   # Model training & prediction
│   │   ├── base_model.py        # Abstract model class
│   │   └── time_series_model.py # XGBoost/LightGBM models
│   ├── utils/                    # Utility functions
│   │   └── common.py
│   └── config_manager.py         # Configuration management
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Data storage
│   ├── raw/                      # Raw collected data
│   └── processed/                # Processed data & features
├── models/                       # Trained models
│   └── trained/                  # Saved model files
│       └── grouped/              # HS code-specific models
├── reports/                      # Analysis reports
├── logs/                         # Application logs
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── main.py                       # Data collection orchestrator
├── train_baseline_models.py      # Model training script
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd scram

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# US_CENSUS_API_KEY=your_key_here
```

Get your free US Census API key: https://api.census.gov/data/key_signup.html

### 3. Run Data Collection

**Collect training data (2010-2019):**
```bash
python main.py --phase train --sources all
```

**Collect test data (2020-2024):**
```bash
python main.py --phase test --sources all
```

**Collect specific sources:**
```bash
# Census trade data only
python main.py --sources census --category integrated_circuits

# Macro indicators only
python main.py --sources macro --indicators gscpi pmi

# Custom date range
python main.py --phase custom --start-date 2020-01-01 --end-date 2024-12-31 --sources all
```

### 4. Extract Features

```bash
# Extract features from collected data
python feature_engineering_example.py

# Validate feature quality
python validate_features.py
```

### 5. Train Models

```bash
# Train baseline models
python train_baseline_models.py

# Train with different configurations
python step3_grouped_modeling.py  # Grouped by HS code
python step5_huber_loss.py        # Huber loss comparison
python step6_temporal_features.py  # With temporal features
```

### 6. Explore Results

```bash
# View model details
python explore_model.py

# Run model demo
python use_model_demo.py

# Check project status
bash check_progress.sh
```

## 📊 Data Sources

### 1. US Census Bureau Trade Data
- **Coverage**: HS codes 8542 (semiconductors), 8486 (equipment), 3818 (wafers), etc.
- **Frequency**: Monthly
- **Metrics**: Import/export value (USD), quantity, country of origin

### 2. USGS Mineral Data
- **Coverage**: Gallium, Germanium, Silicon
- **Frequency**: Annual (interpolated to monthly)
- **Metrics**: Production, imports, import dependence, prices

### 3. Macro Indicators
- **GSCPI**: Global Supply Chain Pressure Index (NY Fed)
- **ISM PMI**: Manufacturing PMI with supply chain components
- **Frequency**: Monthly

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Time ranges
time_range:
  train_start: "2010-01-01"
  train_end: "2019-12-31"
  test_start: "2020-01-01"
  test_end: "2024-12-31"

# HS codes to collect
hs_codes:
  integrated_circuits:
    - code: "854231"
      description: "Processors and controllers"
```

## 📈 Usage Examples

### Programmatic Usage

```python
from src.collectors import USCensusCollector, MacroIndicatorCollector

# Collect Census data
census = USCensusCollector(
    start_date='2020-01-01',
    end_date='2024-12-31',
    trade_type='imports'
)
df = census.collect(category='integrated_circuits')

# Collect macro indicators
macro = MacroIndicatorCollector(
    start_date='2020-01-01',
    end_date='2024-12-31'
)
macro_df = macro.collect(indicators=['gscpi', 'pmi'])
```

### Data Validation

All collectors include built-in validation:
- Required column checks
- Missing value analysis
- Outlier detection
- Date continuity validation

## 🎯 Feature Engineering

The project includes a comprehensive feature pipeline with 49 engineered features across 5 categories:

### Feature Categories

**1. Concentration Features (11 features)**
- Herfindahl-Hirschman Index (HHI) - current and rolling windows (3/6/12 months)
- Market share metrics (Top 1/3/5 suppliers)
- Gini coefficient for supply distribution
- Supplier count and diversity
- HHI change rates (MoM, YoY)

**2. Volatility Features (12 features)**
- Standard deviation for value and quantity
- Coefficient of Variation (CoV)
- Rolling volatility (3/6/12 months)
- Volatility trends and stability scores

**3. Temporal Features (16 features)**
- Time dimensions (month, quarter, year)
- Trend indicators and moving averages
- Lag features (1/3/6/12 months)
- Momentum and seasonality patterns

**4. Growth Features (10 features)**
- Growth rates (MoM, QoQ, YoY)
- Rolling average growth rates
- Compound Annual Growth Rate (CAGR)
- Growth acceleration metrics

**5. Additional Metadata (9 fields)**
- Date, HS code, country
- Trade value and quantity
- Data source and collection timestamps

### Using Features

```python
from src.features import FeaturePipeline

# Initialize pipeline
pipeline = FeaturePipeline()

# Extract features from raw data
features_df = pipeline.extract_all_features(raw_data)

# Features are automatically saved to data/processed/
```

## 🤖 Model Training & Usage

### Trained Models

The project includes several trained models located in `models/trained/`:

1. **Value Forecaster (XGBoost)**
   - `value_forecaster_xgb.json` - Standard MSE loss
   - `value_forecaster_xgb_huber.json` - Huber loss (robust to outliers)
   - `value_forecaster_xgb_log.json` - Log-transformed targets

2. **HHI Forecaster (LightGBM)**
   - `hhi_forecaster_lgb.json` - Concentration risk prediction

3. **Grouped Models**
   - `grouped/` - Separate models trained for each HS code category

### Using Trained Models

```python
from src.models import TimeSeriesForecaster
import pandas as pd

# Load a trained model
model = TimeSeriesForecaster(model_type='xgboost')
model.load('models/trained/value_forecaster_xgb_huber.json')

# Make predictions
predictions = model.predict(test_features)

# Evaluate performance
metrics = model.evaluate(test_features, test_targets)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"R²: {metrics['r2']:.3f}")
```

### Training New Models

```python
# Train a new model
python train_baseline_models.py --model xgboost --target value --loss huber

# Train grouped models (one per HS code)
python step3_grouped_modeling.py

# Compare model performance
python step5_huber_loss.py
```

## 📈 Model Performance

### HHI Concentration Forecaster ⭐

**Best Performing Model**: LightGBM for HHI prediction
- **R²**: 0.66 (explains 66% of variance)
- **MAPE**: 5.49% (highly accurate)
- **MAE**: 59.13
- **Use Case**: Supply chain concentration risk early warning

This model successfully captures HHI trend changes and can predict concentration risk 1 month ahead, enabling proactive supplier diversification decisions.

### Trade Value Forecaster

**Model Evolution**:
1. **Baseline XGBoost**: R² = 0.024, MAE = $65.6M
2. **Log-transformed**: MAE improved to $41.1M (37% reduction)
3. **Huber Loss**: More robust to outliers, MAE = $41.3M
4. **Grouped Models**: Separate models per HS code category

**Key Insights**:
- Trade value predictions are challenging due to high variance across different scales
- Log transformation improves relative error metrics (MAPE)
- Huber loss provides robustness against extreme values
- 7.8% of high-value samples contribute 86% of prediction errors

**Recommendations for Use**:
- Use HHI forecaster for concentration risk monitoring
- Use value forecaster with Huber loss for robust predictions
- Consider grouped models when analyzing specific product categories
- Focus on relative changes (%) rather than absolute values for value predictions

### Model Development Journey

The project followed an iterative improvement process documented in detail:

**Step 1: Data Exploration** (`step1_data_exploration.py`)
- Discovered high skewness (6.37) in trade value distribution
- Identified long-tail distribution pattern

**Step 2: Log Transformation** (`step2_log_transformation.py`)
- Applied Box-Cox transformation
- Reduced MAE by 37% ($65.6M → $41.1M)

**Step 3: Grouped Modeling** (`step3_grouped_modeling.py`)
- Created separate models for each HS code
- Tested specialist vs. generalist approach

**Step 4: Diagnosis** (`step4_diagnosis.py`)
- Root cause analysis showing 7.8% of samples contribute 86% of errors
- Identified systematic underestimation in high-value predictions

**Step 5: Huber Loss** (`step5_huber_loss.py`)
- Implemented robust regression for outlier handling
- Improved prediction stability

**Step 6: Temporal Features** (`step6_temporal_features.py`)
- Added time-series specific features
- Enhanced trend and seasonality capture

## 📊 Analysis Scripts

The project includes several analysis and diagnostic scripts:

- `explore_model.py` - Inspect trained model details
- `feature_engineering_example.py` - Feature extraction examples
- `validate_features.py` - Feature quality validation
- `diagnose_temporal_failure.py` - Temporal feature diagnostics
- `quantity_analysis.py` - Quantity data analysis
- `step4_diagnosis.py` - Model performance diagnosis
- `use_model_demo.py` - Demo for using trained models

## 🛠️ Utility Scripts

**Environment & Setup:**
```bash
# Activate virtual environment (macOS/Linux)
source activate_env.sh

# Verify setup
python verify_setup.py
python check_setup.py
```

**Monitoring & Progress:**
```bash
# Check data collection progress
bash check_progress.sh
python monitor_collection.py

# View project status
cat PROJECT_STATUS.md
cat FINAL_SUMMARY.md
```

**Data Exploration:**
```bash
# Explain collected data
python data_explanation.py

# Analyze zero quantities
python quantity_zero_explanation.py

# Explore model files
python explain_model_file.py
```

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 🔧 Troubleshooting

### Virtual Environment Issues

**Problem**: Architecture mismatch on Apple Silicon (M1/M2/M3)
```bash
# Solution: Create ARM64 native environment
python3 -m venv venv_arm64
source venv_arm64/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: XGBoost/LightGBM OpenMP errors
```bash
# Solution: Install OpenMP runtime
brew install libomp
```

### Data Collection Issues

**Problem**: US Census API rate limiting
- Check your API key is valid
- Reduce `requests_per_minute` in `config/config.yaml`
- Use `--sources` flag to collect specific sources separately

**Problem**: Missing data files
```bash
# Verify data collection
python check_setup.py
bash check_progress.sh
```

### Model Training Issues

**Problem**: "No module named 'src'"
```bash
# Ensure you're in the project root directory
cd /path/to/scram
python train_baseline_models.py
```

**Problem**: Out of memory errors
- Reduce batch size in training scripts
- Use grouped modeling by HS code
- Process data in chunks

## 📁 Output Data Format

Collected data is saved in Parquet format with the following structure:

**Trade Data:**
```
date | hs_code | country | value_usd | quantity | trade_type | data_source
```

**Mineral Data:**
```
date | mineral | us_production | us_imports | import_dependence_pct | data_source
```

**Macro Data:**
```
date | indicator | value | data_source
```

## 📚 Documentation

The project includes comprehensive documentation files:

**Setup & Configuration:**
- `API_KEYS_GUIDE.md` - How to obtain and configure API keys
- `DATA_COLLECTION_GUIDE.md` - Detailed guide for data collection
- `requirements.txt` - Python package dependencies

**Progress Reports:**
- `PROJECT_STATUS.md` - Overall project status and milestones
- `FINAL_SUMMARY.md` - Comprehensive project summary
- `DATA_COLLECTION_STATUS.md` - Data collection progress
- `DATA_COLLECTION_REPORT.md` - Detailed collection results
- `TEST_DATA_COLLECTION_SUCCESS.md` - Test phase collection summary

**Feature Engineering:**
- `FEATURE_ENGINEERING_COMPLETE.md` - Feature engineering documentation
- Feature extraction examples in `feature_engineering_example.py`

**Model Training:**
- `MODEL_TRAINING_PLAN.md` - Training strategy and plan
- `TRAINING_SUCCESS_REPORT.md` - Initial training results
- `MODEL_IMPROVEMENT_REPORT.md` - Model optimization journey (Steps 1-5)
- `NEXT_STEPS.md` - Future improvement roadmap

**Analysis Reports:**
- `reports/step4_diagnosis.json` - Model diagnostic results
- `models/trained/huber_loss_comparison.json` - Loss function comparison
- `models/trained/value_forecaster_xgb_log_comparison.json` - Transformation comparison

## 🛣️ Project Status

### Phase 1: Data Collection ✅ (Completed)
- [x] US Census trade data collector
- [x] USGS mineral data collector
- [x] Macro indicator collector
- [x] Configuration management
- [x] Logging and error handling
- [x] Training data: 37,756 records (2010-2019)
- [x] Test data: 31,900 records (2020-2024)

### Phase 2: Feature Engineering ✅ (Completed)
- [x] HHI concentration index calculation
- [x] CoV volatility metrics
- [x] Time-series feature engineering (49 features total)
- [x] Growth features (MoM, YoY, CAGR)
- [x] Temporal features (trends, seasonality, momentum)
- [x] Feature validation and pipeline

### Phase 3: Model Training ✅ (Completed)
- [x] XGBoost regression models
- [x] LightGBM models  
- [x] Huber loss implementation for robustness
- [x] Grouped modeling by HS code
- [x] Model persistence and versioning
- [x] Evaluation metrics (MAE, RMSE, R², MAPE)

### Phase 4: Analysis & Validation (In Progress)
- [x] Model diagnostics and comparison
- [x] Feature importance analysis
- [ ] 2020-2024 comprehensive backtesting
- [ ] Crisis detection accuracy evaluation
- [ ] Production deployment preparation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

**Data Collection:**
- Additional data sources (shipping indices, news sentiment, pricing data)
- Real-time data streaming capabilities
- Automated data quality reports
- Integration with cloud storage (S3, GCS, Azure)

**Feature Engineering:**
- Network analysis features (supply chain graphs)
- Geopolitical risk indicators
- Economic policy uncertainty indices
- Sentiment analysis from trade publications

**Modeling:**
- LSTM/GRU for sequence modeling
- Ensemble methods combining multiple models
- Anomaly detection for crisis events
- Multi-step forecasting (3/6/12 months ahead)

**Infrastructure:**
- Docker containerization
- CI/CD pipeline
- Model monitoring and drift detection
- REST API for model serving

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

This project implements analytical methods for semiconductor supply chain risk assessment, focusing on:
- **Structural concentration risk** using Herfindahl-Hirschman Index (HHI)
- **Dynamic anomaly detection** through Coefficient of Variation (CoV)
- **Leading indicators** from equipment import data
- **Multi-source data fusion** from public U.S. datasets

**Data Sources:**
- U.S. Census Bureau International Trade Data
- USGS National Minerals Information Center
- Federal Reserve Bank of New York (GSCPI)
- Institute for Supply Management (ISM PMI)

Built with Python, XGBoost, LightGBM, scikit-learn, and pandas.

---

**Note**: This is a data collection framework. Actual API responses may vary from examples shown. Always verify data quality and refer to official documentation of data sources.
