# Module2 vs Real Public Data Comparison

Data month used for comparison: 2022-01-01

## 1) Real/Public Input Baseline (from your new data)
- Total demand USD (state panel): 20,000,000,000.00
- State count: 51
- Lane weighted average unit_cost: 0.170061
- Lane weighted average transit_days: 1.000000
- Lane weighted average reliability: 0.041960
- Lane rows in sample month: 13249

## 2) Model Scenario Outputs (current run)
- Baseline | scenario=baseline | fill_rate=1.000000 | total_logistics_cost_usd=579,398,375.81 | avg_lead_time_days=7.798027 | air_express_share=0.000000
- Cost-Minimizing | scenario=stress | fill_rate=0.737520 | total_logistics_cost_usd=459,484,646.68 | avg_lead_time_days=7.766301 | air_express_share=0.000000
- Resilience-First | scenario=stress | fill_rate=0.787920 | total_logistics_cost_usd=575,441,607.04 | avg_lead_time_days=7.381565 | air_express_share=0.023508
- Targeted Allocation | scenario=stress | fill_rate=0.770280 | total_logistics_cost_usd=526,914,670.75 | avg_lead_time_days=7.909519 | air_express_share=0.013741

## 3) Side-by-side what can be compared now
- Can compare directly: demand scale consistency (real demand vs model total_demand_usd), transport cost/time scale sanity (real lane unit_cost/transit proxy vs model output cost/time levels).
- Cannot directly claim yet: realized fulfillment lift and realized savings, because no observed post-decision execution outcomes table is present (actual delivered/unmet/cost by route and time).

## 4) Quantitative consistency checks
- Demand consistency: model baseline total_demand_usd=20,000,000,000.00 vs real state demand=20,000,000,000.00; gap=0.00 (0.0000%).
- Model-estimated tradeoff in this real-data-calibrated run: Cost-Min saves 115,956,960.36 USD vs Resilience-First, with fill-rate change of 5.040 percentage points.

## 5) Real-data mode-level stats (for calibration transparency)
- mode=truck: unit_cost_wavg=0.170061, transit_days_wavg=1.000000, reliability_wavg=0.041960, capacity_weight=437765.799994