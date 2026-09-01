# Module2 Real vs Model: Audit Explanation

## 1) What this chart compares
- This is a single-model comparison against real values only.
- Selected model result: Targeted Allocation (targeted).
- No model-vs-model deltas are included.

## 2) Data sources
- Real demand baseline: reports/module2/module2_model_metrics.csv (metric=state_panel_total_demand_usd).
- Model outputs: reports/module2/module2_scenario_comparison.csv.
- Real annual transport cost proxy: reports/module2/annual_actual_transportation_costs.csv (Year=2022).
- Comparison summary input: reports/module2/module2_real_vs_model_only.json.

## 3) Calculation steps
1. Demand gap
- real_demand_usd = state_panel_total_demand_usd
- model_demand_usd = selected_model.total_demand_usd
- demand_gap_usd = model_demand_usd - real_demand_usd
- demand_gap_pct = demand_gap_usd / real_demand_usd * 100

2. Monthly cost gap
- real_annual_cost_usd_m = Actual_Total_Transportation_Cost_USD_M (2022)
- real_monthly_cost_proxy_usd = real_annual_cost_usd_m * 1,000,000 / 12
- model_monthly_cost_usd = selected_model.total_logistics_cost_usd
- monthly_cost_gap_usd = model_monthly_cost_usd - real_monthly_cost_proxy_usd
- monthly_cost_gap_pct = monthly_cost_gap_usd / real_monthly_cost_proxy_usd * 100

3. Model fulfillment composition
- delivered_usd = selected_model.delivered_usd
- unmet_usd = selected_model.unmet_usd
- fill_rate = delivered_usd / (delivered_usd + unmet_usd)

## 4) Numbers used in this run
- real_demand_usd = 20,000,000,000.00
- model_demand_usd = 20,000,000,000.00
- demand_gap_usd = 0.00 (0.0000%)
- real_monthly_cost_proxy_usd = 646,127,500.00
- model_monthly_cost_usd = 742,593,070.75
- monthly_cost_gap_usd = 96,465,570.75 (14.9298%)
- delivered_usd = 15,405,600,000.00
- unmet_usd = 4,594,400,000.00
- fill_rate = 0.770280

## 5) Output files
- Figure: reports/module2/figures/module2_real_vs_model_only.png
- Calculation table: reports/module2/module2_real_vs_model_only_calc_table.csv
- This explanation: reports/module2/module2_real_vs_model_only_explained.md