# Module 3 Integrated Supply, Shortage, and Inventory Demo

## Purpose

This experiment connects Module 1 supply-gap risk to Module 3 multi-horizon
supply forecasts. It separates two operational views:

- The calibrated Module 3 median is the expected supply forecast.
- The Module 1-adjusted distribution is a conservative stress-warning view.

The stress view should not replace the expected forecast. On the 2020-2024
out-of-time test, applying the full Module 1 scenario gap increased point MAE,
while improving shortage recall in several horizons.

## Data and timing

- Training period: 2010-2019.
- Quantile calibration period: 2017-2019, with no 2020+ outcomes used.
- Test period: 2020-2024.
- Products: four HS codes present in every train and test month.
- Forecast horizons: 1, 3, and 6 months.
- Module 1 risk: matched from the forecast origin quarter, not the future target
  quarter, to avoid using a risk estimate that was unavailable at forecast time.

The demand baseline is the trailing 12-month mean trade value by HS code. It is
a demand proxy, not customer orders. Inventory results are standardized scenario
buffers, not company inventory records.

## Main findings

| Horizon | Baseline MAE | Stress MAE | Baseline recall | Stress recall | Baseline q10-q90 coverage |
|---|---:|---:|---:|---:|---:|
| 1 month | $805.6M | $1,105.8M | 44.7% | 72.3% | 77.1% |
| 3 months | $937.6M | $1,205.8M | 16.3% | 24.5% | 74.1% |
| 6 months | $1,045.9M | $1,363.4M | 30.9% | 34.6% | 67.6% |

Pre-test residual calibration raised baseline interval coverage from roughly
42-44% before calibration to 68-77% after calibration.

For the six-month persistent-shortfall scenario:

- A one-month initial buffer produced a 32.4% stress stockout rate versus a
  31.0% observed-proxy rate.
- A two-month initial buffer produced a 9.7% stress stockout rate versus an
  11.1% observed-proxy rate.
- A three-month buffer produced no stress stockouts, while the observed proxy
  produced 3.7%; the stress model still misses some tail events.

## Outputs

- `module3_integrated_predictions.csv`: baseline, stress-adjusted, actual-proxy,
  demand-proxy, and shortage fields.
- `module3_inventory_scenarios.csv`: one-, two-, and three-month buffer scenarios.
- `module3_integrated_metrics.json`: out-of-time accuracy, interval, shortage,
  and inventory metrics.
- `figures/module3_supply_forecast_integration.png`: forecast paths.
- `figures/module3_shortage_inventory_impact.png`: MAE, recall, and stockout impact.

## Reproduce

```bash
python3 scripts/training/train_module3_integrated.py
```