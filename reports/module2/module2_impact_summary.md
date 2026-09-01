# Module2 Impact Summary

## 1) We optimized how many routes
- Trunk-route search space (Port->DC multimodal): 27 routes
- Baseline active routes: 18
- Cost-Minimizing active routes: 24 (+6 vs baseline)
- Resilience-First active routes: 24 (+6 vs baseline)
- Targeted Allocation active routes: 24 (+6 vs baseline)

## 2) How much efficiency improved
- Fill-rate gain (Resilience vs Cost-Min): 5.040 percentage points
- Fill-rate gain (Targeted vs Cost-Min): 3.276 percentage points
- Lead-time improvement (Resilience vs Cost-Min): 1.493%
- Lead-time change (Targeted vs Cost-Min): -0.556%

## 3) How much money can be saved
- Cost-Minimizing saves vs Resilience-First: $130,068,960.36
- Targeted Allocation saves vs Resilience-First: $53,466,136.29

## 4) What Module2 does and why
- Built a two-leg network optimization model (Port->DC->State) with rail/truck/air modes.
- Added stress-shock simulation and rerouting to evaluate resilience under disruptions.
- Added state-level fulfillment and unmet-demand outputs to support service governance.
- Added bottleneck shadow-price diagnostics to identify high-ROI capacity expansions.

## 5) Current result interpretation
- Cost-Minimizing is the lowest-cost option but yields lower fill rate.
- Resilience-First achieves the best fill rate and faster lead time, with higher cost.
- Targeted Allocation is the practical middle ground for service-cost balance.