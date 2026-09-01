# Module2 年度真实成本 vs 模型成本差距

对比年份: 2022
真实年度总运输成本 (USD M): 7,753.53
真实方式拆分 (USD M): truck=4,652.12, air=2,326.06, rail=775.35

## 1) 可比口径说明
- 真实年度成本来自 annual_actual_transportation_costs.csv。
- 模型当前是月度结果 (2022-01)，此处按 x12 年化后再比较。

## 2) 年化后模型与真实的差距
- Baseline: model_annual=6,952.78 USD M, gap=-800.75 USD M (-10.33%)
- Cost-Minimizing: model_annual=5,513.82 USD M, gap=-2,239.71 USD M (-28.89%)
- Resilience-First: model_annual=6,905.30 USD M, gap=-848.23 USD M (-10.94%)
- Targeted Allocation: model_annual=6,322.98 USD M, gap=-1,430.55 USD M (-18.45%)

## 3) 基线校准后的策略结论
- baseline->real 缩放系数: 1.115170
- Cost-Min 相对 Resilience 节约: 1,551.74 USD M
- Targeted 相对 Resilience 节约: 649.39 USD M
- 履约率差 (Resilience - Cost-Min): 5.040 个百分点
- 履约率差 (Resilience - Targeted): 1.764 个百分点