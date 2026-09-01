# Module2 HS 854231 真实值 vs 模型结果

## 对比对象
- HS 码: 854231
- 选定模型策略: Targeted Allocation (targeted)
- 模型场景标签: stress

## 真实值
- 2022 年 raw HS 854231 总额 (USD): 20,000,000,000.00
- 2022 年 raw 行数: 51
- 2022 年运输成本总额 (USD M): 7,753.53
- 月化运输成本代理 (USD): 646,127,500.00

## 模型结果
- 2022 年 processed HS 854231 总额 (USD): 20,000,000,000.00
- 2022 年 processed 行数: 51
- total_logistics_cost_usd: 742,593,070.75
- fill_rate: 0.770280
- delivered_usd: 15,405,600,000.00
- unmet_usd: 4,594,400,000.00

## 真实-模型偏差
- 854231 真实值 vs 模型值: 0.00 USD (0.0000%)
- 月化成本偏差: 96,465,570.75 USD (14.9298%)

## 说明
- This report focuses only on HS 854231.
- The real baseline uses the raw 2022 Module2 HS 854231 demand total.
- The model-side demand uses the processed 2022 HS 854231 total feeding the optimizer.
- Monthly model cost is compared against the monthlyized 2022 real transportation cost proxy.