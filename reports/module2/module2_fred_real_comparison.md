# Module2 与 FRED 真实宏观数据对比

## 1) 真实数据来源
- CAPUTLG334S: FRED 官方序列（计算机与电子制造业产能利用率）
- RETAILIRSA: FRED 官方序列（美国零售库存销售比）
- 获取方式：FRED 官方 CSV 接口

## 2) 本次对齐月份
- 模型需求基准月份: 2022-01-01
- CAPUTLG334S 使用月份: 2022-01-01，值: 76.0775，历史 zscore: -0.1961
- RETAILIRSA 使用月份: 2022-01-01，值: 1.1900，历史 zscore: -2.4586
- 宏观紧张度代理 tightness = z(cap) - z(inv) = 2.2625

## 3) 模型结果（同一轮）
- Baseline: fill_rate=1.000000, cost=579,398,375.81 USD
- Cost-Minimizing: fill_rate=0.737520, cost=459,484,646.68 USD
- Resilience-First: fill_rate=0.787920, cost=575,441,607.04 USD
- Targeted Allocation: fill_rate=0.770280, cost=526,914,670.75 USD

## 4) 真实宏观条件下的模型权衡
- Cost-Min 相对 Resilience 的模型估计节约: 115,956,960.36 USD
- 同时履约率差异: -5.040 个百分点（Cost-Min 更低）

## 5) 结论边界
- 以上 FRED 数据是真实美国宏观观测值。
- 对比结果是‘真实宏观数据校准下的模型估计结果’。
- 若要声称‘真实已实现节约’，仍需企业执行后订单级实绩对账。