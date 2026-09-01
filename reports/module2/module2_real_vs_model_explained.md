# Module2: 真实美国数据 vs 模型结果说明（汇报版）

## 1) 真实数据从哪里来（你提供的数据）
本次“真实美国情况”来自你提供并已落地在项目中的政府公开数据处理结果，主要读取以下文件：

- `data/processed/domestic_demand_state_assumption.csv`
  - 用途：州级需求分布（按月，`month` + `demand_zone` + `demand_value_usd`）
  - 本次对比使用月份：`2022-01`
- `data/processed/domestic_lanes.csv`
  - 用途：线路与方式的现实代理参数（`mode`, `unit_cost`, `transit_days`, `reliability`, `capacity`）
  - 本次对比使用月份：`2022-01`

说明：这些是“真实世界公开数据”的输入侧事实，不是模拟生成。

---

## 2) 哪些数据是模型来的
模型输出结果来自 Module2 求解后生成的报告文件：

- `reports/module2/module2_scenario_comparison.csv`
  - 包含 `Baseline / Cost-Minimizing / Resilience-First / Targeted Allocation`
  - 关键字段：`fill_rate`, `total_logistics_cost_usd`, `avg_lead_time_days`, `air_express_share`, `delivered_usd`, `unmet_usd`

- `reports/module2/module2_strategy_tradeoff.csv`
  - 用于策略间成本-履约权衡比较

说明：这些是“模型求解输出”，属于 optimization result，而不是直接观测值。

---

## 3) 本次对比如何做（口径）
本次口径是“真实公开数据校准 + 模型策略对比”：

1. 用真实州级需求总量校验模型需求规模是否一致。
2. 用真实线路参数（成本/时效/可靠性代理）校验模型量级是否合理。
3. 在同一需求规模下比较不同优化策略输出，得到模型估计的节约与履约权衡。

这属于“real-data-calibrated model comparison”，不是“企业执行后财务已实现对账”。

---

## 4) 对比结果（关键数字）

### A. 真实数据基线（2022-01）
- 真实州级总需求：`20,000,000,000.00 USD`
- 州数量：`51`
- 线路加权平均 `unit_cost`：`0.170061`
- 线路加权平均 `transit_days`：`1.000000`
- 线路加权平均 `reliability`：`0.041960`
- 线路样本行数：`13,249`

### B. 模型输出（同一轮运行）
- Baseline：
  - `fill_rate = 1.000000`
  - `total_logistics_cost_usd = 579,398,375.81`
- Cost-Minimizing（stress）：
  - `fill_rate = 0.737520`
  - `total_logistics_cost_usd = 459,484,646.68`
- Resilience-First（stress）：
  - `fill_rate = 0.787920`
  - `total_logistics_cost_usd = 575,441,607.04`
- Targeted Allocation（stress）：
  - `fill_rate = 0.770280`
  - `total_logistics_cost_usd = 526,914,670.75`

### C. 真实 vs 模型一致性检查
- 模型 baseline 总需求：`20,000,000,000.000008`
- 真实州级总需求：`20,000,000,000.00`
- 差异：约 `0.0000%`（仅浮点误差）

结论：模型需求规模与真实数据规模高度一致，说明输入校准是对齐的。

### D. 模型在真实数据校准下给出的策略结论
- Cost-Min 相比 Resilience-First：
  - 模型估计节约：`115,956,960.36 USD`
  - 履约率变化：`-5.040` 个百分点（78.792% -> 73.752%）

解释：这是“在真实公开数据校准条件下”的策略差异估计，不是企业落地后的已实现财务对账值。

---

## 5) 一句话总结（可直接汇报）
你提供的真实美国公开数据已经成功用于校准模型输入；在该真实校准环境下，模型显示 Cost-Min 相对 Resilience 可节约约 1.16 亿美元，但会带来约 5.04 个百分点的履约率下降，体现出明确的成本-服务权衡。

---

## 6) 边界说明（避免误解）
- 现在可以说：`model-estimated impact under real US public data calibration`。
- 现在不能说：`realized savings already achieved in operations`（因为没有企业执行后订单级实绩对账表）。
