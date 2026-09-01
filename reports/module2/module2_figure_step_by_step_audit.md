# Module2 四图逐步计算说明（含每步数值）

本文档针对 impact summary 图中的四个子图，按“概念 -> 数据源 -> 公式 -> 每一步数字 -> 最终柱状值”展开。

数据类型标记说明：
- 模型得到的数据：来自优化器求解结果（flow、delivered、unmet、summary 等）。
- 直接用的数据：从已有报表 CSV 直接读取，不再推导。
- 二次计算的数据：在“模型数据/直接读取数据”基础上做加减乘除或单位换算。

数据源文件：
- reports/module2/module2_strategy_tradeoff.csv
- reports/module2/module2_state_fulfillment_by_strategy.csv
- reports/module2/module2_baseline_intl_flow_allocation.csv（Baseline）
- reports/module2/module2_baseline_flow_allocation.csv（Baseline）
- reports/module2/module2_stress_cost_intl_flow_allocation.csv（Cost-Minimizing）
- reports/module2/module2_stress_cost_flow_allocation.csv（Cost-Minimizing）
- reports/module2/module2_stress_resilience_intl_flow_allocation.csv（Resilience-First）
- reports/module2/module2_stress_resilience_flow_allocation.csv（Resilience-First）
- reports/module2/module2_stress_targeted_intl_flow_allocation.csv（Targeted Allocation）
- reports/module2/module2_stress_targeted_flow_allocation.csv（Targeted Allocation）

---

## 图1：Active Trunk Routes (Port->DC)

### 1) 概念
- 每一行代表一条候选干线路由（port->dc + mode）。
- active route 定义：flow_usd > 1e-6。

### 1.1 这些数字属于哪类数据
- `total_routes`（27）= 直接用的数据（直接读各 flow CSV 的总行数）。
- `active_routes`（18/24/24/24）= 二次计算的数据（在 flow CSV 上按条件 `flow_usd > 1e-6` 计数）。
- `inactive_routes` = 二次计算的数据（`total_routes - active_routes`）。
- `active_share` = 二次计算的数据（`active_routes / total_routes`）。
- 其中 `flow_usd` 本身 = 模型得到的数据（优化器求解后的线路流量）。

### 2) 计算步骤与数值
- Baseline
  - 候选路线总数 total_routes = 27
  - 激活路线 active_routes = 18
  - 未激活路线 inactive_routes = 9
  - 激活占比 active_share = 0.666667 = 66.667%
- Cost-Minimizing
  - 候选路线总数 total_routes = 27
  - 激活路线 active_routes = 24
  - 未激活路线 inactive_routes = 3
  - 激活占比 active_share = 0.888889 = 88.889%
  - 相对 Baseline 增量 = 24 - 18 = +6
- Resilience-First
  - 候选路线总数 total_routes = 27
  - 激活路线 active_routes = 24
  - 未激活路线 inactive_routes = 3
  - 激活占比 active_share = 0.888889 = 88.889%
  - 相对 Baseline 增量 = 24 - 18 = +6
- Targeted Allocation
  - 候选路线总数 total_routes = 27
  - 激活路线 active_routes = 24
  - 未激活路线 inactive_routes = 3
  - 激活占比 active_share = 0.888889 = 88.889%
  - 相对 Baseline 增量 = 24 - 18 = +6

### 3) 图上柱子对应值
- Baseline: 18
- Cost-Minimizing: 24
- Resilience-First: 24
- Targeted Allocation: 24

### 4) 完整来源链路
- 模型先输出每条候选路线的 `flow_usd`（模型得到的数据）。
- 报表文件保存这些 `flow_usd`（直接用的数据来源）。
- 脚本统计 `flow_usd > 1e-6` 的行数，得到柱子高度（二次计算的数据）。

---

## 图2：Service Level (Fill Rate)

### 1) 概念
- fill_rate 表示全国需求中被满足的比例。
- 公式：fill_rate = delivered_usd / total_demand_usd。
- 等价：fill_rate = 1 - unmet_usd / total_demand_usd。

### 1.1 这些数字属于哪类数据
- `delivered_usd`（州级、全国）= 模型得到的数据（由 DC->州最优流量汇总）。
- `unmet_usd`（州级、全国）= 模型得到的数据（需求平衡方程中的未满足变量）。
- `demand_usd`（州级）= 直接用的数据（来自需求输入，进入模型前已固定）。
- `total_demand_usd`（全国）= 二次计算的数据（州级 `demand_usd` 求和）。
- `fill_rate` = 二次计算的数据（`delivered_usd / total_demand_usd`）。
- `fill_rate_pct` = 二次计算的数据（`fill_rate * 100`）。

### 2) 计算步骤与数值（按策略）
- Cost-Minimizing
  - 州级行数 state_rows = 51
  - 全国 delivered_usd = sum(各州 delivered_usd) = 14750400000.000004
  - 全国 unmet_usd = sum(各州 unmet_usd) = 5249599999.999993
  - 全国 total_demand_usd = sum(各州 demand_usd) = 20000000000.000008
  - 平衡校验 delivered + unmet = 19999999999.999996
  - fill_rate = delivered / demand = 14750400000.000004 / 20000000000.000008 = 0.73752000
  - fill_rate_pct = 0.73752000 * 100 = 73.752%
- Resilience-First
  - 州级行数 state_rows = 51
  - 全国 delivered_usd = sum(各州 delivered_usd) = 15758400000.000006
  - 全国 unmet_usd = sum(各州 unmet_usd) = 4241599999.999996
  - 全国 total_demand_usd = sum(各州 demand_usd) = 20000000000.000008
  - 平衡校验 delivered + unmet = 20000000000.000000
  - fill_rate = delivered / demand = 15758400000.000006 / 20000000000.000008 = 0.78792000
  - fill_rate_pct = 0.78792000 * 100 = 78.792%
- Targeted Allocation
  - 州级行数 state_rows = 51
  - 全国 delivered_usd = sum(各州 delivered_usd) = 15405600000.000002
  - 全国 unmet_usd = sum(各州 unmet_usd) = 4594399999.999995
  - 全国 total_demand_usd = sum(各州 demand_usd) = 20000000000.000008
  - 平衡校验 delivered + unmet = 19999999999.999996
  - fill_rate = delivered / demand = 15405600000.000002 / 20000000000.000008 = 0.77028000
  - fill_rate_pct = 0.77028000 * 100 = 77.028%

### 3) 图上柱子对应值
- Cost-Minimizing: 73.752%
- Resilience-First: 78.792%
- Targeted Allocation: 77.028%

### 4) 完整来源链路
- 模型求解得到每个州的 `delivered_usd` 和 `unmet_usd`（模型得到的数据）。
- 汇总表/州级表读取这些字段（直接用的数据来源）。
- 文档中的比例与百分比由上述字段再计算（二次计算的数据）。

---

## 图3：Logistics Cost

### 1) 概念
- total_logistics_cost_usd 是优化输出的总物流成本。
- 图上展示单位是 Million USD（百万美元）。
- 换算公式：cost_musd = total_logistics_cost_usd / 1,000,000。

### 1.1 这些数字属于哪类数据
- `flow_intl_port.flow_usd`、`flow_port_dc.flow_usd`、`flow_dc_state.flow_usd` = 模型得到的数据。
- `flow_port_dc.unit_cost`、`flow_dc_state.unit_cost` = 直接用的数据（网络参数设定后固定）。
- `total_logistics_cost_usd` = 二次计算的数据，公式为：
  - `sum(flow_intl_port.flow_usd * flow_intl_port.unit_cost) + sum(flow_port_dc.flow_usd * flow_port_dc.unit_cost) + sum(flow_dc_state.flow_usd * flow_dc_state.unit_cost)`
- `cost_musd` = 二次计算的数据（单位换算）。

### 2) 计算步骤与数值（按策略）
- Cost-Minimizing
  - total_logistics_cost_usd = 665990246.681790
  - cost_musd = 665990246.681790 / 1,000,000 = 665.990247
- Resilience-First
  - total_logistics_cost_usd = 796059207.038965
  - cost_musd = 796059207.038965 / 1,000,000 = 796.059207
- Targeted Allocation
  - total_logistics_cost_usd = 742593070.752894
  - cost_musd = 742593070.752894 / 1,000,000 = 742.593071

### 3) 图上柱子对应值
- Cost-Minimizing: 666.0 Million USD
- Resilience-First: 796.1 Million USD
- Targeted Allocation: 742.6 Million USD

### 4) 你问的示例：`total_logistics_cost_usd = 665990246.681790` 到底从哪来
- 第1层（报表读取层）
  - 先从策略汇总表读取 Cost-Minimizing 这一行。
  - 该行字段 `total_logistics_cost_usd` 的值就是 665990246.681790（图里这个数先来自这里）。
- 第2层（模型计算层）
  - 该字段不是手填，是模型求解后按下式计算得到：
  - `logistics_cost = sum(flow_intl_port.flow_usd * flow_intl_port.unit_cost) + sum(flow_port_dc.flow_usd * flow_port_dc.unit_cost) + sum(flow_dc_state.flow_usd * flow_dc_state.unit_cost)`
  - 也就是“海外到港口成本 + 港口到DC成本 + DC到州成本”。
- 第3层（数值拆解层，Cost-Minimizing 实算）
  - 第一段成本（overseas->port）= 219219878.00000003
  - 第二段成本（port->dc）= 237809120.00000000
  - 第三段成本（dc->state）= 208961248.68179047
  - 总成本 = 219219878.00000003 + 237809120.00000000 + 208961248.68179047 = 665990246.6817905
  - 与策略汇总表中的 `total_logistics_cost_usd` 一致（浮点尾差为机器精度）。
- 第4层（图表展示层）
  - 图3用的是百万美元单位。
  - `cost_musd = 665990246.681790 / 1,000,000 = 665.990247`。
  - 图上显示时保留 1 位小数，所以是 666.0 Million USD。

### 5) 完整来源链路
- 模型输出两段流量 `flow_usd`（模型得到的数据）。
- 线路单价 `unit_cost` 来自网络参数（直接用的数据）。
- 先算 USD 总成本，再换算成百万美元用于画图（二次计算的数据）。

---

## 图4：Potential Savings vs Resilience-First

### 1) 概念
- 以 Resilience-First 作为参照，比较切换到其他策略可节省多少钱。
- 公式：savings = cost(Resilience-First) - cost(Other Strategy)。

### 1.1 这些数字属于哪类数据
- `cost(Resilience-First / Cost-Min / Targeted)` = 直接用的数据（从策略汇总表读取 `total_logistics_cost_usd`）。
- `savings_usd` = 二次计算的数据（成本做差）。
- `savings_musd` = 二次计算的数据（除以 1,000,000）。

### 2) 计算步骤与数值
- 参照成本 cost(Resilience-First) = 796059207.038965 USD
 - 参照成本 cost(Resilience-First) = 796059207.038965 USD
- Cost-Min vs Resilience
  - savings_usd = 796059207.038965 - 665990246.681790 = 130068960.357175 USD
  - savings_musd = 130068960.357175 / 1,000,000 = 130.068960
- Targeted vs Resilience
  - savings_usd = 796059207.038965 - 742593070.752894 = 53466136.286071 USD
  - savings_musd = 53466136.286071 / 1,000,000 = 53.466136

### 3) 图上柱子对应值
- Cost-Min vs Resilience: 116.0 Million USD
- Targeted vs Resilience: 53.5 Million USD

### 4) 完整来源链路
- 先读取三个策略成本（直接用的数据）。
- 以 Resilience 成本为基准做差，得到节省金额（二次计算的数据）。
- 再做单位换算用于画图（二次计算的数据）。

---

## 附：图中其他常见问题
- 为什么会有 0.000001 级别小数误差：这是浮点数求和的常见现象。
- 为什么 delivered + unmet 与 demand 可能差 1e-6 到 1e-5：同样是浮点精度，不影响业务结论。
- 为什么图2和图3不包含 Baseline：该图是压力场景策略间对比，Baseline 在图1作为路由展开参照。

## 一眼看懂：哪些是模型得到，哪些是直接用
- 模型得到的数据：`flow_usd`、`delivered_usd`、`unmet_usd`。
- 直接用的数据：`demand_usd`（输入需求）、`unit_cost`（网络参数）、汇总表中的已落盘字段值。
- 二次计算的数据：`fill_rate`、`fill_rate_pct`、`active_routes`、`active_share`、`total_logistics_cost_usd`（由 flow*cost 聚合）、`cost_musd`、`savings_usd`、`savings_musd`。