# Cass 真实指数 与 Module2 模型结果差距分析

## 1) 真实数据来源（你新增）
- 文件: data/raw/Module2/Cass Indexes Historical Data.xlsx
- Sheet: Freight Index-Expenditures（Cass 运费综合支出指数）
- Sheet: TL LH Index（Cass 纯干线 Linehaul 指数）

## 2) 与模型对齐月份
- 模型需求月份: 2022-01-01
- Cass Expenditures 使用月份: 2022-01-01, 指数=4.0270, YoY=0.3117, zscore=2.4474
- Cass Linehaul 使用月份: 2022-01-01, 指数=158.0000, YoY=0.1278, zscore=2.8036

## 3) 模型同月结果
- Resilience-First 成本: 575,441,607.04 USD, fill_rate=0.787920
- Cost-Minimizing 成本: 459,484,646.68 USD, fill_rate=0.737520
- Cost-Min 相对 Resilience 节约: 115,956,960.36 USD (20.15%)
- 对应履约率损失: 5.040 个百分点

## 4) 差距是什么（可比与不可比）
- 可比（方向/相对量级）:
  - Cass 指数反映真实美国运费环境冷热，模型成本反映策略在该环境下的相对开销差异。
  - 你可以用 Cass 的高位/低位月份解释为何模型更应偏 Resilience 或 Cost-Min。
- 不可直接比（绝对值）:
  - Cass 是指数（无绝对美元单位），模型是 USD 绝对金额，不能直接相减。
  - Cass 不包含可直接复原的 fulfilled/unfulfilled 订单口径，因此无法直接给真实 fill_rate。

## 5) 结论
- 你新增的 Cass 数据已经成功接入，并可作为真实运费环境基准。
- 在该真实环境基准下，模型仍给出明确权衡：Cost-Min 省 20% 左右成本，但损失约 5.04 个百分点履约率。