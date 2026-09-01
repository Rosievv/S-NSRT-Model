# Module2 与 FRED 新订单/积压订单对比

## 1) 你提供的序列ID可用性
- NO34SVS: not_found_404
- UO34SVS: not_found_404
- 说明：`NO34SVS`/`UO34SVS` 在 FRED 页面返回 404，本报告使用同主题有效序列替代。

## 2) 使用的有效 FRED 真实序列（NAICS 334）
- A34SNO (Manufacturers' New Orders: Computers and Electronic Products (SA)) -> month=2022-01-01, value=24951.0000, zscore=0.0391
- A34SUO (Manufacturers' Unfilled Orders: Computers and Electronic Products (SA)) -> month=2022-01-01, value=140152.0000, zscore=1.9507
- A34SUS (Manufacturers' Unfilled Orders to Shipments Ratio: Computers and Electronic Products (SA)) -> month=2022-01-01, value=6.1900, zscore=1.2756

## 3) 与模型结果的对比结论
- Cost-Min 相对 Resilience 的模型估计节约: 115,956,960.36 USD
- Cost-Min 相对 Resilience 的履约率损失: 5.040 个百分点
- 解释：上述节约/履约差异是模型输出；FRED 序列用于锚定美国真实宏观供需状态。

## 4) Cass Freight Index 说明
- Cass 页面未暴露稳定可抓取的历史文件直链，建议你手动从 Historical Data Archive 下载 Excel。
- 下载后放入 data/raw，我们可以立刻并入同一份对比框架。