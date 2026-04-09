# SCRAM 数据采集机制说明

## 📊 数据采集方式：一次性批量获取

### 核心设计理念

**这是一个「历史数据批量采集」系统，不是实时监控系统。**

### 工作原理

```
用户触发采集命令
    ↓
系统读取配置的时间范围（如 2010-2019）
    ↓
批量下载该时间段内的所有数据
    ↓
保存到本地文件系统（Parquet格式）
    ↓
后续分析直接读取本地数据
```

---

## 🎯 采集模式详解

### 模式 1：训练数据采集（2010-2019）

```bash
python main.py --phase train --sources all
```

**执行过程**：
1. 读取 `config.yaml` 中的 `train_start: "2010-01-01"` 和 `train_end: "2019-12-31"`
2. 对每个 HS 编码（如 854231, 854232...）：
   - 遍历 120 个月（2010年1月 至 2019年12月）
   - 向 US Census API 发送 120 次请求
   - 下载每个月的进出口数据
3. 所有数据合并、验证、保存到 `data/raw/us_census_YYYYMMDD_HHMMSS.parquet`

**时间估算**：
- 4个HS编码 × 120个月 = 480次请求
- 速率限制：30请求/分钟
- 总耗时：约 16-20 分钟

### 模式 2：测试数据采集（2020-2024）

```bash
python main.py --phase test --sources all
```

同样是**一次性批量下载** 2020-2024 的所有数据。

### 模式 3：自定义时间段

```bash
python main.py --phase custom --start-date 2022-01-01 --end-date 2022-12-31
```

下载指定时间段的数据。

---

## ⏱️ 数据更新频率

### 当前设计：手动批量更新

```
采集一次 → 存储到本地 → 长期使用
```

**不是实时系统**，原因如下：

1. **历史分析需求**：
   - 训练模型需要 2010-2019 的历史数据
   - 这些数据是固定的，不会变化
   - 只需采集一次

2. **数据源更新频率**：
   - US Census 月度数据：每月月底发布上月数据
   - USGS 年度数据：每年发布一次
   - GSCPI/PMI：月度更新

3. **API 限制**：
   - Census API 虽然免费，但不建议频繁请求历史数据
   - 合理使用：每隔 1-3 个月更新一次即可

---

## 📅 推荐的数据更新策略

### 阶段 1：初始化（现在）

```bash
# 1. 下载训练数据（2010-2019）- 只需执行一次
python main.py --phase train --sources all

# 2. 下载测试数据（2020-2024）- 只需执行一次
python main.py --phase test --sources all
```

**耗时**：约 30-40 分钟（取决于网络）
**存储**：约 50-200 MB（Parquet 压缩格式）

### 阶段 2：定期更新（每月或每季度）

```bash
# 更新最新月份的数据
python main.py --phase custom \
  --start-date 2026-01-01 \
  --end-date 2026-02-28 \
  --sources census macro
```

**场景**：
- 新的月度贸易数据发布后（通常是次月月底）
- 用于监测最新的供应链动态
- 验证模型预测准确性

---

## 🔄 如果需要实现定时自动采集

虽然当前设计是手动触发，但可以轻松扩展为定时任务：

### 方案 1：Cron Job（Linux/macOS）

```bash
# 每月5号凌晨2点自动采集上月数据
crontab -e

# 添加：
0 2 5 * * cd /path/to/scram && python main.py --phase custom --start-date $(date -d "last month" +\%Y-\%m-01) --end-date $(date -d "last month" +\%Y-\%m-31) --sources all
```

### 方案 2：Python 定时任务

```python
# schedule_collector.py
import schedule
import time
from datetime import datetime, timedelta

def collect_monthly_data():
    """每月自动采集上月数据"""
    last_month = datetime.now().replace(day=1) - timedelta(days=1)
    start = last_month.replace(day=1).strftime('%Y-%m-%d')
    end = last_month.strftime('%Y-%m-%d')
    
    import subprocess
    subprocess.run([
        'python', 'main.py',
        '--phase', 'custom',
        '--start-date', start,
        '--end-date', end,
        '--sources', 'all'
    ])

# 每月5号执行
schedule.every().month.at("05:00").do(collect_monthly_data)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 每小时检查一次
```

### 方案 3：云服务定时任务

- **AWS Lambda + EventBridge**：每月自动触发
- **Google Cloud Scheduler**：定时执行
- **Azure Functions + Timer Trigger**：定时任务

---

## 💾 数据存储与管理

### 存储结构

```
data/
├── raw/                              # 原始采集数据
│   ├── us_census_20260215_143022.parquet
│   ├── usgs_20260215_143055.parquet
│   └── macro_indicators_20260215_143100.parquet
└── processed/                        # 处理后的数据（特征工程后）
    ├── features_train_2010_2019.parquet
    └── features_test_2020_2024.parquet
```

### 数据版本控制

- 每次采集自动添加时间戳
- 保留历史版本用于审计
- 可定期清理旧版本（保留最新3个月）

---

## 🚀 实际使用流程

### 首次使用（你现在的情况）

```bash
# Step 1: 验证配置
python3 verify_setup.py

# Step 2: 安装依赖（如果还没装）
pip install -r requirements.txt

# Step 3: 采集训练数据
python main.py --phase train --sources all
# 预计耗时：20-30分钟
# 进度会实时显示

# Step 4: 采集测试数据
python main.py --phase test --sources all
# 预计耗时：10-15分钟

# Step 5: 查看结果
ls -lh data/raw/
# 应该看到多个 .parquet 文件
```

### 日常使用

```bash
# 每月更新（手动）
python main.py --phase custom \
  --start-date 2026-02-01 \
  --end-date 2026-02-28 \
  --sources census macro

# 查看日志
tail -f logs/main_*.log
```

---

## 🔍 数据质量保证

每次采集自动执行：

1. **实时验证**：
   - 检查必需列是否存在
   - 缺失值比例检查
   - 日期连续性验证

2. **异常检测**：
   - 数值范围合理性
   - 极端值标记（Z-score > 5）

3. **日志记录**：
   - 每次请求详情
   - 错误和警告
   - 数据质量报告

4. **自动重试**：
   - 网络错误自动重试3次
   - 指数退避延迟

---

## ⚡ 性能优化建议

### 当前实现

- ✅ API 速率限制（30 req/min）
- ✅ 自动重试机制
- ✅ 数据压缩存储（Parquet + Snappy）
- ✅ 增量日志记录

### 未来可优化

- 🔄 并行下载多个 HS 编码（需注意速率限制）
- 🔄 断点续传（中断后从上次位置继续）
- 🔄 增量更新（只下载新增月份）

---

## 📋 总结

| 特性 | 当前设计 | 说明 |
|------|---------|------|
| **采集方式** | 批量一次性 | 下载整个时间段的历史数据 |
| **触发方式** | 手动命令 | 用户执行 main.py |
| **更新频率** | 按需手动 | 建议每月或每季度 |
| **数据存储** | 本地文件 | Parquet 格式，高效压缩 |
| **实时性** | 非实时 | 用于历史分析，非监控系统 |
| **可扩展性** | 高 | 可轻松改为定时任务 |

**关键点**：
- ✅ 你只需要运行一次就能获得 10 年的历史数据
- ✅ 后续的特征工程和模型训练都基于本地存储的数据
- ✅ 不需要每次分析都重新下载
- ✅ 每月更新一次即可保持数据新鲜度
