# SCRAM 数据采集模块 - API Key 获取指南

## 📋 必需的 API Keys

### 1. **US Census Bureau API Key** ⭐ 必需

**用途**: 采集美国海关进出口贸易数据（核心数据源）

**数据内容**:
- 半导体产品进出口（HS 8542系列）
- 制造设备进出口（HS 8486系列）
- 硅晶圆等原材料（HS 3818, 8112等）
- 按国家、月度、金额和数量分类

**获取步骤**:
1. 访问: https://api.census.gov/data/key_signup.html
2. 填写姓名、邮箱、组织信息
3. 说明用途：Supply chain research / Academic analysis
4. 立即通过邮件收到 API Key（格式：40位字母数字组合）

**使用限制**:
- 免费无限制使用
- 建议限速：30 requests/分钟（已在代码中实现）

**示例配置**:
```bash
US_CENSUS_API_KEY=1234567890abcdef1234567890abcdef12345678
```

---

## 🔧 可选的 API Keys

### 2. **FRED API Key** (推荐但非必需)

**用途**: 采集宏观经济指标，特别是 ISM PMI 数据

**数据内容**:
- ISM Manufacturing PMI（制造业采购经理人指数）
- New Orders Index（新订单指数）
- Supplier Deliveries Index（供应商交付指数）
- 用于计算供需压力差（PMI Gap）

**获取步骤**:
1. 访问: https://fred.stlouisfed.org/
2. 点击右上角 "My Account" 注册账户
3. 登录后进入: https://fredaccount.stlouisfed.org/apikeys
4. 点击 "Request API Key"
5. 立即获得 32位 API Key

**使用限制**:
- 免费账户：120,000 requests/天
- 实际使用远低于此限制

**备注**:
- 如果未提供此 Key，系统会使用 mock 数据或跳过 PMI 采集
- GSCPI（全球供应链压力指数）无需 API Key，直接从 NY Fed 下载公开 CSV

**示例配置**:
```bash
FRED_API_KEY=abcdef1234567890abcdef1234567890
```

---

## 📊 无需 API Key 的数据源

### 3. **USGS Mineral Data** ✅ 公开数据

**数据来源**: 美国地质调查局（USGS）

**采集方式**:
- 直接下载年度发布的 PDF/Excel 文件
- Mineral Commodity Summaries（矿产商品摘要）
- 无需注册或 API Key

**数据内容**:
- 镓、锗、硅等关键矿产的生产、进口、价格
- 美国对外依赖度统计

### 4. **NY Fed GSCPI** ✅ 公开数据

**数据来源**: 纽约联邦储备银行

**采集方式**:
- 直接下载 CSV 文件
- URL: https://www.newyorkfed.org/medialibrary/media/research/policy/gscpi/data/gscpi.csv
- 无需注册或 API Key

**数据内容**:
- 全球供应链压力指数（月度）
- 综合运费、交付时间等多个维度

---

## 🚀 快速开始配置

### 步骤 1: 复制环境变量模板
```bash
cd scram
cp .env.example .env
```

### 步骤 2: 编辑 .env 文件
```bash
# 最小配置（仅必需项）
US_CENSUS_API_KEY=你的Census_API_Key

# 完整配置（推荐）
US_CENSUS_API_KEY=你的Census_API_Key
FRED_API_KEY=你的FRED_API_Key
```

### 步骤 3: 测试连接
```bash
# 测试 Census API
python -c "from src.collectors import USCensusCollector; c = USCensusCollector(); print('✓ Census API configured')"

# 测试 FRED API（如果配置了）
python -c "from src.collectors import MacroIndicatorCollector; c = MacroIndicatorCollector(); print('✓ FRED API configured')"
```

---

## ⚠️ 常见问题

### Q1: Census API Key 申请需要多久？
**A**: 通常立即通过邮件收到，无需审批。如果未收到，检查垃圾邮件。

### Q2: 没有 FRED API Key 能运行吗？
**A**: 可以。系统会使用 mock 数据或跳过 PMI 部分。但建议申请（5分钟即可完成）以获取真实数据。

### Q3: API Key 有使用期限吗？
**A**: Census 和 FRED 的 API Key 均无期限限制，长期有效。

### Q4: 数据采集会产生费用吗？
**A**: 不会。所有数据源均为**完全免费**的公开数据。

### Q5: 如何验证 API Key 是否有效？
**A**: 运行示例脚本会自动验证：
```bash
python examples.py --example 1
```

---

## 📝 总结

| API Key | 状态 | 获取难度 | 获取时间 | 重要性 |
|---------|------|----------|----------|--------|
| **US Census** | 必需 | ⭐ 简单 | 即时 | ⭐⭐⭐⭐⭐ |
| **FRED** | 可选 | ⭐ 简单 | 即时 | ⭐⭐⭐☆☆ |
| **USGS** | 无需 | - | - | ⭐⭐⭐⭐☆ |
| **GSCPI** | 无需 | - | - | ⭐⭐⭐☆☆ |

**最低配置**: 仅需 US Census API Key（5分钟获取）即可运行核心功能。

**推荐配置**: Census + FRED（10分钟获取）获得完整数据集。
