# 模型训练计划 (Phase 3)

**状态**: 准备开始  
**前置条件**: ✅ 数据采集完成，✅ 特征提取完成

---

## 📊 数据准备情况

### 训练集 (2010-2019)
- **文件**: `data/processed/features_train_full.parquet`
- **规模**: 37,756 条记录 × 58 个特征
- **大小**: 412 KB
- **时间跨度**: 10年 (120个月)

### 测试集 (2020-2024)  
- **文件**: `data/processed/features_test_full.parquet`
- **规模**: 31,900 条记录 × 58 个特征
- **大小**: 388 KB
- **时间跨度**: 5年 (60个月)

### 特征类别 (49个预测特征 + 9个元数据)

**1. 集中度特征 (11个)**
- HHI (当前 + 3/6/12月滚动)
- Top N份额 (Top1/3/5)
- Gini系数
- 供应商数量
- HHI变化率 (MoM/YoY)

**2. 波动性特征 (12个)**
- 标准差 (价值/数量 + 3/6/12月滚动)
- 变异系数 (CoV)
- 波动性趋势
- 稳定性评分

**3. 时间序列特征 (16个)**
- 时间维度 (月/季/年)
- 趋势 + 移动平均 (3/6/12月)
- 滞后特征 (1/3/6/12月)
- 动量 + 季节性

**4. 增长特征 (10个)**
- 增长率 (MoM/QoQ/YoY)
- 平均增长率 (3/6/12月)
- CAGR (3年)
- 增长加速度

---

## 🎯 模型目标

### 任务1: 时间序列预测
**目标**: 预测未来供应链指标（价值、集中度、波动性）

**模型候选**:
1. **ARIMA / SARIMA** - 经典时间序列
2. **XGBoost** - 梯度提升（时间特征）
3. **LightGBM** - 快速梯度提升
4. **LSTM** - 深度学习序列模型

**评估指标**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

### 任务2: 风险分类
**目标**: 识别高风险供应链配置

**风险定义**:
- **高集中度**: HHI > 2500
- **高波动性**: value_cov > 0.5
- **快速变化**: |hhi_change_mom| > 20%

**模型候选**:
1. **Logistic Regression** - 基线
2. **Random Forest** - 集成方法
3. **XGBoost Classifier** - 高性能
4. **Neural Network** - 非线性模式

**评估指标**:
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Confusion Matrix

### 任务3: 异常检测
**目标**: 检测供应链中断和异常事件

**方法候选**:
1. **Isolation Forest** - 无监督
2. **One-Class SVM** - 支持向量机
3. **Autoencoder** - 深度学习
4. **Statistical Methods** - Z-score, IQR

**评估指标**:
- Contamination rate
- Silhouette score
- Manual validation (标注已知事件)

---

## 🚀 实施步骤

### Step 1: 环境准备 (15分钟)
```bash
# 取消注释requirements.txt中的ML库
# 安装依赖
pip install scikit-learn xgboost lightgbm matplotlib seaborn
```

### Step 2: 创建模型框架 (30分钟)
```bash
src/models/
├── __init__.py
├── base_model.py          # 抽象基类
├── time_series_model.py   # 时间序列预测
├── risk_classifier.py     # 风险分类
├── anomaly_detector.py    # 异常检测
└── model_pipeline.py      # 训练和评估pipeline
```

### Step 3: 数据预处理 (20分钟)
- 处理缺失值和无穷值
- 特征标准化/归一化
- 训练/验证/测试集划分
- 时间序列交叉验证设置

### Step 4: 基线模型训练 (1小时)
- 每个任务训练一个简单基线
- 记录性能指标
- 保存模型和结果

### Step 5: 模型优化 (2-3小时)
- 超参数调优
- 特征选择
- 模型集成
- 交叉验证

### Step 6: 模型评估和可视化 (1小时)
- 测试集评估
- 生成评估报告
- 可视化预测结果
- 特征重要性分析

---

## 📦 可交付成果

1. **模型文件** (`models/trained/`)
   - `*.pkl` - 训练好的模型
   - `*.json` - 模型配置和超参数

2. **评估报告** (`reports/`)
   - 模型性能对比
   - 特征重要性分析
   - 预测结果可视化

3. **预测结果** (`data/predictions/`)
   - 测试集预测值
   - 置信区间
   - 风险评分

---

## 🎓 预期结果

### 任务1: 时间序列预测
- **目标精度**: MAPE < 15%
- **关键预测**: 
  - 未来3/6/12月的贸易价值
  - HHI变化趋势
  - 供应商集中度演变

### 任务2: 风险分类
- **目标性能**: F1 > 0.80
- **应用场景**: 
  - 实时风险监控
  - 早期预警系统
  - 供应商评估

### 任务3: 异常检测
- **目标检出率**: 能识别已知的供应链中断事件
  - 2011年日本地震/海啸
  - 2020年COVID-19疫情
  - 2021年芯片短缺
  - 2022年地缘政治事件

---

## 💡 下一步行动

### 立即可做:

**选项A: 快速启动 (推荐)**
```bash
# 1. 安装ML依赖
pip install scikit-learn xgboost lightgbm

# 2. 运行基线模型（自动创建）
python train_baseline_models.py
```

**选项B: 交互式开发**
```bash
# 创建Jupyter notebook进行探索性建模
jupyter notebook notebooks/model_exploration.ipynb
```

**选项C: 分步实施**
1. 先创建模型框架代码
2. 实现一个简单的时间序列预测
3. 逐步扩展到其他模型

---

**准备就绪！你希望从哪个选项开始？**
