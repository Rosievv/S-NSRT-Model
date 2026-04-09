#!/usr/bin/env python3
"""
Step 1: 数据探索分析 - 诊断问题根源

目的:
1. 分析贸易价值分布特征
2. 识别异常值和零值
3. 分析不同HS代码的差异性
4. 为改进策略提供数据支持

知识依据:
- EDA (Exploratory Data Analysis) 是模型改进的第一步
- 可视化可以揭示数据分布问题
- 统计描述可以量化数据特征
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*80)
print("Step 1: 数据探索分析 - 诊断贸易价值预测问题")
print("="*80)

# 加载数据
print("\n【加载数据】")
train_df = pd.read_parquet('data/processed/features_train_full.parquet')
test_df = pd.read_parquet('data/processed/features_test_full.parquet')
print(f"✓ 训练集: {len(train_df):,} 条")
print(f"✓ 测试集: {len(test_df):,} 条")

# =============================================================================
# 分析 1: 贸易价值分布
# =============================================================================
print("\n" + "="*80)
print("【分析 1】贸易价值分布特征")
print("="*80)

print("\n统计描述:")
print(train_df['value_usd'].describe())

# 计算偏度和峰度
from scipy import stats
skewness = stats.skew(train_df['value_usd'].dropna())
kurtosis = stats.kurtosis(train_df['value_usd'].dropna())

print(f"\n偏度 (Skewness): {skewness:.2f}")
print(f"峰度 (Kurtosis): {kurtosis:.2f}")

print("\n【判断逻辑】:")
if skewness > 1:
    print(f"  ✓ 偏度 > 1 表明数据严重右偏（右尾长）")
    print(f"    → 少数超大值拉高均值，需要变换")
if kurtosis > 3:
    print(f"  ✓ 峰度 > 3 表明分布尖峰厚尾")
    print(f"    → 存在极端值，普通模型难以处理")

# 零值和负值分析
n_zero = (train_df['value_usd'] == 0).sum()
n_negative = (train_df['value_usd'] < 0).sum()
print(f"\n零值数量: {n_zero:,} ({n_zero/len(train_df)*100:.2f}%)")
print(f"负值数量: {n_negative:,}")

# 数量级分析
print("\n数量级分布:")
bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, np.inf]
labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '>100M']
train_df['value_magnitude'] = pd.cut(train_df['value_usd'], bins=bins, labels=labels)
magnitude_dist = train_df['value_magnitude'].value_counts().sort_index()
for mag, count in magnitude_dist.items():
    print(f"  {mag:>12s}: {count:>6,} ({count/len(train_df)*100:>5.2f}%)")

print("\n【知识依据】:")
print("  - 多个数量级分布 → 需要对数变换或标准化")
print("  - 右偏分布 → log(x+1) 变换可以使分布更对称")
print("  - Box-Cox变换可以自动找到最优变换参数")

# =============================================================================
# 分析 2: 不同HS代码的差异性
# =============================================================================
print("\n" + "="*80)
print("【分析 2】不同HS代码的贸易价值差异")
print("="*80)

hs_stats = train_df.groupby('hs_code')['value_usd'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
])
hs_stats['cv'] = hs_stats['std'] / hs_stats['mean']  # 变异系数

print("\n按HS代码统计:")
print(hs_stats.to_string())

print("\n【判断逻辑】:")
cv_range = hs_stats['cv'].max() - hs_stats['cv'].min()
mean_range = hs_stats['mean'].max() / hs_stats['mean'].min()
print(f"  ✓ 变异系数范围: {hs_stats['cv'].min():.2f} - {hs_stats['cv'].max():.2f}")
print(f"    → 不同HS代码波动性差异 {cv_range:.2f}")
print(f"  ✓ 平均值比例: 最大/最小 = {mean_range:.1f}倍")
print(f"    → 不同商品贸易规模差异巨大")

print("\n【结论】:")
print("  → 需要按HS代码分别建模（类似多任务学习）")
print("  → 或者在模型中加入HS代码的交互特征")

# =============================================================================
# 分析 3: 时间序列特征
# =============================================================================
print("\n" + "="*80)
print("【分析 3】时间趋势和季节性")
print("="*80)

# 按月份统计
train_df['year_month'] = train_df['date'].dt.to_period('M')
monthly_stats = train_df.groupby('year_month')['value_usd'].agg(['mean', 'std', 'count'])

print("\n月度统计（前10个月）:")
print(monthly_stats.head(10))

# 趋势检验
from scipy.stats import pearsonr
train_df['time_index'] = (train_df['date'] - train_df['date'].min()).dt.days
corr, p_value = pearsonr(train_df['time_index'], train_df['value_usd'])
print(f"\n时间趋势相关系数: {corr:.4f} (p-value: {p_value:.2e})")

if abs(corr) > 0.1:
    print("  ✓ 存在显著时间趋势，模型需要捕捉趋势成分")
else:
    print("  ✓ 时间趋势不显著")

# =============================================================================
# 分析 4: 对数变换效果预览
# =============================================================================
print("\n" + "="*80)
print("【分析 4】对数变换效果预览")
print("="*80)

# 创建对数变换
train_df['log_value_usd'] = np.log1p(train_df['value_usd'])  # log(1+x) 避免log(0)

print("\n原始 vs 对数变换:")
print("\n原始值统计:")
print(train_df['value_usd'].describe())
print(f"偏度: {stats.skew(train_df['value_usd'].dropna()):.2f}")

print("\n对数变换后统计:")
print(train_df['log_value_usd'].describe())
print(f"偏度: {stats.skew(train_df['log_value_usd'].dropna()):.2f}")

print("\n【判断逻辑】:")
log_skew = stats.skew(train_df['log_value_usd'].dropna())
if abs(log_skew) < abs(skewness):
    print(f"  ✓ 对数变换后偏度从 {skewness:.2f} 降至 {log_skew:.2f}")
    print(f"    → 分布更对称，更适合机器学习模型")
    print(f"  ✓ 建议: 使用对数变换后的值作为预测目标")

# =============================================================================
# 可视化
# =============================================================================
print("\n" + "="*80)
print("【生成可视化图表】")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('贸易价值数据探索分析', fontsize=16, y=1.00)

# 1. 原始分布（限制范围以便可视化）
ax = axes[0, 0]
data_plot = train_df['value_usd'][train_df['value_usd'] < train_df['value_usd'].quantile(0.99)]
ax.hist(data_plot, bins=50, edgecolor='black', alpha=0.7)
ax.set_title('原始价值分布 (99%分位数以下)')
ax.set_xlabel('Value USD')
ax.set_ylabel('Frequency')
ax.axvline(train_df['value_usd'].mean(), color='red', linestyle='--', label='Mean')
ax.axvline(train_df['value_usd'].median(), color='green', linestyle='--', label='Median')
ax.legend()

# 2. 对数变换分布
ax = axes[0, 1]
ax.hist(train_df['log_value_usd'], bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.set_title('对数变换后分布')
ax.set_xlabel('Log(1 + Value USD)')
ax.set_ylabel('Frequency')

# 3. Q-Q plot
ax = axes[0, 2]
stats.probplot(train_df['log_value_usd'].dropna(), dist="norm", plot=ax)
ax.set_title('Q-Q Plot (对数变换后)')

# 4. 不同HS代码的箱线图
ax = axes[1, 0]
hs_codes = train_df['hs_code'].unique()
data_by_hs = [train_df[train_df['hs_code'] == hs]['log_value_usd'].values for hs in hs_codes]
ax.boxplot(data_by_hs, labels=hs_codes)
ax.set_title('按HS代码的对数价值分布')
ax.set_xlabel('HS Code')
ax.set_ylabel('Log(1 + Value USD)')
ax.tick_params(axis='x', rotation=45)

# 5. 时间趋势
ax = axes[1, 1]
monthly_mean = train_df.groupby('year_month')['value_usd'].mean()
monthly_mean.plot(ax=ax, marker='o', markersize=3)
ax.set_title('月度平均贸易价值趋势')
ax.set_xlabel('Year-Month')
ax.set_ylabel('Mean Value USD')
ax.tick_params(axis='x', rotation=45)

# 6. 变异系数对比
ax = axes[1, 2]
hs_stats_plot = hs_stats.reset_index()
ax.bar(range(len(hs_stats_plot)), hs_stats_plot['cv'])
ax.set_xticks(range(len(hs_stats_plot)))
ax.set_xticklabels(hs_stats_plot['hs_code'], rotation=45)
ax.set_title('不同HS代码的变异系数')
ax.set_xlabel('HS Code')
ax.set_ylabel('Coefficient of Variation')
ax.axhline(y=hs_stats_plot['cv'].mean(), color='r', linestyle='--', label='Mean CV')
ax.legend()

plt.tight_layout()
plt.savefig('reports/step1_data_exploration.png', dpi=150, bbox_inches='tight')
print("\n✓ 图表已保存: reports/step1_data_exploration.png")

# =============================================================================
# 总结和建议
# =============================================================================
print("\n" + "="*80)
print("【Step 1 总结】基于数据探索的改进建议")
print("="*80)

print("\n问题根源:")
print("  1. 贸易价值跨越多个数量级 (1K - 100M+)")
print("  2. 分布严重右偏 (skewness > 1)")
print("  3. 不同HS代码特征差异大 (变异系数差异大)")
print("  4. 原始尺度导致大值误差过高，影响R²")

print("\n改进策略（按优先级）:")
print("  ✓ 策略1: 对数变换目标变量 (log1p transformation)")
print("    - 理由: 降低偏度，稳定方差")
print("    - 预期: R²提升到 0.3-0.5")
print()
print("  ✓ 策略2: 按HS代码分组建模")
print("    - 理由: 不同商品模式不同")
print("    - 预期: 每个模型专注学习特定商品规律")
print()
print("  ✓ 策略3: 添加HS代码One-hot编码")
print("    - 理由: 让单一模型学习不同商品差异")
print("    - 预期: 比策略2更简单但可能效果略差")
print()
print("  ✓ 策略4: 超参数调优")
print("    - 理由: 默认参数可能不是最优")
print("    - 方法: GridSearch或Optuna")

print("\n下一步: Step 2 - 实施对数变换和数据预处理")
print("="*80)
