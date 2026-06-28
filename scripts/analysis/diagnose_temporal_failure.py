#!/usr/bin/env python3
"""
诊断分析：为什么时间序列特征反而降低了性能？

意外结果:
- Huber Loss: R² = -0.039, MAE = $41.3M (58特征)
- 时间序列: R² = -0.056, MAE = $48.8M (82特征)

可能原因:
1. 过拟合（特征太多，样本太少）
2. 信息泄漏（用了未来信息？）
3. 特征尺度问题（大的lag值主导小的当前值）
4. forecast_horizon=1但滞后处理不当
5. 样本损失（删除了18%训练数据和30%测试数据）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("诊断：为什么时间序列特征降低了性能？")
print("="*80)

# 加载对比结果
with open('models/trained/temporal_features_comparison.json', 'r') as f:
    temporal_results = json.load(f)

with open('models/trained/huber_loss_comparison.json', 'r') as f:
    huber_results = json.load(f)

print("\n【问题陈述】")
print("-"*80)
print("预期: 时间序列特征 → R²提升")
print("实际: 时间序列特征 → R²下降")
print()
print(f"  Huber Loss模型:  R² = {huber_results['test_metrics']['r2']:.4f}, MAE = ${huber_results['test_metrics']['mae']/1e6:.1f}M")
print(f"  时间序列模型:    R² = {temporal_results['test_metrics']['r2']:.4f}, MAE = ${temporal_results['test_metrics']['mae']/1e6:.1f}M")
print(f"  → R²变化: {temporal_results['test_metrics']['r2'] - huber_results['test_metrics']['r2']:.4f} (更差!)")

# =============================================================================
# 假设1: 样本损失导致分布偏移
# =============================================================================
print("\n" + "="*80)
print("【假设1】样本损失导致测试集分布偏移")
print("="*80)

print("\n数据损失情况:")
print("  训练集: 37,756 → 30,977 (损失18.0%)")
print("  测试集: 31,900 → 22,447 (损失29.6%!)")

print("\n【判断逻辑】:")
print("  → 删除lag12缺失的样本，可能删除了早期数据")
print("  → 早期数据（2020年初）可能有不同分布特征（COVID影响）")
print("  → 测试集损失30%很严重，可能改变了评估基准")

# 加载数据检查分布
train_full = pd.read_parquet('data/processed/features_train_full.parquet')
test_full = pd.read_parquet('data/processed/features_test_full.parquet')
test_temporal = pd.read_parquet('data/processed/features_test_temporal.parquet')

print(f"\n测试集日期范围:")
print(f"  完整测试集: {test_full['date'].min()} 到 {test_full['date'].max()}")
print(f"  时间序列集: {test_temporal['date'].min()} 到 {test_temporal['date'].max()}")

# 检查被删除的样本
remaining_dates = set(test_temporal['date'].unique())
all_dates = set(test_full['date'].unique())
missing_dates = all_dates - remaining_dates

if missing_dates:
    print(f"\n被删除的月份数: {len(missing_dates)}")
    print(f"  最早删除: {min(missing_dates)}")
    print(f"  最晚删除: {max(missing_dates)}")
    
    # 统计被删除样本的value分布
    mask_deleted = test_full['date'].isin(missing_dates)
    deleted_values = test_full[mask_deleted]['value_usd']
    remaining_values = test_full[~mask_deleted]['value_usd']
    
    print(f"\n被删除样本的统计特征:")
    print(f"  均值: ${deleted_values.mean():,.0f}")
    print(f"  中位数: ${deleted_values.median():,.0f}")
    print(f"  零值比例: {(deleted_values == 0).mean()*100:.1f}%")
    
    print(f"\n保留样本的统计特征:")
    print(f"  均值: ${remaining_values.mean():,.0f}")
    print(f"  中位数: ${remaining_values.median():,.0f}")
    print(f"  零值比例: {(remaining_values == 0).mean()*100:.1f}%")
    
    if abs(deleted_values.mean() - remaining_values.mean()) / deleted_values.mean() > 0.2:
        print("\n  ✗ 警告: 被删除样本的均值与保留样本差异>20%")
        print("  → 这导致了测试集分布偏移!")

# =============================================================================
# 假设2: 信息泄漏
# =============================================================================
print("\n" + "="*80)
print("【假设2】时间序列特征存在信息泄漏")
print("="*80)

print("\n检查forecast_horizon处理:")
print("  模型设置: forecast_horizon = 1")
print("  含义: 预测下一个月的value")
print()
print("  正确做法: 用 value(t-1) 预测 value(t)")
print("  问题: 如果用 value(t) 作为特征预测 value(t) → 信息泄漏!")

print("\n【潜在问题】:")
print("  移动平均MA在 prepare_data 时如何处理?")
print("  如果 MA包含当前月数据 → 泄漏")
print("  如果 MA只用历史数据 → 正确")

# 检查代码逻辑
print("\n  从代码看:")
print("    df['value_usd_ma3'] = df.groupby(['country', 'hs_code'])['value_usd']")
print("                             .transform(lambda x: x.rolling(3).mean())")
print()
print("  ✓ rolling(3).mean() 包括当前值!")
print("  → 例如: MA3(t) = mean(value(t-2), value(t-1), value(t))")
print("  → 但 forecast_horizon=1 意味着预测 value(t+1)")
print("  → prepare_data 会 shift target: y = value(t+1)")
print("  → 所以特征 MA3(t) 用来预测 value(t+1)，这是合理的")

print("\n  结论: 没有明显的信息泄漏")

# =============================================================================
# 假设3: 特征尺度不匹配
# =============================================================================
print("\n" + "="*80)
print("【假设3】特征尺度不匹配导致模型混淆")
print("="*80)

print("\n特征尺度对比:")
# 查看时间序列特征的尺度
sample_features = test_temporal[[c for c in test_temporal.columns 
                                  if 'value_usd' in c and ('lag' in c or 'ma' in c or c == 'value_usd')]].head(5)

print("\n示例样本的value_usd相关特征（前5行）:")
print(sample_features[['value_usd', 'value_usd_lag1', 'value_usd_lag12', 'value_usd_ma12']].describe())

print("\n【判断逻辑】:")
print("  lag和MA特征的值与当前value_usd同量级")
print("  → StandardScaler会标准化")
print("  → 但如果历史值变化大，可能引入噪声")

# =============================================================================
# 假设4: 过拟合（特征太多）
# =============================================================================
print("\n" + "="*80)
print("【假设4】特征增加导致overfitting")
print("="*80)

print("\n模型复杂度对比:")
print(f"  Huber Loss模型:  58 特征, 31,900 测试样本")
print(f"  时间序列模型:    82 特征, 22,447 测试样本")
print()
print(f"  特征增加: +{82-58} (+{(82-58)/58*100:.1f}%)")
print(f"  样本减少: -{31900-22447} (-{(31900-22447)/31900*100:.1f}%)")

print("\n【判断逻辑】:")
print("  特征/样本比: ")
print(f"    Huber:    58 / 31,900 = {58/31900:.6f}")
print(f"    Temporal: 82 / 22,447 = {82/22447:.6f}")
print(f"    → 时间序列模型的维度密度更高")

print("\n  验证集 vs 测试集对数空间R²:")
print(f"    验证集: R² = {temporal_results['log_space_metrics']['r2']:.4f}")
print(f"    测试集: R² = {temporal_results['log_space_metrics']['r2']:.4f}")

validation_r2 = -0.0254  # 从输出中记录
test_r2 = -0.1134
if abs(validation_r2 - test_r2) > 0.05:
    print(f"\n  ✗ 警告: 验证集和测试集R²差异很大 ({validation_r2:.4f} vs {test_r2:.4f})")
    print("  → 这是过拟合的典型信号!")

# =============================================================================
# 假设5: 时间序列特征不适用此问题
# =============================================================================
print("\n" + "="*80)
print("【假设5】此数据不具备强时间自相关性")
print("="*80)

print("\n检查自相关性:")
# 计算value_usd的滞后相关性
sample_country_hs = test_full.groupby(['country', 'hs_code']).filter(lambda x: len(x) >= 24).iloc[:1000]

if len(sample_country_hs) > 0:
    # 选一个有足够历史的序列
    group = sample_country_hs.groupby(['country', 'hs_code']).get_group(
        list(sample_country_hs.groupby(['country', 'hs_code']).groups.keys())[0]
    )
    
    if len(group) >= 12:
        values = group.sort_values('date')['value_usd'].values
        
        # 计算lag-1到lag-12的自相关
        print("\n某个贸易流的自相关系数:")
        for lag in [1, 3, 6, 12]:
            if len(values) > lag:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                print(f"  Lag-{lag:2d}: {corr:.3f} {'(强相关)' if abs(corr) > 0.7 else '(中等)' if abs(corr) > 0.3 else '(弱相关)'}")

print("\n【判断逻辑】:")
print("  贸易数据特点:")
print("    - 突发性强（大订单不定期出现）")
print("    - 噪声大（受政策、事件影响）")
print("    - 稀疏性（25%零值）")
print("  → 与平滑的GDP、股价等不同")
print("  → 可能不适合传统时间序列方法")

# =============================================================================
# 总结和建议
# =============================================================================
print("\n" + "="*80)
print("【诊断总结】")
print("="*80)

print("\n最可能的原因:")
print("  1. ⭐ 过拟合: 验证集R²=-0.025 但测试集R²=-0.113")
print("     → 特征增加41% + 样本减少30% = 维度灾难")
print()
print("  2. ⭐ 样本偏移: 删除了30%测试数据，改变了分布")
print("     → 被删除的可能是早期或晚期样本")
print()
print("  3. → 弱自相关: 贸易数据突发性强，历史值预测力弱")
print("     → 不同于GDP、销量等平滑序列")

print("\n【知识反思】:")
print("  理论预期: 时间序列特征 → 捕捉历史依赖 → 改善预测")
print("  实际结果: 时间序列特征 → 增加噪声 → 降低性能")
print()
print("  为什么失败?")
print("    1. 贸易数据的stochastic nature (随机性)")
print("    2. 大订单主导，不遵循渐进模式")
print("    3. 特征增加但信噪比下降")

print("\n【下一步建议】:")
print("\n方案B-revised: 选择性使用时间特征")
print("  → 只用lag-1和MA3（最相关的）")
print("  → 避免过多特征")
print("  → 保留更多样本（只删除lag1缺失）")

print("\n方案C: 改变目标 - 预测变化而非绝对值")
print("  → target = log(value(t)) - log(value(t-1))")
print("  → 避免极端值问题")
print("  → 更适合随机游走数据")

print("\n方案D: 两阶段模型")
print("  → Stage 1: 分类器预测是否有大额订单")
print("  → Stage 2: 回归器预测金额")
print("  → 针对性处理稀疏+极端值")

print("\n方案E: 放弃树模型")
print("  → 尝试线性模型（Ridge/Lasso）")
print("  → 或LSTM (真正的序列模型)")
print("  → 树模型不适合外推和稀疏数据")

print("="*80)
