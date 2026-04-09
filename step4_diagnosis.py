#!/usr/bin/env python3
"""
Step 4: 深度诊断 - 分析为什么模型在原始空间表现差

核心问题:
- 对数空间 R² ~ 0.05-0.07 (还可以)
- 原始空间 R² < 0 (很差)  

可能原因:
1. Jensen's Inequality: exp() 放大误差
2. 极端值主导 RMSE/R²
3. 预测分布偏移

诊断策略:
- 按价值区间分析误差
- 比较预测分布 vs 真实分布
- 识别哪些样本贡献了大部分误差
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.time_series_model import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Step4")

print("="*80)
print("Step 4: 深度诊断 - 预测失败的根本原因")
print("="*80)

# =============================================================================
# 4.1 加载模型和数据
# =============================================================================
print("\n【4.1】加载数据和模型")
print("-"*80)

# 重新训练/准备模型以匹配特征
from models.time_series_model import TimeSeriesForecaster

# 加载对数变换后的数据
train_df = pd.read_parquet('data/processed/features_train_log.parquet')
test_df = pd.read_parquet('data/processed/features_test_log.parquet')

# 初始化模型准备数据（与step2一致）
model_log = TimeSeriesForecaster(
    target_variable='log_value_usd',
    forecast_horizon=1,
    model_type='xgboost'
)

# 准备测试数据（使用与训练时相同的方法）
X_test, y_test_log = model_log.prepare_data(test_df, scale_features=True)

# 加载训练好的XGBoost模型
with open('models/trained/value_forecaster_xgb_log.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# 使用训练好的模型预测
y_pred_log = xgb_model.predict(X_test)

# 转回原始空间
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_log)

# 获取对应的test_df记录（注意：prepare_data会去掉forecast_horizon个样本）
test_df_aligned = test_df.iloc[:-1].copy()  # forecast_horizon=1，所以去掉最后1个
test_df_aligned = test_df_aligned.iloc[:len(y_test_log)]  # 确保长度匹配

print(f"测试样本数: {len(y_test_orig):,}")
print(f"预测范围: ${y_pred_orig.min():,.0f} - ${y_pred_orig.max():,.0f}")
print(f"真实范围: ${y_test_orig.min():,.0f} - ${y_test_orig.max():,.0f}")

# =============================================================================
# 4.2 按价值区间分析误差
# =============================================================================
print("\n" + "="*80)
print("【4.2】按价值区间分析误差分布")
print("="*80)

# 定义价值区间（使用对数刻度）
bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, np.inf]
labels = ['0-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M+']

# 分桶
test_df_analysis = test_df_aligned.copy()
test_df_analysis['y_true'] = y_test_orig
test_df_analysis['y_pred'] = y_pred_orig
test_df_analysis['value_bin'] = pd.cut(y_test_orig, bins=bins, labels=labels)

print("\n各价值区间的误差分析:")
print("-"*80)
print(f"{'区间':<12} {'样本数':>8} {'占比%':>7} {'平均MAE':>15} {'平均R²':>10} {'误差贡献%':>12}")
print("-"*80)

total_error = np.sum(np.abs(y_test_orig - y_pred_orig))

for label in labels:
    mask = test_df_analysis['value_bin'] == label
    if mask.sum() == 0:
        continue
    
    y_true_bin = y_test_orig[mask]
    y_pred_bin = y_pred_orig[mask]
    
    mae_bin = mean_absolute_error(y_true_bin, y_pred_bin)
    r2_bin = r2_score(y_true_bin, y_pred_bin)
    error_contrib = np.sum(np.abs(y_true_bin - y_pred_bin)) / total_error * 100
    
    print(f"{label:<12} {mask.sum():>8,} {mask.sum()/len(test_df_analysis)*100:>6.1f}% ${mae_bin:>13,.0f} {r2_bin:>10.4f} {error_contrib:>11.1f}%")

print("\n【判断逻辑】:")
print("  - 如果高价值区间的误差贡献过大 → 极端值主导了整体表现")
print("  - 如果低价值区间样本多但R²差 → 小值预测准确性问题")
print("  - 误差贡献不均匀 → 需要加权损失函数")

# =============================================================================
# 4.3 分析 top 误差样本
# =============================================================================
print("\n" + "="*80)
print("【4.3】Top 10 最大误差样本分析")
print("="*80)

test_df_analysis['abs_error'] = np.abs(y_test_orig - y_pred_orig)
test_df_analysis['rel_error'] = test_df_analysis['abs_error'] / (y_test_orig + 1)

top_errors = test_df_analysis.nlargest(10, 'abs_error')

print("\n按绝对误差排序 (Top 10):")
print("-"*80)
print(f"{'真实值':>15} {'预测值':>15} {'绝对误差':>15} {'HS代码':>10} {'国家':>8}")
print("-"*80)
for idx, row in top_errors.iterrows():
    print(f"${row['y_true']:>14,.0f} ${row['y_pred']:>14,.0f} ${row['abs_error']:>14,.0f} {row['hs_code']:>10} {row['country']:>8}")

# 计算top样本的误差占比
top_10_error = top_errors['abs_error'].sum()
print(f"\nTop 10样本误差贡献: {top_10_error/total_error*100:.1f}%")

top_100_error = test_df_analysis.nlargest(100, 'abs_error')['abs_error'].sum()
print(f"Top 100样本误差贡献: {top_100_error/total_error*100:.1f}%")

print("\n【判断逻辑】:")
if top_10_error/total_error > 0.3:
    print("  ✗ Top 10样本贡献>30%误差 → 极端值主导")
    print("  → 建议: Huber Loss 或 分位数回归")
elif top_100_error/total_error > 0.5:
    print("  → Top 100样本贡献>50%误差 → 长尾分布")
    print("  → 建议: 加权损失或集成多个模型")
else:
    print("  ✓ 误差分布相对均匀")

# =============================================================================
# 4.4 对数空间误差 vs 原始空间误差
# =============================================================================
print("\n" + "="*80)
print("【4.4】对数空间误差放大分析 (Jensen's Inequality)")
print("="*80)

# 对数空间误差
error_log = np.abs(y_test_log - y_pred_log)

# 原始空间误差
error_orig = np.abs(y_test_orig - y_pred_orig)

# 分析放大倍数
test_df_analysis['error_log'] = error_log
test_df_analysis['error_orig'] = error_orig
test_df_analysis['amplification'] = error_orig / (error_log * y_test_orig + 1)  # 简化的放大系数

print(f"对数空间平均误差: {error_log.mean():.4f}")
print(f"对数空间中位误差: {np.median(error_log):.4f}")
print(f"原始空间平均误差: ${error_orig.mean():,.0f}")
print(f"原始空间中位误差: ${np.median(error_orig):,.0f}")

# 按真实值大小分析放大
print("\n按真实值大小分析误差放大:")
print("-"*80)
quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
for q in quantiles:
    val = np.quantile(y_test_orig, q)
    mask = y_test_orig <= val
    if mask.sum() == 0:
        continue
    avg_error_q = error_orig[mask].mean()
    print(f"  {q*100:>5.0f}% 分位数 (≤${val:>12,.0f}): 平均误差 ${avg_error_q:>12,.0f}")

print("\n【知识依据】Jensen's Inequality:")
print("  对于凸函数 f(x) = exp(x)，有 E[f(X)] ≥ f(E[X])")
print("  → 即使对数空间预测很准，exp()会放大误差")  
print("  → 高价值样本（大的log值）放大更严重")

# =============================================================================
# 4.5 预测分布 vs 真实分布
# =============================================================================
print("\n" + "="*80)
print("【4.5】预测分布 vs 真实分布分析")
print("="*80)

print("\n真实值统计:")
print(f"  均值: ${y_test_orig.mean():,.0f}")
print(f"  中位数: ${np.median(y_test_orig):,.0f}")
print(f"  标准差: ${y_test_orig.std():,.0f}")
print(f"  偏度: {pd.Series(y_test_orig).skew():.2f}")

print("\n预测值统计:")
print(f"  均值: ${y_pred_orig.mean():,.0f}")
print(f"  中位数: ${np.median(y_pred_orig):,.0f}")
print(f"  标准差: ${y_pred_orig.std():,.0f}")
print(f"  偏度: {pd.Series(y_pred_orig).skew():.2f}")

# 零值预测
zero_true = (y_test_orig == 0).sum()
near_zero_pred = (y_pred_orig < 1000).sum()
print(f"\n零值样本: 真实={zero_true:,}, 预测接近零(<1K)={near_zero_pred:,}")

print("\n【判断逻辑】:")
if abs(y_pred_orig.mean() - y_test_orig.mean()) / y_test_orig.mean() > 0.2:
    print("  ✗ 预测均值偏离真实均值>20% → 系统性偏差")
    print("  → 可能原因: 训练集与测试集分布不同（时间漂移）")
else:
    print("  ✓ 预测均值接近真实均值")

if abs(y_pred_orig.std() / y_test_orig.std() - 1) > 0.5:
    print("  ✗ 预测方差与真实方差差异大 → 模型过于保守或激进")
else:
    print("  ✓ 预测方差合理")

# =============================================================================
# 4.6 可视化诊断
# =============================================================================
print("\n" + "="*80)
print("【4.6】生成诊断可视化")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Diagnosis: Why Original Space R² is Negative', fontsize=16, fontweight='bold')

# 1. 预测 vs 真实 (对数空间)
ax = axes[0, 0]
ax.scatter(y_test_log, y_pred_log, alpha=0.3, s=10)
ax.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--', lw=2)
ax.set_xlabel('True Value (log space)')
ax.set_ylabel('Predicted Value (log space)')
ax.set_title(f'Log Space: R²={r2_score(y_test_log, y_pred_log):.3f}')
ax.grid(True, alpha=0.3)

# 2. 预测 vs 真实 (原始空间，限制显示范围)
ax = axes[0, 1]
display_max = np.quantile(y_test_orig, 0.95)
mask_display = (y_test_orig < display_max) & (y_pred_orig < display_max)
ax.scatter(y_test_orig[mask_display], y_pred_orig[mask_display], alpha=0.3, s=10)
ax.plot([0, display_max], [0, display_max], 'r--', lw=2)
ax.set_xlabel('True Value ($)')
ax.set_ylabel('Predicted Value ($)')
ax.set_title(f'Original Space (95%): R²={r2_score(y_test_orig, y_pred_orig):.3f}')
ax.ticklabel_format(style='plain', axis='both')
ax.grid(True, alpha=0.3)

# 3. 残差分布
ax = axes[0, 2]
residuals = y_test_orig - y_pred_orig
ax.hist(residuals / 1e6, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Residual (Million $)')
ax.set_ylabel('Frequency')
ax.set_title(f'Residual Distribution\nMean: ${residuals.mean()/1e6:.1f}M')
ax.axvline(0, color='r', linestyle='--', lw=2)
ax.grid(True, alpha=0.3)

# 4. 按价值区间的误差分布
ax = axes[1, 0]
bin_stats = []
for label in labels:
    mask = test_df_analysis['value_bin'] == label
    if mask.sum() > 0:
        mae_bin = mean_absolute_error(y_test_orig[mask], y_pred_orig[mask])
        bin_stats.append({'label': label, 'mae': mae_bin, 'count': mask.sum()})

if bin_stats:
    df_bins = pd.DataFrame(bin_stats)
    bars = ax.bar(range(len(df_bins)), df_bins['mae'] / 1e6)
    ax.set_xticks(range(len(df_bins)))
    ax.set_xticklabels(df_bins['label'], rotation=45, ha='right')
    ax.set_ylabel('MAE (Million $)')
    ax.set_title('MAE by Value Range')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注样本数
    for i, (bar, count) in enumerate(zip(bars, df_bins['count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)

# 5. 累积误差贡献
ax = axes[1, 1]
sorted_errors = np.sort(error_orig)[::-1]
cumulative_error = np.cumsum(sorted_errors) / total_error * 100
ax.plot(range(len(cumulative_error)), cumulative_error, linewidth=2)
ax.axhline(50, color='r', linestyle='--', label='50% error')
ax.axhline(80, color='orange', linestyle='--', label='80% error')
ax.set_xlabel('Number of Samples (ranked by error)')
ax.set_ylabel('Cumulative Error Contribution (%)')
ax.set_title('Pareto Chart of Errors')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 对数误差 vs 原始误差
ax = axes[1, 2]
sample_indices = np.random.choice(len(y_test_orig), min(5000, len(y_test_orig)), replace=False)
ax.scatter(error_log[sample_indices], error_orig[sample_indices]/1e6, alpha=0.3, s=10)
ax.set_xlabel('Log Space Error')
ax.set_ylabel('Original Space Error (Million $)')
ax.set_title('Error Amplification\n(log → original)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 保存
report_dir = Path('reports')
report_dir.mkdir(exist_ok=True)
plt.savefig(report_dir / 'step4_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"✓ 保存可视化: {report_dir / 'step4_diagnosis.png'}")

# =============================================================================
# 总结和建议
# =============================================================================
print("\n" + "="*80)
print("【Step 4 诊断总结】")
print("="*80)

print("\n核心发现:")
print("  1. 对数空间R²尚可 (~0.05-0.07)")
print("  2. 原始空间R²负值 → exp()放大了误差")
print(f"  3. Top 10样本误差占比: {top_10_error/total_error*100:.1f}%")
print(f"  4. Top 100样本误差占比: {top_100_error/total_error*100:.1f}%")

print("\n【根本原因】:")
print("  ✗ Jensen's Inequality效应显著")
print("  ✗ 极端值（高贸易额）预测误差被指数放大")
print("  ✗ 模型在对数空间表现一般，反变换后更差")

print("\n【知识依据与建议】:")
print("\n方案1: 【推荐】改用鲁棒损失函数")
print("  - 理论: Huber Loss在大误差时从L2切换到L1")
print("  - 优点: 对极端值不敏感，避免误差爆炸")
print("  - 实现: XGBoost支持 objective='reg:pseudohubererror'")

print("\n方案2: 直接在对数空间评估")
print("  - 如果业务目标是相对误差而非绝对误差")
print("  - 评估MAPE而非MAE/RMSE")
print("  - 对数空间R²=0.07意味着7%方差可解释")

print("\n方案3: 分位数回归 + 集成")
print("  - 训练多个模型预测不同分位数（P10, P50, P90）")
print("  - 构建预测区间而非点估计")
print("  - 理论依据: Quantile Regression对异质方差鲁棒")

print("\n方案4: 时间序列建模")
print("  - 当前模型未充分利用时间特征")
print("  - 考虑LSTM、Prophet或统计模型（ARIMA）")
print("  - 贸易数据通常有季节性和趋势")

print("\n【下一步行动】:")
print("  优先级1: 尝试Huber Loss (对当前架构改动最小)")
print("  优先级2: 添加滞后特征和移动平均（时间序列特征）")
print("  优先级3: 评估分位数回归模型")

print("="*80)

# 保存诊断报告
diagnosis_report = {
    'log_space_r2': float(r2_score(y_test_log, y_pred_log)),
    'original_space_r2': float(r2_score(y_test_orig, y_pred_orig)),
    'top_10_error_contribution': float(top_10_error/total_error),
    'top_100_error_contribution': float(top_100_error/total_error),
    'mean_log_error': float(error_log.mean()),
    'mean_original_error': float(error_orig.mean()),
    'prediction_mean': float(y_pred_orig.mean()),
    'ground_truth_mean': float(y_test_orig.mean()),
    'recommendations': [
        'Use Huber Loss to reduce sensitivity to outliers',
        'Add temporal features (lags, moving averages)',
        'Consider quantile regression for prediction intervals',
        'Evaluate LSTM or Prophet for time series patterns'
    ]
}

with open('reports/step4_diagnosis.json', 'w') as f:
    json.dump(diagnosis_report, f, indent=2)

print(f"\n✓ 诊断报告已保存: reports/step4_diagnosis.json")
