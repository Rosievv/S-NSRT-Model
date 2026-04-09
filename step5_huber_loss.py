#!/usr/bin/env python3
"""
Step 5: Huber Loss模型 - 鲁棒回归减少极端值影响

基于Step 4诊断结果：
- 7.8%的高价值样本贡献86.1%的误差
- 预测均值严重低估（$728K vs $41M）
- 模型对大额贸易过于保守

Huber Loss策略:
- 对小误差使用L2（平方误差）- 平滑且可导
- 对大误差使用L1（绝对误差）- 对异常值鲁棒
- 转折点δ由数据自适应决定

知识依据:
- Robust Regression: 减少异常值对参数估计的影响
- M-estimator Theory: Huber (1964) 提出的鲁棒估计
- XGBoost实现: objective='reg:pseudohubererror' 使用伪Huber变体
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.time_series_model import TimeSeriesForecaster, mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Step5")

print("="*80)
print("Step 5: Huber Loss鲁棒回归")
print("="*80)

print("\n【策略说明】")
print("-"*80)
print("Huber Loss定义:")
print("""
    L_δ(r) = { 0.5 * r²           if |r| ≤ δ
             { δ * (|r| - 0.5*δ)  if |r| > δ
    
    其中 r = y_true - y_pred, δ 是转折点
""")
print("\n【判断逻辑】")
print("  Step 4发现: 极端值主导误差（86.1%贡献来自7.8%样本）")
print("  → 普通MSE对大误差惩罚过重 (平方项)")
print("  → Huber Loss在大误差时线性惩罚")
print("  → 期望: 对极端值更鲁棒，整体R²提升")

# =============================================================================
# 5.1 加载数据
# =============================================================================
print("\n" + "="*80)
print("【5.1】加载对数变换数据")
print("="*80)

train_df = pd.read_parquet('data/processed/features_train_log.parquet')
test_df = pd.read_parquet('data/processed/features_test_log.parquet')

print(f"训练集: {len(train_df):,} 样本")
print(f"测试集: {len(test_df):,} 样本")

# =============================================================================
# 5.2 Huber Loss模型训练
# =============================================================================
print("\n" + "="*80)
print("【5.2】训练Huber Loss模型")
print("="*80)

print("\n使用XGBoost的Pseudo-Huber损失:")
print("  objective = 'reg:pseudohubererror'")
print("  优点: 处处可导（对梯度提升算法更好）")

# 初始化模型 - 使用Huber Loss
model_huber = TimeSeriesForecaster(
    target_variable='log_value_usd',
    forecast_horizon=1,
    model_type='xgboost',
    config={
        'objective': 'reg:pseudohubererror',  # ← 关键改变
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'random_state': 42
    }
)

print("\n准备训练数据...")
X_train_full, y_train_full = model_huber.prepare_data(train_df, scale_features=True)

# 划分训练/验证集
split_idx = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val = X_train_full[split_idx:]
y_val = y_train_full[split_idx:]

print(f"训练集: {len(X_train):,} 样本")
print(f"验证集: {len(X_val):,} 样本")

print("\n开始训练...")
model_huber.train(X_train, y_train, X_val, y_val)

# 验证集评估（对数空间）
print("\n验证集性能（对数空间）:")
metrics_val = model_huber.evaluate(X_val, y_val)
print(f"  MAE:  {metrics_val['mae']:.4f}")
print(f"  RMSE: {metrics_val['rmse']:.4f}")
print(f"  R²:   {metrics_val['r2']:.4f}")

# =============================================================================
# 5.3 测试集评估
# =============================================================================
print("\n" + "="*80)
print("【5.3】测试集评估")
print("="*80)

X_test, y_test_log = model_huber.prepare_data(test_df, scale_features=True)

# 对数空间评估
metrics_log = model_huber.evaluate(X_test, y_test_log)

print("\n测试集性能（对数空间）:")
print(f"  MAE:  {metrics_log['mae']:.4f}")
print(f"  RMSE: {metrics_log['rmse']:.4f}")
print(f"  R²:   {metrics_log['r2']:.4f}")

# 原始空间评估
y_pred_log = model_huber.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_log)

mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2_orig = r2_score(y_test_orig, y_pred_orig)

# 计算MAPE（排除零值）
mask = y_test_orig != 0
mape_orig = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100

print("\n测试集性能（原始空间 $）:")
print(f"  MAE:  ${mae_orig:,.2f}")
print(f"  RMSE: ${rmse_orig:,.2f}")
print(f"  R²:   {r2_orig:.4f}")
print(f"  MAPE: {mape_orig:.2f}%")

# =============================================================================
# 5.4 与之前模型对比
# =============================================================================
print("\n" + "="*80)
print("【5.4】模型对比")
print("="*80)

# 历史结果
results_comparison = {
    '基线模型 (MSE)': {
        'mae': 65611270.83,
        'r2': 0.0236,
        'approach': '直接预测value_usd'
    },
    '单一对数模型 (MSE)': {
        'mae': 41074162.79,
        'r2': -0.0438,
        'approach': '对数变换 + MSE损失'
    },
    '分组对数模型 (MSE)': {
        'mae': 55114287.12,
        'r2': -0.0516,
        'approach': '按HS代码分组 + MSE损失'
    },
    'Huber Loss模型': {
        'mae': float(mae_orig),
        'r2': float(r2_orig),
        'approach': '对数变换 + Huber Loss'
    }
}

print("\n完整对比表:")
print("-"*80)
print(f"{'模型':<25} {'MAE ($M)':>12} {'R²':>10} {'方法':<30}")
print("-"*80)

for name, metrics in results_comparison.items():
    print(f"{name:<25} ${metrics['mae']/1e6:>10.1f}M {metrics['r2']:>10.4f}  {metrics['approach']:<30}")

# 计算改进
baseline_mae = 65611270.83
huber_mae = mae_orig
improve_pct = (1 - huber_mae / baseline_mae) * 100

print(f"\n【改进幅度】")
print(f"  相比基线MAE: {improve_pct:+.1f}%")

if r2_orig > 0.5:
    result = "🎉 显著改进！模型达到生产标准"
    color = "excellent"
elif r2_orig > 0.3:
    result = "✓ 明显改进，模型可用"
    color = "good"
elif r2_orig > 0.1:
    result = "→ 有改进，仍需优化"
    color = "moderate"
elif r2_orig > 0:
    result = "⚠️  轻微改进"
    color = "slight"
else:
    result = "✗ 仍未达标"
    color = "poor"

print(f"\n【综合评估】: {result}")

# =============================================================================
# 5.5 按价值区间分析改进
# =============================================================================
print("\n" + "="*80)
print("【5.5】Huber Loss对不同价值区间的影响")
print("="*80)

# 需要对齐test_df
test_df_aligned = test_df.iloc[:-1].copy()
test_df_aligned = test_df_aligned.iloc[:len(y_test_log)]

test_df_analysis = test_df_aligned.copy()
test_df_analysis['y_true'] = y_test_orig
test_df_analysis['y_pred'] = y_pred_orig

# 定义价值区间
bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, np.inf]
labels = ['0-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M+']
test_df_analysis['value_bin'] = pd.cut(y_test_orig, bins=bins, labels=labels)

print("\n各价值区间的误差:")
print("-"*80)
print(f"{'区间':<12} {'样本数':>8} {'MAE ($M)':>12} {'R²':>10}")
print("-"*80)

for label in labels:
    mask = test_df_analysis['value_bin'] == label
    if mask.sum() == 0:
        continue
    
    y_true_bin = y_test_orig[mask]
    y_pred_bin = y_pred_orig[mask]
    
    mae_bin = mean_absolute_error(y_true_bin, y_pred_bin)
    r2_bin = r2_score(y_true_bin, y_pred_bin)
    
    print(f"{label:<12} {mask.sum():>8,} ${mae_bin/1e6:>10.1f}M {r2_bin:>10.4f}")

# 分析预测分布
print("\n【预测分布分析】:")
print(f"  真实值均值: ${y_test_orig.mean():,.0f}")
print(f"  预测值均值: ${y_pred_orig.mean():,.0f}")
print(f"  预测/真实比: {y_pred_orig.mean() / y_test_orig.mean():.2%}")

print(f"\n  真实值标准差: ${y_test_orig.std():,.0f}")
print(f"  预测值标准差: ${y_pred_orig.std():,.0f}")
print(f"  方差比: {y_pred_orig.std() / y_test_orig.std():.2%}")

print(f"\n  预测最大值: ${y_pred_orig.max():,.0f}")
print(f"  真实最大值: ${y_test_orig.max():,.0f}")
print(f"  捕获程度: {y_pred_orig.max() / y_test_orig.max():.2%}")

# =============================================================================
# 5.6 保存模型和结果
# =============================================================================
print("\n" + "="*80)
print("【5.6】保存Huber Loss模型")
print("="*80)

model_dir = Path('models/trained')
model_dir.mkdir(parents=True, exist_ok=True)

# 保存模型
model_huber.save_model(model_dir / 'value_forecaster_xgb_huber')

# 保存对比结果
import json

comparison_data = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'strategy': 'huber_loss_regression',
    'test_metrics': {
        'mae': float(mae_orig),
        'rmse': float(rmse_orig),
        'r2': float(r2_orig),
        'mape': float(mape_orig)
    },
    'all_models_comparison': results_comparison,
    'improvement_vs_baseline': float(improve_pct),
    'prediction_stats': {
        'pred_mean': float(y_pred_orig.mean()),
        'true_mean': float(y_test_orig.mean()),
        'pred_std': float(y_pred_orig.std()),
        'true_std': float(y_test_orig.std()),
        'pred_max': float(y_pred_orig.max()),
        'true_max': float(y_test_orig.max())
    }
}

with open(model_dir / 'huber_loss_comparison.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

print(f"✓ 模型已保存: {model_dir / 'value_forecaster_xgb_huber.pkl'}")
print(f"✓ 对比结果已保存: {model_dir / 'huber_loss_comparison.json'}")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "="*80)
print("【Step 5 总结】")
print("="*80)

print("\n完成的工作:")
print("  ✓ 使用Huber Loss (Pseudo-Huber) 重新训练模型")
print("  ✓ 对数空间和原始空间双重评估")
print("  ✓ 与之前所有模型全面对比")
print("  ✓ 按价值区间分析改进效果")

print("\n【关键发现】:")
if r2_orig > results_comparison['单一对数模型 (MSE)']['r2']:
    print("  ✓ Huber Loss相比MSE改进了R²")
    print("  → 鲁棒损失函数对极端值更有效")
else:
    print("  → Huber Loss未显著改进R²")
    print("  → 问题可能不在损失函数，而在特征工程")

if mae_orig < 45e6:
    print(f"  ✓ MAE降至${mae_orig/1e6:.1f}M (可接受范围)")
else:
    print(f"  → MAE仍为${mae_orig/1e6:.1f}M (偏高)")

print("\n【知识回顾】Huber Loss的效果:")
print("  理论预期: 减少异常值的影响")
print("  实际表现: (取决于结果)")
if r2_orig > 0:
    print("    → 正向改进，鲁棒性提升")
else:
    print("    → 改进有限，需要更根本的特征工程")

print("\n【下一步建议】:")
if r2_orig > 0.3:
    print("  → 模型已达到可用水平")
    print("  → 可以继续超参数调优或添加外部特征")
elif r2_orig > 0.1:
    print("  → 添加时间序列特征（滞后、移动平均、季节性）")
    print("  → 考虑集成多个模型")
else:
    print("  1. 添加滞后特征（lag-1, lag-3, lag-6, lag-12)")
    print("  2. 移动平均特征（MA3, MA6, MA12）")
    print("  3. 季节性特征（月份、季度哑变量）")
    print("  4. 外部特征（汇率、GDP、贸易政策指数）")
    print("  5. 考虑使用专门的时间序列模型（Prophet, LSTM）")

print("="*80)
