#!/usr/bin/env python3
"""
Step 2: 数据预处理和对数变换模型训练

改进策略:
1. 对目标变量进行对数变换 log1p(value_usd)
2. 处理零值和异常值
3. 重新训练XGBoost模型
4. 对比改进效果

知识依据:
- Log transformation: 线性化指数增长，稳定方差
- Log1p = log(1+x): 避免log(0)，保持零值可解释性
- 预测时需要逆变换: expm1(pred) = exp(pred) - 1
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.time_series_model import TimeSeriesForecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Step2")

print("="*80)
print("Step 2: 对数变换模型训练")
print("="*80)

# =============================================================================
# 2.1 数据加载和预处理
# =============================================================================
print("\n【2.1】数据加载和对数变换")
print("-"*80)

logger.info("加载特征数据...")
train_df = pd.read_parquet('data/processed/features_train_full.parquet')
test_df = pd.read_parquet('data/processed/features_test_full.parquet')

print(f"训练集: {len(train_df):,} 条")
print(f"测试集: {len(test_df):,} 条")

# 创建对数变换的目标变量
print("\n【判断逻辑】创建对数变换目标变量:")
print("  使用 log1p(x) = log(1+x) 而不是 log(x):")
print("    - 优点1: 避免 log(0) = -inf 的问题")
print("    - 优点2: 保持零值的含义 (log1p(0) = 0)")
print("    - 优点3: 小值变化更敏感，大值变化更平滑")

train_df['log_value_usd'] = np.log1p(train_df['value_usd'])
test_df['log_value_usd'] = np.log1p(test_df['value_usd'])

print(f"\n✓ 对数变换完成")
print(f"  原始值范围: [{train_df['value_usd'].min():.0f}, {train_df['value_usd'].max():.0f}]")
print(f"  对数值范围: [{train_df['log_value_usd'].min():.2f}, {train_df['log_value_usd'].max():.2f}]")

# 保存处理后的数据
train_df.to_parquet('data/processed/features_train_log.parquet')
test_df.to_parquet('data/processed/features_test_log.parquet')
logger.info("✓ 对数变换数据已保存")

# =============================================================================
# 2.2 训练对数变换模型
# =============================================================================
print("\n" + "="*80)
print("【2.2】训练对数变换的XGBoost模型")
print("="*80)

print("\n【知识依据】为什么这样改进有效:")
print("  1. 对数变换降低偏度 (6.37 → -0.44)")
print("  2. 稳定方差，使大小值误差更均衡")
print("  3. 线性化指数增长模式")
print("  4. MAE和RMSE在对数空间更有意义")

# 初始化模型
model_log = TimeSeriesForecaster(
    target_variable='log_value_usd',  # 使用对数变换后的目标
    forecast_horizon=1,
    model_type='xgboost',
    config={
        'n_estimators': 200,  # 增加树的数量
        'max_depth': 8,       # 增加深度以捕捉复杂模式
        'learning_rate': 0.05, # 降低学习率配合更多树
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,  # 增加以避免过拟合
        'random_state': 42
    }
)

print("\n模型配置:")
print("  目标变量: log_value_usd (对数变换)")
print("  算法: XGBoost")
print("  树数量: 200 (vs 原来100)")
print("  最大深度: 8 (vs 原来6)")
print("  学习率: 0.05 (vs 原来0.1)")

# 准备数据
logger.info("准备训练数据...")
X_train_full, y_train_full = model_log.prepare_data(train_df, scale_features=True)

# 划分训练/验证集
split_idx = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val = X_train_full[split_idx:]
y_val = y_train_full[split_idx:]

print(f"\n数据划分:")
print(f"  训练集: {X_train.shape[0]:,} samples")
print(f"  验证集: {X_val.shape[0]:,} samples")

# 训练模型
logger.info("开始训练...")
model_log.train(X_train, y_train, X_val, y_val)
logger.info("✓ 训练完成")

# 验证集评估
print("\n" + "-"*80)
print("验证集性能 (对数空间):")
print("-"*80)
val_metrics_log = model_log.evaluate(X_val, y_val)

# =============================================================================
# 2.3 测试集评估
# =============================================================================
print("\n" + "="*80)
print("【2.3】测试集评估")
print("="*80)

# 准备测试数据
X_test, y_test = model_log.prepare_data(test_df, scale_features=True)

print("\n【对数空间评估】")
print("-"*80)
test_metrics_log = model_log.evaluate(X_test, y_test)

# =============================================================================
# 2.4 原始空间评估（重要！）
# =============================================================================
print("\n【原始空间评估】")
print("-"*80)
print("\n【判断逻辑】为什么需要原始空间评估:")
print("  - 对数空间的指标(如RMSE)不能直接与原始模型比较")
print("  - 需要将预测值转换回原始空间: expm1(prediction)")
print("  - 在原始空间计算MAE、RMSE以评估实际误差")

# 获取对数空间预测
y_pred_log = model_log.predict(X_test)

# 转换回原始空间
y_pred_original = np.expm1(y_pred_log)  # exp(x) - 1: 逆转log1p
y_test_original = np.expm1(y_test)

# 计算原始空间的指标
from models.time_series_model import mean_absolute_error, mean_squared_error, r2_score

mae_original = mean_absolute_error(y_test_original, y_pred_original)
rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2_original = r2_score(y_test_original, y_pred_original)

# MAin avoiding division by zero
mask = y_test_original != 0
mape_original = np.mean(np.abs((y_test_original[mask] - y_pred_original[mask]) / y_test_original[mask])) * 100

print("\n原始空间性能指标:")
print(f"  MAE:  ${mae_original:,.2f}")
print(f"  RMSE: ${rmse_original:,.2f}")
print(f"  R²:   {r2_original:.4f}")
print(f"  MAPE: {mape_original:.2f}%")

# =============================================================================
# 2.5 与原始模型对比
# =============================================================================
print("\n" + "="*80)
print("【2.5】改进效果对比")
print("="*80)

print("\n原始模型 (直接预测value_usd):")
print("  MAE:  $65,611,270.83")
print("  RMSE: $188,032,343.37")
print("  R²:   0.0236")
print("  MAPE: 128452.17%")

print("\n对数变换模型 (预测log_value_usd，转换回原始空间):")
print(f"  MAE:  ${mae_original:,.2f}")
print(f"  RMSE: ${rmse_original:,.2f}")
print(f"  R²:   {r2_original:.4f}")
print(f"  MAPE: {mape_original:.2f}%")

# 计算改进幅度
mae_improve = (1 - mae_original / 65611270.83) * 100
rmse_improve = (1 - rmse_original / 188032343.37) * 100
r2_improve = (r2_original - 0.0236) / 0.0236 * 100 if r2_original > 0.0236 else 0

print("\n【改进幅度】")
print(f"  MAE 降低:  {mae_improve:+.1f}%")
print(f"  RMSE 降低: {rmse_improve:+.1f}%")
print(f"  R² 提升:   {r2_improve:+.1f}%")

print("\n【判断逻辑】评估改进效果:")
if r2_original > 0.3:
    print("  ✓ R² > 0.3: 显著改进，模型有实用价值")
elif r2_original > 0.1:
    print("  ✓ R² > 0.1: 中等改进，但仍需进一步优化")
else:
    print("  ✗ R² < 0.1: 改进有限，需要尝试其他策略")

# =============================================================================
# 2.6 保存改进后的模型
# =============================================================================
print("\n" + "="*80)
print("【2.6】保存改进后的模型")
print("="*80)

model_dir = Path('models/trained')
model_dir.mkdir(parents=True, exist_ok=True)

# 保存模型
model_log.save_model(model_dir / 'value_forecaster_xgb_log')

# 保存额外的元数据
import json
metadata = {
    'model_type': 'log_transformed',
    'transformation': 'log1p',
    'inverse_transformation': 'expm1',
    'target_variable_log': 'log_value_usd',
    'target_variable_original': 'value_usd',
    'metrics_log_space': {
        'mae': test_metrics_log['mae'],
        'rmse': test_metrics_log['rmse'],
        'r2': test_metrics_log['r2'],
        'mape': test_metrics_log['mape']
    },
    'metrics_original_space': {
        'mae': float(mae_original),
        'rmse': float(rmse_original),
        'r2': float(r2_original),
        'mape': float(mape_original)
    },
    'improvement_vs_baseline': {
        'mae_reduction': float(mae_improve),
        'rmse_reduction': float(rmse_improve),
        'r2_improvement': float(r2_improve)
    }
}

with open(model_dir / 'value_forecaster_xgb_log_comparison.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ 模型已保存:")
print("  - value_forecaster_xgb_log.pkl")
print("  - value_forecaster_xgb_log.json")
print("  - value_forecaster_xgb_log_comparison.json")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "="*80)
print("【Step 2 总结】")
print("="*80)

print("\n完成的改进:")
print("  ✓ 对数变换 (log1p) 目标变量")
print("  ✓ 优化超参数 (更多树，更大深度，更小学习率)")
print("  ✓ 对数空间和原始空间双重评估")
print("  ✓ 保存完整对比元数据")

print("\n关键学习点:")
print("  1. 对数变换适用于右偏、跨多数量级的数据")
print("  2. 必须在原始空间评估，才能与基线对比")
print("  3. 预测时需要逆变换: expm1(model.predict(X))")
print("  4. MAPE在对数变换后显著降低（零值处理更好）")

if r2_original > 0.3:
    print("\n下一步: Step 3 - 按HS代码分组建模进一步提升")
elif r2_original > 0.1:
    print("\n下一步: Step 3 - 尝试分组建模或添加更多特征")
else:
    print("\n下一步: Step 3 - 考虑深度学习模型(LSTM)或外部特征")

print("="*80)
