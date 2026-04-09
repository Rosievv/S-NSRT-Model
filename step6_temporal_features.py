#!/usr/bin/env python3
"""
Step 6: 方案A - 时间序列特征工程

基于Step 4-5诊断结果，根本问题是缺失时间序列特征。
当前模型只用静态特征（HHI、增长率、波动性），无法捕捉时间依赖。

时间序列特征策略:
1. 滞后特征 (Lag Features): value(t-1), value(t-3), value(t-6), value(t-12)
2. 移动平均 (Moving Average): MA3, MA6, MA12
3. 趋势特征 (Trend): value/MA12, diff from MA
4. 季节性 (Seasonality): month_sin, month_cos, quarter dummies

知识依据:
- Box-Jenkins Methodology: ARIMA模型的核心是滞后项和移动平均
- Autocorrelation: 贸易值通常有很强的自相关性
- Seasonal Decomposition: 贸易数据有明显的季节模式（年底旺季等）
- Feature Engineering for Time Series (Hyndman & Athanasopoulos, 2018)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.time_series_model import TimeSeriesForecaster, mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Step6")

print("="*80)
print("Step 6: 方案A - 时间序列特征工程")
print("="*80)

print("\n【策略说明】")
print("-"*80)
print("当前问题诊断:")
print("  ✗ R² = -0.04 (预测不如均值)")
print("  ✗ 预测均值严重低估（5% of actual）")
print("  ✗ 模型过于保守，无法捕捉极端值")
print("\n根本原因:")
print("  → 缺失时间依赖信息（没有lag features）")
print("  → 忽略季节性模式（没有seasonal encoding）")
print("  → 无法捕捉趋势（没有moving average）")
print("\n解决方案:")
print("  ✓ 添加滞后特征: 利用历史值预测未来")
print("  ✓ 添加移动平均: 平滑噪声，捕捉趋势")
print("  ✓ 添加季节性编码: 捕捉周期性模式")

print("\n【知识依据】")
print("  1. Autocorrelation: 时间序列值之间通常相关")
print("  2. Box-Jenkins: ARIMA的AR部分就是滞后项")
print("  3. Seasonal Patterns: 贸易有季节性（如Q4旺季）")
print("  4. Feature Engineering: 好的特征 > 复杂的模型")

# =============================================================================
# 6.1 加载原始数据
# =============================================================================
print("\n" + "="*80)
print("【6.1】加载数据（使用原始特征集）")
print("="*80)

train_df = pd.read_parquet('data/processed/features_train_full.parquet')
test_df = pd.read_parquet('data/processed/features_test_full.parquet')

print(f"训练集: {len(train_df):,} 样本")
print(f"测试集: {len(test_df):,} 样本")
print(f"当前特征数: {len([c for c in train_df.columns if c not in ['date', 'hs_code', 'country', 'year', 'month', 'time_idx', 'value_usd', 'quantity']])} 列")

# =============================================================================
# 6.2 时间序列特征工程
# =============================================================================
print("\n" + "="*80)
print("【6.2】构建时间序列特征")
print("="*80)

def add_temporal_features(df, target_col='value_usd'):
    """
    为数据集添加时间序列特征
    
    参数:
        df: 数据框（必须有 date, country, hs_code, target_col）
        target_col: 目标变量列名
    
    返回:
        添加了时间特征的数据框
    """
    df = df.sort_values(['country', 'hs_code', 'date']).copy()
    
    # 按 country-hs_code 分组（每个贸易流独立）
    group_cols = ['country', 'hs_code']
    
    print(f"\n构建特征基于: {target_col}")
    print("-"*80)
    
    # ==========================
    # 1. 滞后特征 (Lag Features)
    # ==========================
    print("1. 滞后特征 (Lag Features)")
    print("   理论: 时间序列值通常与过去值相关（autocorrelation）")
    print("   实现: value(t) 依赖 value(t-1), value(t-3), value(t-6), value(t-12)")
    
    for lag in [1, 3, 6, 12]:
        col_name = f'{target_col}_lag{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        print(f"   → {col_name}: 平均缺失率 {df[col_name].isna().mean()*100:.1f}%")
    
    # ==========================
    # 2. 移动平均 (Moving Average)
    # ==========================
    print("\n2. 移动平均 (Moving Average)")
    print("   理论: 平滑短期波动，捕捉长期趋势")
    print("   实现: MA3, MA6, MA12 窗口")
    
    for window in [3, 6, 12]:
        col_name = f'{target_col}_ma{window}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        print(f"   → {col_name}: 捕捉 {window} 个月趋势")
    
    # ==========================
    # 3. 趋势特征 (Trend Features)
    # ==========================
    print("\n3. 趋势特征 (Trend)")
    print("   理论: 相对于趋势的偏离度表示异常或周期波动")
    
    # 相对MA的位置
    df[f'{target_col}_trend_ratio'] = df[target_col] / (df[f'{target_col}_ma12'] + 1)
    print(f"   → trend_ratio: value / MA12")
    
    # 一阶差分（变化量）
    df[f'{target_col}_diff1'] = df.groupby(group_cols)[target_col].diff(1)
    print(f"   → diff1: value(t) - value(t-1)")
    
    # 相对变化率
    df[f'{target_col}_pct_change'] = df.groupby(group_cols)[target_col].pct_change(1)
    print(f"   → pct_change: (value(t) - value(t-1)) / value(t-1)")
    
    # ==========================
    # 4. 季节性特征 (Seasonality)
    # ==========================
    print("\n4. 季节性特征 (Seasonality)")
    print("   理论: 贸易有周期性模式（如年底旺季、春节淡季）")
    print("   实现: 三角函数编码月份 + Quarter类别")
    
    # 月份的周期性编码（避免12和1的断层）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print(f"   → month_sin/cos: 捕捉月度周期")
    
    # 季度（Q1淡季 vs Q4旺季）
    df['quarter'] = (df['month'] - 1) // 3 + 1
    print(f"   → quarter: 1-4 表示Q1-Q4")
    
    # ==========================
    # 5. 统计特征 (Rolling Statistics)
    # ==========================
    print("\n5. 滚动统计特征 (Rolling Statistics)")
    print("   理论: 波动性和稳定性是重要的预测信号")
    
    # 滚动标准差（波动性）
    df[f'{target_col}_std6'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.rolling(window=6, min_periods=1).std()
    )
    print(f"   → std6: 6个月滚动标准差（波动性）")
    
    # 滚动最大值
    df[f'{target_col}_max6'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.rolling(window=6, min_periods=1).max()
    )
    print(f"   → max6: 6个月最大值")
    
    # 滚动最小值
    df[f'{target_col}_min6'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.rolling(window=6, min_periods=1).min()
    )
    print(f"   → min6: 6个月最小值")
    
    return df

print("\n处理训练集...")
train_df_enhanced = add_temporal_features(train_df, 'value_usd')

print("\n处理测试集...")
test_df_enhanced = add_temporal_features(test_df, 'value_usd')

# 统计新特征
new_temporal_features = [col for col in train_df_enhanced.columns 
                         if any(x in col for x in ['lag', 'ma', 'trend', 'diff', 'pct_change', 
                                                     'month_sin', 'month_cos', 'quarter', 
                                                     'std6', 'max6', 'min6'])]

print(f"\n✓ 新增时间序列特征: {len(new_temporal_features)} 个")
print(f"  总特征数: {len([c for c in train_df_enhanced.columns if c not in ['date', 'hs_code', 'country', 'year', 'month', 'time_idx', 'value_usd', 'quantity']])} 列")

# =============================================================================
# 6.3 处理缺失值
# =============================================================================
print("\n" + "="*80)
print("【6.3】处理缺失值")
print("="*80)

# 检查缺失情况
missing_stats = train_df_enhanced[new_temporal_features].isna().sum()
print("\n新特征缺失值统计:")
print(missing_stats[missing_stats > 0])

print("\n【判断逻辑】:")
print("  - 滞后特征会有缺失（如lag12，前12个月无历史）")
print("  - 选项1: 删除缺失样本（会损失数据）")
print("  - 选项2: 填充0或均值（引入bias）")
print("  - 选项3: 使用min_periods保留部分样本")
print("  → 采用: 删除lag12缺失的样本（保证质量）")

# 只删除lag12缺失的（这是最长的滞后）
train_df_enhanced = train_df_enhanced.dropna(subset=['value_usd_lag12'])
test_df_enhanced = test_df_enhanced.dropna(subset=['value_usd_lag12'])

# 剩余缺失值填充（如pct_change在第一个值会是NaN）
for col in new_temporal_features:
    if train_df_enhanced[col].isna().any():
        train_df_enhanced[col].fillna(0, inplace=True)
        test_df_enhanced[col].fillna(0, inplace=True)

print(f"\n✓ 训练集保留: {len(train_df_enhanced):,} 样本 (原{len(train_df):,})")
print(f"✓ 测试集保留: {len(test_df_enhanced):,} 样本 (原{len(test_df):,})")

# =============================================================================
# 6.4 对数变换（继续使用，因为已验证有效）
# =============================================================================
print("\n" + "="*80)
print("【6.4】对数变换目标变量")
print("="*80)

print("继续使用log1p变换:")
print("  理由: Step 2验证了对数空间R²=0.84（对数空间建模有效）")
print("  方法: 在原始value_usd上应用，加上时间特征后再建模")

train_df_enhanced['log_value_usd'] = np.log1p(train_df_enhanced['value_usd'])
test_df_enhanced['log_value_usd'] = np.log1p(test_df_enhanced['value_usd'])

# 对所有lag和ma特征也做对数变换
print("\n对时间序列特征也应用log变换:")
for col in new_temporal_features:
    if 'lag' in col or 'ma' in col or 'diff' in col or 'std' in col or 'max' in col or 'min' in col:
        if col in train_df_enhanced.columns:
            # 对于可能有负值的diff和pct_change，不做log变换
            if 'diff' in col or 'pct_change' in col:
                continue
            # 对于其他特征，log变换
            train_df_enhanced[f'log_{col}'] = np.log1p(train_df_enhanced[col])
            test_df_enhanced[f'log_{col}'] = np.log1p(test_df_enhanced[col])
            print(f"  → log_{col}")

# 更新特征列表
all_features = [c for c in train_df_enhanced.columns 
                if c not in ['date', 'hs_code', 'country', 'year', 'month', 
                             'time_idx', 'value_usd', 'quantity', 'log_value_usd']]

temporal_features_count = len([f for f in all_features if any(x in f for x in 
                                ['lag', 'ma', 'trend', 'diff', 'pct_change', 
                                 'month_sin', 'month_cos', 'quarter', 'std', 'max', 'min', 'log_'])])

print(f"\n✓ 总特征数: {len(all_features)}")
print(f"  其中时间序列特征: {temporal_features_count}")

# =============================================================================
# 6.5 模型训练 - XGBoost with Temporal Features
# =============================================================================
print("\n" + "="*80)
print("【6.5】训练增强的时间序列模型")
print("="*80)

print("\n模型配置:")
print("  算法: XGBoost")
print("  损失: reg:squarederror (标准MSE)")
print("  目标: log_value_usd (对数空间)")
print("  特征: 原始特征 + 时间序列特征")

# 初始化模型
model_temporal = TimeSeriesForecaster(
    target_variable='log_value_usd',
    forecast_horizon=1,
    model_type='xgboost',
    config={
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'random_state': 42
    }
)

# 准备数据
print("\n准备训练数据...")
X_train_full, y_train_full = model_temporal.prepare_data(train_df_enhanced, scale_features=True)

# 划分训练/验证集
split_idx = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val = X_train_full[split_idx:]
y_val = y_train_full[split_idx:]

print(f"训练集: {len(X_train):,} 样本, {X_train.shape[1]} 特征")
print(f"验证集: {len(X_val):,} 样本")

# 训练
print("\n开始训练...")
model_temporal.train(X_train, y_train, X_val, y_val)

# 验证集评估
print("\n验证集性能（对数空间）:")
metrics_val = model_temporal.evaluate(X_val, y_val)
print(f"  MAE:  {metrics_val['mae']:.4f}")
print(f"  RMSE: {metrics_val['rmse']:.4f}")
print(f"  R²:   {metrics_val['r2']:.4f}")

# =============================================================================
# 6.6 测试集评估
# =============================================================================
print("\n" + "="*80)
print("【6.6】测试集评估")
print("="*80)

X_test, y_test_log = model_temporal.prepare_data(test_df_enhanced, scale_features=True)

print(f"测试集: {len(X_test):,} 样本")

# 对数空间评估
metrics_log = model_temporal.evaluate(X_test, y_test_log)

print("\n测试集性能（对数空间）:")
print(f"  MAE:  {metrics_log['mae']:.4f}")
print(f"  RMSE: {metrics_log['rmse']:.4f}")
print(f"  R²:   {metrics_log['r2']:.4f}")
print(f"  MAPE: {metrics_log['mape']:.2f}%")

# 原始空间评估
y_pred_log = model_temporal.predict(X_test)
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
# 6.7 与之前所有模型全面对比
# =============================================================================
print("\n" + "="*80)
print("【6.7】完整模型对比表】")
print("="*80)

results_comparison = {
    '基线模型 (MSE, 无log)': {
        'mae': 65611270.83,
        'r2': 0.0236,
        'features': '静态特征 (58个)',
        'approach': '直接预测value_usd'
    },
    '单一对数模型 (MSE)': {
        'mae': 41074162.79,
        'r2': -0.0438,
        'features': '静态特征 (58个)',
        'approach': '对数变换 + MSE'
    },
    '分组对数模型': {
        'mae': 55114287.12,
        'r2': -0.0516,
        'features': '静态特征 (58个)',
        'approach': '按HS代码分组'
    },
    'Huber Loss模型': {
        'mae': 41280669.83,
        'r2': -0.0386,
        'features': '静态特征 (58个)',
        'approach': '对数 + Huber Loss'
    },
    '🚀 时间序列模型 (新)': {
        'mae': float(mae_orig),
        'r2': float(r2_orig),
        'features': f'静态+时间特征 ({X_test.shape[1]}个)',
        'approach': '对数 + 时间序列特征'
    }
}

print("\n" + "="*80)
print(f"{'模型':<25} {'MAE ($M)':>12} {'R²':>10} {'特征':>20} {'方法':<25}")
print("="*80)

for name, metrics in results_comparison.items():
    mae_m = metrics['mae'] / 1e6
    marker = "📈" if name.startswith('🚀') else "  "
    print(f"{marker} {name:<23} ${mae_m:>10.1f}M {metrics['r2']:>10.4f} {metrics['features']:>20} {metrics['approach']:<25}")

# 计算改进
baseline_mae = 65611270.83
temporal_mae = mae_orig
improve_vs_baseline = (1 - temporal_mae / baseline_mae) * 100

best_previous_r2 = 0.0236  # 基线模型
improve_r2 = r2_orig - best_previous_r2

print("\n" + "="*80)
print("【改进幅度分析】")
print("="*80)
print(f"相比基线模型:")
print(f"  MAE降低: {improve_vs_baseline:+.1f}%")
print(f"  R²提升: {improve_r2:+.4f}")

if r2_orig > 0.5:
    result = "🎉 重大突破！模型达到生产级标准"
    color = "excellent"
elif r2_orig > 0.3:
    result = "✓ 显著改进！模型可用于预测"
    color = "good"
elif r2_orig > 0.1:
    result = "→ 明显改进，但仍有优化空间"
    color = "moderate"
elif r2_orig > 0:
    result = "⚠️  轻微改进"
    color = "slight"
else:
    result = "✗ 仍需进一步改进"
    color = "poor"

print(f"\n【综合评估】: {result}")

# =============================================================================
# 6.8 特征重要性分析
# =============================================================================
print("\n" + "="*80)
print("【6.8】特征重要性分析")
print("="*80)

# 获取特征重要性
if hasattr(model_temporal.model, 'feature_importances_'):
    importances = model_temporal.model.feature_importances_
    
    # 获取特征名（需要从prepare_data中重建）
    feature_df = train_df_enhanced[[c for c in train_df_enhanced.columns 
                                     if c not in ['date', 'hs_code', 'country', 'year', 'month', 
                                                  'time_idx', 'value_usd', 'quantity', 'log_value_usd']]]
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_names = feature_df.columns.tolist()
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 最重要特征:")
    print("-"*80)
    print(f"{'排名':<5} {'特征名':<40} {'重要性':>10} {'类型':<15}")
    print("-"*80)
    
    for idx, row in importance_df.head(20).iterrows():
        feat_name = row['feature']
        importance = row['importance']
        
        # 判断特征类型
        if any(x in feat_name for x in ['lag', 'ma', 'trend', 'diff', 'pct_change', 'std', 'max', 'min']):
            feat_type = '🕐 时间序列'
        elif any(x in feat_name for x in ['month_sin', 'month_cos', 'quarter']):
            feat_type = '📅 季节性'
        elif any(x in feat_name for x in ['hhi', 'concentration']):
            feat_type = '📊 浓度指标'
        elif any(x in feat_name for x in ['growth', 'volatility']):
            feat_type = '📈 趋势指标'
        else:
            feat_type = '🔧 静态特征'
        
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<5} {feat_name:<40} {importance:>10.4f} {feat_type:<15}")
    
    # 统计各类特征的总重要性
    print("\n" + "-"*80)
    print("各类特征重要性占比:")
    print("-"*80)
    
    temporal_importance = importance_df[importance_df['feature'].str.contains('lag|ma|trend|diff|pct_change|std|max|min')]['importance'].sum()
    seasonal_importance = importance_df[importance_df['feature'].str.contains('month_sin|month_cos|quarter')]['importance'].sum()
    static_importance = importance_df[~importance_df['feature'].str.contains('lag|ma|trend|diff|pct_change|std|max|min|month_sin|month_cos|quarter')]['importance'].sum()
    
    total_importance = importance_df['importance'].sum()
    
    print(f"  🕐 时间序列特征: {temporal_importance/total_importance*100:>6.1f}%")
    print(f"  📅 季节性特征:   {seasonal_importance/total_importance*100:>6.1f}%")
    print(f"  🔧 静态特征:     {static_importance/total_importance*100:>6.1f}%")
    
    print("\n【判断逻辑】:")
    if temporal_importance / total_importance > 0.4:
        print("  ✓ 时间序列特征贡献>40% → 验证了方案A的有效性!")
        print("  → 历史信息确实是关键预测因子")
    elif temporal_importance / total_importance > 0.2:
        print("  → 时间序列特征有一定贡献，但静态特征仍重要")
    else:
        print("  ✗ 时间序列特征贡献<20% → 可能特征工程不够或数据问题")

# =============================================================================
# 6.9 预测分布分析
# =============================================================================
print("\n" + "="*80)
print("【6.9】预测分布分析")
print("="*80)

print("\n真实值统计:")
print(f"  均值:     ${y_test_orig.mean():,.0f}")
print(f"  中位数:   ${np.median(y_test_orig):,.0f}")
print(f"  标准差:   ${y_test_orig.std():,.0f}")
print(f"  最大值:   ${y_test_orig.max():,.0f}")

print("\n预测值统计:")
print(f"  均值:     ${y_pred_orig.mean():,.0f}")
print(f"  中位数:   ${np.median(y_pred_orig):,.0f}")
print(f"  标准差:   ${y_pred_orig.std():,.0f}")
print(f"  最大值:   ${y_pred_orig.max():,.0f}")

print("\n均值/方差恢复度:")
print(f"  预测/真实均值比: {y_pred_orig.mean() / y_test_orig.mean():.2%}")
print(f"  预测/真实标准差比: {y_pred_orig.std() / y_test_orig.std():.2%}")
print(f"  预测/真实最大值比: {y_pred_orig.max() / y_test_orig.max():.2%}")

print("\n【判断逻辑】:")
if y_pred_orig.mean() / y_test_orig.mean() > 0.8:
    print("  ✓ 均值恢复良好 (>80%)")
elif y_pred_orig.mean() / y_test_orig.mean() > 0.5:
    print("  → 均值部分恢复 (50-80%)")
else:
    print("  ✗ 均值仍严重低估 (<50%)")

if y_pred_orig.std() / y_test_orig.std() > 0.3:
    print("  ✓ 方差恢复改善 (>30%)")
else:
    print("  → 方差仍过小，预测过于保守")

# =============================================================================
# 6.10 保存模型和结果
# =============================================================================
print("\n" + "="*80)
print("【6.10】保存增强模型")
print("="*80)

model_dir = Path('models/trained')
model_dir.mkdir(parents=True, exist_ok=True)

# 保存模型
model_temporal.save_model(model_dir / 'value_forecaster_xgb_temporal')

# 保存对比结果
import json

comparison_data = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'strategy': 'temporal_feature_engineering',
    'new_features_count': temporal_features_count,
    'total_features': X_test.shape[1],
    'test_metrics': {
        'mae': float(mae_orig),
        'rmse': float(rmse_orig),
        'r2': float(r2_orig),
        'mape': float(mape_orig)
    },
    'log_space_metrics': {
        'mae': float(metrics_log['mae']),
        'rmse': float(metrics_log['rmse']),
        'r2': float(metrics_log['r2']),
        'mape': float(metrics_log['mape'])
    },
    'all_models_comparison': {k: {**v, 'mae': float(v['mae']), 'r2': float(v['r2'])} 
                              for k, v in results_comparison.items()},
    'improvement': {
        'mae_reduction_pct': float(improve_vs_baseline),
        'r2_gain': float(improve_r2)
    },
    'feature_importance_summary': {
        'temporal_pct': float(temporal_importance / total_importance * 100) if 'temporal_importance' in locals() else 0,
        'seasonal_pct': float(seasonal_importance / total_importance * 100) if 'seasonal_importance' in locals() else 0,
        'static_pct': float(static_importance / total_importance * 100) if 'static_importance' in locals() else 0
    },
    'prediction_stats': {
        'pred_mean': float(y_pred_orig.mean()),
        'true_mean': float(y_test_orig.mean()),
        'pred_std': float(y_pred_orig.std()),
        'true_std': float(y_test_orig.std()),
        'pred_max': float(y_pred_orig.max()),
        'true_max': float(y_test_orig.max()),
        'mean_recovery_ratio': float(y_pred_orig.mean() / y_test_orig.mean()),
        'std_recovery_ratio': float(y_pred_orig.std() / y_test_orig.std())
    }
}

with open(model_dir / 'temporal_features_comparison.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

# 也保存增强的数据集
train_df_enhanced.to_parquet('data/processed/features_train_temporal.parquet')
test_df_enhanced.to_parquet('data/processed/features_test_temporal.parquet')

print(f"✓ 模型已保存: {model_dir / 'value_forecaster_xgb_temporal.pkl'}")
print(f"✓ 对比结果已保存: {model_dir / 'temporal_features_comparison.json'}")
print(f"✓ 增强数据已保存: data/processed/features_*_temporal.parquet")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "="*80)
print("【Step 6 总结 - 方案A完成】")
print("="*80)

print("\n完成的工作:")
print("  ✓ 添加4个滞后特征 (lag-1, lag-3, lag-6, lag-12)")
print("  ✓ 添加3个移动平均 (MA3, MA6, MA12)")
print("  ✓ 添加3个趋势特征 (trend_ratio, diff1, pct_change)")
print("  ✓ 添加3个季节性特征 (month_sin, month_cos, quarter)")
print("  ✓ 添加3个统计特征 (std6, max6, min6)")
print(f"  ✓ 总计新增 {temporal_features_count} 个时间序列特征")

print("\n关键结果:")
print(f"  R²: {r2_orig:.4f} (基线: 0.0236)")
if r2_orig > 0.0236:
    print(f"  → R²提升: +{(r2_orig - 0.0236):.4f} ({'✓ 正向改进' if r2_orig > 0.1 else '轻微改进'})")
else:
    print(f"  → R²下降: {(r2_orig - 0.0236):.4f} (未达预期)")

print(f"\n  MAE: ${mae_orig/1e6:.1f}M (基线: $65.6M)")
print(f"  → MAE降低: {improve_vs_baseline:.1f}%")

print(f"\n  均值恢复: {y_pred_orig.mean() / y_test_orig.mean():.1%} (步骤5: 5%)")
print(f"  方差恢复: {y_pred_orig.std() / y_test_orig.std():.1%} (步骤5: 3%)")

print("\n【知识验证】:")
print("  假设: 时间序列特征能捕捉历史依赖性，改善预测")
if r2_orig > 0.1:
    print("  结果: ✓ 假设验证！时间特征确实有效")
    print("  理论支持: Box-Jenkins方法论正确")
elif r2_orig > 0:
    print("  结果: → 部分验证，改进有限")
    print("  可能原因: 需要更复杂的模型（LSTM）或更多外部特征")
else:
    print("  结果: ✗ 假设未验证")
    print("  可能原因: 1) 数据质量问题 2) 过拟合 3) 需要非线性模型")

print("\n【下一步建议】:")
if r2_orig > 0.5:
    print("  🎉 模型已达标！可以:")
    print("    1. 进行超参数精调")
    print("    2. 添加外部特征（GDP、汇率、政策指数）")
    print("    3. 准备生产部署")
elif r2_orig > 0.3:
    print("  ✓ 模型可用！可以:")
    print("    1. 尝试集成多个模型")
    print("    2. 添加外部宏观经济特征")
    print("    3. 考虑LSTM捕捉更复杂的时间模式")
elif r2_orig > 0.1:
    print("  → 继续改进:")
    print("    1. 尝试更长的滞后期（lag-18, lag-24）")
    print("    2. 添加交叉特征（hs_code × lag features）")
    print("    3. 考虑LSTM或Transformer模型")
else:
    print("  ⚠️  需要重新评估:")
    print("    1. 检查数据质量和特征分布")
    print("    2. 尝试完全不同的建模方法（Prophet, ARIMA）")
    print("    3. 考虑分位数回归或概率预测")

print("="*80)
