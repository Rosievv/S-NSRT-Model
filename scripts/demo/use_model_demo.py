#!/usr/bin/env python3
"""
演示如何使用训练好的模型
展示模型就像"即食食品"，开箱即用
"""
import pickle
import pandas as pd
import numpy as np

print("="*70)
print("演示：如何使用训练好的.pkl模型文件")
print("="*70)

# ============================================================
# 步骤1: 加载模型（就像从冰箱拿出即食食品）
# ============================================================
print("\n【步骤1】从文件加载模型")
print("-"*70)
print("执行: pickle.load('hhi_forecaster_lgb.pkl')")

with open('models/trained/hhi_forecaster_lgb.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"✓ 模型加载成功！")
print(f"  类型: {type(model).__name__}")
print(f"  包含: {model.n_estimators} 棵决策树")
print(f"  特征数: {model.n_features_in_}")

# ============================================================
# 步骤2: 准备新数据（就像摆好盘子）
# ============================================================
print("\n【步骤2】准备要预测的数据")
print("-"*70)

# 加载一些真实的测试数据
test_features = pd.read_parquet('data/processed/features_test_full.parquet')
print(f"✓ 加载了 {len(test_features)} 条测试数据")

# 取一条示例
sample = test_features.iloc[0:1]
print(f"\n示例数据（第1条）:")
print(f"  日期: {sample['date'].values[0]}")
print(f"  HS代码: {sample['hs_code'].values[0]}")
print(f"  真实HHI: {sample['hhi'].values[0]:.2f}")

# 准备特征（去掉日期、HS代码等非数值列）
X_sample = sample.select_dtypes(include=[np.number]).values
print(f"\n转换为特征向量:")
print(f"  形状: {X_sample.shape}")
print(f"  前5个特征值: {X_sample[0, :5]}")

# ============================================================
# 步骤3: 预测（就像打开即食食品，直接吃）
# ============================================================
print("\n【步骤3】使用模型预测")
print("-"*70)
print("执行: model.predict(X_sample)")

prediction = model.predict(X_sample)

print(f"\n✓ 预测完成！")
print(f"  预测的HHI: {prediction[0]:.2f}")
print(f"  真实的HHI: {sample['hhi'].values[0]:.2f}")
print(f"  误差: {abs(prediction[0] - sample['hhi'].values[0]):.2f} " +
      f"({abs(prediction[0] - sample['hhi'].values[0]) / sample['hhi'].values[0] * 100:.1f}%)")

# ============================================================
# 步骤4: 批量预测（一次预测很多）
# ============================================================
print("\n【步骤4】批量预测（10条数据）")
print("-"*70)

X_batch = test_features.head(10).select_dtypes(include=[np.number]).values
predictions = model.predict(X_batch)

print(f"✓ 批量预测完成！")
print(f"\n预测结果对比:")
print(f"{'序号':<6} {'真实HHI':<12} {'预测HHI':<12} {'误差':<10} {'误差%':<10}")
print("-"*60)
for i in range(10):
    real = test_features.iloc[i]['hhi']
    pred = predictions[i]
    error = abs(pred - real)
    error_pct = error / real * 100
    print(f"{i+1:<6} {real:<12.2f} {pred:<12.2f} {error:<10.2f} {error_pct:<10.2f}%")

avg_error = np.mean([abs(predictions[i] - test_features.iloc[i]['hhi']) 
                     for i in range(10)])
print(f"\n平均绝对误差: {avg_error:.2f}")

# ============================================================
# 关键要点总结
# ============================================================
print("\n" + "="*70)
print("关键要点")
print("="*70)
print("""
1. ✅ .pkl文件是"训练好的模型"，不是代码
   - 包含100棵决策树的完整结构
   - 包含所有学到的参数（278KB二进制数据）
   - 可以直接用来预测，不需要重新训练

2. ✅ 使用模型只需要3步：
   - 用pickle加载模型
   - 准备数据（数值特征）
   - 调用model.predict()

3. ✅ 模型是"只读的"：
   - 不能修改内部参数
   - 如果想改进，需要重新训练生成新的.pkl

4. ✅ 模型可以复制分享：
   - 把.pkl文件发给别人
   - 他们也能用同样的方式预测
   - 不需要源代码或训练数据
""")
print("="*70)

# ============================================================
# 额外演示：查看模型"记住"了什么
# ============================================================
print("\n【额外】模型内部的智慧")
print("-"*70)
print("模型通过100棵树学会了这些规则（示例）：")
print("""
树1说: 如果feature_0（贸易额）> 50M，预测HHI增加200
树2说: 如果feature_4（Gini系数）> 0.6，预测HHI增加150
树3说: 如果feature_0 < 50M 且 feature_2 > 100，预测HHI减少80
...
树100说: 综合前99棵树的结果，微调...

最终预测 = 所有树投票的加权平均
""")

print("特征重要性（哪些特征最有用）:")
feature_importance = model.feature_importances_
top_5_idx = np.argsort(feature_importance)[-5:][::-1]
for idx in top_5_idx:
    print(f"  feature_{idx}: {feature_importance[idx]:.4f}")

print("\n模型就是这样工作的！")
print("="*70)
