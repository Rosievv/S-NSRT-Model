#!/usr/bin/env python3
"""
探索模型文件里到底保存了什么
"""
import pickle
import json

print("="*70)
print("探索训练好的模型文件")
print("="*70)

# 1. 查看元数据（JSON文件）
print("\n【1】模型元数据（.json文件 - 人类可读）")
print("-"*70)
with open('models/trained/hhi_forecaster_lgb.json', 'r') as f:
    metadata = json.load(f)
    print(f"模型名称: {metadata['model_name']}")
    print(f"模型类型: {metadata['model_type']}")
    print(f"保存时间: {metadata.get('saved_at', 'N/A')}")
    print(f"\n配置参数:")
    for key, value in metadata['config'].items():
        print(f"  {key}: {value}")
    
    print(f"\n测试集性能:")
    if 'metrics' in metadata and 'test' in metadata['metrics']:
        for key, value in metadata['metrics']['test'].items():
            print(f"  {key.upper()}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# 2. 加载模型对象（PKL文件）
print("\n\n【2】模型对象（.pkl文件 - 二进制）")
print("-"*70)
with open('models/trained/hhi_forecaster_lgb.pkl', 'rb') as f:
    model = pickle.load(f)
    
    print(f"对象类型: {type(model)}")
    print(f"对象大小: 278 KB（包含所有训练好的参数）")
    
    # 查看模型包含什么
    print(f"\n模型属性（部分）:")
    for attr in dir(model):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            if attr == 'model_algorithm':
                print(f"      值: {getattr(model, attr, 'N/A')}")
            if attr == 'target_variable':
                print(f"      值: {getattr(model, attr, 'N/A')}")

# 3. 查看内部的机器学习模型
print("\n\n【3】内部的LightGBM模型")
print("-"*70)
if hasattr(model, 'model') and model.model is not None:
    lgb_model = model.model
    print(f"LightGBM Booster类型: {type(lgb_model)}")
    
    # 获取模型信息
    try:
        model_str = lgb_model.model_to_string()
        lines = model_str.split('\n')[:20]  # 只显示前20行
        print("\n模型内部结构（前20行）:")
        print("-"*70)
        for line in lines:
            print(line)
        print("...")
        print(f"\n总共有 {len(model_str.split('Tree='))-1} 棵决策树")
    except:
        print("无法读取模型内部结构")

# 4. 演示如何使用模型
print("\n\n【4】如何使用这个模型文件？")
print("-"*70)
print("""
步骤1: 加载模型
    with open('hhi_forecaster_lgb.pkl', 'rb') as f:
        model = pickle.load(f)

步骤2: 准备数据
    X_new = [[特征1, 特征2, ...]]  # 新数据

步骤3: 预测
    prediction = model.predict(X_new)
    print(f"预测的HHI: {prediction}")

就这么简单！不需要重新训练！
""")

print("="*70)
print("总结：")
print("- .py文件 = 代码（如何训练）")
print("- .pkl文件 = 训练好的模型（可以直接用来预测）")
print("- .json文件 = 元数据（配置信息）")
print("="*70)
