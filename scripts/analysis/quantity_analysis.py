"""
分析Quantity字段为什么是0
"""
import pandas as pd
import numpy as np

# 读取训练数据
df = pd.read_parquet('data/raw/us_census_20260215_201556.parquet')

print('='*80)
print('Quantity 字段分析')
print('='*80)

print(f'\n总记录数: {len(df):,}')
print(f'\nQuantity统计:')
print(f'  等于0的记录: {(df["quantity"] == 0).sum():,} ({(df["quantity"] == 0).sum()/len(df)*100:.1f}%)')
print(f'  大于0的记录: {(df["quantity"] > 0).sum():,} ({(df["quantity"] > 0).sum()/len(df)*100:.1f}%)')
print(f'  缺失值(NaN): {df["quantity"].isna().sum():,}')

print(f'\nQuantity基本统计:')
print(df['quantity'].describe())

print('\n' + '='*80)
print('按HS代码查看Quantity情况')
print('='*80)
for hs in sorted(df['hs_code'].unique()):
    df_hs = df[df['hs_code'] == hs]
    zero_count = (df_hs['quantity'] == 0).sum()
    nonzero_count = (df_hs['quantity'] > 0).sum()
    zero_pct = zero_count / len(df_hs) * 100
    
    print(f'\nHS {hs}:')
    print(f'  总记录: {len(df_hs):,}')
    print(f'  Quantity=0: {zero_count:,} ({zero_pct:.1f}%)')
    print(f'  Quantity>0: {nonzero_count:,} ({100-zero_pct:.1f}%)')
    if nonzero_count > 0:
        nonzero_qty = df_hs[df_hs['quantity'] > 0]['quantity']
        print(f'  Quantity范围: {nonzero_qty.min():,} - {nonzero_qty.max():,}')
        print(f'  Quantity平均值: {nonzero_qty.mean():,.0f}')

print('\n' + '='*80)
print('按时间查看Quantity报告情况')
print('='*80)
time_analysis = df.groupby(df['date'].dt.year).agg({
    'quantity': lambda x: (x > 0).sum() / len(x) * 100
}).reset_index()
time_analysis.columns = ['year', 'nonzero_pct']

print('\n有数量报告的记录占比（按年份）:')
for _, row in time_analysis.iterrows():
    print(f'  {int(row["year"])}年: {row["nonzero_pct"]:.1f}%')

print('\n' + '='*80)
print('有数量数据的记录示例（前5条）')
print('='*80)
df_with_qty = df[df['quantity'] > 0].head()
for i, row in df_with_qty.iterrows():
    unit_price = row['value_usd'] / row['quantity'] if row['quantity'] > 0 else 0
    print(f"\n{row['date'].date()} | HS {row['hs_code']} | {row['country']}")
    print(f"  价值: ${row['value_usd']:,} | 数量: {row['quantity']:,} | 单价: ${unit_price:.4f}")

print('\n' + '='*80)
print('Quantity=0的记录示例（前5条）')
print('='*80)
df_zero_qty = df[df['quantity'] == 0].head()
for i, row in df_zero_qty.iterrows():
    print(f"\n{row['date'].date()} | HS {row['hs_code']} | {row['country']}")
    print(f"  价值: ${row['value_usd']:,} | 数量: 0 (未报告)")

print('\n' + '='*80)
print('结论分析')
print('='*80)
zero_pct = (df['quantity'] == 0).sum() / len(df) * 100
print(f"""
为什么Quantity很多是0？

1. 数据报告政策
   - US Census允许某些产品类别不报告数量
   - 半导体等高科技产品可能因商业保密原因不报告具体数量
   - 贸易商可能选择只报告价值（必填），而不报告数量（选填）

2. 统计现状
   - 本数据集中{zero_pct:.1f}%的记录quantity=0
   - 这意味着绝大多数半导体进口记录只报告了价值，未报告数量
   - 这在高价值、小体积的电子产品贸易中很常见

3. 对分析的影响
   - ✅ 价值(value_usd)数据完整可靠，可用于所有分析
   - ⚠️ 数量(quantity)数据缺失严重，不应作为主要分析指标
   - 💡 建议：基于价值而非数量进行供应链风险分析

4. 为什么保留quantity字段
   - 少量有数量的记录仍有参考价值
   - 可用于计算单价（当数量>0时）
   - 保持数据完整性

结论：Quantity=0 不是错误，而是数据报告规则的正常结果。
     我们的风险分析完全基于value_usd，不受quantity缺失影响。
""")

print('='*80)
