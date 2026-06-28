"""
数据说明脚本
"""
import pandas as pd

# 读取训练数据
df = pd.read_parquet('data/raw/us_census_20260215_201556.parquet')

print('='*80)
print('供应链风险分析数据说明')
print('='*80)
print(f'\n数据形状: {df.shape[0]}行 × {df.shape[1]}列')
print(f'时间范围: {df["date"].min().date()} 到 {df["date"].max().date()} (10年)')
print(f'总月数: {df["date"].nunique()}个月')
print(f'总供应国: {df["country"].nunique()}个国家')

print('\n' + '='*80)
print('字段说明')
print('='*80)
field_desc = {
    'date': '月度日期 (每月1号)',
    'hs_code': 'HS商品编码 (6位) - 半导体集成电路产品代码',
    'country': '出口国名称',
    'country_code': '出口国代码',
    'value_usd': '进口金额 (美元)',
    'quantity': '进口数量 (个)',
    'trade_type': '贸易类型 (imports=进口)',
    'data_source': '数据来源 (us_census)',
    'collected_at': '数据采集时间戳'
}
for col, desc in field_desc.items():
    print(f'{col:15s}: {desc}')

print('\n' + '='*80)
print('前3行示例数据')
print('='*80)
for i, row in df.head(3).iterrows():
    print(f'\n记录 {i+1}:')
    print(f'  日期: {row["date"].date()}')
    print(f'  HS代码: {row["hs_code"]}')
    print(f'  国家: {row["country"]} ({row["country_code"]})')
    print(f'  贸易额: ${row["value_usd"]:,}')
    print(f'  数量: {row["quantity"]:,} 个')
    print(f'  类型: {row["trade_type"]}')

print('\n' + '='*80)
print('HS代码说明 (半导体集成电路)')
print('='*80)
hs_info = {
    '854231': 'Processors & Controllers - 处理器和控制器',
    '854232': 'Memories - 存储器(内存)',
    '854233': 'Amplifiers - 放大器',
    '854239': 'Other Electronic ICs - 其他电子集成电路'
}
for hs, desc in hs_info.items():
    if hs in df['hs_code'].values:
        count = len(df[df['hs_code'] == hs])
        value = df[df['hs_code'] == hs]['value_usd'].sum()
        print(f'{hs}: {desc}')
        print(f'  记录数: {count:,} | 总额: ${value/1e9:.2f}B')

print('\n' + '='*80)
print('Top 10 供应国（按贸易额）')
print('='*80)
top_countries = df.groupby('country')['value_usd'].sum().sort_values(ascending=False).head(10)
total_value = df['value_usd'].sum()
for i, (country, value) in enumerate(top_countries.items(), 1):
    pct = value / total_value * 100
    print(f'{i:2d}. {country:20s} ${value/1e9:.2f}B ({pct:.1f}%)')

print('\n' + '='*80)
print('贸易统计')
print('='*80)
monthly_trade = df.groupby("date")["value_usd"].sum()
print(f'总贸易额: ${total_value/1e9:.2f}B (1.69万亿美元)')
print(f'月均贸易额: ${monthly_trade.mean()/1e6:.2f}M')
print(f'最大单月: ${monthly_trade.max()/1e6:.2f}M ({monthly_trade.idxmax().date()})')
print(f'最小单月: ${monthly_trade.min()/1e6:.2f}M ({monthly_trade.idxmin().date()})')

print('\n' + '='*80)
print('数据理解要点')
print('='*80)
print('''
1. 这是美国从全球进口半导体集成电路的月度贸易数据
2. 时间跨度: 2010-2019年，共120个月
3. 产品范围: 4类核心集成电路产品 (处理器、内存、放大器、其他IC)
4. 地理覆盖: 214个国家/地区
5. 数据粒度: 每条记录 = 1个月 × 1个HS代码 × 1个国家
6. 总规模: 37,756条记录，代表1.69万亿美元的贸易额

供应链风险指标:
- 如果某个国家占比过高 → 集中度风险
- 如果贸易额波动大 → 稳定性风险
- 如果增长率为负 → 萎缩风险
- 如果供应商数量少 → 依赖风险
''')

print('='*80)
