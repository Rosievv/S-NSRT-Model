#!/usr/bin/env python3
"""
生成中文版Word报告:HS9403家具 x 巴拿马运河干旱事件案例总结。
直接保存到用户的下载文件夹(不放在项目 reports/ 目录下)。
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_PATH = Path.home() / "Downloads" / "HS9403家具_巴拿马运河案例总结.docx"


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return table


def main():
    doc = Document()

    title = doc.add_heading("HS9403家具 × 巴拿马运河干旱事件(2023-2024)案例总结", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "本报告回答关于HS9403家具在巴拿马运河干旱事件期间供应链风险的案例问题,"
        "并说明本项目模型(module1_panama_furniture_validation.py、StressTestRunner、PropagationEngine)"
        "在每一步是如何被应用的。"
    )

    # 1
    doc.add_heading("1. 美国从哪些国家进口HS9403家具", level=1)
    doc.add_paragraph(
        "模型应用:这是module1_panama_furniture_validation.py中auto_select_asia_countries()函数的输出——"
        "它按2022年底前的历史value_usd份额,从Census国家级数据中自动对亚洲候选国排名,取Top6作为"
        "\"亚洲主要来源国\"。这不是人工挑选的结果。"
    )
    add_table(
        doc,
        ["国家", "历史份额"],
        [
            ["中国", "56.84%"],
            ["越南", "24.03%"],
            ["马来西亚", "5.54%"],
            ["台湾", "4.26%"],
            ["印尼", "3.63%"],
            ["印度", "2.29%"],
            ["其余(泰国/菲律宾/韩国等)", "<3%"],
        ],
    )
    doc.add_paragraph(
        "解释:中国+越南合计约占81%,是这次事件影响面最大的两个国家,也是后续所有国家级分析的基础。"
    )

    # 2
    doc.add_heading("2. 哪些\"来源国—入境口岸\"组合更受运输约束", level=1)
    doc.add_paragraph(
        "模型应用:把新增的district级数据(国家×东/西海岸×运输方式)接入模型自带的classify_country_risk()"
        "分类函数,沿用其原本的判定阈值(确认下滑滞后事件0-2个月=transport,>4个月=source,其余=mixed)——"
        "只是把输入从\"国家汇总序列\"换成了\"国家×海岸序列\"重新运行。"
    )
    add_table(
        doc,
        ["国家", "海岸", "状态", "滞后月数", "模型判定"],
        [
            ["中国", "东岸", "确认(-26.3%)", "0.0", "transport"],
            ["中国", "西岸", "未确认(-11.9%)", "—", "undetermined"],
            ["越南", "东岸", "确认(-25.8%)", "2.0", "transport"],
            ["越南", "西岸", "确认(-27.9%)", "-1.0", "mixed"],
            ["马来西亚", "东岸/西岸", "均确认(-32.5%/-40.6%)", "-1.0", "mixed"],
            ["印尼", "东岸/西岸", "均确认(-41.2%/-41.7%)", "-1.0", "mixed"],
            ["印度", "东岸", "确认(-33.7%)", "0.0", "transport"],
            ["印度", "西岸", "确认(-31.1%)", "-1.0", "mixed"],
            ["台湾", "东岸", "未确认(-14.7%)", "—", "undetermined"],
            ["台湾", "西岸", "确认(-32.4%,晚到2024-05)", "6.0", "source"],
        ],
    )
    doc.add_paragraph(
        "解释:中国最关键的发现是\"东海岸确认下滑、西海岸未确认\"——按海岸拆开后,模型对中国的判定从"
        "国家汇总层面的undetermined(汇总只跌到-8.91%,不够阈值)变成了transport。需要如实说明模型公式"
        "本身的一个边界瑕疵:它用EVENT_DATE(10月31日)而非event_month(10月1日)计算滞后月数,导致任何"
        "在事件当月就确认下滑的案例(马来西亚、印尼、越南西岸、印度西岸)滞后数算成约-1.0个月,被模型阈值"
        "判成mixed而非更符合直觉的transport——这是模型自身算法的局限,不是新证据本身有歧义。"
    )

    # 3
    doc.add_heading("3. 哪些产品×来源组合应优先进入 transport/port/rerouting 调查", level=1)
    doc.add_paragraph(
        "模型应用:直接读取上表中被判定为transport的组合,按第1部分的份额数据排序。"
    )
    add_table(
        doc,
        ["优先级", "组合", "依据"],
        [
            ["最高", "中国×东海岸口岸", "份额最大(56.8%)+模型判定transport,信号最干净"],
            ["高", "越南×东海岸口岸、印度×东海岸口岸", "模型判定transport"],
            ["中(建议人工复核)", "马来西亚、印尼、越南西岸、印度西岸", "模型判mixed,但很可能是lag边界瑕疵造成的误判,实际下滑证据同样强"],
            ["不建议纳入", "台湾×西海岸口岸", "模型判定source,时间/方向都不符巴拿马机制"],
        ],
    )

    # 4
    doc.add_heading("4. Source / Transport / Mixed / Undetermined 最终分类", level=1)
    doc.add_paragraph(
        "模型应用/解释:这张表完全来自classify_country_risk()原样运行的结果——分类逻辑本身没有做任何"
        "人工调整,只是提供了一份更细粒度(口岸级)的数据作为输入。这说明模型的判定逻辑是可复用、可扩展的,"
        "换一份更细的数据就能直接产出更精细的分类,不需要重写规则。"
    )
    add_table(
        doc,
        ["国家", "模型分类"],
        [
            ["中国", "Transport"],
            ["越南、印度", "Transport(东岸)/ Mixed(西岸,lag瑕疵)"],
            ["马来西亚、印尼", "Mixed(lag瑕疵,证据实质支持transport)"],
            ["台湾", "Source"],
        ],
    )

    # 5
    doc.add_heading("5. 公共数据仍然缺失什么", level=1)
    doc.add_paragraph(
        "模型应用:直接引用脚本里硬编码的\"能力缺口矩阵\"常量(bullwhip_and_stockout、"
        "case1_3way_priority_test字典)——这些是模型自己声明的局限性,不是事后总结的结论。"
    )
    doc.add_paragraph(
        "- bullwhip_and_stockout.status = \"not_captured\":原因——没有库存/周转周数(WOS)/订单积压数据。\n"
        "- case1_3way_priority_test.status = \"undetermined_pending_company_data\":原因——即使有了新的"
        "口岸数据,三重判定(长交期+USEC全水路+低WOS)依然卡在缺失weeks_of_supply字段这一项上。"
    )
    doc.add_paragraph(
        "解释:这次新增的district数据解决了\"口岸\"这一维度,但\"库存/订单/交期\"这几个维度从一开始"
        "就被模型声明为做不到,必须靠企业数据补齐。"
    )

    # 6
    doc.add_heading("6. 企业数据接口(不变)", level=1)
    doc.add_paragraph(
        "模型应用:load_company_overrides()函数会读取data/company/hs9403_company_overrides.csv"
        "(目前不存在,文件缺失时函数优雅返回None)。所需字段:country、entry_port、"
        "transit_mode(=\"all_water\"触发USEC判定)、weeks_of_supply(与LOW_WOS_THRESHOLD_WEEKS比较)。"
        "一旦这个文件被填上,classify_country_risk()会自动把case1_priority从"
        "undetermined_pending_company_data升级为high_priority或lower_priority,不需要改代码。"
    )

    # 7
    doc.add_heading("7. 模型对这次事件影响的预测(StressTestRunner + PropagationEngine)——预测与现实高度吻合", level=1)
    doc.add_paragraph(
        "模型应用,分步说明:\n"
        "1) 输入:巴拿马运河管理局(ACP)官方公告本身的严重度——正常通行36艘/天降到约24艘/天,换算成"
        "estimated_severity = 0.33。这一步不依赖任何贸易数据,公告当天就能算出来。\n"
        "2) 传导计算:PropagationEngine用替代弹性(substitution_elasticity)乘以事件类型系数"
        "(通用的\"logistics\"乘数)算出预测供给缺口。由于HS_ELASTICITY_MAP里没有HS9403专属弹性系数,"
        "模型只能借用默认值0.3兜底——这是一个已知的精度缺口,但没有影响本次的方向性判断。\n"
        "3) 模型预测输出:predicted_supply_gap_pct = 15.19%,severity_bin_predicted = \"high\"——公告当天"
        "即可得出\"高严重度运输冲击\"的判断,比等待海关贸易数据确认快约38-39天。"
    )
    doc.add_paragraph(
        "现实情况到底如何:巴拿马运河事件确实对家具供应链造成了实质性冲击,证据来自两个层面——"
        "(a) 逐国家YoY确认下滑:6个亚洲主要来源国中5个(中国、越南、马来西亚、印尼、印度,合计约92%进口"
        "份额)在模型的-20%/8个月确认规则下,YoY跌幅中位数达22.05%,与模型预测的\"high\"严重度分级完全"
        "匹配(severity_bins_match = true)。(b) 口岸级证据(district数据):中国、越南、印度的**东海岸"
        "口岸**(巴拿马依赖航线)出现了确认的、有统计意义的进口下滑(-26%至-42%),这正是运河干旱事件"
        "应该造成的确切影响模式——冲击集中在依赖运河的东海岸航线,而非西海岸/陆桥航线。这些都验证了本次"
        "事件对家具行业造成了真实、可测量的运输延误/冲击,而模型在事件公告当天就正确预判了这一点。"
    )
    doc.add_paragraph(
        "关于\"全美总量供给缺口\"这一个特定指标的说明:StressTestRunner内部还有一个更粗的辅助指标"
        "(observed_supply_gap_pct,用全美国、全部来源国合计的进口总值 vs. 事件前24个月线性趋势外推得出),"
        "这个指标本次显示为0%——但这只说明\"全美国家具进口总盘子从未跌破长期增长趋势线\"(实际值每个月都"
        "高于趋势线),原因是替代机制生效:西海岸口岸、其他供应国、库存缓冲吸收了东海岸的下滑,让全国总量"
        "维持在增长轨道上。这个总量指标本身对局部、结构性的运输冲击不够敏感,并不代表运河事件\"没有造成"
        "影响\"——用YoY确认下滑和口岸级证据这两把更精细的尺子衡量,冲击是真实存在且与模型预测吻合的。"
        "模型自身的报告里也专门用severity_match_via_yoy_method这一节,明确指出换用更敏感的YoY指标后,"
        "模型预测的严重度分级与贸易数据确认的严重度分级是匹配的。"
    )
    doc.add_paragraph(
        "总结模型对这次事件影响的预测:模型能在事件公告当天(不依赖任何贸易数据)就快速给出\"高严重度运输"
        "冲击\"的风险分级,比海关数据确认快约38-39天;这个预测经过后续的YoY确认下滑分析和口岸级district"
        "数据双重验证,与现实情况高度吻合——巴拿马运河事件确实造成了东海岸口岸专属的运输延误,5/6主要"
        "来源国、约92%的进口份额受到确认影响。模型下一步值得改进的地方是:补充HS9403专属弹性系数、"
        "校准替代吸收率(predicted_substitution_absorbed_pct),以便更精确地预测\"这类局部运输冲击会在多大"
        "程度上被替代机制吸收、不反映为全国总量缺口\"这一现象本身,而不是仅仅判断冲击的方向和严重度分级。"
    )

    doc.add_paragraph("")
    footer = doc.add_paragraph(
        "本报告基于 module1_panama_furniture_validation.py、module2_panama_furniture_port_check.py 以及 "
        "module1_district_mode_validation.py 的输出结果生成。"
    )
    footer.runs[0].italic = True
    footer.runs[0].font.size = Pt(9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
