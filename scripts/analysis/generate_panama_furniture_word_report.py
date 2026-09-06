#!/usr/bin/env python3
"""
Generate a Word (.docx) report summarizing the HS9403 furniture x Panama
Canal drought 2023-2024 case study conclusions, including how the model
(module1_panama_furniture_validation.py's classify_country_risk(),
StressTestRunner, PropagationEngine) was applied at each step.

Output: reports/module1/HS9403_Panama_Canal_Case_Study_Summary.docx
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT_DIR / "reports" / "module1" / "HS9403_Panama_Canal_Case_Study_Summary.docx"


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

    title = doc.add_heading("HS9403 Furniture x Panama Canal Drought 2023-2024 -- Case Study Summary", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "This report answers the case-study questions on HS9403 furniture supply-chain risk during the "
        "Panama Canal drought (2023-2024), and explains how the project's model "
        "(module1_panama_furniture_validation.py, StressTestRunner, PropagationEngine) was applied at each step."
    )

    # 1
    doc.add_heading("1. Which countries does the US import HS9403 furniture from", level=1)
    doc.add_paragraph(
        "Model application: this is the output of auto_select_asia_countries() in "
        "module1_panama_furniture_validation.py -- it automatically ranks Asian candidate countries by "
        "historical (through 2022-12-31) value_usd share from Census country-level data, and selects the "
        "Top 6 as the 'main Asia source countries'. This is not a manual selection."
    )
    add_table(
        doc,
        ["Country", "Historical Share"],
        [
            ["China", "56.84%"],
            ["Vietnam", "24.03%"],
            ["Malaysia", "5.54%"],
            ["Taiwan", "4.26%"],
            ["Indonesia", "3.63%"],
            ["India", "2.29%"],
            ["Others (Thailand/Philippines/Korea etc.)", "<3%"],
        ],
    )
    doc.add_paragraph(
        "Explanation: China and Vietnam together account for about 81% of imports -- the two countries "
        "most exposed to this event, and the basis for all subsequent country-level analysis."
    )

    # 2
    doc.add_heading("2. Which source-country x entry-port combinations are more transport-constrained", level=1)
    doc.add_paragraph(
        "Model application: the newly-added district-level data (country x east/west coast x transport mode) "
        "was fed into the model's own classify_country_risk() classification function, reusing its original "
        "thresholds (confirmed decline within 0-2 months of the event = transport; >4 months = source; "
        "otherwise = mixed) -- only the input was changed from country-aggregated series to "
        "country x coast-group series."
    )
    add_table(
        doc,
        ["Country", "Coast", "Status", "Lag (months)", "Model Classification"],
        [
            ["China", "East", "Confirmed (-26.3%)", "0.0", "transport"],
            ["China", "West", "Not confirmed (-11.9%)", "-", "undetermined"],
            ["Vietnam", "East", "Confirmed (-25.8%)", "2.0", "transport"],
            ["Vietnam", "West", "Confirmed (-27.9%)", "-1.0", "mixed"],
            ["Malaysia", "East / West", "Both confirmed (-32.5%/-40.6%)", "-1.0", "mixed"],
            ["Indonesia", "East / West", "Both confirmed (-41.2%/-41.7%)", "-1.0", "mixed"],
            ["India", "East", "Confirmed (-33.7%)", "0.0", "transport"],
            ["India", "West", "Confirmed (-31.1%)", "-1.0", "mixed"],
            ["Taiwan", "East", "Not confirmed (-14.7%)", "-", "undetermined"],
            ["Taiwan", "West", "Confirmed (-32.4%, late to 2024-05)", "6.0", "source"],
        ],
    )
    doc.add_paragraph(
        "Explanation: the key finding for China is 'East Coast confirmed decline, West Coast not confirmed' -- "
        "splitting by coast changed the model's classification for China from 'undetermined' at the country-"
        "aggregate level (only -8.91%, below threshold) to 'transport'. Note honestly: the model has a "
        "boundary quirk -- it computes lag using EVENT_DATE (Oct 31) rather than event_month (Oct 1), so any "
        "case confirmed in the event month itself (Malaysia, Indonesia, Vietnam-West, India-West) gets a lag "
        "of about -1.0 months, which the model's thresholds bucket as 'mixed' rather than the more intuitive "
        "'transport'. This is a limitation of the model's own formula, not genuine ambiguity in the new evidence."
    )

    # 3
    doc.add_heading("3. Which product x source combinations should be prioritized for transport/port/rerouting investigation", level=1)
    doc.add_paragraph(
        "Model application: directly reading the combinations classified as 'transport' in the table above, "
        "ordered by the share figures from Section 1."
    )
    add_table(
        doc,
        ["Priority", "Combination", "Basis"],
        [
            ["Highest", "China x East Coast ports", "Largest share (56.8%) + model classifies as transport, cleanest signal"],
            ["High", "Vietnam x East Coast ports, India x East Coast ports", "Model classifies as transport"],
            ["Medium (recommend manual review)", "Malaysia, Indonesia, Vietnam-West, India-West", "Model classifies as mixed, but likely due to the lag boundary quirk; the underlying decline evidence is equally strong"],
            ["Not recommended", "Taiwan x West Coast ports", "Model classifies as source; timing/direction inconsistent with the Panama Canal mechanism"],
        ],
    )

    # 4
    doc.add_heading("4. Final Source / Transport / Mixed / Undetermined classification", level=1)
    doc.add_paragraph(
        "Model application / explanation: this table comes entirely from running classify_country_risk() "
        "as-is -- no manual adjustment to the classification logic was made, only a richer (port-level) "
        "dataset was supplied as input. This demonstrates the model's classification logic is reusable and "
        "extensible: a finer-grained dataset directly produces a finer-grained classification without "
        "rewriting the rules."
    )
    add_table(
        doc,
        ["Country", "Model Classification"],
        [
            ["China", "Transport"],
            ["Vietnam, India", "Transport (East Coast) / Mixed (West Coast, lag quirk)"],
            ["Malaysia, Indonesia", "Mixed (lag quirk; evidence substantively supports transport)"],
            ["Taiwan", "Source"],
        ],
    )

    # 5
    doc.add_heading("5. Remaining public-data gaps", level=1)
    doc.add_paragraph(
        "Model application: directly quoting the 'capability gap matrix' constants hard-coded in the script "
        "(bullwhip_and_stockout, case1_3way_priority_test dictionaries) -- these are limitations the model "
        "declares about itself, not conclusions summarized after the fact."
    )
    doc.add_paragraph(
        "- bullwhip_and_stockout.status = 'not_captured': reason -- no inventory / weeks-of-supply / order-"
        "backlog data available.\n"
        "- case1_3way_priority_test.status = 'undetermined_pending_company_data': reason -- even with the new "
        "port-level data, the 3-way test (long lead time + USEC all-water destination + low weeks-of-supply) "
        "is still blocked on the missing weeks_of_supply field."
    )
    doc.add_paragraph(
        "Explanation: the new district-level data solved the 'port' dimension, but the 'inventory / orders / "
        "lead time' dimensions were declared out of scope by the model from the start, and must be supplied "
        "via company data."
    )

    # 6
    doc.add_heading("6. Company-data plug-in interface (unchanged)", level=1)
    doc.add_paragraph(
        "Model application: the load_company_overrides() function reads "
        "data/company/hs9403_company_overrides.csv (does not currently exist; the function gracefully "
        "returns None if absent). Required schema: country, entry_port, transit_mode (== 'all_water' triggers "
        "USEC classification), weeks_of_supply (compared against LOW_WOS_THRESHOLD_WEEKS). Once this file is "
        "populated, classify_country_risk() automatically upgrades case1_priority from "
        "'undetermined_pending_company_data' to 'high_priority' or 'lower_priority' -- no code changes needed."
    )

    # 7
    doc.add_heading("7. Model's predicted impact of this event (StressTestRunner + PropagationEngine)", level=1)
    doc.add_paragraph(
        "Model application, step by step:\n"
        "1) Input: the severity implied by the ACP's own official advisory -- normal transit capacity of 36 "
        "ships/day cut to about 24 ships/day -- converted into estimated_severity = 0.33. This step does not "
        "depend on any trade data and can be computed the same day as the advisory.\n"
        "2) Propagation: PropagationEngine multiplies a substitution elasticity by an event-type multiplier "
        "(the generic 'logistics' multiplier) to compute the predicted supply gap. Because HS_ELASTICITY_MAP "
        "has no HS9403-specific entry, the model falls back to the default elasticity of 0.3 -- a known "
        "precision gap.\n"
        "3) Model prediction output: predicted_supply_gap_pct = 15.19%, severity_bin_predicted = 'high'."
    )
    doc.add_paragraph(
        "Observed (using the same 'pre-event 24-month trend extrapolation' method as "
        "StressTestRunner._build_observed_supply_gap): observed_supply_gap_pct = 0%, "
        "severity_bin_observed = 'low'. directional_hit = 0 -- the model's directional call was wrong this "
        "time: it predicted a material gap, but in aggregate there was none."
    )
    doc.add_paragraph(
        "Explanation of what this 'wrong prediction' means: combined with the new port-level evidence, the "
        "model was not entirely wrong -- it correctly identified that the Panama Canal event would cause a "
        "transport disruption (the confirmed East Coast decline is real), but it underestimated the "
        "substitution/diversion mechanism (West Coast ports, inventory buffers, cross-country substitute "
        "sourcing). The model does have a field intended to capture this: "
        "predicted_substitution_absorbed_pct = 20.01%, but this absorption rate is also computed from the "
        "default elasticity and has not been calibrated against this event's actual outcome."
    )
    doc.add_paragraph(
        "Summary of the model's predicted impact: the model can quickly assign a risk severity level "
        "on the day an event is announced (about 38-39 days faster than waiting for customs-data "
        "confirmation), and it correctly judged the direction of 'a real transport/port-level disruption "
        "exists'. However, it mispredicted the magnitude of the aggregate supply gap (predicted high, "
        "observed low), primarily because HS9403 lacks a dedicated elasticity coefficient and the "
        "substitution-absorption rate has not been calibrated -- these are the two most valuable next "
        "improvements for the model."
    )

    doc.add_paragraph("")
    footer = doc.add_paragraph(
        "Generated from module1_panama_furniture_validation.py, module2_panama_furniture_port_check.py, "
        "and module1_district_mode_validation.py outputs."
    )
    footer.runs[0].italic = True
    footer.runs[0].font.size = Pt(9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
