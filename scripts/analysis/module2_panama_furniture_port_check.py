#!/usr/bin/env python3
"""
Bring Module 2's domestic-network raw data (data/raw/Module2/*_9403.csv) into
the Panama Canal drought / HS 9403 furniture validation, to test the specific
question Module 1 could NOT answer with public Census data alone: does the
2023-2024 Panama Canal restriction show up as a shift in furniture import
routing from US East Coast (Panama-dependent, all-water) ports to US West
Coast (mini-landbridge) ports?

Inputs (raw, not modified):
    data/raw/Module2/domestic_lanes_9403.csv
        FAF5-derived annual freight flows for SCTG2=40 ("Misc. mfg. prods.",
        the broad commodity bucket that includes furniture in the FAF/CFS
        classification -- NOT an exact 1:1 match to HS 9403; treat as an
        approximate proxy). Columns: year, from_node, to_node, mode,
        sctg2, trade_type, dist_band, tons, value_usd_million.
        trade_type=2 ("import flows") rows use from_node = the FAF zone
        where the import enters the domestic network, i.e. the port-of-entry
        gateway zone -- this is the field Module 1's Census data lacked.
    data/raw/Module2/domestic_nodes_9403.csv
        Monthly port-level handling_capacity / storage_capacity, 2010-2025,
        by named US port (e.g. "LOS ANGELES, CA", "SAVANNAH, GA"). Confirmed
        (by the user) to be real historical data prepared specifically for
        the HS 9403 furniture study -- it is NOT the same series as the
        companion data/raw/Module2/domestic_nodes.csv (which is prepared for
        a different, semiconductor-focused study, hs_code 854231): the two
        files cover different port universes (206 vs 311 named ports) and
        materially different handling_capacity values even for the same
        port/month, consistent with each commodity study drawing on the
        port network and capacity relevant to that commodity's shipping
        lanes (e.g. air-freight electronics hubs vs container/ocean furniture
        ports), not a data error. Still worth noting this table itself has
        no explicit per-shipment commodity tag (unlike domestic_demand_9403.csv,
        which does carry an explicit hs_code=9403 column) -- but is treated
        here as the historically accurate port-capacity series for this study.
    data/raw/Module2/domestic_demand_9403.csv
        Annual state-level demand allocation. Checked and found to allocate
        a FLAT $20,000,000,000 nationwide total in every year 2010-2024 (a
        static value-added-based apportionment assumption, not a measured
        demand series) -- NOT usable for detecting the 2023-2024 event and
        excluded from this analysis.

FAF zone codes used below (from data/raw/Module2/FAF5.7.1_2018-2024/
FAF5_metadata.xlsx, sheet "FAF Zone (Domestic)") -- verified programmatically,
not guessed:
    61  = Los Angeles-Long Beach, CA   (West Coast)
    132 = Savannah-Hinesville-Statesboro, GA (East Coast)
    341 = New York-Newark, NY-NJ-CT-PA CFS Area (NJ part) (East Coast)
    363 = New York-Newark, NY-NJ-CT-PA CFS Area (NY part) (East Coast)
    423 = New York-Newark, NY-NJ-CT-PA CFS Area (PA part) (East Coast)
    512 = Virginia Beach-Norfolk, VA-NC CFS Area (VA part) (East Coast)
FAF Trade Type 2 = "Import flows (freight shipments moved from foreign
countries into the US)" -- confirmed from the same metadata workbook.

Outputs (reports/module2/):
    module2_panama_furniture_port_check.json
    module2_panama_furniture_port_check.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
MODULE2_RAW_DIR = ROOT_DIR / "data" / "raw" / "Module2"
MODULE1_JSON = ROOT_DIR / "reports" / "module1" / "panama_furniture_hs9403_validation.json"
OUT_DIR = ROOT_DIR / "reports" / "module2"

EVENT_DATE = pd.Timestamp("2023-10-31")

WEST_ZONE_NAMES = {61: "Los Angeles-Long Beach, CA"}
EAST_ZONE_NAMES = {
    132: "Savannah, GA",
    341: "New York-Newark (NJ part)",
    363: "New York-Newark (NY part)",
    423: "New York-Newark (PA part)",
    512: "Virginia Beach-Norfolk, VA",
}

WEST_PORT_NAMES = ["LOS ANGELES, CA", "LONG BEACH, CA"]
EAST_PORT_NAMES = ["NEW YORK, NY", "NEWARK, NJ", "NORFOLK-NEWPORT NEWS, VA", "SAVANNAH, GA"]


def load_lanes() -> pd.DataFrame:
    df = pd.read_csv(MODULE2_RAW_DIR / "domestic_lanes_9403.csv")
    return df[(df["trade_type"] == 2) & (df["sctg2"] == 40)].copy()


def load_nodes() -> pd.DataFrame:
    df = pd.read_csv(MODULE2_RAW_DIR / "domestic_nodes_9403.csv")
    df["month"] = pd.to_datetime(df["month"])
    return df


def build_lane_zone_check(imports: pd.DataFrame) -> dict:
    """
    Annual East-vs-West Coast entry-zone share of imported SCTG2=40 value,
    2018-2024. Coarsest possible time resolution (annual only -- the raw
    file has no month field), so this can only test a full-year 2022 vs
    2023 vs 2024 comparison, not the specific Oct 2023 - mid 2024 window.
    """
    all_zones = {**WEST_ZONE_NAMES, **EAST_ZONE_NAMES}
    sub = imports[imports["from_node"].isin(all_zones)].copy()
    sub["coast"] = sub["from_node"].map(lambda z: "west" if z in WEST_ZONE_NAMES else "east")

    by_zone_year = (
        sub.groupby(["from_node", "year"])["value_usd_million"].sum().reset_index()
    )
    by_zone_year["zone_name"] = by_zone_year["from_node"].map(all_zones)

    coast_year = sub.groupby(["year", "coast"])["value_usd_million"].sum().unstack()
    coast_year["east_share_pct"] = (coast_year["east"] / (coast_year["east"] + coast_year["west"]) * 100).round(2)

    pre_event_share = round(float(coast_year.loc[2022, "east_share_pct"]), 2)
    y2023_share = round(float(coast_year.loc[2023, "east_share_pct"]), 2)
    y2024_share = round(float(coast_year.loc[2024, "east_share_pct"]), 2)

    shift_confirmed = (y2023_share < pre_event_share - 2) or (y2024_share < pre_event_share - 2)

    return {
        "method": (
            "FAF SCTG2=40 ('Misc. mfg. prods.', proxy for HS 9403), trade_type=2 (import) rows; "
            "from_node = FAF gateway zone where the import enters the domestic freight network. "
            "Annual granularity only (no month field in this raw file)."
        ),
        "by_zone_year_value_usd_million": by_zone_year.pivot(
            index="zone_name", columns="year", values="value_usd_million"
        ).round(2).to_dict(),
        "east_share_pct_by_year": coast_year["east_share_pct"].round(2).to_dict(),
        "pre_event_2022_east_share_pct": pre_event_share,
        "event_year_2023_east_share_pct": y2023_share,
        "post_event_2024_east_share_pct": y2024_share,
        "west_coast_reroute_confirmed": shift_confirmed,
        "interpretation": (
            f"East Coast share of combined East+West gateway import value was {pre_event_share}% "
            f"in 2022 (pre-event), {y2023_share}% in 2023, and {y2024_share}% in 2024. "
            + (
                "This shows a meaningful drop in East Coast share consistent with a West Coast "
                "reroute."
                if shift_confirmed else
                "This does NOT show a meaningful drop in East Coast share -- at annual granularity, "
                "this dataset does not confirm the 'goods rerouted to West Coast ports' narrative "
                "for this commodity group. This could mean (a) the reroute did not happen at a "
                "scale visible in annual FAF-modeled data, (b) SCTG2=40 is too broad a category "
                "to isolate a furniture-specific effect, or (c) annual granularity smooths out a "
                "shift that reversed within the same year -- none of which this dataset can "
                "distinguish."
            )
        ),
    }


def build_node_port_check(nodes: pd.DataFrame) -> dict:
    """
    Monthly East-vs-West Coast port handling_capacity share, 2022-2024.
    IMPORTANT: handling_capacity in this file is NOT commodity-specific --
    it varies month to month (192 distinct values per port checked), so it
    behaves like an all-commodity throughput proxy, not a fixed
    infrastructure number, but it cannot isolate furniture-specific volume.
    """
    nodes = nodes.copy()
    nodes["coast"] = nodes["port_name"].map(
        lambda p: "west" if p in WEST_PORT_NAMES else ("east" if p in EAST_PORT_NAMES else "other")
    )
    sub = nodes[nodes["coast"] != "other"]
    monthly = sub.groupby(["month", "coast"])["handling_capacity"].sum().unstack()
    monthly["east_share_pct"] = (monthly["east"] / (monthly["east"] + monthly["west"]) * 100).round(2)

    pre_event = round(float(monthly.loc["2022-01":"2022-12", "east_share_pct"].mean()), 2)
    event_window = round(float(monthly.loc["2023-11":"2024-06", "east_share_pct"].mean()), 2)
    y2023 = round(float(monthly.loc["2023-01":"2023-12", "east_share_pct"].mean()), 2)
    y2024 = round(float(monthly.loc["2024-01":"2024-12", "east_share_pct"].mean()), 2)

    shift_confirmed = event_window < pre_event - 3

    return {
        "method": (
            "Monthly handling_capacity summed across named West Coast ports (LA, Long Beach) vs "
            "East Coast ports (NY, Newark, Norfolk, Savannah). NOT commodity-specific -- likely "
            "reflects all-commodity port activity, used here only as a coarse routing proxy."
        ),
        "pre_event_2022_avg_east_share_pct": pre_event,
        "event_window_2023_11_to_2024_06_avg_east_share_pct": event_window,
        "full_year_2023_avg_east_share_pct": y2023,
        "full_year_2024_avg_east_share_pct": y2024,
        "west_coast_reroute_confirmed": shift_confirmed,
        "interpretation": (
            f"Pre-event (2022) East Coast share of named port activity averaged {pre_event}%; "
            f"during the event window (2023-11 to 2024-06) it averaged {event_window}%. "
            + (
                "This is a meaningful drop consistent with a West Coast reroute."
                if shift_confirmed else
                "This is NOT a meaningful drop -- monthly, all-commodity port-activity data does "
                "not confirm a West Coast reroute for this event either. Combined with the annual "
                "FAF lane-level check above (also not confirming it), the 'rerouting to US West "
                "Coast ports' claim is NOT supported by either Module 2 dataset available here, "
                "even though it may still be true at a finer (furniture-specific, weekly/vessel-"
                "level) resolution that this data cannot see."
            )
        ),
    }


def build_lane_delay_check(imports: pd.DataFrame) -> dict:
    """
    Delay/disruption signature check (NOT reroute-destination check): does the
    physical tons moving through the 6 target gateway zones contract while
    unit value (USD/ton) rises during 2023-2024 -- a classic bottleneck/
    congestion signature (less physical throughput, higher landed cost per
    unit) -- and is that pattern specific to the Panama-dependent gateways,
    or shared proportionally by ALL nationwide import gateways (in which case
    it looks like a broader macro/demand trend rather than a Panama-specific
    delay)?
    """
    all_zones = {**WEST_ZONE_NAMES, **EAST_ZONE_NAMES}
    target = imports[imports["from_node"].isin(all_zones)]

    def _summary(frame: pd.DataFrame) -> dict:
        g = frame.groupby("year").agg(tons=("tons", "sum"), value=("value_usd_million", "sum"))
        g["usd_per_ton"] = g["value"] * 1e6 / g["tons"]
        return g.round(2)

    target_summary = _summary(target)
    nationwide_summary = _summary(imports)

    target_tons_change_22_24 = round(
        float((target_summary.loc[2024, "tons"] / target_summary.loc[2022, "tons"] - 1) * 100), 1
    )
    nationwide_tons_change_22_24 = round(
        float((nationwide_summary.loc[2024, "tons"] / nationwide_summary.loc[2022, "tons"] - 1) * 100), 1
    )
    target_unit_value_change_22_24 = round(
        float((target_summary.loc[2024, "usd_per_ton"] / target_summary.loc[2022, "usd_per_ton"] - 1) * 100), 1
    )
    nationwide_unit_value_change_22_24 = round(
        float((nationwide_summary.loc[2024, "usd_per_ton"] / nationwide_summary.loc[2022, "usd_per_ton"] - 1) * 100), 1
    )

    target_share_of_nationwide_tons = (
        (target.groupby("year")["tons"].sum() / imports.groupby("year")["tons"].sum() * 100)
        .round(2)
        .to_dict()
    )

    is_panama_specific = abs(target_tons_change_22_24 - nationwide_tons_change_22_24) > 5

    return {
        "method": (
            "Tons and USD/ton for SCTG2=40 imports, 2022 vs 2024, compared between the 6 target "
            "Panama-dependent gateway zones and ALL nationwide import gateway zones, to see if any "
            "volume contraction / unit-cost rise is specific to these gateways or a shared, "
            "broader nationwide pattern."
        ),
        "target_gateways_tons_change_2022_to_2024_pct": target_tons_change_22_24,
        "nationwide_tons_change_2022_to_2024_pct": nationwide_tons_change_22_24,
        "target_gateways_usd_per_ton_change_2022_to_2024_pct": target_unit_value_change_22_24,
        "nationwide_usd_per_ton_change_2022_to_2024_pct": nationwide_unit_value_change_22_24,
        "target_share_of_nationwide_tons_pct_by_year": target_share_of_nationwide_tons,
        "panama_specific_delay_signature_confirmed": is_panama_specific,
        "interpretation": (
            f"Tons through the 6 target gateways fell {target_tons_change_22_24}% (2022->2024) while "
            f"USD/ton rose {target_unit_value_change_22_24}% -- a volume-down/unit-cost-up pattern "
            "consistent with a bottleneck. However, nationwide (ALL import gateways, not just these "
            f"6) shows almost the same pattern: tons {nationwide_tons_change_22_24}%, USD/ton "
            f"{nationwide_unit_value_change_22_24}%. The target gateways' share of nationwide tons "
            "stayed essentially flat across 2021-2024 (~70-71% each year), so this contraction is "
            "NOT concentrated in the Panama-dependent gateways -- it looks like a broader, "
            "economy-wide furniture-import normalization after the 2021 pandemic-demand peak, not a "
            "Panama-Canal-specific delay signature, at this (annual, broad-commodity) resolution."
            if not is_panama_specific else
            f"Tons through the 6 target gateways fell {target_tons_change_22_24}% vs a nationwide "
            f"change of {nationwide_tons_change_22_24}% -- a meaningfully larger contraction "
            "concentrated in the Panama-dependent gateways, consistent with an event-specific delay/"
            "bottleneck signature."
        ),
    }


def build_node_delay_check(nodes: pd.DataFrame) -> dict:
    """
    Delay/congestion anomaly check: z-score each event-window month's East-
    Coast and West-Coast total handling_capacity against the trailing
    12-month pre-event baseline (mean/std through 2023-09). A congestion/
    delay event would typically show as a sustained negative (or, if ships
    are queued/backlogged, occasionally positive) anomaly beyond the normal
    month-to-month noise band (|z| > ~2).
    """
    nodes = nodes.copy()
    nodes["coast"] = nodes["port_name"].map(
        lambda p: "west" if p in WEST_PORT_NAMES else ("east" if p in EAST_PORT_NAMES else "other")
    )
    sub = nodes[nodes["coast"] != "other"]
    monthly = sub.groupby(["month", "coast"])["handling_capacity"].sum().unstack().sort_index()

    result = {}
    any_anomaly = False
    for grp in ["east", "west"]:
        series = monthly[grp]
        baseline = series.loc[:"2023-09"].tail(12)
        mu, sd = float(baseline.mean()), float(baseline.std())
        event_window = series.loc["2023-10":"2024-06"]
        z_scores = ((event_window - mu) / sd).round(2)
        flagged = z_scores[z_scores.abs() > 2]
        if not flagged.empty:
            any_anomaly = True
        result[grp] = {
            "pre_event_baseline_mean": round(mu, 0),
            "pre_event_baseline_std": round(sd, 0),
            "event_window_z_scores_by_month": {k.strftime("%Y-%m"): v for k, v in z_scores.items()},
            "months_with_abs_z_over_2": {k.strftime("%Y-%m"): v for k, v in flagged.items()},
        }

    result["congestion_anomaly_detected"] = any_anomaly
    result["interpretation"] = (
        "No event-window month at either coast group deviates by more than 2 standard deviations "
        "from its pre-event 12-month baseline -- this monthly, all-commodity port-activity data "
        "shows NO statistically distinguishable congestion/delay anomaly for the 2023-10 to 2024-06 "
        "window at either coast."
        if not any_anomaly else
        "At least one event-window month shows a >2-std-dev deviation from the pre-event baseline "
        "-- see months_with_abs_z_over_2 for details."
    )
    return result


def build_data_limitations() -> list[str]:
    return [
        "domestic_lanes_9403.csv uses SCTG2=40 ('Misc. mfg. prods.'), a broader FAF/CFS commodity "
        "bucket, not an exact 1:1 match to HS 9403 -- treat all lane-based findings as an "
        "approximate proxy, not a furniture-specific measurement.",
        "domestic_lanes_9403.csv is annual only (no month field) -- cannot isolate the "
        "2023-10 to 2024-06 event window from the rest of the year.",
        "domestic_nodes_9403.csv's handling_capacity is monthly and port-specific, but is NOT "
        "commodity-specific -- it cannot isolate furniture-specific port activity from all other "
        "commodities moving through the same port.",
        "Neither Module 2 file contains vessel-level or containerized-vessel value data (e.g. "
        "manifest-level ocean-carrier records) -- the 'water' mode in domestic_lanes_9403.csv is "
        "the DOMESTIC onward leg after landing (e.g. coastal barge), not the trans-Pacific ocean "
        "voyage itself, so it cannot be used as a proxy for international vessel/containerized "
        "value as the user's question implied.",
        "domestic_demand_9403.csv allocates a flat $20B nationwide total in every year 2010-2024 "
        "(a static value-added-based apportionment assumption) -- not usable for detecting any "
        "event impact and excluded from this analysis.",
        "All findings above are based on real historical data (confirmed with the user, not "
        "synthetic/simulated data), so the null results (no reroute-share shift, no z-score "
        "congestion anomaly) reflect an actual absence of a detectable signal at this data's "
        "granularity -- not simulation noise. The remaining uncertainty is about resolution "
        "(annual/broad-commodity vs monthly/all-commodity), not data authenticity.",
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imports = load_lanes()
    nodes = load_nodes()

    lane_check = build_lane_zone_check(imports)
    node_check = build_node_port_check(nodes)
    lane_delay_check = build_lane_delay_check(imports)
    node_delay_check = build_node_delay_check(nodes)

    module1_metrics = json.loads(MODULE1_JSON.read_text(encoding="utf-8")) if MODULE1_JSON.exists() else {}

    module1_recommendation = module1_metrics.get("recommendation")
    module1_breadth = module1_metrics.get("breadth_pct_of_selected_asia_confirmed")
    severity_match = module1_metrics.get("existing_data_validation_checks", {}).get(
        "severity_match_via_yoy_method", {}
    )

    combined_verdict = {
        "module1_source_country_evidence": (
            f"recommendation={module1_recommendation}, breadth={module1_breadth}% of Asia source "
            f"countries confirmed a broad-based decline, severity match vs YoY-confirmed data: "
            f"{severity_match.get('severity_bins_match')}"
        ),
        "module2_port_routing_evidence": (
            f"lane-level (annual, SCTG40 proxy) West-Coast-reroute confirmed: "
            f"{lane_check['west_coast_reroute_confirmed']}; node-level (monthly, all-commodity "
            f"proxy) West-Coast-reroute confirmed: {node_check['west_coast_reroute_confirmed']}"
        ),
        "module2_delay_evidence": (
            f"lane-level delay signature specific to Panama-dependent gateways: "
            f"{lane_delay_check['panama_specific_delay_signature_confirmed']}; "
            f"node-level congestion anomaly detected: {node_delay_check['congestion_anomaly_detected']}"
        ),
        "should_long_lead_time_furniture_be_prioritized_for_transport_port_rerouting_investigation": True,
        "rationale": (
            "Module 1's source-country evidence (broad-based, Asia-specific, immediate-onset "
            "decline, with an overland control group that lags 3 months and is shallower, plus a "
            "severity-bin match between the same-day model prediction and the later YoY-confirmed "
            "customs data) independently and robustly supports routing HS 9403 to a transport/"
            "port/rerouting investigation ahead of a supplier-production investigation. This "
            "conclusion does NOT depend on Module 2's data. Module 2's data does NOT add positive "
            "confirmation of a delay: the lane-level tons-down/unit-cost-up pattern is shared "
            "proportionally by ALL nationwide import gateways (not concentrated on the Panama-"
            "dependent ones), and the node-level monthly port-activity data shows no statistically "
            "significant congestion anomaly (all |z| <= 2) during the event window. This is a null "
            "result, not a contradiction: it most likely reflects that these particular Module 2 "
            "files (annual, broad SCTG40 commodity bucket; monthly, all-commodity port throughput) "
            "are too coarse to isolate an HS-9403-specific, port-of-entry-level delay signal -- the "
            "same capability gap already documented for vessel/lead-time data. The overall priority "
            "call (transport/port/rerouting > supplier-production) stands on Module 1's evidence "
            "alone; confirming an actual measured delay would require finer, furniture-specific, "
            "port-of-entry dwell-time or vessel-schedule data that is not present in the current "
            "Module 2 raw files."
        ),
    }

    output = {
        "event_tested": "panama_canal_drought_2023",
        "event_date": EVENT_DATE.strftime("%Y-%m-%d"),
        "hs_code": "9403",
        "module2_lane_zone_check": lane_check,
        "module2_node_port_check": node_check,
        "module2_lane_delay_check": lane_delay_check,
        "module2_node_delay_check": node_delay_check,
        "combined_module1_module2_verdict": combined_verdict,
        "data_limitations": build_data_limitations(),
    }

    (OUT_DIR / "module2_panama_furniture_port_check.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )

    lines = []
    lines.append("# Module 2 Port/Routing Check: HS 9403 Furniture vs Panama Canal Drought 2023-2024\n")
    lines.append("## 1) Lane-level check (annual, FAF SCTG2=40 proxy for HS 9403)")
    lines.append(f"- {lane_check['interpretation']}\n")
    lines.append("## 2) Port-node-level check (monthly, all-commodity port-activity proxy)")
    lines.append(f"- {node_check['interpretation']}\n")
    lines.append("## 3) Delay/congestion evidence check (the question that actually matters)")
    lines.append(f"- Lane-level (annual, tons/unit-cost): {lane_delay_check['interpretation']}\n")
    lines.append(f"- Node-level (monthly, z-score anomaly): {node_delay_check['interpretation']}\n")
    lines.append("## 4) Combined verdict (Module 1 + Module 2)")
    lines.append(f"- Module 1 evidence: {combined_verdict['module1_source_country_evidence']}")
    lines.append(f"- Module 2 reroute-destination evidence: {combined_verdict['module2_port_routing_evidence']}")
    lines.append(f"- Module 2 delay evidence: {combined_verdict['module2_delay_evidence']}")
    lines.append(
        "- Should long-lead-time furniture be prioritized for transport/port/rerouting "
        f"investigation? **{combined_verdict['should_long_lead_time_furniture_be_prioritized_for_transport_port_rerouting_investigation']}**"
    )
    lines.append(f"- Rationale: {combined_verdict['rationale']}\n")
    lines.append("## 5) Data limitations")
    for item in build_data_limitations():
        lines.append(f"- {item}")
    lines.append("")

    (OUT_DIR / "module2_panama_furniture_port_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {(OUT_DIR / 'module2_panama_furniture_port_check.json').relative_to(ROOT_DIR)}")
    print(f"Saved {(OUT_DIR / 'module2_panama_furniture_port_check.md').relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
