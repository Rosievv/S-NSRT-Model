#!/usr/bin/env python3
"""
District-level (US Customs port-of-entry region) validation for the Panama
Canal drought 2023-2024 / HS 9403 furniture case, using the newly-supplied
Census district-mode breakdown files:
    data/raw/us_census_district_9403_2010_2025_20260905_221251.parquet
    data/raw/us_census_district_854231_2010_2025_20260905_221251.parquet

Both files share the same schema (date, hs_code, district_code, district_name,
country, country_code, value_general, value_consumption, quantity,
containerized_value, containerized_weight, vessel_value, vessel_weight,
air_value, air_weight, trade_type, data_source, collected_at) -- this is the
exact "source country x US entry port x vessel/containerized value" join
that Module 1 (country-only) and Module 2 (port-only, FAF-based) could not
provide.

Methodology ("train on 854231, validate on 9403" per user instruction):
    1) CALIBRATION / SANITY CHECK using HS 854231 (semiconductors -- a
       commodity known to move overwhelmingly by AIR, not ocean vessel).
       If our mode-split and coast-classification logic is sound, it should
       (a) show 854231 as air-dominant, confirming the value/mode fields
       behave as expected, and (b) show NO Panama-Canal-specific East-Coast
       vessel disruption signal for 854231 around 2023-10 to 2024-06 (a
       placebo/specificity check: semiconductors should NOT react to a
       canal disruption the way an all-water furniture trade would).
    2) VALIDATION on HS 9403 (furniture -- known to move overwhelmingly by
       vessel/containerized ocean freight): apply the identical
       country x coast-group YoY-decline confirmation logic (same rule as
       module1_panama_furniture_validation.py: <=-20% YoY within 8 months of
       the event) to see which source-country x coast-of-entry combinations
       actually confirm a decline, and whether it is concentrated in
       East-Coast (Panama-dependent, all-water) districts specifically.

Outputs (reports/module1/):
    district_mode_validation.json
    district_mode_validation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUT_DIR = ROOT_DIR / "reports" / "module1"

NINE403_PATH = RAW_DIR / "us_census_district_9403_2010_2025_20260905_221251.parquet"
SEMI_PATH = RAW_DIR / "us_census_district_854231_2010_2025_20260905_221251.parquet"

EVENT_DATE = pd.Timestamp("2023-10-31")
YOY_DECLINE_THRESHOLD_PCT = -20.0
CONFIRMATION_WINDOW_MONTHS = 8

# Same 6 auto-selected Asia source countries as module1_panama_furniture_validation.py
ASIA_SOURCE_COUNTRIES = ["China", "Vietnam", "Malaysia", "Taiwan", "Indonesia", "India"]

WEST_COAST_DISTRICTS = ["LOS ANGELES, CA", "SAN FRANCISCO, CA", "SEATTLE, WA", "SAN DIEGO, CA", "COLUMBIA-SNAKE, OR"]
EAST_COAST_DISTRICTS = [
    "NEW YORK CITY, NY", "NORFOLK, VA", "SAVANNAH, GA", "CHARLESTON, SC",
    "BALTIMORE, MD", "WILMINGTON, NC", "PHILADELPHIA, PA", "BOSTON, MA",
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["coast"] = df["district_name"].map(
        lambda d: "west" if d in WEST_COAST_DISTRICTS else ("east" if d in EAST_COAST_DISTRICTS else "other")
    )
    return df


def mode_split_sanity_check(df: pd.DataFrame, label: str) -> dict:
    """Sanity check: does the vessel/air/containerized split match known priors?"""
    tot_gen = df["value_general"].sum()
    tot_ves = df["vessel_value"].sum()
    tot_air = df["air_value"].sum()
    tot_cnt = df["containerized_value"].sum()
    return {
        "label": label,
        "vessel_share_pct": round(float(tot_ves / tot_gen * 100), 1),
        "air_share_pct": round(float(tot_air / tot_gen * 100), 1),
        "containerized_share_pct": round(float(tot_cnt / tot_gen * 100), 1),
    }


def yoy_pct(series: pd.Series) -> pd.Series:
    return series.pct_change(12).mul(100)


def confirm_decline(yoy: pd.Series) -> dict:
    event_month = EVENT_DATE.to_period("M").to_timestamp()
    window_end = event_month + pd.DateOffset(months=CONFIRMATION_WINDOW_MONTHS)
    window = yoy.loc[event_month:window_end]
    hits = window.loc[window <= YOY_DECLINE_THRESHOLD_PCT]
    if hits.empty:
        return {
            "status": "not_confirmed",
            "confirmation_month": None,
            "min_yoy_pct_in_window": round(float(window.min()), 2) if not window.empty else None,
        }
    return {
        "status": "confirmed",
        "confirmation_month": hits.index[0].strftime("%Y-%m-%d"),
        "yoy_pct_at_confirmation": round(float(hits.iloc[0]), 2),
        "min_yoy_pct_in_window": round(float(window.min()), 2),
    }


def country_coast_confirmation(df: pd.DataFrame, countries: list[str], value_col: str = "value_general") -> dict:
    """
    For each country, split by coast group (east/west), build a monthly
    value_col series, and run the same YoY-decline confirmation rule used
    in module1_panama_furniture_validation.py -- separately for each coast.
    """
    out = {}
    for country in countries:
        sub = df[(df["country"] == country.upper()) & (df["coast"].isin(["east", "west"]))]
        out[country] = {}
        for coast in ["east", "west"]:
            series = sub[sub["coast"] == coast].groupby("date")[value_col].sum().sort_index()
            if series.empty or len(series) < 24:
                out[country][coast] = {"status": "insufficient_data"}
                continue
            series = series.asfreq("MS").fillna(0.0)
            yoy = yoy_pct(series)
            out[country][coast] = confirm_decline(yoy)
    return out


def build_placebo_check(semi_df: pd.DataFrame) -> dict:
    """
    Specificity/placebo check on HS 854231 (semiconductors, air-dominant):
    aggregate ALL countries' East-Coast-district vessel_value and check
    whether it ALSO shows a spurious '-20% YoY within 8 months of the canal
    event' confirmation. If it does, our method may be prone to false
    positives; if it does not, that supports the method being specific to
    genuinely vessel/ocean-dependent trade disruptions.
    """
    east = semi_df[semi_df["coast"] == "east"]
    series = east.groupby("date")["vessel_value"].sum().sort_index().asfreq("MS").fillna(0.0)
    yoy = yoy_pct(series)
    result = confirm_decline(yoy)
    return {
        "method": (
            "HS854231 (semiconductors, 94.8% air-freight by value) East-Coast-district vessel_value, "
            "all countries combined -- a placebo test. Semiconductors should NOT show a Panama-Canal-"
            "specific vessel disruption, since they barely move by ocean vessel at all."
        ),
        "result": result,
        "placebo_negative_as_expected": result["status"] == "not_confirmed",
        "interpretation": (
            "As expected, semiconductor East-Coast vessel value shows no confirmed Panama-Canal-timed "
            "decline -- supports that the country x coast YoY-confirmation method is specific to "
            "genuinely vessel-dependent trade, not a generic artifact that fires on any commodity."
            if result["status"] == "not_confirmed" else
            "Unexpectedly, semiconductor East-Coast vessel value ALSO shows a confirmed decline in this "
            "window -- this weakens confidence that a 9403 finding is Panama-Canal-specific rather than "
            "a broader East-Coast-wide effect unrelated to the canal (e.g. a different shared shock)."
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    furniture = load(NINE403_PATH)
    semi = load(SEMI_PATH)

    sanity = [
        mode_split_sanity_check(furniture, "HS9403 furniture"),
        mode_split_sanity_check(semi, "HS854231 semiconductors"),
    ]

    placebo = build_placebo_check(semi)

    country_coast_general = country_coast_confirmation(furniture, ASIA_SOURCE_COUNTRIES, "value_general")
    country_coast_containerized = country_coast_confirmation(furniture, ASIA_SOURCE_COUNTRIES, "containerized_value")

    # Build a priority table: which country x coast combos confirm a decline,
    # and is it east-coast-specific (i.e. east confirmed but west not, or east
    # confirmed earlier/deeper than west)?
    priority_rows = []
    for country in ASIA_SOURCE_COUNTRIES:
        east = country_coast_general[country].get("east", {})
        west = country_coast_general[country].get("west", {})
        east_confirmed = east.get("status") == "confirmed"
        west_confirmed = west.get("status") == "confirmed"
        if east_confirmed and not west_confirmed:
            pattern = "east_coast_specific_disruption"
        elif east_confirmed and west_confirmed:
            pattern = "both_coasts_declined"
        elif west_confirmed and not east_confirmed:
            pattern = "west_coast_specific_decline_not_panama_related"
        else:
            pattern = "no_coast_confirmed"
        priority_rows.append({
            "country": country,
            "east_status": east.get("status"),
            "east_confirmation_month": east.get("confirmation_month"),
            "west_status": west.get("status"),
            "west_confirmation_month": west.get("confirmation_month"),
            "pattern": pattern,
            "transport_port_priority": pattern in ("east_coast_specific_disruption", "both_coasts_declined"),
        })

    output = {
        "event_tested": "panama_canal_drought_2023",
        "event_date": EVENT_DATE.strftime("%Y-%m-%d"),
        "hs_code_validated": "9403",
        "hs_code_calibration": "854231",
        "data_sources": {
            "hs9403": str(NINE403_PATH.relative_to(ROOT_DIR)),
            "hs854231": str(SEMI_PATH.relative_to(ROOT_DIR)),
        },
        "step1_calibration_mode_split_sanity_check": sanity,
        "step1_calibration_placebo_check": placebo,
        "step2_validation_country_coast_confirmation_value_general": country_coast_general,
        "step2_validation_country_coast_confirmation_containerized_value": country_coast_containerized,
        "priority_table": priority_rows,
    }

    (OUT_DIR / "district_mode_validation.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )

    lines = []
    lines.append("# District-Level Validation: HS 9403 Furniture x Panama Canal Drought 2023-2024\n")
    lines.append("## Step 1: Calibration (HS854231 semiconductors)")
    for s in sanity:
        lines.append(f"- {s['label']}: vessel={s['vessel_share_pct']}%, air={s['air_share_pct']}%, containerized={s['containerized_share_pct']}%")
    lines.append(f"- Placebo check: {placebo['interpretation']}\n")
    lines.append("## Step 2: Validation (HS9403 furniture) -- country x coast-of-entry YoY confirmation")
    lines.append("| Country | East status | East confirm month | West status | West confirm month | Pattern | Transport/port priority |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in priority_rows:
        lines.append(
            f"| {row['country']} | {row['east_status']} | {row['east_confirmation_month']} | "
            f"{row['west_status']} | {row['west_confirmation_month']} | {row['pattern']} | "
            f"{row['transport_port_priority']} |"
        )
    lines.append("")
    (OUT_DIR / "district_mode_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {(OUT_DIR / 'district_mode_validation.json').relative_to(ROOT_DIR)}")
    print(f"Saved {(OUT_DIR / 'district_mode_validation.md').relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
