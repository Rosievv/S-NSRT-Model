#!/usr/bin/env python3
"""Build Module2 domestic inputs with real-data-first plus modeled fill.

Policy:
1) Use observed public data wherever available.
2) If required values are missing, fill them via transparent modeled rules.
3) Keep all modeled values explicitly tagged in source/confidence fields.

Outputs:
- data/processed/domestic_nodes.csv
- data/processed/domestic_lanes.csv
- data/processed/domestic_demand.csv
- data/processed/domestic_demand_state_assumption.csv
- data/processed/domestic_data_quality_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_MODULE2_DIR = PROJECT_ROOT / "data" / "raw" / "Module2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FAF_DIR = RAW_MODULE2_DIR / "FAF5_2022_HighwayAssignmentResults_04_07_2022" / "CSV Format"
FAF_DOMESTIC_FILE = FAF_DIR / "FAF5 Domestic Truck Flows by Commodity_2022.csv"
FAF_IMPORT_FILE = FAF_DIR / "FAF5 Import Truck Flows by Commodity_2022.csv"
RAW_NODES_FILE = RAW_MODULE2_DIR / "domestic_nodes.csv"
FAF_GDB_FILE = RAW_MODULE2_DIR / "Forecast2050BaselineNetworksAndAssignments" / "Networks" / "Geodatabase Format" / "FAF5Network.gdb"

BEA_CANDIDATES = [
    RAW_MODULE2_DIR / "U.Value Added by Industry.csv",
    RAW_MODULE2_DIR / "Value Added by Industry.csv",
]
SQGDP_FILE = RAW_MODULE2_DIR / "SQGDP1 State quarterly gross domestic product (GDP) summary.csv"
RAW_STATE_DEMAND_PANEL_FILE = RAW_MODULE2_DIR / "domestic_demand.csv"

MONTH_LABEL = "2022-01"
ALLOW_MODELED_FILL = True
HS_CODES = ("854231", "854232", "854233", "854239")
HS_WEIGHTS = {
    "854231": 0.42,
    "854232": 0.28,
    "854233": 0.08,
    "854239": 0.22,
}


def _required_columns(label: str) -> Dict[str, str]:
    suffix = "Dom" if label == "domestic" else "Imp"
    return {
        "id": "ID",
        "ab_tons": f"AB Durable Manuf (high tech)-Tons_22 {suffix}",
        "ba_tons": f"BA Durable Manuf (high tech)-Tons_22 {suffix}",
        "ab_trips": f"AB Durable Manuf (high tech)-Trips_22 {suffix}",
        "ba_trips": f"BA Durable Manuf (high tech)-Trips_22 {suffix}",
    }


def _load_faf_geography_lookup() -> pd.DataFrame:
    try:
        import pyogrio
    except ImportError as exc:
        raise ImportError(
            "pyogrio is required to recover FAF link geography from the geodatabase"
        ) from exc

    if not FAF_GDB_FILE.exists():
        raise FileNotFoundError(f"Missing FAF geodatabase: {FAF_GDB_FILE}")

    columns = ["ID", "STATE", "FAFZONE", "BorderState1", "BorderState2", "Country", "STFIPS", "County_Name"]
    geo = pyogrio.read_dataframe(str(FAF_GDB_FILE), layer="FAF5_Links", columns=columns)
    geo = geo.rename(
        columns={
            "ID": "id",
            "STATE": "faf_state",
            "FAFZONE": "faf_zone",
            "BorderState1": "border_state_1",
            "BorderState2": "border_state_2",
            "Country": "faf_country",
            "STFIPS": "faf_stfips",
            "County_Name": "faf_county_name",
        }
    )
    geo["id"] = pd.to_numeric(geo["id"], errors="coerce").astype("Int64")
    geo = geo.dropna(subset=["id"]).copy()
    geo["id"] = geo["id"].astype(int)
    geo["faf_state"] = geo["faf_state"].astype(str).str.strip()
    geo["faf_zone"] = pd.to_numeric(geo["faf_zone"], errors="coerce").astype("Int64")
    geo["faf_stfips"] = pd.to_numeric(geo["faf_stfips"], errors="coerce").astype("Int64")
    return geo.drop_duplicates(subset=["id"], keep="first")


def _load_faf_subset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing FAF file: {path}")

    req = _required_columns(label)
    df = pd.read_csv(path, usecols=list(req.values()))
    df = df.rename(columns={v: k for k, v in req.items()})

    for col in ("ab_tons", "ba_tons", "ab_trips", "ba_trips"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    return df


def _build_lanes(df: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    total = df["ab_tons"] + df["ba_tons"]
    reliability = 1.0 - (df["ab_tons"] - df["ba_tons"]).abs() / total.replace(0.0, np.nan)
    reliability = reliability.fillna(0.5).clip(0.0, 1.0)

    ab = pd.DataFrame(
        {
            "link_id": df["id"],
            "lane_direction": "AB",
            "month": MONTH_LABEL,
            "from_node": "FAF_LINK_" + df["id"].astype(str) + "_A",
            "to_node": "FAF_LINK_" + df["id"].astype(str) + "_B",
            "mode": "truck",
            "capacity": df["ab_tons"],
            "unit_cost": (df["ab_trips"] / df["ab_tons"].replace(0.0, np.nan)).fillna(0.08),
            "transit_days": 1,
            "reliability": reliability,
            "source": f"faf5_{scope_label}_2022_observed",
            "confidence": "high",
        }
    )

    ba = pd.DataFrame(
        {
            "link_id": df["id"],
            "lane_direction": "BA",
            "month": MONTH_LABEL,
            "from_node": "FAF_LINK_" + df["id"].astype(str) + "_B",
            "to_node": "FAF_LINK_" + df["id"].astype(str) + "_A",
            "mode": "truck",
            "capacity": df["ba_tons"],
            "unit_cost": (df["ba_trips"] / df["ba_tons"].replace(0.0, np.nan)).fillna(0.08),
            "transit_days": 1,
            "reliability": reliability,
            "source": f"faf5_{scope_label}_2022_observed",
            "confidence": "high",
        }
    )

    lanes = pd.concat([ab, ba], ignore_index=True)
    lanes = lanes[lanes["capacity"] > 0].copy()
    lanes["unit_cost"] = lanes["unit_cost"].clip(lower=0.001, upper=10.0)
    lanes["reliability"] = lanes["reliability"].clip(lower=0.0, upper=1.0)
    return lanes


def build_domestic_lanes_from_faf() -> pd.DataFrame:
    domestic = _load_faf_subset(FAF_DOMESTIC_FILE, "domestic")
    imported = _load_faf_subset(FAF_IMPORT_FILE, "import")
    geo_lookup = _load_faf_geography_lookup()
    lanes = pd.concat(
        [_build_lanes(domestic, "domestic"), _build_lanes(imported, "import")],
        ignore_index=True,
    )

    lanes = lanes.merge(geo_lookup, left_on="link_id", right_on="id", how="left")
    lanes = lanes.drop(columns=["id"], errors="ignore")

    lanes = (
        lanes.groupby(
            [
                "month",
                "link_id",
                "lane_direction",
                "from_node",
                "to_node",
                "mode",
                "source",
                "confidence",
                "faf_state",
                "faf_zone",
                "border_state_1",
                "border_state_2",
                "faf_country",
                "faf_stfips",
                "faf_county_name",
            ],
            as_index=False,
        )
        .agg({"capacity": "sum", "unit_cost": "mean", "transit_days": "mean", "reliability": "mean"})
    )
    lanes["transit_days"] = lanes["transit_days"].round().astype(int)
    return lanes.sort_values(["from_node", "to_node", "source"]).reset_index(drop=True)


def build_nodes_from_raw() -> pd.DataFrame:
    if not RAW_NODES_FILE.exists():
        raise FileNotFoundError(f"Missing required real nodes file: {RAW_NODES_FILE}")

    nodes = pd.read_csv(RAW_NODES_FILE)
    required = ["month", "node_id", "node_type", "state", "handling_capacity", "storage_capacity"]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        if not ALLOW_MODELED_FILL:
            raise ValueError(f"Missing columns in raw domestic_nodes.csv: {missing}")
        # Build a minimal table from available columns, then model missing values.
        for col in missing:
            nodes[col] = np.nan

    nodes["month"] = nodes["month"].astype(str)
    nodes["hs_code"] = "8542"
    nodes["source"] = "us_census_port_observed"
    nodes["confidence"] = "high"

    # Modeled fill for missing capacities: infer from node-level historical medians.
    for cap_col, default_val in (("handling_capacity", 1.0e8), ("storage_capacity", 1.5e8)):
        cap = pd.to_numeric(nodes[cap_col], errors="coerce")
        if cap.isna().any() and ALLOW_MODELED_FILL:
            by_node = cap.groupby(nodes["node_id"]).transform("median")
            cap = cap.fillna(by_node)
            cap = cap.fillna(default_val)
            modeled_mask = nodes[cap_col].isna() | pd.to_numeric(nodes[cap_col], errors="coerce").isna()
            nodes.loc[modeled_mask, "source"] = "us_census_port_modeled_fill"
            nodes.loc[modeled_mask, "confidence"] = "medium"
        nodes[cap_col] = cap

    out_cols = [
        "month",
        "hs_code",
        "source",
        "confidence",
        "node_id",
        "node_type",
        "state",
        "handling_capacity",
        "storage_capacity",
    ]
    return nodes[out_cols].copy()


def _find_bea_file() -> Path:
    for candidate in BEA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No BEA value-added file found in data/raw/Module2. Expected one of: "
        + ", ".join(str(p.name) for p in BEA_CANDIDATES)
    )


def _load_bea_semiconductor_value() -> Tuple[Path, float, str]:
    bea_path = _find_bea_file()
    df = pd.read_csv(bea_path, skiprows=3)
    name_col = "Unnamed: 1" if "Unnamed: 1" in df.columns else ("Line" if "Line" in df.columns else None)
    if name_col is None:
        raise ValueError(f"Unexpected BEA schema in {bea_path.name}")

    name_series = df[name_col].astype(str)
    mask = name_series.str.contains("Semiconductor and other electronic component manufacturing", case=False, na=False)
    semicon = df[mask]
    if semicon.empty:
        raise ValueError(f"Could not find semiconductor row in {bea_path.name}")

    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    if not year_cols:
        raise ValueError(f"No year columns found in {bea_path.name}")

    year = "2022" if "2022" in year_cols else str(year_cols[-1])
    value = pd.to_numeric(semicon.iloc[0][year], errors="coerce")
    if pd.isna(value):
        raise ValueError(f"Semiconductor value is missing for year {year} in {bea_path.name}")

    annual_value_usd = float(value) * 1e9
    return bea_path, annual_value_usd, year


def build_demand_from_bea() -> pd.DataFrame:
    try:
        bea_path, annual_value_usd, bea_year = _load_bea_semiconductor_value()
        monthly_total = annual_value_usd / 12.0
        src = f"bea_{bea_year}_{bea_path.stem.replace(' ', '_').lower()}_observed"
        conf = "high"
    except Exception as exc:
        if not ALLOW_MODELED_FILL:
            raise
        # Modeled fallback from observed domestic lane capacity scale.
        lanes = build_domestic_lanes_from_faf()
        monthly_total = float(lanes["capacity"].sum()) * 250.0
        src = "bea_missing_modeled_from_faf_capacity"
        conf = "medium"
        print(f"[warn] BEA missing/unreadable; using modeled demand fill from FAF capacity: {exc}")

    return pd.DataFrame(
        [
            {
                "month": MONTH_LABEL,
                "demand_zone": "USA_TOTAL",
                "hs_code": "8542",
                "demand_value_usd": round(monthly_total, 2),
                "source": src,
                "confidence": conf,
            }
        ]
    )


def build_state_assumption_demand_from_sqgdp() -> pd.DataFrame:
    """Build state-level demand assumptions scaled from real BEA + real SQGDP.

    This output is for model development only. It keeps traceability by storing
    explicit assumption metadata and leaving the primary observed demand output
    untouched.
    """
    if not SQGDP_FILE.exists():
        raise FileNotFoundError(f"Missing SQGDP file: {SQGDP_FILE}")

    _, annual_value_usd, bea_year = _load_bea_semiconductor_value()
    monthly_total = annual_value_usd / 12.0

    sq = pd.read_csv(SQGDP_FILE, skiprows=3)
    required = {"GeoFIPS", "GeoName", "LineCode", "Description"}
    missing = required.difference(sq.columns)
    if missing:
        raise ValueError(f"SQGDP file missing required columns: {sorted(missing)}")

    quarter_cols = [c for c in sq.columns if str(c).startswith("2022:Q")]
    if not quarter_cols:
        raise ValueError("SQGDP file does not contain 2022 quarter columns")

    # LineCode 1 is Real GDP in this table. Exclude national row and aggregate only states.
    line_code = pd.to_numeric(sq["LineCode"], errors="coerce")
    geofips = sq["GeoFIPS"].astype(str).str.zfill(5)
    state_rows = sq[(line_code == 1.0) & (geofips != "00000")].copy()
    if state_rows.empty:
        raise ValueError("SQGDP state rows for LineCode 1 are empty")

    state_rows[quarter_cols] = state_rows[quarter_cols].apply(pd.to_numeric, errors="coerce")
    state_rows["state_level"] = state_rows[quarter_cols].mean(axis=1)
    state_rows = state_rows.dropna(subset=["state_level"])
    total_level = float(state_rows["state_level"].sum())
    if total_level <= 0:
        raise ValueError("SQGDP state levels sum to zero")

    rows = []
    for _, r in state_rows.iterrows():
        state_share = float(r["state_level"] / total_level)
        zone = str(r["GeoName"]).strip().upper().replace(" ", "_")
        zone = "STATE_" + zone
        for hs_code in HS_CODES:
            rows.append(
                {
                    "month": MONTH_LABEL,
                    "demand_zone": zone,
                    "hs_code": hs_code,
                    "demand_value_usd": round(monthly_total * state_share * HS_WEIGHTS[hs_code], 2),
                    "source": "bea_sqgdp_state_share_assumption",
                    "confidence": "medium",
                    "assumption_flag": True,
                    "assumption_basis": f"BEA {bea_year} semicon total scaled by SQGDP 2022 state GDP shares",
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["demand_zone", "hs_code"]).reset_index(drop=True)


def build_state_demand_from_raw_panel(target_year: int = 2022) -> pd.DataFrame:
    """Build state-level demand directly from the newly added raw demand panel.

    If the panel is unavailable or invalid, callers can fall back to the
    SQGDP-based assumption builder.
    """
    if not RAW_STATE_DEMAND_PANEL_FILE.exists():
        raise FileNotFoundError(f"Missing raw state demand panel: {RAW_STATE_DEMAND_PANEL_FILE}")

    panel = pd.read_csv(RAW_STATE_DEMAND_PANEL_FILE)
    required = {"year", "state", "demand_value_usd", "hs_code"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Raw state demand panel missing required columns: {sorted(missing)}")

    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel[panel["year"] == float(target_year)].copy()
    if panel.empty:
        raise ValueError(f"Raw state demand panel has no rows for year={target_year}")

    panel["demand_value_usd"] = pd.to_numeric(panel["demand_value_usd"], errors="coerce")
    panel = panel.dropna(subset=["demand_value_usd", "state"]).copy()
    panel["state_clean"] = panel["state"].astype(str).str.strip()
    panel["demand_zone"] = "STATE_" + panel["state_clean"].str.upper().str.replace(" ", "_", regex=False)
    panel["hs_code"] = panel["hs_code"].astype(str)

    out = (
        panel.groupby(["demand_zone", "hs_code"], as_index=False)
        .agg(
            demand_value_usd=("demand_value_usd", "sum"),
            geo_fips=("geo_fips", "first") if "geo_fips" in panel.columns else ("state_clean", "first"),
            value_added=("value_added", "sum") if "value_added" in panel.columns else ("demand_value_usd", "sum"),
            state_share=("state_share", "sum") if "state_share" in panel.columns else ("demand_value_usd", "sum"),
        )
        .sort_values(["demand_zone", "hs_code"])
        .reset_index(drop=True)
    )

    out.insert(0, "month", MONTH_LABEL)
    out["source"] = f"domestic_demand_panel_{target_year}_observed"
    out["confidence"] = "high"
    out["assumption_flag"] = False
    out["assumption_basis"] = "State-level demand panel provided in raw Module2 input"
    return out


def enforce_source_contract(frame: pd.DataFrame, name: str) -> None:
    if "source" not in frame.columns:
        raise ValueError(f"{name} is missing source column")
    allowed_tokens = ("observed", "modeled_fill", "modeled", "assumption")
    values = frame["source"].astype(str).str.lower().unique().tolist()
    bad = [v for v in values if not any(tok in v for tok in allowed_tokens)]
    if bad:
        raise ValueError(f"{name} contains invalid source tags: {bad}")


def run_quality_checks(name: str, frame: pd.DataFrame, key_cols: List[str]) -> Dict[str, object]:
    duplicate_keys = int(frame.duplicated(subset=key_cols).sum())
    null_rates = {col: float(rate) for col, rate in frame.isna().mean().round(4).to_dict().items() if rate > 0}
    summary = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "duplicate_key_rows": duplicate_keys,
        "month_min": str(frame["month"].min()) if "month" in frame.columns and not frame.empty else None,
        "month_max": str(frame["month"].max()) if "month" in frame.columns and not frame.empty else None,
        "null_rates": null_rates,
    }
    print(f"[check] {name}: rows={summary['rows']} duplicate_keys={duplicate_keys} months={summary['month_min']}..{summary['month_max']}")
    if null_rates:
        print(f"[check] {name}: null_rates={null_rates}")
    return summary


def write_outputs(
    nodes: pd.DataFrame,
    lanes: pd.DataFrame,
    demand: pd.DataFrame,
    state_assumption_demand: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(PROCESSED_DIR / "domestic_nodes.csv", index=False)
    lanes.to_csv(PROCESSED_DIR / "domestic_lanes.csv", index=False)
    demand.to_csv(PROCESSED_DIR / "domestic_demand.csv", index=False)
    state_assumption_demand.to_csv(PROCESSED_DIR / "domestic_demand_state_assumption.csv", index=False)


def write_quality_report(report: Dict[str, object]) -> None:
    with open(PROCESSED_DIR / "domestic_data_quality_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    print("[*] Building Module 2 domestic datasets (real-first with modeled fill allowed)...")
    lanes = build_domestic_lanes_from_faf()
    nodes = build_nodes_from_raw()
    demand = build_demand_from_bea()
    try:
        state_assumption_demand = build_state_demand_from_raw_panel(target_year=2022)
    except Exception as exc:
        if not ALLOW_MODELED_FILL:
            raise
        print(f"[warn] Raw state demand panel unavailable/invalid; fallback to SQGDP assumptions: {exc}")
        state_assumption_demand = build_state_assumption_demand_from_sqgdp()

    enforce_source_contract(nodes, "domestic_nodes")
    enforce_source_contract(lanes, "domestic_lanes")
    enforce_source_contract(demand, "domestic_demand")
    enforce_source_contract(state_assumption_demand, "domestic_demand_state_assumption")

    write_outputs(nodes, lanes, demand, state_assumption_demand)

    quality_report = {
        "real_only_mode": False,
        "modeled_fill_enabled": ALLOW_MODELED_FILL,
        "domestic_nodes": run_quality_checks("domestic_nodes", nodes, ["month", "node_id", "hs_code"]),
        "domestic_lanes": run_quality_checks("domestic_lanes", lanes, ["month", "link_id", "lane_direction", "from_node", "to_node", "mode", "source"]),
        "domestic_demand": run_quality_checks("domestic_demand", demand, ["month", "demand_zone", "hs_code"]),
        "domestic_demand_state_assumption": run_quality_checks(
            "domestic_demand_state_assumption",
            state_assumption_demand,
            ["month", "demand_zone", "hs_code"],
        ),
    }
    write_quality_report(quality_report)

    print(f"[ok] Wrote {(PROCESSED_DIR / 'domestic_nodes.csv').relative_to(PROJECT_ROOT)}")
    print(f"[ok] Wrote {(PROCESSED_DIR / 'domestic_lanes.csv').relative_to(PROJECT_ROOT)}")
    print(f"[ok] Wrote {(PROCESSED_DIR / 'domestic_demand.csv').relative_to(PROJECT_ROOT)}")
    print(f"[ok] Wrote {(PROCESSED_DIR / 'domestic_demand_state_assumption.csv').relative_to(PROJECT_ROOT)}")
    print(f"[ok] Wrote {(PROCESSED_DIR / 'domestic_data_quality_summary.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
