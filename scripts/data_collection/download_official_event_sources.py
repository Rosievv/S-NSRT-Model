#!/usr/bin/env python3
"""Download official Figure 3 supporting sources into the local raw-data area."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "official_event_sources"
USER_AGENT = "S-NSRT-Model/1.0 official-event-evidence-collector"

SOURCES = [
    {
        "event_key": "japan_earthquake_2011",
        "date": "2011-03-11",
        "institution": "U.S. Geological Survey",
        "title": "M 9.1 - 2011 Great Tohoku Earthquake, Japan",
        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/official20110311054624120_30/executive",
        "filename": "japan_earthquake_2011_usgs.html",
    },
    {
        "event_key": "thai_flood_2011",
        "date": "2011-10-17",
        "institution": "United Nations Office for Disaster Risk Reduction",
        "title": "Floods reveal risk reduction gaps in Thailand",
        "url": "https://www.undrr.org/news/floods-reveal-risk-reduction-gaps-thailand",
        "filename": "thai_flood_2011_undrr.html",
    },
    {
        "event_key": "japan_export_controls_2019",
        "date": "2019-09-11",
        "institution": "World Trade Organization",
        "title": "DS590: Japan - Measures Related to the Exportation of Products and Technology to Korea",
        "url": "https://www.wto.org/english/tratop_e/dispu_e/cases_e/ds590_e.htm",
        "filename": "japan_export_controls_2019_wto.html",
    },
    {
        "event_key": "covid_q1_2020",
        "date": "2020-01-12",
        "institution": "World Health Organization",
        "title": "COVID-19 - China",
        "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2020-DON233",
        "filename": "covid_q1_2020_who.html",
    },
    {
        "event_key": "taiwan_drought_2021",
        "date": "2021-03-18",
        "institution": "Executive Yuan, R.O.C. (Taiwan)",
        "title": "Premier Su urges public to conserve water",
        "url": "https://english.ey.gov.tw/Page/61BF20C3E89B856/bb664055-d90d-49f5-8480-9372821a4550",
        "filename": "taiwan_drought_2021_executive_yuan.html",
    },
    {
        "event_key": "malaysia_asia_shock_2021",
        "date": "2021-05-28",
        "institution": "Prime Minister's Office of Malaysia",
        "title": "Implementation of Total Lockdown",
        "url": "https://www.pmo.gov.my/wp-content/uploads/2021/06/Kenyataan-Media-PMO-Pelaksanaan-Total-Lockdown.pdf",
        "filename": "malaysia_asia_shock_2021_pmo.pdf",
    },
    {
        "event_key": "malaysia_asia_shock_2021",
        "date": "2021-06-01",
        "institution": "Malaysia Ministry of International Trade and Industry",
        "title": "MITI Leads CIMS 3.0 Coordination to Expedite Approvals",
        "url": "https://www.miti.gov.my/miti/resources/Media%20Release/MEDIA%20RELEASE_CIMS%203.0%20UPDATES_1%20JUNE%202021.pdf",
        "filename": "malaysia_asia_shock_2021_miti.pdf",
    },
    {
        "event_key": "beirut_port_explosion_2020",
        "date": "2020-08-05",
        "institution": "United Nations",
        "title": "Immediate humanitarian assistance mobilized in force, to support Beirut after deadly blast",
        "url": "https://news.un.org/en/story/2020/08/1069712",
        "filename": "beirut_port_explosion_2020_un.html",
    },
    {
        "event_key": "suez_ever_given_2021",
        "date": "2021-03-29",
        "institution": "Suez Canal Authority",
        "title": "Successful Refloating of EVER GIVEN",
        "url": "https://www.suezcanal.gov.eg/English/MediaCenter/News/Pages/nav_29-03-2021.aspx",
        "filename": "suez_ever_given_2021_sca.html",
    },
    {
        "event_key": "yantian_port_covid_2021",
        "date": "2021-05-28",
        "institution": "Yantian District Government",
        "title": "Yantian Port restarts receiving containers",
        "url": "https://www.yantian.gov.cn/English/news/content/post_8819036.html",
        "filename": "yantian_port_covid_2021_government.html",
    },
    {
        "event_key": "panama_canal_drought_2023",
        "date": "2023-10-30",
        "institution": "Panama Canal Authority",
        "title": "Reduction in Transits Due to the Ongoing Deficit in Precipitation in the Canal Watershed",
        "url": "https://pancanal.com/wp-content/uploads/2023/01/ADV48-2023-Reduction-in-Transits-Due-to-the-Ongoing-Deficit-in-Precipitation-in-the-Canal-Watershed.pdf",
        "filename": "panama_canal_drought_2023_acp.pdf",
    },
    {
        "event_key": "red_sea_rerouting_2023",
        "date": "2023-12-15",
        "institution": "A.P. Moller - Maersk",
        "title": "Maersk Operations through Red Sea / Gulf of Aden",
        "url": "https://www.maersk.com/news/articles/2023/12/15/maersk-operations-through-red-sea-gulf-of-aden",
        "filename": "red_sea_rerouting_2023_maersk.html",
    },
    {
        "event_key": "baltimore_bridge_collapse_2024",
        "date": "2024-03-26",
        "institution": "National Transportation Safety Board",
        "title": "Contact of Containership Dali with the Francis Scott Key Bridge",
        "url": "https://www.ntsb.gov/investigations/Pages/DCA24MM031.aspx",
        "filename": "baltimore_bridge_collapse_2024_ntsb.html",
    },
]


def download_source(source: Dict[str, str]) -> Dict[str, object]:
    result: Dict[str, object] = {
        **source,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    try:
        response = requests.get(
            source["url"],
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        result.update(
            {
                "http_status": response.status_code,
                "final_url": response.url,
                "content_type": response.headers.get("Content-Type", ""),
            }
        )
        response.raise_for_status()
        output_path = OUTPUT_DIR / source["filename"]
        output_path.write_bytes(response.content)
        result.update(
            {
                "status": "saved",
                "local_path": str(output_path.relative_to(ROOT_DIR)),
                "byte_count": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            }
        )
    except requests.RequestException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = [download_source(source) for source in SOURCES]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_policy": "Responses are stored unchanged; failed downloads retain provenance and error details only.",
        "sources": results,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    saved = sum(result["status"] == "saved" for result in results)
    print(f"Saved {saved}/{len(results)} official sources to {OUTPUT_DIR}")
    for result in results:
        detail = result.get("local_path", result.get("error", "unknown error"))
        print(f"{result['status']:>6}  {result['event_key']}: {detail}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()