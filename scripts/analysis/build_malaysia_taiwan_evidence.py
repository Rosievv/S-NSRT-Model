#!/usr/bin/env python3
"""Build an auditable evidence chain for the 2021 Malaysia and Taiwan cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_OFFICIAL_DIR = ROOT_DIR / "data" / "raw" / "official_event_sources"
RAW_NEWS_DIR = ROOT_DIR / "data" / "raw" / "disruption_news"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "event_evidence"
REPORT_DIR = ROOT_DIR / "reports" / "module1"
FIG_DIR = REPORT_DIR / "figures"

COLORS = {
    "early_risk": "#287271",
    "operational_restriction": "#d97732",
    "impact_confirmation": "#9a4f38",
    "ink": "#253238",
    "muted": "#60747b",
    "line": "#d3d7d4",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_html(source_path: Path, output_name: str) -> Path:
    soup = BeautifulSoup(source_path.read_text(encoding="utf-8"), "lxml")
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    output_path = PROCESSED_DIR / output_name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _extract_pdf(source_path: Path, output_name: str) -> Path:
    reader = PdfReader(source_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    output_path = PROCESSED_DIR / output_name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_evidence_index() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    taiwan_official = _extract_html(
        RAW_OFFICIAL_DIR / "taiwan_drought_2021_executive_yuan.html",
        "taiwan_drought_2021_executive_yuan.txt",
    )
    malaysia_miti = _extract_pdf(
        RAW_OFFICIAL_DIR / "malaysia_asia_shock_2021_miti.pdf",
        "malaysia_asia_shock_2021_miti.txt",
    )

    records = [
        {
            "event_key": "taiwan_drought_2021",
            "evidence_date": "2021-03-18",
            "source": "Executive Yuan, R.O.C. (Taiwan)",
            "source_type": "official_notice",
            "evidence_stage": "early_risk",
            "local_path": str(taiwan_official.relative_to(ROOT_DIR)),
            "raw_source_path": "data/raw/official_event_sources/taiwan_drought_2021_executive_yuan.html",
            "key_fact": "The government described a severe drought after a typhoon-free year and warned that water shortages could become critical.",
            "model_use": "Drought severity and location signal",
        },
        {
            "event_key": "taiwan_drought_2021",
            "evidence_date": "2021-03-25",
            "source": "South China Morning Post",
            "source_type": "mainstream_news_body",
            "evidence_stage": "operational_restriction",
            "local_path": "data/raw/disruption_news/taiwan_drought_2021/taiwan-chip-makers-water-supplies-cut-as-drought-threatens-island-s-reserves-ef27a8e3fd.txt",
            "raw_source_path": "data/raw/disruption_news/taiwan_drought_2021/taiwan-chip-makers-water-supplies-cut-as-drought-threatens-island-s-reserves-ef27a8e3fd.html",
            "key_fact": "Authorities announced a 15% water-supply cut for companies in two Taichung science parks and identified TSMC and Micron operations.",
            "model_use": "Facility exposure and quantified restriction",
        },
        {
            "event_key": "taiwan_drought_2021",
            "evidence_date": "2021-04-09",
            "source": "BBC News",
            "source_type": "mainstream_news_body",
            "evidence_stage": "impact_confirmation",
            "local_path": "data/raw/disruption_news/taiwan_drought_2021/taiwan-drought-man-retrieves-phone-dropped-in-lake-a-year-ago-c7969f143f.txt",
            "raw_source_path": "data/raw/disruption_news/taiwan_drought_2021/taiwan-drought-man-retrieves-phone-dropped-in-lake-a-year-ago-c7969f143f.html",
            "key_fact": "The report linked water rationing to Taiwan's semiconductor sector and the Hsinchu Science Park supply base.",
            "model_use": "Sector-level impact corroboration",
        },
        {
            "event_key": "malaysia_asia_shock_2021",
            "evidence_date": "2021-05-28",
            "source": "Prime Minister's Office of Malaysia",
            "source_type": "official_metadata",
            "evidence_stage": "early_risk",
            "local_path": None,
            "raw_source_path": None,
            "key_fact": "The government announced the first phase of a nationwide total lockdown for 1-14 June 2021.",
            "model_use": "National lockdown timing; source remained unavailable locally due TLS failure",
        },
        {
            "event_key": "malaysia_asia_shock_2021",
            "evidence_date": "2021-06-01",
            "source": "Malaysia Ministry of International Trade and Industry",
            "source_type": "official_notice",
            "evidence_stage": "operational_restriction",
            "local_path": str(malaysia_miti.relative_to(ROOT_DIR)),
            "raw_source_path": "data/raw/official_event_sources/malaysia_asia_shock_2021_miti.pdf",
            "key_fact": "Only 95,142 of 517,144 registered companies had operating approval; approved-site workforce was reduced by about 2 million from 3.2 million.",
            "model_use": "Quantified operating-capacity restriction",
        },
        {
            "event_key": "malaysia_asia_shock_2021",
            "evidence_date": "2021-08-24",
            "source": "The Straits Times",
            "source_type": "mainstream_news_body",
            "evidence_stage": "impact_confirmation",
            "local_path": "data/raw/disruption_news/malaysia_asia_shock_2021/chip-shortage-set-to-worsen-as-covid-19-rampages-through-malaysia.txt",
            "raw_source_path": "data/raw/disruption_news/malaysia_asia_shock_2021/chip-shortage-set-to-worsen-as-covid-19-rampages-through-malaysia.html",
            "key_fact": "The report identified Malaysia as a chip testing and packaging hub and documented plant suspensions and downstream auto-production effects.",
            "model_use": "Affected-object mapping and downstream impact confirmation",
        },
    ]
    frame = pd.DataFrame(records)
    frame["text_available"] = frame["local_path"].notna()
    frame["sha256"] = frame["local_path"].map(
        lambda value: _sha256(ROOT_DIR / value) if isinstance(value, str) and (ROOT_DIR / value).exists() else None
    )
    return frame


def plot_evidence_chain(frame: pd.DataFrame) -> None:
    plot_frame = frame.copy()
    plot_frame["date"] = pd.to_datetime(plot_frame["evidence_date"])
    plot_frame["country"] = plot_frame["event_key"].map(
        {"taiwan_drought_2021": "Taiwan", "malaysia_asia_shock_2021": "Malaysia"}
    )
    plot_frame["label"] = plot_frame.apply(
        lambda row: f'{row["evidence_date"]}\n{row["source"]}', axis=1
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.6), sharex=False)
    fig.suptitle("Malaysia and Taiwan: Multi-Source Evidence Chains", x=0.08, y=0.97, ha="left", fontsize=20, weight="bold")
    fig.text(0.08, 0.925, "Official risk signals, quantified operating restrictions, and later impact confirmation", color=COLORS["muted"])

    for ax, country in zip(axes, ["Taiwan", "Malaysia"]):
        subset = plot_frame.loc[plot_frame["country"] == country].sort_values("date")
        positions = range(len(subset))
        ax.hlines(0, 0, len(subset) - 1, color=COLORS["line"], linewidth=3)
        for position, (_, row) in zip(positions, subset.iterrows()):
            color = COLORS[row["evidence_stage"]]
            ax.scatter(position, 0, s=190, color=color, edgecolor="white", linewidth=2, zorder=3)
            vertical = 0.34 if position % 2 == 0 else -0.34
            ax.vlines(position, 0, vertical * 0.72, color=color, linewidth=1.5)
            ax.text(position, vertical, row["label"], ha="center", va="center", fontsize=9.5, weight="semibold")
        ax.text(-0.08, 0.5, country, transform=ax.transAxes, fontsize=14, weight="bold", color=COLORS["ink"], va="center")
        ax.set_xlim(-0.5, len(subset) - 0.5)
        ax.set_ylim(-0.62, 0.62)
        ax.axis("off")

    fig.legend(
        handles=[
            Patch(facecolor=COLORS["early_risk"], label="Early risk signal"),
            Patch(facecolor=COLORS["operational_restriction"], label="Operational restriction"),
            Patch(facecolor=COLORS["impact_confirmation"], label="Impact confirmation"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.06, 0.08, 1, 0.89))
    fig.savefig(FIG_DIR / "malaysia_taiwan_evidence_chains.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_evidence_index()
    frame.to_csv(REPORT_DIR / "malaysia_taiwan_evidence_index.csv", index=False)
    summary = {
        "evidence_records": int(len(frame)),
        "records_with_local_text": int(frame["text_available"].sum()),
        "countries": frame.groupby(frame["event_key"].map({"taiwan_drought_2021": "Taiwan", "malaysia_asia_shock_2021": "Malaysia"})).size().to_dict(),
        "evidence_stages": frame["evidence_stage"].value_counts().to_dict(),
        "method_note": "Evidence dates are preserved; later impact confirmation is not backdated as an early warning signal.",
    }
    (REPORT_DIR / "malaysia_taiwan_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    plot_evidence_chain(frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()