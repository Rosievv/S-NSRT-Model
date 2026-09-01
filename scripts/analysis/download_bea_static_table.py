"""Attempt direct BEA static workbook download and extract 2022 sheet."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "module2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://apps.bea.gov/industry/xls/io-annual/Use_SUT_Framework_2017_2022_DET.xlsx"


def download_bea_static_table() -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    print("[*] Downloading BEA static workbook...")
    resp = requests.get(URL, headers=headers, timeout=60)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    print(f"[*] Response content-type: {content_type}")

    if "spreadsheet" not in content_type and "application/vnd" not in content_type:
        html_path = OUT_DIR / "bea_static_download_response.html"
        html_path.write_bytes(resp.content)
        raise RuntimeError(
            "BEA URL did not return an Excel file. "
            f"Saved response body to {html_path}."
        )

    excel_file = io.BytesIO(resp.content)
    df_2022 = pd.read_excel(excel_file, sheet_name="2022", skiprows=5)

    out_csv = OUT_DIR / "bea_use_table_2022_raw.csv"
    df_2022.to_csv(out_csv, index=False)

    print(f"[+] Parsed 2022 sheet, shape={df_2022.shape}")
    print(f"[+] Wrote: {out_csv}")


if __name__ == "__main__":
    download_bea_static_table()
