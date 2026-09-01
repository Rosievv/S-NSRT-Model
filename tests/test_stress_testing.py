from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from risk_propagation.stress_testing import StressTestRunner


def test_observed_gap_filters_country_and_hs_together() -> None:
    dates = pd.date_range("2020-01-01", "2021-04-01", freq="MS")
    rows = []
    for date in dates:
        rows.append({"date": date, "hs_code": "854231", "country": "Taiwan", "value_usd": 100.0})
        other_value = 100.0 if date < pd.Timestamp("2021-01-01") else 10.0
        rows.append({"date": date, "hs_code": "854231", "country": "Japan", "value_usd": other_value})
    runner = StressTestRunner(pd.DataFrame(rows))
    event = {
        "date_range": ("2021-01", "2021-04"),
        "affected_countries": ["Taiwan"],
        "affected_hs_codes": ["854231"],
    }

    observed = runner._build_observed_supply_gap(event)

    assert observed["observed_supply_gap_pct"] == 0.0