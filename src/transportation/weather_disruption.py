"""
Weather Disruption Detector

Fetches live tropical-cyclone data from NOAA's National Hurricane Center
(NHC) public feed and translates active storms into port-level capacity
reductions that can be fed into Module2's network optimisation as a
weather-driven rerouting trigger (as opposed to the purely statistical
volume/cost-spike disruptions in ``disruption_detector.py``).

Data source
-----------
``https://www.nhc.noaa.gov/CurrentStorms.json`` — a public, no-API-key
JSON feed listing all currently active tropical cyclones tracked by NHC,
including position (lat/lon) and intensity classification.

Design notes
------------
- No API key required.
- Network access failures (offline environment, DNS blocked, etc.) are
  handled gracefully: methods return an "all clear" (no disruption)
  result rather than raising, so callers can always run without network.
- Severity -> capacity multiplier mapping is a simple, explicit rule set
  (not calibrated against real port-closure data) and should be treated
  as a first-pass heuristic, not a validated impact model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger("SCRAM.Transportation.WeatherDisruption")

NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"

# Major US ports relevant to Module2's network (must match port keys used
# in scripts/analysis/module2_model_development.py).
PORT_COORDINATES: Dict[str, Dict[str, float]] = {
    "PORT_LA_LB": {"lat": 33.74, "lon": -118.25},   # Los Angeles / Long Beach
    "PORT_HOU": {"lat": 29.75, "lon": -95.10},      # Houston / Gulf Coast
    "PORT_NY_NJ": {"lat": 40.67, "lon": -74.15},    # New York / New Jersey
}

# Classification -> capacity multiplier for a port within the impact radius.
# 1.0 = no impact. Lower = larger capacity reduction.
CLASSIFICATION_IMPACT: Dict[str, float] = {
    "TD": 0.90,   # Tropical Depression
    "TS": 0.80,   # Tropical Storm
    "HU": 0.55,   # Hurricane (Cat 1-2 typical NHC classification code)
    "MH": 0.30,   # Major Hurricane (Cat 3+)
}
DEFAULT_IMPACT_RADIUS_KM = 800.0


@dataclass(frozen=True)
class PortWeatherImpact:
    port: str
    storm_name: str
    classification: str
    distance_km: float
    capacity_multiplier: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _parse_latlon(value: object) -> Optional[float]:
    """Parse NHC's lat/lon strings like '25.4N' or '89.3W' into signed floats."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    sign = -1.0 if text.endswith(("S", "W")) else 1.0
    try:
        return sign * float(text[:-1])
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


class WeatherDisruptionDetector:
    """
    Detect active tropical cyclones near Module2's ports and translate
    them into per-port capacity multipliers usable as a real-world
    rerouting trigger.
    """

    def __init__(
        self,
        port_coordinates: Optional[Dict[str, Dict[str, float]]] = None,
        impact_radius_km: float = DEFAULT_IMPACT_RADIUS_KM,
        timeout_s: float = 8.0,
    ):
        self.port_coordinates = port_coordinates or PORT_COORDINATES
        self.impact_radius_km = impact_radius_km
        self.timeout_s = timeout_s

    def fetch_active_storms(self) -> List[Dict[str, object]]:
        """
        Fetch active storms from NHC. Returns an empty list (not an
        exception) on any network/parsing failure so callers can always
        proceed offline.
        """
        try:
            resp = requests.get(NHC_CURRENT_STORMS_URL, timeout=self.timeout_s)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - network/env can fail many ways
            logger.warning("Could not fetch NHC active-storms feed: %s", exc)
            return []

        storms = payload.get("activeStorms", []) if isinstance(payload, dict) else []
        parsed: List[Dict[str, object]] = []
        for storm in storms:
            lat = _parse_latlon(storm.get("latitude"))
            lon = _parse_latlon(storm.get("longitude"))
            if lat is None or lon is None:
                continue
            parsed.append(
                {
                    "name": storm.get("name", "UNKNOWN"),
                    "classification": storm.get("classification", "TS"),
                    "lat": lat,
                    "lon": lon,
                }
            )
        return parsed

    def get_port_impacts(self, storms: Optional[List[Dict[str, object]]] = None) -> pd.DataFrame:
        """
        Compute per-port weather impacts.

        Returns a DataFrame with columns:
        ``port, storm_name, classification, distance_km, capacity_multiplier``.
        Empty DataFrame means no active weather disruption detected
        (either no storms, or none within ``impact_radius_km`` of a port).
        """
        if storms is None:
            storms = self.fetch_active_storms()

        impacts: List[PortWeatherImpact] = []
        for port, coord in self.port_coordinates.items():
            best: Optional[PortWeatherImpact] = None
            for storm in storms:
                dist = _haversine_km(coord["lat"], coord["lon"], float(storm["lat"]), float(storm["lon"]))
                if dist > self.impact_radius_km:
                    continue
                classification = str(storm.get("classification", "TS")).upper()
                multiplier = CLASSIFICATION_IMPACT.get(classification, 0.80)
                candidate = PortWeatherImpact(
                    port=port,
                    storm_name=str(storm.get("name", "UNKNOWN")),
                    classification=classification,
                    distance_km=round(dist, 1),
                    capacity_multiplier=multiplier,
                )
                if best is None or candidate.capacity_multiplier < best.capacity_multiplier:
                    best = candidate
            if best is not None:
                impacts.append(best)

        if not impacts:
            return pd.DataFrame(
                columns=["port", "storm_name", "classification", "distance_km", "capacity_multiplier"]
            )
        return pd.DataFrame([impact.__dict__ for impact in impacts])

    def get_port_capacity_multipliers(self, storms: Optional[List[Dict[str, object]]] = None) -> Dict[str, float]:
        """
        Convenience accessor returning ``{port: capacity_multiplier}`` for
        every configured port, defaulting to ``1.0`` (no impact) when a
        port has no nearby storm.
        """
        df = self.get_port_impacts(storms)
        multipliers = {port: 1.0 for port in self.port_coordinates}
        for row in df.itertuples(index=False):
            multipliers[row.port] = row.capacity_multiplier
        return multipliers
