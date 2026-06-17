"""T-F observation fetcher: normalization (NOAA/CWA -> sanitized JSON) and
token hygiene. Network-free — fixtures only.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev_tpxo10" / "scripts"))
import fetch_tf_observations as FT  # noqa: E402


NOAA_FIXTURE = {
    "metadata": {"id": "8454000", "name": "Providence", "lon": "-71.4012", "lat": "41.8071"},
    "data": [
        {"t": "2023-07-25 00:00", "v": "0.512"},
        {"t": "2023-07-25 00:06", "v": ""},          # gap -> skipped
        {"t": "2023-07-25 00:12", "v": "0.498"},
    ],
}

CWA_FIXTURE = {
    "Records": {"SeaSurfaceObs": {"Location": [{
        "Station": {"StationID": "C6V100", "StationLongitude": "120.51",
                    "StationLatitude": "24.29"},
        "StationObsTimes": {"StationObsTime": [
            {"DateTime": "2023-07-25T08:00:00+08:00",
             "WeatherElements": {"TideHeight": "123.4"}},
            {"DateTime": "2023-07-25T08:30:00+08:00",
             "WeatherElements": {"TideHeight": "None"}},   # missing -> skipped
            {"DateTime": "2023-07-25T09:00:00+08:00",
             "WeatherElements": {"TideHeight": "150.1"}},
        ]}}]}},
}


def test_normalize_noaa_metres_to_cm_and_utc():
    sid, rec = FT.normalize_noaa(NOAA_FIXTURE)
    assert sid == "8454000"
    assert rec["lon"] == -71.4012 and rec["lat"] == 41.8071
    assert rec["times_utc"] == ["2023-07-25T00:00:00", "2023-07-25T00:12:00"]
    assert rec["heights_cm"] == [51.2, 49.8]            # m -> cm, gap dropped


def test_normalize_cwa_local_to_utc_and_cm():
    sid, rec = FT.normalize_cwa(CWA_FIXTURE)
    assert sid == "C6V100"
    # +08:00 -> UTC shifts back 8 h
    assert rec["times_utc"] == ["2023-07-25T00:00:00", "2023-07-25T01:00:00"]
    assert rec["heights_cm"] == [123.4, 150.1]          # cm as-is; None dropped


def test_normalize_cwa_height_unit_m():
    sid, rec = FT.normalize_cwa(CWA_FIXTURE, height_unit="m")
    assert rec["heights_cm"] == [12340.0, 15010.0]      # m -> cm scaling


def test_normalize_handles_empty():
    assert FT.normalize_noaa({"metadata": {"id": "x", "lon": "0", "lat": "0"},
                              "data": []}) is None
    assert FT.normalize_cwa({"Records": {"SeaSurfaceObs": {"Location": []}}}) is None


def test_token_hygiene_raises_if_leaked():
    payload = {"s": {"note": "secret-token-123 inside"}}
    with pytest.raises(RuntimeError, match="token leaked"):
        FT.assert_token_absent(payload, "secret-token-123")
    # absent token -> no error
    FT.assert_token_absent({"s": {"lon": 1.0}}, "secret-token-123")
    FT.assert_token_absent({"s": {"lon": 1.0}}, None)


def test_sanitized_output_has_no_metadata_or_token():
    """The normalized record carries ONLY lon/lat/times/heights — no station
    name, no token, no raw API metadata."""
    _, rec = FT.normalize_noaa(NOAA_FIXTURE)
    assert set(rec) == {"lon", "lat", "times_utc", "heights_cm"}
    blob = json.dumps(rec)
    assert "Providence" not in blob and "metadata" not in blob
