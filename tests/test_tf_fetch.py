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
             "WeatherElements": {"TideHeight": "1.234"}},
            {"DateTime": "2023-07-25T08:30:00+08:00",
             "WeatherElements": {"TideHeight": "None"}},   # missing -> skipped
            {"DateTime": "2023-07-25T09:00:00+08:00",
             "WeatherElements": {"TideHeight": "1.501"}},
        ]}}]}},
}


def test_normalize_noaa_metres_to_cm_and_utc():
    sid, rec = FT.normalize_noaa(NOAA_FIXTURE)
    assert sid == "8454000"
    assert rec["lon"] == -71.4012 and rec["lat"] == 41.8071
    assert rec["times_utc"] == ["2023-07-25T00:00:00", "2023-07-25T00:12:00"]
    assert rec["heights_cm"] == [51.2, 49.8]            # m -> cm, gap dropped


def test_normalize_cwa_local_to_utc_metres_to_cm():
    sid, rec = FT.normalize_cwa(CWA_FIXTURE)   # default unit = metres
    assert sid == "C6V100"
    # +08:00 -> UTC shifts back 8 h
    assert rec["times_utc"] == ["2023-07-25T00:00:00", "2023-07-25T01:00:00"]
    assert rec["heights_cm"] == [123.4, 150.1]          # m -> cm; None dropped


def test_normalize_cwa_height_unit_cm_keeps_raw():
    sid, rec = FT.normalize_cwa(CWA_FIXTURE, height_unit="cm")
    assert rec["heights_cm"] == [1.234, 1.501]          # cm override = raw


def test_normalize_cwa_uses_station_meta_for_coords():
    # obs response without coords -> lon/lat come from station_meta
    j = {"Records": {"SeaSurfaceObs": {"Location": [{
        "Station": {"StationID": "C4A01"},
        "StationObsTimes": {"StationObsTime": [
            {"DateTime": "2026-06-16T08:00:00+08:00",
             "WeatherElements": {"TideHeight": "1.4"}}]}}]}}}
    sid, rec = FT.normalize_cwa(j, station_meta={"C4A01": {"lon": 121.42, "lat": 25.18}})
    assert rec["lon"] == 121.42 and rec["lat"] == 25.18 and rec["heights_cm"] == [140.0]
    # missing coords AND no meta -> None
    assert FT.normalize_cwa(j) is None


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


def test_cwa_station_cap_refuses_bulk(monkeypatch, tmp_path):
    """Conservative guard: CWA with >5 stations is refused (exit 2) unless
    --allow-bulk, before any token load / fetch."""
    import sys
    monkeypatch.setattr(sys, "argv", [
        "fetch_tf_observations.py", "--source", "cwa",
        "--stations", "a,b,c,d,e,f", "--out", str(tmp_path / "o.json")])
    assert FT.main() == 2
