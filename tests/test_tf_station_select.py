"""Reproducible NOAA station-selection candidate filter (offline; the
model-coverage step needs the store and is exercised by the script run).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev_tpxo10" / "scripts"))
import select_tf_noaa_stations as SEL  # noqa: E402


def test_water_level_candidates_filters_and_skips():
    sj = {"portsStationList": [
        {"stationID": "A1", "waterlevel": True, "lat": 1.0, "lng": 2.0},
        {"stationID": "A2", "waterlevel": False, "lat": 1.0, "lng": 2.0},   # no WL
        {"stationID": SEL.SKIP_STATIONS[0], "waterlevel": True, "lat": 1, "lng": 2},  # skip
        {"stationID": "A3", "waterlevel": True, "lat": 3.0, "lng": 4.0},
    ]}
    all_st, cands = SEL.water_level_candidates(sj)
    assert len(all_st) == 4
    ids = [s["stationID"] for s in cands]
    assert ids == ["A1", "A3"]                     # WL-only, skip excluded


def test_skip_list_is_the_known_legacy_set():
    assert "8723214" in SEL.SKIP_STATIONS and len(SEL.SKIP_STATIONS) == 12
