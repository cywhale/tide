#!/usr/bin/env python
"""Reproducible NOAA T-F station selection + provenance manifest.

Makes the station panel used by the T-F NOAA primary gate reproducible
(not ad hoc): filters `test/stations_noaa.json` to water-level stations
(less the known-bad skip list), keeps only those the tpxo10 store
resolves to a flag-0 valid nearest cell (estuary/inner-bay gauges the
global 1/30deg model cannot resolve are dropped BEFORE any API call),
and emits a manifest with the full selection chain + per-stage exclusion
reasons. Run in the PRODUCTION env.

Re-running reproduces the candidate/skip/model-resolved sets exactly. The
fetched/returned/comparable counts are read back from the gate evidence
files so the manifest captures the WHOLE chain in one tracked artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# known-bad NOAA stations (weird waveforms / empty series), from the
# legacy test/compare_observ_tide01.py skip list.
SKIP_STATIONS = ["8764314", "8726674", "8735180", "8770822", "8775132",
                 "8770613", "8770520", "8770475", "8773259", "8723214",
                 "8775241", "8773767"]


def water_level_candidates(stations_json: dict):
    lst = stations_json["portsStationList"]
    cands = [s for s in lst if s.get("waterlevel") and s["stationID"] not in SKIP_STATIONS]
    return lst, cands


def model_resolved(candidates, adapter, tol=0.5 / 30.0):
    """Keep candidates whose nearest tpxo10 cell is flag-0 valid."""
    import numpy as np
    resolved, invalid = [], []
    for s in candidates:
        lon = s["lng"] + 360.0 if s["lng"] < 0 else s["lng"]
        try:
            sub = adapter.sel_point(lon, s["lat"], tol=tol)
            if int(np.asarray(sub["z_flag"].values)) == 0:
                resolved.append(s["stationID"])
            else:
                invalid.append(s["stationID"])
        except Exception:
            invalid.append(s["stationID"])
    return resolved, invalid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stations", type=Path, default=REPO / "test" / "stations_noaa.json")
    ap.add_argument("--store", type=Path, default=REPO / "data" / "tpxo10.zarr")
    ap.add_argument("--fetched-obs", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "tf_observations_noaa_20260617.json",
                    help="sanitized obs file produced by the fetch step (for returned ids)")
    ap.add_argument("--gate-result", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "tf_observation_noaa_20260617.json",
                    help="gate result (for comparable ids)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "tf_noaa_station_selection_20260617.json")
    args = ap.parse_args()

    from src import store_adapter as SA
    sjson = json.loads(args.stations.read_text())
    all_st, cands = water_level_candidates(sjson)
    adapter = SA.open_store(str(args.store))
    resolved, invalid = model_resolved(cands, adapter)

    returned = sorted(json.loads(args.fetched_obs.read_text()).keys()) \
        if args.fetched_obs.exists() else []
    comparable = []
    if args.gate_result.exists():
        gr = json.loads(args.gate_result.read_text())["stations"]
        comparable = sorted(k for k, v in gr.items()
                            if v["old"]["rmse_cm"] is not None
                            and v["new"]["rmse_cm"] is not None)

    resolved_set = set(resolved)
    returned_set = set(returned)
    comparable_set = set(comparable)
    excluded = {
        "model_invalid_cell": sorted(invalid),
        "resolved_but_not_returned (no NOAA data for the window / dropped)":
            sorted(resolved_set - returned_set),
        "returned_but_not_comparable (tpxo9 baseline invalid at cell)":
            sorted(returned_set - comparable_set),
    }
    manifest = {
        "source_file": str(args.stations.relative_to(REPO)),
        "store": str(args.store.relative_to(REPO)),
        "selection": {
            "total_stations": len(all_st),
            "water_level_candidates": len(cands),
            "known_skip_list": SKIP_STATIONS,
            "model_resolved_flag0": len(resolved),
            "returned_with_data": len(returned),
            "comparable_both_stores": len(comparable),
        },
        "model_resolved_ids": sorted(resolved),
        "returned_ids": returned,
        "comparable_ids": comparable,
        "excluded": excluded,
        "method": ("offline tpxo10 flag-0 model-coverage filter applied to "
                   "water-level candidates BEFORE any API call; deterministic "
                   "and reproducible by re-running this script."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    sel = manifest["selection"]
    print(f"NOAA station selection: total {sel['total_stations']} -> "
          f"water-level {sel['water_level_candidates']} -> model-resolved "
          f"{sel['model_resolved_flag0']} -> returned {sel['returned_with_data']} "
          f"-> comparable {sel['comparable_both_stores']}")
    print(f"PASS select_tf_noaa_stations -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
