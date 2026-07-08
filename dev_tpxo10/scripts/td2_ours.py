#!/usr/bin/env python
"""T-D2 (spec §7.4) — OURS producer. Run in the PRODUCTION env
(`uv run --project <repo> python dev_tpxo10/scripts/td2_ours.py`).

Computes the hourly z elevation series at fixed stations through the
EXACT runtime path (store adapter -> amp/ph -> pyTMD 2.2.8
predict.time_series + infer_minor) over two fixed windows, and saves
ours.json for td2_golden.py to compare against a pyTMD 3.0.6 reference
built from the TPXO10 source. This isolates the 2.2.8<->3.x prediction
engine equivalence (the hc itself is identical, proven by T-B/T-D1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import store_adapter as SA               # noqa: E402
from src.model_utils import get_tide_series, get_tide_time  # noqa: E402

# named stations (subset of the spec §7.4 set spanning Taiwan Strait,
# Kuroshio, open Pacific/Atlantic, shelf, and a polar cell)
STATIONS = {
    "taiwan_strait": (119.5, 24.5),
    "kuroshio_e_taiwan": (122.5, 23.5),
    "luzon_strait": (121.0, 20.5),
    "open_pacific": (160.0, 20.0),
    "n_atlantic": (320.0, 40.0),
    "gulf_of_maine": (290.5, 43.0),
    "yellow_sea": (123.0, 35.0),
    "arabian_sea": (65.0, 15.0),
    "south_atlantic": (340.0, -30.0),
    "weddell_edge": (315.0, -68.0),
}
WINDOWS = [("2023-07-25", "2023-07-27"),   # ~48 h, near solstice
           ("2023-03-20", "2023-03-22")]   # ~48 h, near equinox


def main() -> int:
    store = SA.get_zarr_path()
    adapter = SA.open_store(store)
    out = {"store": store, "schema": adapter.schema,
           "cons": list(adapter.constituents), "stations": {}}
    for name, (lon, lat) in STATIONS.items():
        series = {}
        for w0, w1 in WINDOWS:
            t_days, dt = get_tide_time(pd.to_datetime(w0), pd.to_datetime(w1))
            sub = adapter.sel_point(lon, lat, tol=0.5 / 30.0)
            amp, ph = adapter.amp_ph(sub, "z")
            z = get_tide_series(np.ma.filled(amp, np.nan),
                                np.ma.filled(ph, np.nan),
                                np.asarray(adapter.constituents), t_days,
                                format="netcdf", unit="cm", drop_mask=True)
            z = np.asarray(z).ravel()
            series[w0] = {"t_days": [float(x) for x in np.asarray(t_days).ravel()],
                          "z_cm": [None if not np.isfinite(v) else float(v) for v in z]}
        out["stations"][name] = {"lon": lon, "lat": lat, "windows": series}
    dst = REPO / "dev_tpxo10" / "benchmarks" / "td2_ours.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    n = sum(len(s["windows"]) for s in out["stations"].values())
    print(f"PASS td2_ours: {len(STATIONS)} stations x {len(WINDOWS)} windows "
          f"({n} series) from {adapter.schema} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
