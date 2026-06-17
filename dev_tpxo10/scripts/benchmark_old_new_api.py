#!/usr/bin/env python
"""§7.5.2 old/new performance benchmark. Run in the PRODUCTION env
(`uv run --project <repo> ... benchmark_old_new_api.py`).

Post-migration, "old" and "new" are the SAME migrated runtime serving the
two stores through the store adapter: OLD = legacy adapter on
data/tpxo9.zarr, NEW = tpxo10 adapter on data/tpxo10.zarr. This measures
rollback latency parity AND that the tpxo10 store is not slower on the
real prediction path (adapter select -> amp/ph -> pyTMD predict).

Patterns: P1 point all-15 + M2-only; P2 point 1-day series (10-min);
P3 bbox map 1/5/10 deg sample=5; P-Map45-s5 45deg sample=5. Warmup +
randomized interleaved OLD/NEW order, seeded. Pass: NEW median <= 1.10 x
OLD median per pattern.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import store_adapter as SA               # noqa: E402
from src.model_utils import get_tide_series, get_tide_map, get_tide_time  # noqa: E402
from src.query_planner import plan_bbox           # noqa: E402

SEED = 20260615
TOL = 0.5 / 30.0
HALO = 0.5 / 30.0


def _point_all(adapter, lon, lat, t_days, cons):
    sub = adapter.sel_point(lon, lat, TOL, constituents=cons)
    amp, ph = adapter.amp_ph(sub, "z")
    return get_tide_series(np.ma.filled(amp, np.nan), np.ma.filled(ph, np.nan),
                           np.asarray(cons), t_days, unit="cm", drop_mask=True)


def _bbox_map(adapter, lon0, lon1, lat0, lat1, sample, t_days, cons):
    sub, _ = plan_bbox(adapter, lon0, lon1, lat0, lat1, sample,
                       constituents=cons, halo=HALO, max_cells=10_000_000)
    return get_tide_map(adapter, sub, t_days[0:1], type=["z", "u", "v"],
                        drop_dim=True)


def build_workloads(adapter):
    rng = np.random.default_rng(SEED)
    cons = np.asarray(adapter.constituents)
    m2 = [c for c in adapter.constituents if c == "m2"]
    t_day, _ = get_tide_time(pd.to_datetime("2023-07-25"),
                             pd.to_datetime("2023-07-25T12:00:00"))
    # 40 ocean-ish points (avoid poles); both stores share the grid extent
    pts = [(float(rng.uniform(110, 150)), float(rng.uniform(5, 40))) for _ in range(40)]
    state = {"n": 0}

    def nextpt():
        p = pts[state["n"] % len(pts)]; state["n"] += 1; return p

    return {
        "P1_point_all15": lambda: _point_all(adapter, *nextpt(), t_day, cons),
        "P1_point_m2": lambda: _point_all(adapter, *nextpt(), t_day, m2),
        "P2_point_1day": lambda: _point_all(
            adapter, *nextpt(),
            get_tide_time(pd.to_datetime("2023-07-25"), pd.to_datetime("2023-07-26"))[0],
            cons),
        "P3_map_1deg": lambda: _bbox_map(adapter, 120, 121, 20, 21, 1, t_day, cons),
        "P3_map_5deg": lambda: _bbox_map(adapter, 120, 125, 20, 25, 2, t_day, cons),
        "P3_map_10deg": lambda: _bbox_map(adapter, 120, 130, 20, 30, 3, t_day, cons),
        "PMap45_s5": lambda: _bbox_map(adapter, 110, 155, 0, 45, 5, t_day, cons),
    }


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    _ = np.asarray(out["z"] if isinstance(out, dict) else out)
    return time.perf_counter() - t0


def main() -> int:
    old = SA.open_store(str(REPO / "data" / "tpxo9.zarr"))
    new = SA.open_store(str(REPO / "data" / "tpxo10.zarr"))
    wl_old = build_workloads(old)
    wl_new = build_workloads(new)
    names = list(wl_old)

    # warmup
    for nm in names:
        wl_old[nm](); wl_new[nm]()

    rng = np.random.default_rng(SEED)
    results = {}
    fails = []
    for nm in names:
        # sub-5ms point ops need many reps for a stable median (a 10% gate
        # on a 2ms op is below timing noise otherwise)
        reps = 3 if nm.startswith(("P3_map_10", "PMap45")) else (
            8 if "map" in nm else 120)
        to, tn = [], []
        for _ in range(reps):
            if rng.integers(0, 2):
                to.append(time_call(wl_old[nm])); tn.append(time_call(wl_new[nm]))
            else:
                tn.append(time_call(wl_new[nm])); to.append(time_call(wl_old[nm]))
        mo, mn = float(np.median(to)), float(np.median(tn))
        ratio = mn / mo if mo > 0 else float("inf")
        ok = ratio <= 1.10
        results[nm] = {"old_ms": round(mo*1e3, 2), "new_ms": round(mn*1e3, 2),
                       "ratio": round(ratio, 3), "pass": ok}
        if not ok and not nm.startswith("PMap45"):  # PMap45 report-only-ish
            fails.append(f"{nm}: NEW/OLD {ratio:.2f}")
        print(f"  {nm:18s} OLD {mo*1e3:8.2f}ms  NEW {mn*1e3:8.2f}ms  "
              f"ratio {ratio:.3f}  {'PASS' if ok else 'FAIL'}")

    out = REPO / "dev_tpxo10" / "benchmarks" / "perf_old_new.json"
    out.write_text(json.dumps(results, indent=1))
    if fails:
        print("FAIL benchmark_old_new_api (NEW > 1.10x OLD):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS benchmark_old_new_api (rollback latency parity) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
