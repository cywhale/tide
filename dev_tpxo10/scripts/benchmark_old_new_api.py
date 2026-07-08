#!/usr/bin/env python
"""§7.5.2 old/new performance benchmark. Run in the PRODUCTION env
(`uv run --project <repo> ... benchmark_old_new_api.py`).

Post-migration, "old" and "new" are the SAME migrated runtime serving the
two stores through the store adapter: OLD = legacy adapter on
data/tpxo9.zarr, NEW = tpxo10 adapter on data/tpxo10.zarr. This measures
rollback latency parity AND that the tpxo10 store is not slower on the
real prediction path (adapter select -> amp/ph -> pyTMD predict).

Patterns: P1 point all-15 + M2-only; P2 point 1-day series (10-min);
P3 bbox map 1/5/10 deg sample=5; P-Map45-s5 45deg sample=5; P-Strip
45x5 deg elongated sample=5. Each reports mean/median/P95/P99/max wall.
Warmup + randomized interleaved OLD/NEW order, seeded.

Rep design (justified deviation from the §7.5.2 ">=400 queries" note,
which was written for sub-ms warm point reads): sub-5ms point patterns
use 400 reps (stable percentiles); map patterns are wall-time bounded
(1/5deg 50, 10deg/strip 12, 45deg 8) — 400 reps of the 0.9s 45deg map
would be ~12 min/store with no extra signal. Recorded in spec §7.5.2.

Blocking gate (NEW median <= 1.10x OLD median): P1, P2, P3, P-Map45-s5,
P-Strip. P-Map45-s1 (unsampled 45deg) is NOT a latency pattern here — at
the production cap it is REJECTED before materialization; this script
re-confirms the cap rejection, and the peak-RSS/payload measurement for
it is the §7.5.3 W8 harness evidence at G1 (not re-run here). P4
cold-cache is deferred to a Linux host (macOS cannot evict the page
cache; §7.5.1).
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
from src.query_planner import plan_bbox, get_max_bbox_cells, BboxCapError  # noqa: E402

SEED = 20260615
TOL = 0.5 / 30.0
HALO = 0.5 / 30.0
GATE = 1.10
# blocking latency patterns (NEW median must be <= GATE x OLD median)
BLOCKING = {"P1_point_all15", "P1_point_m2", "P2_point_1day",
            "P3_map_1deg", "P3_map_5deg", "P3_map_10deg",
            "PMap45_s5", "PStrip_45x5"}


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
        "PStrip_45x5": lambda: _bbox_map(adapter, 110, 155, 20, 25, 5, t_day, cons),
    }


def _stats(times):
    a = np.asarray(times) * 1e3   # ms
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2), "n": int(a.size)}


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

    def reps_for(nm):
        if nm in ("PMap45_s5", "PStrip_45x5"):
            return 8
        if nm == "P3_map_10deg":
            return 12
        if "map" in nm:
            return 50
        return 400          # sub-5ms point patterns -> stable percentiles

    rng = np.random.default_rng(SEED)
    results = {}
    fails = []
    for nm in names:
        reps = reps_for(nm)
        to, tn = [], []
        for _ in range(reps):
            if rng.integers(0, 2):
                to.append(time_call(wl_old[nm])); tn.append(time_call(wl_new[nm]))
            else:
                tn.append(time_call(wl_new[nm])); to.append(time_call(wl_old[nm]))
        so, sn = _stats(to), _stats(tn)
        ratio = sn["median"] / so["median"] if so["median"] > 0 else float("inf")
        ok = ratio <= GATE
        results[nm] = {"old": so, "new": sn, "ratio_median": round(ratio, 3),
                       "blocking": nm in BLOCKING, "pass": ok}
        if nm in BLOCKING and not ok:
            fails.append(f"{nm}: NEW/OLD median {ratio:.2f}")
        print(f"  {nm:16s} OLD med {so['median']:8.2f}ms p95 {so['p95']:8.2f} | "
              f"NEW med {sn['median']:8.2f}ms p95 {sn['p95']:8.2f} | "
              f"ratio {ratio:.3f}  {'PASS' if ok else 'FAIL'}"
              + ("" if nm in BLOCKING else "  (report-only)"))

    # P-Map45-s1 (unsampled 45deg): at the production cap it is REJECTED
    # before materialization; re-confirm here (peak-RSS evidence is the
    # §7.5.3 W8 harness at G1, not re-run).
    cap = get_max_bbox_cells()
    rejected = False
    requested_cells = None
    try:
        plan_bbox(new, 110, 155, 0, 45, 1, constituents=np.asarray(new.constituents),
                  halo=HALO, max_cells=cap)
    except BboxCapError as e:
        rejected = True
        requested_cells = int(e.requested)
        print(f"  PMap45_s1        cap rejects ({requested_cells} > {cap} cells) "
              "before materialization — PASS (peak-RSS = W8/G1 evidence)")
    results["PMap45_s1_cap"] = {"rejected_by_cap": rejected, "cap": cap,
                                "requested_cells": requested_cells}
    if not rejected:
        fails.append("PMap45_s1: cap did NOT reject the unsampled 45deg map")

    out = REPO / "dev_tpxo10" / "benchmarks" / "perf_old_new.json"
    out.write_text(json.dumps(results, indent=1))
    if fails:
        print("FAIL benchmark_old_new_api:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS benchmark_old_new_api (blocking latency + cap revalidation) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
