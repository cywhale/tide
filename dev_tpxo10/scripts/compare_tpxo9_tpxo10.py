#!/usr/bin/env python
"""T-C: TPXO9-zarr <-> TPXO10-zarr cross-version consistency (spec §7.3).

Calibration mode (Stage 1): computes the |Δhc| distributions that freeze
the §7.3 thresholds at G1. Values are EXPECTED to differ (different model
generations); the gate is that differences are bounded, spatially
explicable and unbiased.

Method (spec §7.3):
* z at z-nodes — both stores share the identical 1/30° z-grid (asserted).
  Old store hc = z_amp · exp(−i·z_ph·π/180) [m] (the pyTMD 2.x phase
  convention makes this equal to the source Re+i·Im); new store
  hc = 1e-3·(z_Re + i·z_Im) [m]. Metric: |Δhc| in mm per constituent.
* Strata: 50% uniform / 25% shelf (50 ≤ h < 200 m) / 25% coastal
  (h < 50 m), all within the Stage 1 region. The spec's polar stratum
  (|lat| > 60°) has no cells in this region — DEFERRED to the Stage 2
  global run (reported as such).
* Currents (report-only, round 7 split): runtime-facing uz/vz vs the old
  store's z-grid u/v with an empirical unit-scale detection (the legacy
  store's current units are determined from data, then documented).

Output: per-constituent, per-depth-stratum percentiles + signed deep-M2
amplitude bias + top-20 outliers; JSON at benchmarks/tc_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

BINDING = ["m2", "s2", "k1", "o1"]
SEED = 20260611
N_SAMPLE = 120_000


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--old", type=Path, default=repo_root / "data" / "tpxo9.zarr")
    ap.add_argument("--out", type=Path,
                    default=repo_root / "dev_tpxo10" / "benchmarks" / "tc_calibration.json")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    new = xr.open_zarr(args.new, consolidated=True, decode_times=False)
    old = xr.open_zarr(args.old, consolidated=True, decode_times=False)
    j0, j1, i0, i1 = new.attrs["interior_index_window"]

    # grid identity assertion (rtol=0)
    assert np.allclose(old["lat"].values[j0:j1], new["lat_z"].values,
                       rtol=0.0, atol=1e-9), "lat axes differ"
    assert np.allclose(old["lon"].values[i0:i1], new["lon_z"].values,
                       rtol=0.0, atol=1e-9), "lon axes differ"
    cons_new = [str(c) for c in new["constituents"].values]
    cons_old = [str(c) for c in old["constituents"].values]
    assert set(cons_new) == set(cons_old), "constituent sets differ"

    hz = new["hz"].values
    z_flag = new["z_flag"].values
    old_amp0 = old["z_amp"].isel(lat=slice(j0, j1), lon=slice(i0, i1),
                                 constituents=cons_old.index("m2")).values
    common = (z_flag == 0) & np.isfinite(old_amp0)
    jj_all, ii_all = np.nonzero(common)
    h_at = hz[common]
    is_global = (j1 - j0) == P.NY and (i1 - i0) == P.NX
    lat_at = new["lat_z"].values[jj_all]

    idx_all = np.arange(len(jj_all))
    if is_global:
        # §7.3 global strata: 50% uniform / 25% shelf-coastal / 25% polar
        # (polar is BINDING at G2: its samples flow into the deep/shelf
        # gates and are additionally reported as their own stratum)
        alloc = (("uniform", idx_all, N_SAMPLE // 2),
                 ("shallow", idx_all[h_at < 200], N_SAMPLE // 4),
                 ("polar", idx_all[np.abs(lat_at) > 60], N_SAMPLE // 4))
    else:
        alloc = (("uniform", idx_all, N_SAMPLE // 2),
                 ("shelf", idx_all[(h_at >= 50) & (h_at < 200)], N_SAMPLE // 4),
                 ("coastal", idx_all[h_at < 50], N_SAMPLE // 4))
    picks = [rng.choice(pool, size=min(n, len(pool)), replace=False)
             for _, pool, n in alloc]
    sel = np.unique(np.concatenate(picks))
    jj, ii = jj_all[sel], ii_all[sel]
    h_sel = hz[jj, ii]
    strata = {
        "deep": h_sel >= 1000,
        "shelf": (h_sel >= 50) & (h_sel < 1000),
        "coastal": h_sel < 50,
    }
    if is_global:
        strata["polar"] = np.abs(new["lat_z"].values[jj]) > 60
    print(f"[1/4] sampled {len(jj)} common-valid z-cells: " + ", ".join(
        f"{s} {int(m.sum())}" for s, m in strata.items())
        + ("" if is_global else "; polar stratum DEFERRED to Stage 2"))

    results = {"_meta": {"seed": SEED, "n": int(len(jj)),
                         "polar_stratum": "deferred to Stage 2 global store",
                         "units": "mm |Δhc| (complex vector difference)"}}
    lon_z, lat_z = new["lon_z"].values, new["lat_z"].values
    pct = [50, 75, 90, 95, 99]
    for c in cons_new:
        ko, kn = cons_old.index(c), cons_new.index(c)
        amp = old["z_amp"].isel(lat=slice(j0, j1), lon=slice(i0, i1),
                                constituents=ko).values[jj, ii]
        ph = old["z_ph"].isel(lat=slice(j0, j1), lon=slice(i0, i1),
                              constituents=ko).values[jj, ii]
        hc_old = amp * np.exp(-1j * np.deg2rad(ph))          # meters
        hc_new = 1e-3 * (new["z_Re"].isel(constituents=kn).values[jj, ii]
                         + 1j * new["z_Im"].isel(constituents=kn).values[jj, ii])
        d_mm = 1e3 * np.abs(hc_new - hc_old)
        entry = {}
        for s, mask in strata.items():
            entry[s] = {f"P{p}": round(float(np.percentile(d_mm[mask], p)), 2)
                        for p in pct}
        if c == "m2":
            deep = strata["deep"]
            bias = 1e3 * np.mean(np.abs(hc_new[deep]) - np.abs(hc_old[deep]))
            entry["deep_amp_bias_mm"] = round(float(bias), 3)
            order = np.argsort(d_mm)[::-1][:20]
            entry["outliers_top20"] = [
                {"d_mm": round(float(d_mm[o]), 1),
                 "lon": round(float(lon_z[ii[o]]), 3),
                 "lat": round(float(lat_z[jj[o]]), 3),
                 "h_m": round(float(h_sel[o]), 1)} for o in order]
        results[c] = entry
        tag = "BINDING" if c in BINDING else "report"
        print(f"  z {c:>4} [{tag:7s}] deep P95={entry['deep']['P95']:8.2f}mm "
              f"shelf P95={entry['shelf']['P95']:8.2f}mm "
              f"coastal P95={entry['coastal']['P95']:9.2f}mm")
    print(f"[2/4] deep M2 signed amp bias: {results['m2']['deep_amp_bias_mm']} mm")

    # currents: report-only, empirical unit detection on M2
    ko, kn = cons_old.index("m2"), cons_new.index("m2")
    uo_amp = old["u_amp"].isel(lat=slice(j0, j1), lon=slice(i0, i1),
                               constituents=ko).values[jj, ii]
    uo_ph = old["u_ph"].isel(lat=slice(j0, j1), lon=slice(i0, i1),
                             constituents=ko).values[jj, ii]
    un = (new["uz_Re"].isel(constituents=kn).values[jj, ii]
          + 1j * new["uz_Im"].isel(constituents=kn).values[jj, ii]).astype(np.complex128)
    ok = np.isfinite(uo_amp) & (np.abs(un) > 1e-6) & (uo_amp > 0)
    ratio = np.median(uo_amp[ok] / np.abs(un[ok]))
    scale = 10 ** round(np.log10(ratio))  # detect decade (1=m/s, 100=cm/s)
    uc_old = (uo_amp * np.exp(-1j * np.deg2rad(uo_ph)))[ok] / scale
    d_u = np.abs(un[ok] - uc_old)
    results["_currents"] = {
        "detected_old_unit_scale_vs_m_s": float(scale),
        "median_amp_ratio": round(float(ratio), 4),
        "u_m2_abs_diff_m_s": {f"P{p}": round(float(np.percentile(d_u, p)), 4)
                              for p in pct},
        "note": ("report-only (round 7): differences mix genuine TPXO10 "
                 "changes with the legacy-spline vs D12-centering regrid "
                 "method change; binding current gates are station-level T-F"),
    }
    print(f"[3/4] currents (report-only): old-unit scale={scale}x m/s, "
          f"u M2 |Δ| P95={results['_currents']['u_m2_abs_diff_m_s']['P95']} m/s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    # frozen-threshold enforcement (round 12, finding 5; thresholds frozen
    # in the spec Stage 1 decision memo #7)
    failures = []
    for c in BINDING:
        if results[c]["deep"]["P95"] > 30.0:
            failures.append(f"{c} deep P95 {results[c]['deep']['P95']}mm > 30mm")
        if results[c]["shelf"]["P95"] > 150.0:
            failures.append(f"{c} shelf P95 {results[c]['shelf']['P95']}mm > 150mm")
    if abs(results["m2"]["deep_amp_bias_mm"]) > 5.0:
        failures.append(f"deep M2 bias {results['m2']['deep_amp_bias_mm']}mm > 5mm")
    if failures:
        print("FAIL compare_tpxo9_tpxo10 (frozen thresholds):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[4/4] PASS compare_tpxo9_tpxo10 (frozen-threshold gate) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
