#!/usr/bin/env python
"""G2 coverage gate (spec §4 Gate G2):
  1. Every TPXO9-store valid ocean point maps to a valid (flag in {0,1})
     TPXO10 node of the same node type.
  2. Zero NaN anywhere in Re/Im/flag/uz/vz arrays.
  3. Inpainted fraction <= 2% of ocean nodes per variable (calibrated
     bound from Stage 1).
Streams latitude bands to bound memory. Violations of (1) are counted
and located (the legacy store contains ~205k extrapolation-filled cells
whose fill band may exceed DMAX in the new contract — any such cells are
reported with coordinates for the G2 review).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

BAND = 452  # rows per streamed band


def recompute_flag_class(src: Path, node: str, gj: int, gi: int, R: int) -> int:
    """Independently recompute the frozen §3.2+D3 flag class for one
    global (gj, gi) node from the SOURCE, over a local (2R+1)^2 window.
    Returns FLAG_SOURCE / FLAG_FILLED / FLAG_INVALID. Used to self-verify
    the isolated-no-data coverage class (round 16, finding 3)."""
    import netCDF4
    prefix, rvar, ivar, hvar = {
        "z": ("h", "hRe", "hIm", "hz"),
        "u": ("u", "uRe", "uIm", "hu"),
        "v": ("u", "vRe", "vIm", "hv"),
    }[node]
    NY, NX = P.NY, P.NX
    j0, j1 = max(gj - R, 0), min(gj + R + 1, NY)
    i0, i1 = max(gi - R, 0), min(gi + R + 1, NX)
    grid = src / "grid_tpxo10atlas_v2.nc"
    with netCDF4.Dataset(grid) as g:
        h = np.asarray(g[hvar][i0:i1, j0:j1]).T.astype(np.float64)
    ncons = len(P.CONSTITUENTS)
    re = np.empty(h.shape + (ncons,), dtype=np.int32)
    im = np.empty_like(re)
    for k, c in enumerate(P.CONSTITUENTS):
        with netCDF4.Dataset(src / f"{prefix}_{c}_tpxo10_atlas_30_v2.nc") as ds:
            re[..., k] = np.asarray(ds[rvar][i0:i1, j0:j1]).T
            im[..., k] = np.asarray(ds[ivar][i0:i1, j0:j1]).T
    valid = P.compute_validity(h, re, im)
    flag = P.classify_flags(h, valid, dmax=P.DMAX_FILL_CELLS)
    return int(flag[gj - j0, gi - i0])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path,
                    default=repo_root / "data" / "tpxo10.zarr")  # promoted canonical (round 17)
    ap.add_argument("--old", type=Path, default=repo_root / "data" / "tpxo9.zarr")
    args = ap.parse_args()

    new = xr.open_zarr(args.store, consolidated=True, decode_times=False, mask_and_scale=False)
    old = xr.open_zarr(args.old, consolidated=True, decode_times=False, mask_and_scale=False)
    NY, NX = P.NY, P.NX
    failures = []

    nan_found = {}
    fill_counts = {"z": 0, "u": 0, "v": 0}
    ocean_counts = {"z": 0, "u": 0, "v": 0}
    cover_viol = {"z": [], "u": [], "v": []}

    for j0 in range(0, NY, BAND):
        j1 = min(j0 + BAND, NY)
        for node in ("z", "u", "v"):
            flag = new[f"{node}_flag"].values[j0:j1]
            re = new[f"{node}_Re"].values[j0:j1]
            im = new[f"{node}_Im"].values[j0:j1]
            for name, arr in ((f"{node}_Re", re), (f"{node}_Im", im)):
                if not np.all(np.isfinite(arr)):
                    nan_found[name] = nan_found.get(name, 0) + 1
            fill_counts[node] += int((flag == P.FLAG_FILLED).sum())
            ocean_counts[node] += int((flag <= P.FLAG_FILLED).sum())

            # legacy-valid = finite in ANY constituent (round 16, finding 2:
            # the first-constituent-only test missed 81 u-cells whose M2 was
            # invalid but other constituents valid — 3 of them reclassified
            # coastline that the gate would otherwise silently drop)
            amp_all = old[f"{node}_amp"].isel(lat=slice(j0, j1)).values
            old_valid = np.isfinite(amp_all).any(axis=-1)
            bad = old_valid & (flag == P.FLAG_INVALID)
            if bad.any():
                jj, ii = np.nonzero(bad)
                for p in range(len(jj)):
                    cover_viol[node].append((j0 + int(jj[p]), int(ii[p])))
        for name in ("uz_Re", "uz_Im", "vz_Re", "vz_Im"):
            arr = new[name].values[j0:j1]
            if not np.all(np.isfinite(arr)):
                nan_found[name] = nan_found.get(name, 0) + 1
    print(f"[1/3] NaN scan: {'OK (zero NaN)' if not nan_found else nan_found}")
    if nan_found:
        failures.append(f"NaN present: {nan_found}")

    for node in ("z", "u", "v"):
        frac = fill_counts[node] / max(ocean_counts[node], 1)
        ok = frac <= 0.02
        print(f"[2/3] {node}: filled {fill_counts[node]} / ocean "
              f"{ocean_counts[node]} = {100 * frac:.4f}% "
              + ("OK" if ok else "FAIL (>2%)"))
        if not ok:
            failures.append(f"{node} fill fraction {frac:.4%} > 2%")

    # G2 finding (2026-06-12): 100% of legacy-valid -> new-invalid cells
    # are TPXO10 COASTLINE RECLASSIFICATION (TPXO9 h>0, TPXO10 h==0 — the
    # advertised all-node coastline redefinition). Classified machine-
    # checkably: reclassified cells are REPORTED; any violation NOT
    # explained by reclassification (e.g. TPXO10-ocean cell beyond the
    # fill band) FAILS. The amendment of the original strict criterion
    # requires owner/reviewer sign-off at G2 (recorded in TESTING.md).
    import netCDF4
    g9_path = repo_root / "data_src" / "TPXO9_atlas_v5" / "grid_tpxo9_atlas_30_v5.nc"
    lon_z = new["lon_z"].values
    lat_z = new["lat_z"].values
    with netCDF4.Dataset(g9_path) as g9:
        for node, hvar in (("z", "hz"), ("u", "hu"), ("v", "hv")):
            v = cover_viol[node]
            if not v:
                print(f"[3/3] {node}: legacy-valid -> new-invalid: 0")
                continue
            jj = np.array([p[0] for p in v]); ii = np.array([p[1] for p in v])
            h9 = np.asarray(g9[hvar][:]).T[jj, ii]
            h10 = new[f"h{node}"].values[jj, ii]
            reclass = (h9 > 0) & (h10 == 0)
            # second explained class: a flag-2 cell whose bathymetry is
            # kept (h10>0) is contract-correct ONLY if the FROZEN §3.2+D3
            # rule independently yields flag 2 — i.e. it is source-invalid
            # AND further than DMAX_FILL_CELLS from any source-valid node.
            # Recompute that rule here from the source (round 16, finding 3:
            # the gate must self-verify, not infer from T-B).
            rest = np.nonzero(~reclass)[0]
            isolated = np.zeros(len(jj), dtype=bool)
            if len(rest):
                src = repo_root / "data_src" / "TPXO10_atlas_v2"
                R = P.DMAX_FILL_CELLS + 1
                for p in rest:
                    gj, gi = int(jj[p]), int(ii[p])
                    cls = recompute_flag_class(src, node, gj, gi, R)
                    isolated[p] = (h10[p] > 0) and (cls == P.FLAG_INVALID)
            unexplained = ~reclass & ~isolated
            print(f"[3/3] {node}: legacy-valid -> new-invalid: {len(v)} "
                  f"({int(reclass.sum())} coastline-reclassified [reported], "
                  f"{int(isolated.sum())} isolated-no-data per frozen "
                  f"contract [reported], {int(unexplained.sum())} unexplained)")
            if unexplained.any():
                failures.append(f"{node}: {int(unexplained.sum())} UNEXPLAINED coverage violations")
                for p in np.nonzero(unexplained)[0][:10]:
                    print(f"    UNEXPLAINED ({lon_z[ii[p]]:.4f}E, {lat_z[jj[p]]:.4f}N) "
                          f"h9={h9[p]:.2f} h10={h10[p]:.2f}")

    if failures:
        print("FAIL verify_coverage")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS verify_coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
