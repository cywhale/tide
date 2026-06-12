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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_global.zarr")
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

            amp0 = old[f"{node}_amp"].isel(
                lat=slice(j0, j1), constituents=0).values
            old_valid = np.isfinite(amp0)
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

    lon_z = new["lon_z"].values
    lat_z = new["lat_z"].values
    for node in ("z", "u", "v"):
        v = cover_viol[node]
        print(f"[3/3] {node}: legacy-valid -> new-invalid violations: {len(v)}")
        if v:
            failures.append(f"{node}: {len(v)} coverage violations")
            for (j, i) in v[:10]:
                print(f"    ({lon_z[i]:.4f}E, {lat_z[j]:.4f}N)")

    if failures:
        print("FAIL verify_coverage")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS verify_coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
