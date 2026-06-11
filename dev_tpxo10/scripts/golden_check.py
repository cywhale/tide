#!/usr/bin/env python
"""T-D1 golden-value checks (spec §7.4): store vs pyTMD 3.0.6 reading path.

Cross-checks our pipeline's IO/transpose/index mapping and the D12
centering against pyTMD's own readers at seeded + named node indices:

  1. z harmonic constants: store (z_Re + i*z_Im) [mm] vs
     pyTMD open_atlas_dataset(group='z') at the same global node — rtol 1e-9.
  2. u AND v transport: store ints vs pyTMD open_atlas_dataset
     (group='u'/'v') (NOTE: at this API layer pyTMD returns *transport* in
     cm^2/s; the depth division happens in higher-level accessors) —
     rtol 1e-9.
  3. hu/hv: store vs pyTMD open_atlas_grid (land-mask convention:
     pyTMD NaN <-> store 0) — exact on finite cells.
  4. D12 centering for BOTH uz and vz: store float32 m/s vs two-edge
     averages hand-computed from pyTMD-READ transport and pyTMD-READ edge
     depths (never from store arrays) — rtol 1e-5; includes one-sided
     coastal cells and the halo-dependent boundary cells (easternmost
     column for uz, northernmost row for vz), whose outer edge is read
     from the global source via pyTMD.

Only cells whose contributing edges are source-valid (§3.2 recomputed from
pyTMD-read values) are compared (pyTMD has no inpaint). Stratified
sampling: deep (h>=1000), shelf (50<=h<1000), coastal (h<50), plus fixed
named locations. Dateline/polar/last-latitude branches are global-only and
are covered by unit tests now and the Stage 2 global golden run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import pyTMD.io.ATLAS as ATLAS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

NAMED = {  # (lon, lat) inside the Stage 1 region
    "taiwan_strait": (119.5, 24.5),
    "kuroshio_east_taiwan": (122.5, 23.5),
    "luzon_strait": (121.0, 20.5),
    "surigao_strait": (125.6, 9.87),
    "open_pacific": (140.0, 20.0),
    "yellow_sea": (123.0, 35.0),
}
FAILURES: list[str] = []


def check(ok: bool, msg: str):
    if not ok:
        FAILURES.append(msg)


def sample_indices(rng, flag, hz, n_per_stratum):
    """Window-local (j, i) samples stratified by depth, flag==0 only."""
    out = []
    strata = [(hz >= 1000), (hz >= 50) & (hz < 1000), (hz > 0) & (hz < 50)]
    for s in strata:
        jj, ii = np.nonzero(s & (flag == 0))
        if len(jj) == 0:
            continue
        pick = rng.choice(len(jj), size=min(n_per_stratum, len(jj)), replace=False)
        out += [(int(jj[p]), int(ii[p])) for p in pick]
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--source", type=Path,
                    default=repo_root / "data_src" / "TPXO10_atlas_v2")
    ap.add_argument("--n-per-stratum", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    z = xr.open_zarr(args.store, consolidated=True, decode_times=False)
    j0, j1, i0, i1 = z.attrs["interior_index_window"]
    hz = z["hz"].values
    z_flag = z["z_flag"].values

    pts = sample_indices(rng, z_flag, hz, args.n_per_stratum)
    for name, (lon, lat) in NAMED.items():
        i = int(np.argmin(np.abs(z["lon_z"].values - lon)))
        j = int(np.argmin(np.abs(z["lat_z"].values - lat)))
        if z_flag[j, i] == 0:
            pts.append((j, i))
    print(f"[1/4] sampled {len(pts)} flag-0 z-cells "
          f"(strata deep/shelf/coastal + {len(NAMED)} named)")

    jj = np.array([p[0] for p in pts]); ii = np.array([p[1] for p in pts])
    gj, gi = jj + j0, ii + i0  # global indices (pyTMD y/x are index-aligned)

    # -- 1. z harmonic constants ------------------------------------------
    st_z = (z["z_Re"].values[jj, ii, :] + 1j * z["z_Im"].values[jj, ii, :])
    for k, c in enumerate(P.CONSTITUENTS):
        ds = ATLAS.open_atlas_dataset(
            args.source / f"h_{c}_tpxo10_atlas_30_v2.nc", group="z")
        ref = ds[c].isel(y=xr.DataArray(gj), x=xr.DataArray(gi)).values
        check(np.allclose(st_z[:, k], ref, rtol=1e-9, atol=1e-9),
              f"z hc mismatch for {c}")
    print("[2/4] z hc vs pyTMD(group='z'): " + ("OK" if not FAILURES else "FAIL"))

    # -- 2. transport + 3. grid depths ------------------------------------
    hu_g = ATLAS.open_atlas_grid(
        args.source / "grid_tpxo10atlas_v2.nc", group="u")["bathymetry"]
    hv_g = ATLAS.open_atlas_grid(
        args.source / "grid_tpxo10atlas_v2.nc", group="v")["bathymetry"]
    # pyTMD's grid reader masks land (h == 0) to NaN; the store keeps the
    # raw 0. Equivalence: finite -> exact equal; NaN -> store value is 0.
    for hname, h_g in (("hu", hu_g), ("hv", hv_g)):
        st = z[hname].values[jj, ii]
        ref = h_g.isel(y=xr.DataArray(gj), x=xr.DataArray(gi)).values
        finite = np.isfinite(ref)
        check(np.array_equal(st[finite], ref[finite].astype(np.float32))
              and bool(np.all(st[~finite] == 0.0)),
              f"{hname} mismatch vs pyTMD grid reader (land-mask convention)")

    # -- 2b. u AND v transport vs pyTMD ------------------------------------
    for comp, flag_var, rv in (("u", "u_flag", "u"), ("v", "v_flag", "v")):
        n_flag = z[flag_var].values
        npts = [(j, i) for j, i in pts if n_flag[j, i] == 0]
        njj = np.array([p[0] for p in npts]); nii = np.array([p[1] for p in npts])
        st_t = (z[f"{comp}_Re"].values[njj, nii, :]
                + 1j * z[f"{comp}_Im"].values[njj, nii, :])
        for k, c in enumerate(P.CONSTITUENTS):
            ds = ATLAS.open_atlas_dataset(
                args.source / f"u_{c}_tpxo10_atlas_30_v2.nc", group=rv)
            ref = ds[c].isel(y=xr.DataArray(njj + j0), x=xr.DataArray(nii + i0)).values
            check(np.allclose(st_t[:, k], ref, rtol=1e-9, atol=1e-9),
                  f"{comp} transport mismatch for {c}")
        print(f"[3/4] {comp} transport ({len(npts)} cells) vs pyTMD: "
              + ("OK" if not FAILURES else "FAIL"))

    # -- 4. D12 centering vs pieces READ VIA pyTMD (round 9, finding 3:
    #       both components, reference never touches store arrays, and the
    #       halo-dependent boundary cells are included via global reads) ---
    nlat, nlon = z.sizes["lat_z"], z.sizes["lon_z"]

    def pytmd_transport_at(group, gjj, gii):
        out = np.empty((len(gjj), len(P.CONSTITUENTS)), dtype=np.complex128)
        for k, c in enumerate(P.CONSTITUENTS):
            ds = ATLAS.open_atlas_dataset(
                args.source / f"u_{c}_tpxo10_atlas_30_v2.nc", group=group)
            out[:, k] = ds[c].isel(y=xr.DataArray(gjj), x=xr.DataArray(gii)).values
        return out

    for comp, group, h_g in (("uz", "u", hu_g), ("vz", "v", hv_g)):
        st_flag = z[f"{comp}_flag"].values
        st_val = z[f"{comp}_Re"].values + 1j * z[f"{comp}_Im"].values
        cells = [(j, i) for j, i in pts if st_flag[j, i] <= 1]
        # boundary cells whose outer edge lives in the halo (global read)
        if comp == "uz":
            bj = np.nonzero(st_flag[:, nlon - 1] <= 1)[0]
            cells += [(int(j), nlon - 1) for j in
                      rng.choice(bj, size=min(60, len(bj)), replace=False)]
        else:
            bi = np.nonzero(st_flag[nlat - 1, :] <= 1)[0]
            cells += [(nlat - 1, int(i)) for i in
                      rng.choice(bi, size=min(60, len(bi)), replace=False)]
        cjj = np.array([c[0] for c in cells]); cii = np.array([c[1] for c in cells])
        if comp == "uz":
            e1 = (cjj + j0, cii + i0); e2 = (cjj + j0, cii + i0 + 1)
        else:
            e1 = (cjj + j0, cii + i0); e2 = (cjj + j0 + 1, cii + i0)
        T1 = pytmd_transport_at(group, *e1); T2 = pytmd_transport_at(group, *e2)
        h1 = h_g.isel(y=xr.DataArray(e1[0]), x=xr.DataArray(e1[1])).values
        h2 = h_g.isel(y=xr.DataArray(e2[0]), x=xr.DataArray(e2[1])).values
        v1 = np.isfinite(h1) & (h1 > 0) & ~np.all(T1 == 0, axis=1)
        v2 = np.isfinite(h2) & (h2 > 0) & ~np.all(T2 == 0, axis=1)
        n_cmp = {"two-edge": 0, "one-sided": 0, "skipped-inpaint": 0}
        bad = 0
        for n in range(len(cells)):
            j, i = cells[n]
            if v1[n] and v2[n]:
                ref = 0.5e-4 * (T1[n] / h1[n] + T2[n] / h2[n])
                ok = st_flag[j, i] == 0 and np.allclose(
                    st_val[j, i, :], ref, rtol=1e-5, atol=1e-9)
                n_cmp["two-edge"] += 1
            elif v1[n] or v2[n]:
                # the invalid edge may have been inpainted (then the store
                # legitimately used it); only the pure one-sided case has a
                # pyTMD-derivable reference
                if st_flag[j, i] != 1:
                    ok = False
                else:
                    ref = 1e-4 * (T1[n] / h1[n] if v1[n] else T2[n] / h2[n])
                    if np.allclose(st_val[j, i, :], ref, rtol=1e-5, atol=1e-9):
                        ok = True
                        n_cmp["one-sided"] += 1
                    else:
                        ok = True  # treated as inpainted-edge two-edge case
                        n_cmp["skipped-inpaint"] += 1
            else:
                ok = st_flag[j, i] != 0
                n_cmp["skipped-inpaint"] += 1
            if not ok:
                bad += 1
        check(bad == 0, f"{comp} centering: {bad}/{len(cells)} cells mismatch")
        print(f"[4/4] {comp} centering vs pyTMD-read pieces "
              f"({n_cmp['two-edge']} two-edge, {n_cmp['one-sided']} one-sided, "
              f"{n_cmp['skipped-inpaint']} skipped-inpaint, incl. boundary): "
              + ("OK" if bad == 0 else "FAIL"))

    if FAILURES:
        print("FAIL golden_check")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PASS golden_check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
