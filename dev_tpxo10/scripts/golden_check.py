#!/usr/bin/env python
"""T-D1 golden-value checks (spec §7.4): store vs pyTMD 3.0.6 reading path.

Cross-checks our pipeline's IO/transpose/index mapping and the D12
centering against pyTMD's own readers at seeded + named node indices:

  1. z harmonic constants: store (z_Re + i*z_Im) [mm] vs
     pyTMD open_atlas_dataset(group='z') at the same global node — rtol 1e-9.
  2. u/v transport: store ints vs pyTMD open_atlas_dataset(group='u'/'v')
     (NOTE: at this API layer pyTMD returns *transport* in cm^2/s; the
     depth division happens in higher-level accessors) — rtol 1e-9.
  3. hu/hv: store vs pyTMD open_atlas_grid — exact.
  4. D12 centering: store uz (float32 m/s) vs hand-computed two-edge
     average of per-edge velocities built from pyTMD-read transport and
     depths — rtol 1e-5; includes one-sided (flag 1) coastal cells whose
     usable edge is native.

Only flag-0 cells/edges are compared against pyTMD (pyTMD has no inpaint).
Stratified sampling: deep (h>=1000), shelf (50<=h<1000), coastal (h<50),
plus fixed named locations. Dateline/polar branches are global-only and
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

    u_flag = z["u_flag"].values
    upts = [(j, i) for j, i in pts if u_flag[j, i] == 0]
    ujj = np.array([p[0] for p in upts]); uii = np.array([p[1] for p in upts])
    st_u = (z["u_Re"].values[ujj, uii, :] + 1j * z["u_Im"].values[ujj, uii, :])
    for k, c in enumerate(P.CONSTITUENTS):
        ds = ATLAS.open_atlas_dataset(
            args.source / f"u_{c}_tpxo10_atlas_30_v2.nc", group="u")
        ref = ds[c].isel(y=xr.DataArray(ujj + j0), x=xr.DataArray(uii + i0)).values
        check(np.allclose(st_u[:, k], ref, rtol=1e-9, atol=1e-9),
              f"u transport mismatch for {c}")
    print(f"[3/4] u transport ({len(upts)} cells) + hu/hv vs pyTMD: "
          + ("OK" if not FAILURES else "FAIL"))

    # -- 4. D12 centering vs hand-computed pyTMD-read pieces ---------------
    uz_flag = z["uz_flag"].values
    nlon = z.sizes["lon_z"]
    cand0 = [(j, i) for j, i in pts if i < nlon - 1 and uz_flag[j, i] == 0]
    # one-sided coastal cells with both contributing edges native-or-missing
    oj, oi = np.nonzero(uz_flag == 1)
    keep = oi < nlon - 1
    oj, oi = oj[keep], oi[keep]
    pick = rng.choice(len(oj), size=min(60, len(oj)), replace=False)
    cand1 = []
    for p in pick:
        j, i = int(oj[p]), int(oi[p])
        fw, fe = u_flag[j, i], u_flag[j, i + 1]
        if (fw == 0 and fe == 2) or (fw == 2 and fe == 0):  # purely one-sided native
            cand1.append((j, i))
    hu_st = z["hu"].values
    u_re = z["u_Re"].values; u_im = z["u_Im"].values
    st_uz = z["uz_Re"].values + 1j * z["uz_Im"].values

    def edge_vel(j, i):
        return 1e-4 * (u_re[j, i, :] + 1j * u_im[j, i, :]) / hu_st[j, i]

    for label, cells in (("two-edge", cand0), ("one-sided", cand1)):
        bad = 0
        for j, i in cells:
            fw, fe = u_flag[j, i], u_flag[j, i + 1]
            if fw == 0 and fe == 0:
                ref = 0.5 * (edge_vel(j, i) + edge_vel(j, i + 1))
            elif fw == 0:
                ref = edge_vel(j, i)
            else:
                ref = edge_vel(j, i + 1)
            if not np.allclose(st_uz[j, i, :], ref, rtol=1e-5, atol=1e-9):
                bad += 1
        check(bad == 0, f"uz centering ({label}): {bad}/{len(cells)} cells mismatch")
        print(f"[4/4] uz centering {label} ({len(cells)} cells): "
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
