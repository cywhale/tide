#!/usr/bin/env python
"""T-B: NetCDF <-> Zarr consistency gate (spec §7.2) — regional full scan.

Per §3.0 layer semantics:
  flag==0 : stored int32 Re/Im EXACTLY equal source (tolerance 0)
  flag==1 : source cell invalid per §3.2 (inpaint only fills invalid cells)
  flag==2 : stored value exactly 0 and source cell invalid
Plus: coordinates and hz/hu/hv full-scan exact; flags equal a deterministic
recompute (validity + distance band) on the haloed window; derived uz/vz
match an independent reimplementation of the D12 centering rule from the
stored truth layers (rtol <= 1e-6 float32 budget; the easternmost column /
northernmost row need the halo edge and are excluded here — covered by
T-D1 golden checks instead, count reported).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, msg: str):
    if not ok:
        FAILURES.append(msg)


def independent_center_u(eu_w, eu_e, f_w, f_e):
    """Deliberately separate implementation of D12 (loop-free but written
    independently of tpxo10_pipeline._center_pair) for T-B round-1 bullet."""
    use_w, use_e = f_w <= 1, f_e <= 1
    out = np.where(
        use_w & use_e, (eu_w + eu_e) / 2.0,
        np.where(use_w, eu_w, np.where(use_e, eu_e, 0.0)),
    )
    flag = np.where(
        use_w & use_e & (f_w == 0) & (f_e == 0), 0,
        np.where(use_w | use_e, 1, 2),
    ).astype(np.uint8)
    return out, flag


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--source", type=Path,
                    default=repo_root / "data_src" / "TPXO10_atlas_v2")
    args = ap.parse_args()

    z = xr.open_zarr(args.store, consolidated=True, decode_times=False)
    j0, j1, i0, i1 = z.attrs["interior_index_window"]
    j_int, i_int = slice(j0, j1), slice(i0, i1)
    jw, iw = P.haloed_window(j_int, i_int)
    H = P.HALO_CELLS
    nlat, nlon = j1 - j0, i1 - i0
    interior = (slice(H, H + nlat), slice(H, H + nlon))

    grid = args.source / "grid_tpxo10atlas_v2.nc"
    with netCDF4.Dataset(grid) as g:
        for name in ("lon_z", "lat_z", "lon_u", "lat_u", "lon_v", "lat_v"):
            w = i_int if name.startswith("lon") else j_int
            check(np.array_equal(z[name].values, np.asarray(g[name][w])),
                  f"coord {name} differs from source window")
        h_haloed = {n: np.asarray(g[hv][iw, jw]).T.astype(np.float32)
                    for n, hv in (("z", "hz"), ("u", "hu"), ("v", "hv"))}
    print("[1/5] coordinates: " + ("OK" if not FAILURES else "FAIL"))

    truth_haloed = {}
    for node in ("z", "u", "v"):
        h = h_haloed[node]
        check(np.array_equal(z[f"h{node}"].values, h[interior]),
              f"h{node} differs from source")
        prefix, rvar, ivar = {"z": ("h", "hRe", "hIm"),
                              "u": ("u", "uRe", "uIm"),
                              "v": ("u", "vRe", "vIm")}[node]
        ncons = len(P.CONSTITUENTS)
        s_re = np.empty(h.shape + (ncons,), dtype=np.int32)
        s_im = np.empty_like(s_re)
        for k, c in enumerate(P.CONSTITUENTS):
            with netCDF4.Dataset(args.source / f"{prefix}_{c}_tpxo10_atlas_30_v2.nc") as ds:
                s_re[..., k] = np.asarray(ds[rvar][iw, jw]).T
                s_im[..., k] = np.asarray(ds[ivar][iw, jw]).T

        valid = P.compute_validity(h, s_re, s_im)
        flag_re = P.classify_flags(h, valid)[interior]
        flag = z[f"{node}_flag"].values
        check(np.array_equal(flag, flag_re), f"{node}_flag != deterministic recompute")

        st_re = z[f"{node}_Re"].values
        st_im = z[f"{node}_Im"].values
        src_re, src_im = s_re[interior], s_im[interior]
        v_int = valid[interior]
        f0, f1, f2 = flag == 0, flag == 1, flag == 2
        check(np.array_equal(st_re[f0], src_re[f0]) and np.array_equal(st_im[f0], src_im[f0]),
              f"{node}: flag0 cells not bit-exact vs source")
        check(bool(np.all(v_int[f0])), f"{node}: some flag0 cells not source-valid")
        check(not np.any(v_int[f1]), f"{node}: inpaint overwrote a valid cell")
        check(not np.any(v_int[f2]), f"{node}: flag2 cell is source-valid")
        check(bool(np.all(st_re[f2] == 0)) and bool(np.all(st_im[f2] == 0)),
              f"{node}: flag2 cells not stored as 0")
        truth_haloed[node] = (s_re, s_im, h, valid)
        print(f"[2/5] {node} truth layer (flag0={int(f0.sum())} "
              f"flag1={int(f1.sum())} flag2={int(f2.sum())}): "
              + ("OK" if not FAILURES else "FAIL"))

    # derived layer: independent recompute from STORED truth (interior-only
    # edges; the last column/row needs halo and is excluded -> reported)
    for comp, flag_name, e_node in (("uz", "uz_flag", "u"), ("vz", "vz_flag", "v")):
        st_val = (z[f"{comp}_Re"].values + 1j * z[f"{comp}_Im"].values)
        st_flag = z[flag_name].values
        n_re = z[f"{e_node}_Re"].values.astype(np.float64)
        n_im = z[f"{e_node}_Im"].values.astype(np.float64)
        n_fl = z[f"{e_node}_flag"].values
        n_h = z[f"h{e_node}"].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            evel = 1e-4 * (n_re + 1j * n_im) / n_h[..., None]
        evel[n_fl > 1] = 0.0
        if comp == "uz":
            ref, rflag = independent_center_u(
                evel[:, :-1, :], evel[:, 1:, :],
                np.repeat(n_fl[:, :-1, None], evel.shape[-1], -1),
                np.repeat(n_fl[:, 1:, None], evel.shape[-1], -1))
            sl = (slice(None), slice(0, nlon - 1))
            excluded = nlat  # easternmost column cell count
        else:
            ref, rflag = independent_center_u(
                evel[:-1, :, :], evel[1:, :, :],
                np.repeat(n_fl[:-1, :, None], evel.shape[-1], -1),
                np.repeat(n_fl[1:, :, None], evel.shape[-1], -1))
            sl = (slice(0, nlat - 1), slice(None))
            excluded = nlon
        got = st_val[sl]
        want = ref.astype(np.complex64)
        check(np.allclose(got, want, rtol=1e-6, atol=1e-9),
              f"{comp}: derived values mismatch independent recompute "
              f"(max|d|={np.abs(got - want).max():.3e})")
        check(np.array_equal(st_flag[sl[0], sl[1]], rflag[..., 0]),
              f"{comp}: flags mismatch independent recompute")
        print(f"[3/5] {comp} derived layer: "
              + ("OK" if not FAILURES else "FAIL")
              + f" (excluded halo-dependent cells: {excluded}, covered by T-D1)")

    check(z.attrs.get("tide_store_schema") == P.SCHEMA_VERSION, "schema attr missing")
    check(z.attrs.get("quantization_rule") == P.QUANTIZATION_VERSION,
          "quantization attr missing")
    print("[4/5] provenance attrs: " + ("OK" if not FAILURES else "FAIL"))

    if FAILURES:
        print("FAIL verify_against_netcdf")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[5/5] PASS verify_against_netcdf (full regional scan, tolerance 0 on source layer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
