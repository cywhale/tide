#!/usr/bin/env python
"""T-B: NetCDF <-> Zarr consistency gate (spec §7.2) — regional full scan.

Per §3.0 layer semantics:
  flag==0 : stored int32 Re/Im EXACTLY equal source (tolerance 0)
  flag==1 : source cell invalid per §3.2 (inpaint only fills invalid cells)
  flag==2 : stored value exactly 0 and source cell invalid
Plus: coordinates and hz/hu/hv full-scan exact; flags equal a deterministic
recompute on the haloed window; truth layers equal a FULL deterministic
reproduction (source halo + recomputed flags + recomputed inpaint, byte
equality on every interior cell of every flag class); derived uz/vz match
an independent reimplementation of the D12 centering rule computed on the
haloed expected truth — full interior coverage including the easternmost
column and northernmost row (round 9, finding 2: no exclusions).
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


def independent_center_u_wrap(evel, flags):
    """Global (periodic) variant of the independent centering recompute."""
    e_east = np.roll(evel, -1, axis=1)
    f_east = np.roll(flags, -1, axis=1)
    nc = evel.shape[-1]
    return independent_center_u(
        evel, e_east,
        np.repeat(flags[..., None], nc, -1),
        np.repeat(f_east[..., None], nc, -1))


def verify_global(z, source: Path, repo_root: Path, sample_tiles: int | None) -> int:
    """G2 global T-B: tiled deterministic reproduction (same lat_tiles
    decomposition as the converter — inpaint donor selection is
    tile-window-deterministic), periodic-wrap derived recompute, top-row
    one-sided, canonical-publication attrs. sample_tiles=None => full
    scan (required once before G2 sign-off)."""
    import netCDF4 as nc4
    NY, NX = P.NY, P.NX
    tile_lat = int(z.attrs["tile_lat_rows"])
    check(z.attrs.get("canonical") is True
          and z.attrs.get("conversion_status") == "complete",
          "store is not a published canonical store")
    tiles = P.lat_tiles(NY, tile_lat)
    check(sorted(z.attrs.get("tiles_written", [])) == list(range(len(tiles))),
          "tiles_written ledger incomplete")

    grid = source / "grid_tpxo10atlas_v2.nc"
    with nc4.Dataset(grid) as g:
        axes = {n: np.asarray(g[n][:]) for n in
                ("lon_z", "lat_z", "lon_u", "lat_u", "lon_v", "lat_v")}
        for name, ax in axes.items():
            check(np.array_equal(z[name].values, ax), f"coord {name} differs")
    print("[1/5] coordinates (full global): " + ("OK" if not FAILURES else "FAIL"))

    pick = list(range(len(tiles)))
    if sample_tiles:
        rng = np.random.default_rng(20260611)
        pick = sorted(set(rng.choice(len(tiles) - 1, size=sample_tiles - 1,
                                     replace=False).tolist()) | {len(tiles) - 1})
        print(f"    sampled tiles: {pick} (top tile always included)")

    ncons = len(P.CONSTITUENTS)
    fill_totals = {"z": 0, "u": 0, "v": 0}
    for t_i in pick:
        j0, j1, jh0, jh1, top = tiles[t_i]
        jw = slice(jh0, jh1)
        H0, nrow = j0 - jh0, j1 - j0
        interior = slice(H0, H0 + nrow)
        with nc4.Dataset(grid) as g:
            h_haloed = {n: np.asarray(g[hv][:, jw]).T.astype(np.float32)
                        for n, hv in (("z", "hz"), ("u", "hu"), ("v", "hv"))}
        expected = {}
        for node in ("z", "u", "v"):
            h = h_haloed[node]
            check(np.array_equal(z[f"h{node}"].values[j0:j1], h[interior]),
                  f"tile{t_i} h{node} differs")
            prefix, rvar, ivar = {"z": ("h", "hRe", "hIm"),
                                  "u": ("u", "uRe", "uIm"),
                                  "v": ("u", "vRe", "vIm")}[node]
            s_re = np.empty(h.shape + (ncons,), dtype=np.int32)
            s_im = np.empty_like(s_re)
            for k, c in enumerate(P.CONSTITUENTS):
                with nc4.Dataset(source / f"{prefix}_{c}_tpxo10_atlas_30_v2.nc") as ds:
                    s_re[..., k] = np.asarray(ds[rvar][:, jw]).T
                    s_im[..., k] = np.asarray(ds[ivar][:, jw]).T
            valid = P.compute_validity(h, s_re, s_im)
            flag_h = P.classify_flags(h, valid)
            P.inpaint_fill_inplace(axes[f"lon_{node}"], axes[f"lat_{node}"][jw],
                                   s_re, s_im, valid, flag_h, context=f"t{t_i}/{node}")
            inv = flag_h == P.FLAG_INVALID
            s_re[inv], s_im[inv] = 0, 0
            check(np.array_equal(z[f"{node}_flag"].values[j0:j1], flag_h[interior]),
                  f"tile{t_i} {node}_flag != reproduction")
            check(np.array_equal(z[f"{node}_Re"].values[j0:j1], s_re[interior])
                  and np.array_equal(z[f"{node}_Im"].values[j0:j1], s_im[interior]),
                  f"tile{t_i} {node} truth != full reproduction")
            fill_totals[node] += int((flag_h[interior] == P.FLAG_FILLED).sum())
            expected[node] = (s_re, s_im, flag_h, h)

        for comp, e_node in (("uz", "u"), ("vz", "v")):
            e_re, e_im, e_fl, e_h = expected[e_node]
            with np.errstate(divide="ignore", invalid="ignore"):
                evel = 1e-4 * (e_re.astype(np.float64) + 1j * e_im.astype(np.float64)) \
                    / e_h.astype(np.float64)[..., None]
            evel[e_fl > 1] = 0.0
            if comp == "uz":
                ref, rflag = independent_center_u_wrap(evel, e_fl)
                got = (z["uz_Re"].values[j0:j1] + 1j * z["uz_Im"].values[j0:j1])
                want = ref[interior].astype(np.complex64)
                fwant = rflag[interior][..., 0]
            else:
                nc_ = evel.shape[-1]
                ref, rflag = independent_center_u(
                    evel[:-1], evel[1:],
                    np.repeat(e_fl[:-1, :, None], nc_, -1),
                    np.repeat(e_fl[1:, :, None], nc_, -1))
                if top:  # one-sided last global row (independent recompute)
                    usable = (e_fl[-1:, :] <= 1)[..., None].repeat(nc_, -1)
                    last = np.where(usable, evel[-1:], 0.0)
                    lastf = np.where(usable, 1, 2).astype(np.uint8)
                    ref = np.concatenate([ref, last], axis=0)
                    rflag = np.concatenate([rflag, lastf], axis=0)
                got = (z["vz_Re"].values[j0:j1] + 1j * z["vz_Im"].values[j0:j1])
                want = ref[interior].astype(np.complex64)
                fwant = rflag[interior][..., 0]
            check(np.allclose(got, want, rtol=1e-6, atol=1e-9),
                  f"tile{t_i} {comp} derived mismatch")
            check(np.array_equal(z[f"{comp}_flag"].values[j0:j1], fwant),
                  f"tile{t_i} {comp} flags mismatch")
        print(f"[2/5] tile {t_i} ({j0}:{j1}{', top' if top else ''}): "
              + ("OK" if not FAILURES else "FAIL"))

    scope = "FULL SCAN" if not sample_tiles else f"{len(pick)}/{len(tiles)} tiles"
    print(f"[3/5] truth + derived reproduction ({scope}); fills seen {fill_totals}")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--source", type=Path,
                    default=repo_root / "data_src" / "TPXO10_atlas_v2")
    ap.add_argument("--sample-tiles", type=int, default=None,
                    help="global mode: verify N sampled tiles (top always "
                         "included); omit for the full scan required at G2")
    args = ap.parse_args()

    z = xr.open_zarr(args.store, consolidated=True, decode_times=False)
    j0, j1, i0, i1 = z.attrs["interior_index_window"]
    if [j0, j1, i0, i1] == [0, P.NY, 0, P.NX]:
        verify_global(z, args.source, repo_root, args.sample_tiles)
        return finish(z, repo_root)
    j_int, i_int = slice(j0, j1), slice(i0, i1)
    jw, iw = P.haloed_window(j_int, i_int)
    H = P.HALO_CELLS
    nlat, nlon = j1 - j0, i1 - i0
    interior = (slice(H, H + nlat), slice(H, H + nlon))

    grid = args.source / "grid_tpxo10atlas_v2.nc"
    axes = {}
    with netCDF4.Dataset(grid) as g:
        for name in ("lon_z", "lat_z", "lon_u", "lat_u", "lon_v", "lat_v"):
            w = i_int if name.startswith("lon") else j_int
            check(np.array_equal(z[name].values, np.asarray(g[name][w])),
                  f"coord {name} differs from source window")
            axes[name] = np.asarray(g[name][iw if name.startswith("lon") else jw])
        h_haloed = {n: np.asarray(g[hv][iw, jw]).T.astype(np.float32)
                    for n, hv in (("z", "hz"), ("u", "hu"), ("v", "hv"))}
    print("[1/5] coordinates: " + ("OK" if not FAILURES else "FAIL"))

    # truth layers: full deterministic reproduction (round 9, finding 2 —
    # source halo + recompute of flags AND inpaint => byte-equality on ALL
    # interior cells, every flag class)
    expected_haloed = {}
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
        flag_h = P.classify_flags(h, valid)
        flag = z[f"{node}_flag"].values
        check(np.array_equal(flag, flag_h[interior]),
              f"{node}_flag != deterministic recompute")

        exp_re, exp_im = s_re.copy(), s_im.copy()
        P.inpaint_fill_inplace(axes[f"lon_{node}"], axes[f"lat_{node}"],
                               exp_re, exp_im, valid, flag_h, context=node)
        inv = flag_h == P.FLAG_INVALID
        exp_re[inv], exp_im[inv] = 0, 0

        st_re = z[f"{node}_Re"].values
        st_im = z[f"{node}_Im"].values
        check(np.array_equal(st_re, exp_re[interior])
              and np.array_equal(st_im, exp_im[interior]),
              f"{node}: truth layer != full deterministic reproduction")
        # explicit layer semantics (§3.0), redundant with the above but kept
        # as the contract-readable form:
        f0, f1, f2 = flag == 0, flag == 1, flag == 2
        v_int = valid[interior]
        src_re, src_im = s_re[interior], s_im[interior]
        check(np.array_equal(st_re[f0], src_re[f0]) and np.array_equal(st_im[f0], src_im[f0]),
              f"{node}: flag0 cells not bit-exact vs source")
        check(not np.any(v_int[f1]), f"{node}: inpaint overwrote a valid cell")
        check(bool(np.all(st_re[f2] == 0)) and bool(np.all(st_im[f2] == 0)),
              f"{node}: flag2 cells not stored as 0")
        expected_haloed[node] = (exp_re, exp_im, flag_h, h)
        print(f"[2/5] {node} truth layer (flag0={int(f0.sum())} "
              f"flag1={int(f1.sum())} flag2={int(f2.sum())}, full reproduction): "
              + ("OK" if not FAILURES else "FAIL"))

    # derived layer: independent centering recompute on the HALOED expected
    # truth — full interior coverage including the easternmost column and
    # northernmost row (round 9, finding 2: no exclusions)
    for comp, flag_name, e_node in (("uz", "uz_flag", "u"), ("vz", "vz_flag", "v")):
        e_re, e_im, e_fl, e_h = expected_haloed[e_node]
        with np.errstate(divide="ignore", invalid="ignore"):
            evel = 1e-4 * (e_re.astype(np.float64) + 1j * e_im.astype(np.float64)) \
                / e_h.astype(np.float64)[..., None]
        evel[e_fl > 1] = 0.0
        if comp == "uz":
            ref, rflag = independent_center_u(
                evel[:, :-1, :], evel[:, 1:, :],
                np.repeat(e_fl[:, :-1, None], evel.shape[-1], -1),
                np.repeat(e_fl[:, 1:, None], evel.shape[-1], -1))
        else:
            ref, rflag = independent_center_u(
                evel[:-1, :, :], evel[1:, :, :],
                np.repeat(e_fl[:-1, :, None], evel.shape[-1], -1),
                np.repeat(e_fl[1:, :, None], evel.shape[-1], -1))
        got = z[f"{comp}_Re"].values + 1j * z[f"{comp}_Im"].values
        want = ref[interior].astype(np.complex64)
        check(np.allclose(got, want, rtol=1e-6, atol=1e-9),
              f"{comp}: derived values mismatch independent recompute "
              f"(max|d|={np.abs(got - want).max():.3e})")
        check(np.array_equal(z[flag_name].values, rflag[interior][..., 0]),
              f"{comp}: flags mismatch independent recompute")
        print(f"[3/5] {comp} derived layer (full scan incl. boundary "
              f"column/row): " + ("OK" if not FAILURES else "FAIL"))

    return finish(z, repo_root)


def finish(z, repo_root: Path) -> int:
    check(z.attrs.get("tide_store_schema") == P.SCHEMA_VERSION, "schema attr missing")
    check(z.attrs.get("quantization_rule") == P.QUANTIZATION_VERSION,
          "quantization attr missing")
    # round 10, finding 3: the recorded content hashes must correspond to
    # the recorded commit's blobs — proving the commit reproduces the
    # pipeline inputs, not merely that a commit was recorded
    import hashlib
    import json as _json
    import subprocess
    commit = z.attrs.get("pipeline_git_commit", "")
    src_sha = _json.loads(z.attrs.get("pipeline_source_sha256", "{}"))
    check(bool(commit) and bool(src_sha), "provenance commit/hashes missing")
    for relpath, sha in src_sha.items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relpath}"],
            cwd=repo_root, capture_output=True,
        )
        check(blob.returncode == 0
              and hashlib.sha256(blob.stdout).hexdigest() == sha,
              f"provenance hash mismatch vs commit blob: {relpath}")
    print(f"[4/5] provenance attrs + {len(src_sha)} hash<->commit-blob checks: "
          + ("OK" if not FAILURES else "FAIL"))

    if FAILURES:
        print("FAIL verify_against_netcdf")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[5/5] PASS verify_against_netcdf (tolerance 0 on source layer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
