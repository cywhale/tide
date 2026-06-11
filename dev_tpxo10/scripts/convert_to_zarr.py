#!/usr/bin/env python
"""TPXO10-atlas-v2 NetCDF -> Zarr converter (spec v0.3.0 §3.1 schema).

Stage 1 regional mode: --region reads a haloed window from the global
source (D12 regional rule: compute on halo, write interior only, never
wrap at a regional edge), builds the two-layer store (§3.0) with the D12
uz/vz derived layer, and records full provenance (D8).

Example (Stage 1 prototype):
  uv run python scripts/convert_to_zarr.py \
      --region 104,151,-1,46 --chunks 113,113,15 \
      --out stores/tpxo10_proto.zarr
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

MASK_RULE = "valid iff node bathymetry > 0 AND not all constituents are (0+0j)"


def load_node_stack(source: Path, node: str, iw: slice, jw: slice):
    """Load haloed (lat, lon, ncons) int32 Re/Im stacks for one node type."""
    shape = (jw.stop - jw.start, iw.stop - iw.start, len(P.CONSTITUENTS))
    re = np.empty(shape, dtype=np.int32)
    im = np.empty_like(re)
    prefix, rvar, ivar = {
        "z": ("h", "hRe", "hIm"),
        "u": ("u", "uRe", "uIm"),
        "v": ("u", "vRe", "vIm"),
    }[node]
    for k, c in enumerate(P.CONSTITUENTS):
        with netCDF4.Dataset(source / f"{prefix}_{c}_tpxo10_atlas_30_v2.nc") as ds:
            re[..., k] = np.asarray(ds[rvar][iw, jw]).T  # (nx,ny) -> (lat,lon)
            im[..., k] = np.asarray(ds[ivar][iw, jw]).T
    return re, im


def finite_or_die(arr: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(arr)):
        raise P.PipelineError(f"{name}: non-finite values in derived layer")
    return arr


def convert(region, chunks, out, source: Path, repo_root: Path) -> None:
    lon0, lon1, lat0, lat1 = region
    cl_lat, cl_lon, cl_con = chunks
    grid = source / "grid_tpxo10atlas_v2.nc"

    with netCDF4.Dataset(grid) as g:
        axes = {n: np.asarray(g[n][:]) for n in
                ("lon_z", "lat_z", "lon_u", "lat_u", "lon_v", "lat_v")}

    j_int, i_int = P.region_to_index_window(lon0, lon1, lat0, lat1,
                                            axes["lon_z"], axes["lat_z"])
    jw, iw = P.haloed_window(j_int, i_int)
    H = P.HALO_CELLS
    nlat, nlon = j_int.stop - j_int.start, i_int.stop - i_int.start
    print(f"[1/6] window: interior lat[{j_int.start}:{j_int.stop}] "
          f"lon[{i_int.start}:{i_int.stop}] ({nlat}x{nlon}), halo={H}")

    with netCDF4.Dataset(grid) as g:
        h_haloed = {n: np.asarray(g[hv][iw, jw]).T.astype(np.float32)
                    for n, hv in (("z", "hz"), ("u", "hu"), ("v", "hv"))}

    interior = (slice(H, H + nlat), slice(H, H + nlon))
    truth, flags, haloed = {}, {}, {}
    for node in ("z", "u", "v"):
        re, im = load_node_stack(source, node, iw, jw)
        h = h_haloed[node]
        valid = P.compute_validity(h, re, im)
        flag = P.classify_flags(h, valid)
        lon_ax = axes[f"lon_{node}"][iw]
        lat_ax = axes[f"lat_{node}"][jw]
        P.inpaint_fill_inplace(lon_ax, lat_ax, re, im, valid, flag, context=node)
        # enforce flag-2 cells store exactly 0 (§3.0 invalid layer)
        inv = flag == P.FLAG_INVALID
        re[inv], im[inv] = 0, 0
        haloed[node] = (re, im, flag, h)
        truth[node] = (re[interior], im[interior])
        flags[node] = flag[interior]
        n_fill = int((flag[interior] == P.FLAG_FILLED).sum())
        print(f"[2/6] {node}-node truth layer: filled cells (interior) = {n_fill}")

    # D12 derived layer: edge velocities -> z-centered uz/vz
    re_u, im_u, fl_u, hu = haloed["u"]
    re_v, im_v, fl_v, hv = haloed["v"]
    eu = P.edge_velocity(re_u, im_u, hu, fl_u)
    ev = P.edge_velocity(re_v, im_v, hv, fl_v)
    uz_c, uz_f3 = P.center_u(eu, fl_u, wrap=False)
    vz_c, vz_f3 = P.center_v(ev, fl_v)
    uz_c = uz_c[interior]
    vz_c = vz_c[interior]
    uz_flag = P.collapse_constituent_flags(uz_f3[interior], "uz")
    vz_flag = P.collapse_constituent_flags(vz_f3[interior], "vz")
    uz_re = finite_or_die(uz_c.real, "uz_Re").astype(np.float32)
    uz_im = finite_or_die(uz_c.imag, "uz_Im").astype(np.float32)
    vz_re = finite_or_die(vz_c.real, "vz_Re").astype(np.float32)
    vz_im = finite_or_die(vz_c.imag, "vz_Im").astype(np.float32)
    print(f"[3/6] derived uz/vz: flag0={(uz_flag == 0).sum()} "
          f"flag1={(uz_flag == 1).sum()} flag2={(uz_flag == 2).sum()} (uz, interior)")

    manifest_path = repo_root / "dev_tpxo10" / "manifests" / "tpxo10_atlas_v2.sha256.json"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    import pyTMD
    attrs = {
        "tide_store_schema": P.SCHEMA_VERSION,
        "source_model": "TPXO10-atlas-v2 (OSU, registered academic license)",
        "source_manifest_sha256": json.loads(manifest_path.read_text()),
        "pipeline_git_commit": commit,
        "pipeline_pyTMD_version": pyTMD.version.full_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "region_request": list(region),
        "interior_index_window": [j_int.start, j_int.stop, i_int.start, i_int.stop],
        "interior_bounds_lonlat": [
            float(axes["lon_z"][i_int][0]), float(axes["lon_z"][i_int][-1]),
            float(axes["lat_z"][j_int][0]), float(axes["lat_z"][j_int][-1]),
        ],
        "halo_cells": P.HALO_CELLS,
        "mask_rule": MASK_RULE,
        "inpaint_method": "pyTMD.interpolate.inpaint (Garcia 2010 DCT-PLS impl)",
        "inpaint_params": json.dumps(P.INPAINT_PARAMS),
        "inpaint_band_dmax_cells": P.DMAX_FILL_CELLS,
        "quantization_rule": P.QUANTIZATION_VERSION,
        "depth_clamp": "none (G1 sign-off: h<=0 invalid, positive depth divides as-is)",
        "flag_semantics": "0=source(bit-exact) 1=derived(inpaint/one-sided/inpainted-edge) 2=invalid(0)",
        "unit_conversions": (
            "z: hc[m]=1e-3*(z_Re+1j*z_Im); "
            "u: U[m^2/s]=1e-4*(u_Re+1j*u_Im), u[m/s]=U/hu (per edge); "
            "uz/vz: velocity m/s (D12 centering), no runtime conversion"
        ),
    }

    cz = {"lat_z": axes["lat_z"][j_int], "lon_z": axes["lon_z"][i_int],
          "lat_u": axes["lat_u"][j_int], "lon_u": axes["lon_u"][i_int],
          "lat_v": axes["lat_v"][j_int], "lon_v": axes["lon_v"][i_int],
          "constituents": np.array(P.CONSTITUENTS, dtype="<U3")}
    dims = {"z": ("lat_z", "lon_z"), "u": ("lat_u", "lon_u"), "v": ("lat_v", "lon_v")}
    data = {}
    for node in ("z", "u", "v"):
        d2, d3 = dims[node], dims[node] + ("constituents",)
        data[f"{node}_Re"] = (d3, truth[node][0])
        data[f"{node}_Im"] = (d3, truth[node][1])
        data[f"{node}_flag"] = (d2, flags[node])
        data[f"h{node}"] = (d2, h_haloed[node][interior])
    zdims2, zdims3 = dims["z"], dims["z"] + ("constituents",)
    data.update({
        "uz_Re": (zdims3, uz_re), "uz_Im": (zdims3, uz_im),
        "vz_Re": (zdims3, vz_re), "vz_Im": (zdims3, vz_im),
        "uz_flag": (zdims2, uz_flag), "vz_flag": (zdims2, vz_flag),
    })
    ds = xr.Dataset(data, coords=cz, attrs=attrs)

    comp = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    enc = {}
    for name, var in ds.data_vars.items():
        c = (cl_lat, cl_lon, cl_con)[: var.ndim] if var.ndim == 3 else (cl_lat, cl_lon)
        enc[name] = {"chunks": c, "compressor": comp}
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/6] writing {out} (chunks {chunks}, Blosc/lz4 c5 shuffle)")
    ds.to_zarr(out, mode="w", encoding=enc, consolidated=True, zarr_format=2)
    print("[5/6] consolidated zarr v2 written")
    print(f"[6/6] PASS convert_to_zarr region={region}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", type=str, default="104,151,-1,46",
                    help="lon0,lon1,lat0,lat1 (Stage 1 output region)")
    ap.add_argument("--chunks", type=str, default="113,113,15")
    ap.add_argument("--out", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--source", type=Path,
                    default=repo_root / "data_src" / "TPXO10_atlas_v2")
    args = ap.parse_args()
    region = tuple(float(x) for x in args.region.split(","))
    chunks = tuple(int(x) for x in args.chunks.split(","))
    convert(region, chunks, args.out, args.source, repo_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
