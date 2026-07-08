#!/usr/bin/env python
"""Stage 1 edge-depth / velocity-outlier survey (G1 kickoff sign-off:
must run BEFORE the converter writes data; informs the depth-clamp
go/no-go — STOP condition if a clamp turns out to be needed).

Over the Stage 1 output region, for each current node type (u, v):
  - distribution of positive edge depths at §3.2-valid nodes
    (min, P0.1/P1/P5, counts below 1/2/5 m)
  - per-edge velocity amplitude |1e-4 * (Re + i*Im) / h| per constituent
    at valid nodes: global max, P99.99, top-10 outliers with coordinates,
    depth and constituent

Decision aid: tidal currents above ~5 m/s do not occur physically even in
extreme channels; unclamped division producing larger values would argue
for a clamp policy (=> STOP at G1 per sign-off).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

REGION = (104.0, 151.0, -1.0, 46.0)  # spec Stage 1 output region


def survey_node(
    source: Path, node: str, jw: slice, iw: slice,
    lon_axis: np.ndarray, lat_axis: np.ndarray,
) -> dict:
    grid = source / "grid_tpxo10atlas_v2.nc"
    hvar = {"u": "hu", "v": "hv"}[node]
    with netCDF4.Dataset(grid) as g:
        h = np.asarray(g[hvar][iw, jw]).T.astype(np.float64)  # (lat, lon)

    ncons = len(P.CONSTITUENTS)
    re = np.empty(h.shape + (ncons,), dtype=np.int32)
    im = np.empty_like(re)
    for k, c in enumerate(P.CONSTITUENTS):
        with netCDF4.Dataset(source / f"u_{c}_tpxo10_atlas_30_v2.nc") as ds:
            re[..., k] = np.asarray(ds[f"{node}Re"][iw, jw]).T
            im[..., k] = np.asarray(ds[f"{node}Im"][iw, jw]).T

    valid = P.compute_validity(h, re, im)
    hv = h[valid]
    depth_stats = {
        "n_valid": int(valid.sum()),
        "min_h": float(hv.min()),
        "P0.1": float(np.percentile(hv, 0.1)),
        "P1": float(np.percentile(hv, 1)),
        "P5": float(np.percentile(hv, 5)),
        "n_h_lt_1m": int((hv < 1).sum()),
        "n_h_lt_2m": int((hv < 2).sum()),
        "n_h_lt_5m": int((hv < 5).sum()),
    }

    speed = (
        P.TRANSPORT_SCALE
        * np.abs(re[valid].astype(np.float64) + 1j * im[valid].astype(np.float64))
        / hv[:, np.newaxis]
    )  # (n_valid, ncons) m/s amplitude per constituent
    jj, ii = np.nonzero(valid)
    flat = speed.ravel()
    order = np.argsort(flat)[::-1][:10]
    outliers = []
    for idx in order:
        p, k = divmod(int(idx), ncons)
        outliers.append({
            "speed_m_s": round(float(flat[idx]), 3),
            "con": P.CONSTITUENTS[k],
            "lon": round(float(lon_axis[ii[p]]), 4),
            "lat": round(float(lat_axis[jj[p]]), 4),
            "h_m": round(float(hv[p]), 2),
        })
    vel_stats = {
        "max_speed_m_s": float(flat.max()),
        "P99.99_m_s": float(np.percentile(flat, 99.99)),
        "outliers_top10": outliers,
    }
    return {"depth": depth_stats, "velocity": vel_stats}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=repo_root / "data_src" / "TPXO10_atlas_v2")
    ap.add_argument("--max-plausible", type=float, default=5.0,
                    help="m/s; above this, a clamp policy discussion (STOP) is triggered")
    args = ap.parse_args()

    with netCDF4.Dataset(args.source / "grid_tpxo10atlas_v2.nc") as g:
        lon_z = np.asarray(g["lon_z"][:])
        lat_z = np.asarray(g["lat_z"][:])
        lon_u, lat_u = np.asarray(g["lon_u"][:]), np.asarray(g["lat_u"][:])
        lon_v, lat_v = np.asarray(g["lon_v"][:]), np.asarray(g["lat_v"][:])
    jw, iw = P.region_to_index_window(*REGION, lon_z, lat_z)

    worst = 0.0
    for node, lon_ax, lat_ax in (("u", lon_u, lat_u), ("v", lon_v, lat_v)):
        # slice the global axes to the window: survey indices are window-relative
        r = survey_node(args.source, node, jw, iw, lon_ax[iw], lat_ax[jw])
        d, v = r["depth"], r["velocity"]
        print(f"== {node}-nodes (region {REGION}, n_valid={d['n_valid']}) ==")
        print(f"  depth: min={d['min_h']:.2f}m P0.1={d['P0.1']:.2f} P1={d['P1']:.2f} "
              f"P5={d['P5']:.2f} | h<1m:{d['n_h_lt_1m']} h<2m:{d['n_h_lt_2m']} h<5m:{d['n_h_lt_5m']}")
        print(f"  speed: max={v['max_speed_m_s']:.3f} m/s  P99.99={v['P99.99_m_s']:.3f} m/s")
        for o in v["outliers_top10"]:
            print(f"    {o['speed_m_s']:>7.3f} m/s  {o['con']:>4}  "
                  f"({o['lon']}, {o['lat']})  h={o['h_m']}m")
        worst = max(worst, v["max_speed_m_s"])

    if worst > args.max_plausible:
        print(f"FAIL clamp-decision: max speed {worst:.3f} m/s > "
              f"{args.max_plausible} m/s — STOP at G1, clamp policy needs re-sign-off")
        return 1
    print(f"PASS no-clamp policy: max speed {worst:.3f} m/s <= {args.max_plausible} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
