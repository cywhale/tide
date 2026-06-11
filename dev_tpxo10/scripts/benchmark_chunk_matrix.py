#!/usr/bin/env python
"""§7.5.1 chunk/open-mode matrix benchmark (blocking G1 evidence).

Full factorial: spatial chunk {113, 225, 450} x constituent chunk {5, 15}
x open-mode {dask 'auto', direct (chunks=None)} over workloads W1-W8, all
on real source-backed data inside the prototype's benchmark area
(105-150E / 0-45N). Variant stores are pure rechunk copies of the
canonical prototype (identical values; layout is the only variable).

Blocking instrumentation per cell (review round 2, finding 6):
  chunks touched, compressed bytes read, decompressed bytes (estimated as
  full-chunk size per touched chunk — upper bound, noted), dask task
  count (dask mode), peak RSS delta, cold first-access latency.

Cold-cache: tries macOS `purge`; if unavailable the cold round is labeled
`first-path-access` per spec §7.5.1 (the binding cold-cache gate then
runs on a Linux host where eviction is verifiable).

A tpxo9-baseline column runs the same workloads against the production
`data/tpxo9.zarr` (z/u/v amp+ph float64, chunks 113x113x8) to anchor the
D5 read-amplification constraint (point-query decompressed bytes <=
1.5x baseline).

Results: printed table + JSON at dev_tpxo10/benchmarks/chunk_matrix.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import xarray as xr
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpxo10_pipeline as P  # noqa: E402

SPATIAL = [113, 225, 450]
CONS_CHUNKS = [5, 15]
SUBSET5 = ["m2", "s2", "k1", "o1", "n2"]
M2 = "m2"
SEED = 20260611

# benchmark area inside the prototype interior (window-local indices are
# computed from the store coords at runtime)
BENCH_LON = (105.0, 150.0)
BENCH_LAT = (0.0, 45.0)


class CountingStore(zarr.DirectoryStore):
    """DirectoryStore wrapper counting chunk reads + compressed bytes."""

    META_SUFFIXES = (".zarray", ".zattrs", ".zgroup", ".zmetadata")

    def __init__(self, path):
        super().__init__(str(path))
        self.reset()

    def reset(self):
        self.chunks = 0
        self.compressed = 0
        self.keys = []

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if not key.endswith(self.META_SUFFIXES):
            self.chunks += 1
            self.compressed += len(v)
            self.keys.append(key)
        return v


def decompressed_bytes(keys, ds) -> int:
    total = 0
    for key in keys:
        var = key.split("/", 1)[0]
        if var in ds.variables:
            v = ds[var]
            cs = v.encoding.get("chunks") or v.shape
            total += int(np.prod(cs)) * v.dtype.itemsize
    return total


def rechunk_variant(canonical: Path, out: Path, cl: int, cc: int):
    src = xr.open_zarr(canonical, consolidated=True, decode_times=False)
    if out.exists():
        # round 11, finding 5: never silently reuse a stale variant —
        # verify provenance commit and chunk layout, else rebuild
        try:
            old = xr.open_zarr(out, consolidated=True, decode_times=False)
            same = (old.attrs.get("pipeline_git_commit")
                    == src.attrs.get("pipeline_git_commit")
                    and old["z_Re"].encoding.get("chunks") == (cl, cl, cc))
            old.close()
        except Exception:
            same = False
        if same:
            return
        import shutil
        shutil.rmtree(out)
        print(f"  (stale variant {out.name} rebuilt)")
    comp = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    enc = {}
    for name, var in src.data_vars.items():
        c = (cl, cl, cc)[: var.ndim] if var.ndim == 3 else (cl, cl)
        enc[name] = {"chunks": c, "compressor": comp}
    src.attrs["rechunked_from"] = str(canonical.name)
    src.load().to_zarr(out, mode="w", encoding=enc, consolidated=True,
                       zarr_format=2)


def try_purge() -> str:
    try:
        r = subprocess.run(["purge"], capture_output=True, timeout=120)
        return "cold-cache(purge)" if r.returncode == 0 else "first-path-access"
    except Exception:
        return "first-path-access"


def open_store(path, mode, counter):
    chunks = "auto" if mode == "dask" else None
    return xr.open_zarr(counter, consolidated=True, decode_times=False,
                        chunks=chunks)


def measure(fn, reps, counter, ds):
    times, task_counts = [], []
    proc = psutil.Process()
    counter.reset()
    rss0 = proc.memory_info().rss
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "compute"):
            graph = getattr(out.data if hasattr(out, "data") else out, "__dask_graph__", None)
            if graph is not None:
                task_counts.append(len(graph()))
            out = out.compute()
        _ = np.asarray(out)
        times.append(time.perf_counter() - t0)
    rss1 = proc.memory_info().rss
    return {
        "ms_median": round(1e3 * float(np.median(times)), 2),
        "ms_p95": round(1e3 * float(np.percentile(times, 95)), 2),
        "chunks": counter.chunks // reps,
        "MiB_compressed": round(counter.compressed / reps / 2**20, 2),
        "MiB_decompressed": round(
            decompressed_bytes(counter.keys, ds) / reps / 2**20, 2),
        "dask_tasks": int(np.median(task_counts)) if task_counts else None,
        # end-minus-start, NOT peak (round 11, finding 3); binding memory
        # evidence comes from the W8 process-tree sampler only
        "rss_end_minus_start_MiB": round((rss1 - rss0) / 2**20, 1),
    }


def workloads(ds, schema, rng):
    """Return {name: fn} for W1-W8, schema in {'tpxo10','tpxo9'}."""
    if schema == "tpxo10":
        lat, lon, conc = "lat_z", "lon_z", "constituents"
        zvars = ["z_Re", "z_Im"]
        cvars = ["uz_Re", "uz_Im", "vz_Re", "vz_Im"]
    else:
        lat, lon, conc = "lat", "lon", "constituents"
        zvars = ["z_amp", "z_ph"]
        cvars = ["u_amp", "u_ph", "v_amp", "v_ph"]
    lats, lons = ds[lat].values, ds[lon].values
    j0 = int(np.searchsorted(lats, BENCH_LAT[0]))
    j1 = int(np.searchsorted(lats, BENCH_LAT[1]))
    i0 = int(np.searchsorted(lons, BENCH_LON[0]))
    i1 = int(np.searchsorted(lons, BENCH_LON[1]))
    cons = [str(c) for c in ds[conc].values]
    k5 = [cons.index(c) for c in SUBSET5]
    km2 = [cons.index(M2)]
    pj = rng.integers(j0, j1, size=64)
    pi = rng.integers(i0, i1, size=64)
    state = {"n": 0}

    def point(kidx=None, allvars=True):
        n = state["n"] = (state["n"] + 1) % 64
        j, i = int(pj[n]), int(pi[n])
        names = zvars + cvars if allvars else zvars
        sel = ds[names].isel({lat: j, lon: i})
        if kidx is not None:
            sel = sel.isel({conc: kidx})
        return sel.to_dataarray()

    def box(dlon, dlat, step=1, phase=0):
        # phase: bbox-origin offset in cells — the legacy decimation grid
        # follows the bbox origin (spec §3.1), so sampled workloads must
        # exercise a second, non-stride-aligned phase (round 11, finding 2)
        jj = slice(j0 + phase, j0 + phase + int(dlat * 30), step)
        ii = slice(i0 + phase, i0 + phase + int(dlon * 30), step)
        return ds[zvars + cvars].isel({lat: jj, lon: ii}).to_dataarray()

    return {
        "W1_point_all15": lambda: point(),
        "W2_point_m2": lambda: point(kidx=km2),
        "W3_point_5cons": lambda: point(kidx=k5),
        "W4_map_1deg": lambda: box(1, 1),
        "W5_map_10deg": lambda: box(10, 10),
        "W6_map_45deg_s5": lambda: box(45, 45, step=5),
        "W6b_map_45deg_s5_phase3": lambda: box(45, 45, step=5, phase=3),
        "W7_strip_45x5": lambda: box(45, 5),
        "W8_map_45deg_s1": lambda: box(45, 45),
    }


def run_cell(label, path, mode, schema, results, cold_label):
    rng = np.random.default_rng(SEED)
    counter = CountingStore(path)
    ds = open_store(path, mode, counter)
    wl = workloads(ds, schema, rng)
    cell = {}
    # cold first-access (whatever the cache state label says)
    counter.reset()
    t0 = time.perf_counter()
    _ = np.asarray(wl["W1_point_all15"]())
    cell["cold_W1_ms"] = round(1e3 * (time.perf_counter() - t0), 2)
    cell["cold_label"] = cold_label
    for name, fn in wl.items():
        reps = 2 if name.startswith(("W6", "W8")) else (5 if "map" in name or "strip" in name else 20)
        # warmup
        for _ in range(2):
            out = fn()
            _ = np.asarray(out.compute() if hasattr(out, "compute") else out)
        cell[name] = measure(fn, reps, counter, ds)
    results[label] = cell
    w1 = cell["W1_point_all15"]
    print(f"{label:34s} W1 {w1['ms_median']:7.2f}ms {w1['chunks']:3d}ch "
          f"{w1['MiB_decompressed']:8.2f}MiB | W6 "
          f"{cell['W6_map_45deg_s5']['ms_median']:9.2f}ms | W8 "
          f"{cell['W8_map_45deg_s1']['ms_median']:9.2f}ms "
          f"rssΔ{cell['W8_map_45deg_s1']['rss_end_minus_start_MiB']:.0f}MiB")
    ds.close()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", type=Path,
                    default=repo_root / "dev_tpxo10" / "stores" / "tpxo10_proto.zarr")
    ap.add_argument("--tpxo9", type=Path, default=repo_root / "data" / "tpxo9.zarr")
    ap.add_argument("--out", type=Path,
                    default=repo_root / "dev_tpxo10" / "benchmarks" / "chunk_matrix.json")
    args = ap.parse_args()

    variants = []
    for cl in SPATIAL:
        for cc in CONS_CHUNKS:
            out = args.canonical.parent / f"bench_{cl}x{cl}x{cc}.zarr"
            print(f"[rechunk] {out.name}")
            rechunk_variant(args.canonical, out, cl, cc)
            variants.append((cl, cc, out))

    cold_label = try_purge()
    print(f"cold-cache method: {cold_label}")
    results = {"_meta": {
        "seed": SEED, "cold_label": cold_label,
        "decompressed_note": "full-chunk upper bound",
        "latency_note": ("EXPLORATORY: fixed ordering, no interleaving, "
                         "W6/W8 reps=2; binding latency evidence is the W8 "
                         "harness / §7.5.2 interleaved benchmark"),
        "dask_note": ("'dask' mode = local threaded scheduler, "
                      "chunks='auto' (single-block); NOT representative of "
                      "the production distributed service — open-mode is "
                      "decided at the W8 harness"),
    }}
    for cl, cc, path in variants:
        for mode in ("direct", "dask"):
            try_purge()
            run_cell(f"tpxo10 {cl}x{cl}x{cc} {mode}", path, mode, "tpxo10",
                     results, cold_label)
    if args.tpxo9.exists():
        for mode in ("direct", "dask"):
            try_purge()
            run_cell(f"tpxo9-baseline 113x113x8 {mode}", args.tpxo9, mode,
                     "tpxo9", results, cold_label)
    else:
        print("WARN: tpxo9 baseline store not found; D5 ratio incomplete")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"PASS benchmark_chunk_matrix -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
