#!/usr/bin/env python
"""Stage 0 source inspection for TPXO10-atlas-v2 (spec v0.3.0 §4 Stage 0 / Gate G0).

Asserts, against the local source directory:
  1. File inventory: grid file + 15 h_ + 15 u_ files (31 files, exact names).
  2. Per-file schema (spec §2): dims nx=10800/ny=5401, variable names,
     dtypes and units exactly as documented.
  3. D12 node-offset convention (the centering formulas depend on it):
       lon_u = lon_z - delta/2   (u on the WESTERN cell edge)
       lat_u = lat_z
       lon_v = lon_z
       lat_v = lat_z - delta/2   (v on the SOUTHERN cell edge)
     with delta = 1/30 deg, uniform spacing, and periodic longitude span
     (nx * delta == 360).
  4. SHA-256 manifest of all 31 files (provenance input for D8), written to
     dev_tpxo10/manifests/tpxo10_atlas_v2.sha256.json.

Output convention: one line per check group, verbose only on failure,
exit code 0/1 (spec D10).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

DELTA = 1.0 / 30.0
NX, NY = 10800, 5401
CONSTITUENTS = [
    "2n2", "k1", "k2", "m2", "m4", "mf", "mm", "mn4",
    "ms4", "n2", "o1", "p1", "q1", "s1", "s2",
]
GRID_FILE = "grid_tpxo10atlas_v2.nc"

# variable -> (dimensions, dtype, units); None = don't check units
GRID_VARS = {
    "lon_z": (("nx",), "float64", "degree_east"),
    "lat_z": (("ny",), "float64", "degree_north"),
    "lon_u": (("nx",), "float64", "degree_east"),
    "lat_u": (("ny",), "float64", "degree_north"),
    "lon_v": (("nx",), "float64", "degree_east"),
    "lat_v": (("ny",), "float64", "degree_north"),
    "hz": (("nx", "ny"), "float32", "meter"),
    "hu": (("nx", "ny"), "float32", "meter"),
    "hv": (("nx", "ny"), "float32", "meter"),
}
H_VARS = {
    "con": (("nct",), "|S1", None),
    "lon_z": (("nx",), "float64", "degree_east"),
    "lat_z": (("ny",), "float64", "degree_north"),
    "hRe": (("nx", "ny"), "int32", "millimeter"),
    "hIm": (("nx", "ny"), "int32", "millimeter"),
}
U_VARS = {
    "con": (("nct",), "|S1", None),
    "lon_u": (("nx",), "float64", "degree_east"),
    "lat_u": (("ny",), "float64", "degree_north"),
    "lon_v": (("nx",), "float64", "degree_east"),
    "lat_v": (("ny",), "float64", "degree_north"),
    "uRe": (("nx", "ny"), "int32", "centimeter^2/sec"),
    "uIm": (("nx", "ny"), "int32", "centimeter^2/sec"),
    "vRe": (("nx", "ny"), "int32", "centimeter^2/sec"),
    "vIm": (("nx", "ny"), "int32", "centimeter^2/sec"),
}


def expected_files() -> list[str]:
    files = [GRID_FILE]
    files += [f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONSTITUENTS]
    files += [f"u_{c}_tpxo10_atlas_30_v2.nc" for c in CONSTITUENTS]
    return files


def check_inventory(source: Path) -> list[str]:
    errors = []
    present = {p.name for p in source.glob("*.nc")}
    expected = set(expected_files())
    if missing := sorted(expected - present):
        errors.append(f"missing files: {missing}")
    if extra := sorted(present - expected):
        errors.append(f"unexpected files: {extra}")
    return errors


def check_file_schema(path: Path, spec: dict) -> list[str]:
    errors = []
    with netCDF4.Dataset(path) as ds:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        if dims.get("nx") != NX or dims.get("ny") != NY:
            errors.append(f"{path.name}: dims {dims}, expected nx={NX} ny={NY}")
        for name, (vdims, dtype, units) in spec.items():
            if name not in ds.variables:
                errors.append(f"{path.name}: variable {name} missing")
                continue
            var = ds.variables[name]
            if var.dimensions != vdims:
                errors.append(f"{path.name}:{name} dims {var.dimensions} != {vdims}")
            if str(var.dtype) != dtype:
                errors.append(f"{path.name}:{name} dtype {var.dtype} != {dtype}")
            if units is not None and getattr(var, "units", None) != units:
                errors.append(
                    f"{path.name}:{name} units {getattr(var, 'units', None)!r} != {units!r}"
                )
        if extra := sorted(set(ds.variables) - set(spec)):
            errors.append(f"{path.name}: unexpected variables {extra}")
    return errors


def check_node_offsets(
    lon_z: np.ndarray,
    lat_z: np.ndarray,
    lon_u: np.ndarray,
    lat_u: np.ndarray,
    lon_v: np.ndarray,
    lat_v: np.ndarray,
    delta: float = DELTA,
    atol: float = 1e-6,
) -> list[str]:
    """D12 node-offset convention checks. Pure function for unit testing."""
    errors = []
    if not np.allclose(np.diff(lon_z), delta, atol=atol):
        errors.append("lon_z spacing is not uniformly delta")
    if not np.allclose(np.diff(lat_z), delta, atol=atol):
        errors.append("lat_z spacing is not uniformly delta")
    if not np.allclose(lon_u, lon_z - delta / 2.0, atol=atol):
        errors.append("lon_u != lon_z - delta/2 (u not on western edge)")
    if not np.allclose(lat_u, lat_z, atol=atol):
        errors.append("lat_u != lat_z")
    if not np.allclose(lon_v, lon_z, atol=atol):
        errors.append("lon_v != lon_z")
    if not np.allclose(lat_v, lat_z - delta / 2.0, atol=atol):
        errors.append("lat_v != lat_z - delta/2 (v not on southern edge)")
    if not np.isclose(len(lon_z) * delta, 360.0, atol=atol):
        errors.append("longitude span nx*delta != 360 (grid not periodic)")
    return errors


def sha256_of(path: Path, bufsize: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, default=repo_root / "data_src" / "TPXO10_atlas_v2"
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=repo_root / "dev_tpxo10" / "manifests" / "tpxo10_atlas_v2.sha256.json",
    )
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 manifest")
    args = ap.parse_args()

    failures: list[str] = []

    errs = check_inventory(args.source)
    failures += errs
    print(f"[1/4] inventory: {'OK 31 files' if not errs else 'FAIL'}")

    if not failures:
        n_err = 0
        for fname in expected_files():
            if fname == GRID_FILE:
                spec = GRID_VARS
            elif fname.startswith("h_"):
                spec = H_VARS
            else:
                spec = U_VARS
            errs = check_file_schema(args.source / fname, spec)
            failures += errs
            n_err += len(errs)
        print(f"[2/4] schema (31 files): {'OK' if n_err == 0 else 'FAIL'}")

        with netCDF4.Dataset(args.source / GRID_FILE) as g:
            errs = check_node_offsets(
                g["lon_z"][:].filled(np.nan),
                g["lat_z"][:].filled(np.nan),
                g["lon_u"][:].filled(np.nan),
                g["lat_u"][:].filled(np.nan),
                g["lon_v"][:].filled(np.nan),
                g["lat_v"][:].filled(np.nan),
            )
        failures += errs
        print(f"[3/4] D12 node offsets: {'OK (u=west edge, v=south edge, periodic lon)' if not errs else 'FAIL'}")

    if args.no_hash:
        print("[4/4] sha256 manifest: SKIPPED (--no-hash)")
    elif not failures:
        manifest = {}
        for fname in expected_files():
            p = args.source / fname
            manifest[fname] = {"sha256": sha256_of(p), "bytes": p.stat().st_size}
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        total = sum(m["bytes"] for m in manifest.values())
        print(f"[4/4] sha256 manifest: OK 31 files, {total} bytes -> {args.manifest_out}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS inspect_source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
