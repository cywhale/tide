"""Stage 3 step 4: store-path audit gate + dual-schema rollback/API golden.

- Path-audit gate: production runtime files must not hardcode a store path
  (`*.zarr` open); the only store path lives in the `get_zarr_path()`
  helper default constant. Legacy TPXO9 tooling is archived under
  `dev/legacy_tpxo9/`.
- Rollback/golden: the SAME migrated runtime serves both schemas through
  the adapter; legacy amp/ph round-trips to hc exactly (rollback parity),
  and the const/point output shape matches across schemas.
"""
import importlib
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src import store_adapter as SA

REPO = Path(__file__).resolve().parents[1]

# production runtime modules (NOT tests, NOT dev_tpxo10 conversion env,
# NOT the archived legacy tooling)
PROD_FILES = [
    REPO / "tide_app.py",
    REPO / "src" / "model_utils.py",
    REPO / "src" / "tide_forecast.py",
    REPO / "src" / "store_adapter.py",
    REPO / "src" / "query_planner.py",
    REPO / "src" / "config.py",
]

# an actual store-open with a hardcoded .zarr literal (not a comment / not
# the helper default constant)
_OPEN_ZARR_LITERAL = re.compile(r"open_zarr\(\s*['\"][^'\"]*\.zarr['\"]")


def test_no_hardcoded_store_open_in_production():
    offenders = []
    for f in PROD_FILES:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _OPEN_ZARR_LITERAL.search(line):
                offenders.append(f"{f.name}:{i}: {stripped}")
    assert offenders == [], (
        "production code opens a hardcoded store path (use get_zarr_path()): "
        + "; ".join(offenders))


def test_default_store_path_only_in_helper_constant():
    """The `data/tpxo10.zarr` default literal lives only in store_adapter's
    DEFAULT_ZARR_RELPATH constant among production modules."""
    hits = []
    for f in PROD_FILES:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "data/tpxo10.zarr" in line and not line.strip().startswith("#"):
                hits.append((f.name, i, line.strip()))
    assert len(hits) == 1 and hits[0][0] == "store_adapter.py", hits
    assert "DEFAULT_ZARR_RELPATH" in hits[0][2]


def test_legacy_tooling_archived():
    legacy = REPO / "dev" / "legacy_tpxo9"
    assert (legacy / "extract_parallel.py").exists()
    assert (legacy / "zarr_fillna_parallel.py").exists()
    assert (legacy / "README.md").exists()
    # the loose dev/ scripts are gone
    assert not (REPO / "dev" / "extract_parallel.py").exists()


# ---------- dual-schema rollback / golden ----------

CONS = ["m2", "s2", "k1"]
NC = len(CONS)


def _legacy_and_tpxo10_equivalent_point(tmp_path):
    """Build a legacy store and a tpxo10 store whose z node at (1,1)
    encodes the SAME complex harmonic constant, so both adapters must
    yield the same hc there."""
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([120.0, 121.0, 122.0])
    # chosen hc (metres) per constituent at node (1,1)
    hc_true = np.array([1.234 - 0.5j, 0.2 + 0.9j, -0.3 - 0.1j])

    # legacy: amp/ph
    amp = np.abs(hc_true)
    ph = np.rad2deg(-np.angle(hc_true)) % 360.0
    la = np.zeros((3, 3, NC)); lp = np.zeros((3, 3, NC))
    la[1, 1] = amp; lp[1, 1] = ph
    legacy = xr.Dataset(
        {f"{v}_amp": (("lat", "lon", "constituents"), la.copy()) for v in "zuv"}
        | {f"{v}_ph": (("lat", "lon", "constituents"), lp.copy()) for v in "zuv"},
        coords={"lat": lat, "lon": lon, "constituents": np.array(CONS, "<U3")})
    lpath = tmp_path / "legacy.zarr"; legacy.to_zarr(lpath, mode="w", consolidated=True)

    # tpxo10: z_Re/z_Im = hc/1e-3 (mm), uz/vz already velocity
    re = np.zeros((3, 3, NC), np.int32); im = np.zeros((3, 3, NC), np.int32)
    re[1, 1] = np.rint(hc_true.real / 1e-3); im[1, 1] = np.rint(hc_true.imag / 1e-3)
    z0f = np.zeros((3, 3), np.uint8)
    t10 = xr.Dataset(
        {"z_Re": (("lat_z", "lon_z", "constituents"), re),
         "z_Im": (("lat_z", "lon_z", "constituents"), im),
         "uz_Re": (("lat_z", "lon_z", "constituents"), np.zeros((3, 3, NC), np.float32)),
         "uz_Im": (("lat_z", "lon_z", "constituents"), np.zeros((3, 3, NC), np.float32)),
         "vz_Re": (("lat_z", "lon_z", "constituents"), np.zeros((3, 3, NC), np.float32)),
         "vz_Im": (("lat_z", "lon_z", "constituents"), np.zeros((3, 3, NC), np.float32)),
         "z_flag": (("lat_z", "lon_z"), z0f), "uz_flag": (("lat_z", "lon_z"), z0f.copy()),
         "vz_flag": (("lat_z", "lon_z"), z0f.copy())},
        coords={"lat_z": lat, "lon_z": lon, "constituents": np.array(CONS, "<U3")},
        attrs={"tide_store_schema": SA.SCHEMA_TPXO10})
    tpath = tmp_path / "tpxo10.zarr"; t10.to_zarr(tpath, mode="w", consolidated=True)
    return lpath, tpath, hc_true


def test_rollback_both_schemas_open_and_agree(tmp_path):
    lpath, tpath, hc_true = _legacy_and_tpxo10_equivalent_point(tmp_path)
    la = SA.open_store(str(lpath))
    ta = SA.open_store(str(tpath))
    assert isinstance(la, SA.LegacyAdapter) and isinstance(ta, SA.Tpxo10Adapter)

    lsub = la.sel_point(121.0, 11.0, tol=0.5)
    tsub = ta.sel_point(121.0, 11.0, tol=0.5)
    lhc = np.asarray(la.hc(lsub, "z"))
    thc = np.asarray(ta.hc(tsub, "z"))
    # legacy hc is exact; tpxo10 hc within int32 quantization (<= 0.5 mm)
    assert np.allclose(lhc, hc_true, rtol=0, atol=1e-12)
    assert np.allclose(thc, hc_true, rtol=0, atol=5e-4)
    # cross-schema agreement to the quantization budget
    assert np.allclose(lhc, thc, rtol=0, atol=5e-4)


def test_legacy_amp_ph_roundtrip_is_exact(tmp_path):
    """Rollback parity foundation: legacy amp/ph -> hc -> amp/ph is exact,
    so the legacy store served through the migrated runtime is byte-faithful."""
    lpath, _, _ = _legacy_and_tpxo10_equivalent_point(tmp_path)
    a = SA.open_store(str(lpath))
    sub = a.sel_point(121.0, 11.0, tol=0.5)
    amp, ph = a.amp_ph(sub, "z")
    hc = np.ma.filled(amp, np.nan) * np.exp(-1j * np.ma.filled(ph, np.nan) * np.pi / 180.0)
    assert np.allclose(hc, np.asarray(a.hc(sub, "z")), rtol=0, atol=1e-12)
