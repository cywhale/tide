"""Stage 3 step 4 (round 24 F1): API-level dual-schema golden / rollback.

Drives the SAME migrated runtime via TestClient against a synthetic
legacy (tpxo9-shape) store and a synthetic tpxo10 store that encode the
IDENTICAL quantized hc field, so the API responses must match across
schemas (rollback parity), plus tpxo10 response-shape / missing / cap
behavior. Covers /api/tide point, /api/tide bbox, /api/tide/const.
Dask disabled + USNO mocked -> fast, no network.
"""
import importlib

import numpy as np
import pytest
import xarray as xr

from src import store_adapter as SA

CONS = ["m2", "s2", "k1", "o1"]
NC = len(CONS)
GRID = 1.0 / 30.0
NLAT, NLON = 8, 8
LAT0, LON0 = 10.0, 120.0


def _hc_field(seed=7):
    """Deterministic complex hc field (metres), QUANTIZED to the tpxo10
    int32-mm grid so both schemas can encode it exactly."""
    rng = np.random.default_rng(seed)
    re_mm = rng.integers(50, 3000, (NLAT, NLON, NC)).astype(np.int32)
    im_mm = rng.integers(-3000, 3000, (NLAT, NLON, NC)).astype(np.int32)
    return re_mm, im_mm


def _build_stores(tmp_path):
    lat = LAT0 + GRID * np.arange(NLAT)
    lon = LON0 + GRID * np.arange(NLON)
    re_mm, im_mm = _hc_field()
    hc = 1e-3 * (re_mm + 1j * im_mm)              # metres, the shared truth
    amp = np.abs(hc)
    ph = np.rad2deg(-np.angle(hc)) % 360.0

    # one invalid node (flag==2) for the tpxo10 missing-behavior checks
    inv = (2, 3)

    # legacy: z/u/v amp+ph (u/v in cm/s -> derive from hc*100 so that the
    # adapter's u/v hc matches tpxo10's uz/vz*100 convention)
    la = {v: amp.copy() for v in "zuv"}
    lp = {v: ph.copy() for v in "zuv"}
    for v in "uv":                                # u/v amp in cm/s
        la[v] = amp.copy() * 100.0
    # legacy invalid: NaN amp (the legacy missing convention)
    for v in "zuv":
        la[v][inv] = np.nan
    legacy = xr.Dataset(
        {f"{v}_amp": (("lat", "lon", "constituents"), la[v]) for v in "zuv"}
        | {f"{v}_ph": (("lat", "lon", "constituents"), lp[v]) for v in "zuv"},
        coords={"lat": lat, "lon": lon, "constituents": np.array(CONS, "<U3")})
    lpath = tmp_path / "legacy.zarr"
    legacy.to_zarr(lpath, mode="w", consolidated=True)

    # tpxo10: z_Re/z_Im (mm); uz/vz = velocity m/s = hc (so *100 -> cm/s
    # matches legacy u/v amp). flags 0 except the invalid node.
    uz = hc.real.astype(np.float32); vz = hc.imag.astype(np.float32)
    zf = np.zeros((NLAT, NLON), np.uint8)
    zf[inv] = 2
    re2 = re_mm.copy(); im2 = im_mm.copy(); re2[inv] = 0; im2[inv] = 0
    uzr = hc.real.astype(np.float32).copy(); uzi = hc.imag.astype(np.float32).copy()
    uzr[inv] = 0; uzi[inv] = 0
    t10 = xr.Dataset(
        {"z_Re": (("lat_z", "lon_z", "constituents"), re2),
         "z_Im": (("lat_z", "lon_z", "constituents"), im2),
         "uz_Re": (("lat_z", "lon_z", "constituents"), uzr),
         "uz_Im": (("lat_z", "lon_z", "constituents"), uzi),
         "vz_Re": (("lat_z", "lon_z", "constituents"), uzr.copy()),
         "vz_Im": (("lat_z", "lon_z", "constituents"), uzi.copy()),
         "z_flag": (("lat_z", "lon_z"), zf),
         "uz_flag": (("lat_z", "lon_z"), zf.copy()),
         "vz_flag": (("lat_z", "lon_z"), zf.copy())},
        coords={"lat_z": lat, "lon_z": lon, "constituents": np.array(CONS, "<U3")},
        attrs={"tide_store_schema": SA.SCHEMA_TPXO10})
    tpath = tmp_path / "tpxo10.zarr"
    t10.to_zarr(tpath, mode="w", consolidated=True)
    return lpath, tpath, lat, lon, inv


def _client(store, monkeypatch):
    monkeypatch.setenv("TIDE_ZARR_PATH", str(store))
    monkeypatch.setenv("TIDE_DASK_DISABLE", "1")
    import src.tide_forecast as tf
    monkeypatch.setattr(tf, "_fetch_usno_oneday", lambda *a, **k: (None, ""))
    from fastapi.testclient import TestClient
    import tide_app
    importlib.reload(tide_app)
    return TestClient(tide_app.app)


@pytest.fixture
def stores(tmp_path):
    return _build_stores(tmp_path)


def _point(client, lon, lat):
    return client.get("/api/tide", params={
        "lon0": lon, "lat0": lat, "append": "z,u,v",
        "start": "2023-07-25", "end": "2023-07-25T03:00:00"})


# ---------- point: cross-schema identical ----------

def test_point_rollback_identical(stores, monkeypatch):
    lpath, tpath, lat, lon, inv = stores
    qlon, qlat = float(lon[4]), float(lat[4])
    with _client(lpath, monkeypatch) as cl:
        jl = _point(cl, qlon, qlat).json()
    with _client(tpath, monkeypatch) as ct:
        jt = _point(ct, qlon, qlat).json()
    assert set(jl) == set(jt)
    assert jl["z"] and jt["z"]
    assert np.allclose(jl["z"], jt["z"], rtol=0, atol=1e-2)   # cm, quantized
    assert np.allclose(jl["u"], jt["u"], rtol=0, atol=1e-2)


# ---------- bbox map: cross-schema identical shape + values ----------

def test_bbox_rollback_identical(stores, monkeypatch):
    lpath, tpath, lat, lon, inv = stores
    p = {"lon0": float(lon[0]), "lon1": float(lon[-1]),
         "lat0": float(lat[0]), "lat1": float(lat[-1]),
         "start": "2023-07-25T00:00:00", "append": "z,u,v", "sample": 1}
    with _client(lpath, monkeypatch) as cl:
        jl = cl.get("/api/tide", params=p).json()
    with _client(tpath, monkeypatch) as ct:
        jt = ct.get("/api/tide", params=p).json()
    assert set(jl) == set(jt)
    assert jl["longitude"] == jt["longitude"]
    assert jl["latitude"] == jt["latitude"]
    assert np.allclose(jl["z"], jt["z"], rtol=0, atol=1e-2)
    # the invalid node is filtered out in BOTH schemas (missing, not 0)
    assert len(jl["z"]) == len(jt["z"]) == NLAT * NLON - 1


# ---------- const multipoint: cross-schema + missing ----------

def test_const_rollback_and_missing(stores, monkeypatch):
    lpath, tpath, lat, lon, inv = stores
    valid_lon, valid_lat = float(lon[4]), float(lat[4])
    inv_lon, inv_lat = float(lon[inv[1]]), float(lat[inv[0]])
    p = {"lon": f"{valid_lon},{inv_lon}", "lat": f"{valid_lat},{inv_lat}",
         "append": "z", "constituent": "m2", "complex": "amp", "mode": "row"}
    with _client(lpath, monkeypatch) as cl:
        rl = cl.get("/api/tide/const", params=p).json()
    with _client(tpath, monkeypatch) as ct:
        rt = ct.get("/api/tide/const", params=p).json()
    # both: only the VALID point survives (invalid dropped, never 0)
    assert len(rl) == len(rt) == 1
    assert rl[0].keys() == rt[0].keys()
    assert rl[0]["m2_amp"] > 0 and rt[0]["m2_amp"] > 0
    assert np.isclose(rl[0]["m2_amp"], rt[0]["m2_amp"], rtol=0, atol=1e-2)


# ---------- tpxo10 cap behavior ----------

def test_tpxo10_cap_400(stores, monkeypatch):
    lpath, tpath, lat, lon, inv = stores
    monkeypatch.setenv("TIDE_MAX_BBOX_CELLS", "10")   # tiny cap
    with _client(tpath, monkeypatch) as ct:
        r = ct.get("/api/tide", params={
            "lon0": float(lon[0]), "lon1": float(lon[-1]),
            "lat0": float(lat[0]), "lat1": float(lat[-1]),
            "start": "2023-07-25T00:00:00", "sample": 1})
    assert r.status_code == 400
    assert "MAX_BBOX_CELLS" in r.json()["detail"]
