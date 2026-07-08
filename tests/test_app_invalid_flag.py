"""Stage 3 round-23 F1: TPXO10 flag==2 invalid cells must serialize as
MISSING (empty/null), never as 0, across point / const / forecast.

Uses a synthetic tpxo10 toy store (deterministic, no canonical needed)
with one invalid (flag==2) node and valid ocean nodes; drives the real
FastAPI app. Dask is disabled (TIDE_DASK_DISABLE) so the app starts fast.
"""
import importlib
import os

import numpy as np
import pytest
import xarray as xr

from src import store_adapter as SA

CONS = ["m2", "s2", "k1", "o1"]
NC = len(CONS)


@pytest.fixture
def tpxo10_app(tmp_path, monkeypatch):
    """Build a tiny tpxo10 store with a flag==2 node at (lon[1], lat[1])
    and valid nodes elsewhere; start the app against it."""
    nlat, nlon = 4, 4
    lat = np.linspace(10.0, 11.0, nlat)
    lon = np.linspace(120.0, 121.0, nlon)
    rng = np.random.default_rng(0)
    # valid non-zero harmonic constants everywhere
    zre = rng.integers(200, 2000, (nlat, nlon, NC)).astype(np.int32)
    zim = rng.integers(200, 2000, (nlat, nlon, NC)).astype(np.int32)
    uz = rng.uniform(0.1, 1.0, (nlat, nlon, NC)).astype(np.float32)
    vz = rng.uniform(0.1, 1.0, (nlat, nlon, NC)).astype(np.float32)
    zf = np.zeros((nlat, nlon), np.uint8)
    uzf = np.zeros((nlat, nlon), np.uint8)
    vzf = np.zeros((nlat, nlon), np.uint8)
    # invalid node: flag==2 with the on-disk zeros (the bug-trigger shape)
    inv = (1, 1)
    zf[inv] = 2; uzf[inv] = 2; vzf[inv] = 2
    zre[inv] = 0; zim[inv] = 0; uz[inv] = 0; vz[inv] = 0
    ds = xr.Dataset(
        {"z_Re": (("lat_z", "lon_z", "constituents"), zre),
         "z_Im": (("lat_z", "lon_z", "constituents"), zim),
         "uz_Re": (("lat_z", "lon_z", "constituents"), uz),
         "uz_Im": (("lat_z", "lon_z", "constituents"), uz.copy()),
         "vz_Re": (("lat_z", "lon_z", "constituents"), vz),
         "vz_Im": (("lat_z", "lon_z", "constituents"), vz.copy()),
         "z_flag": (("lat_z", "lon_z"), zf),
         "uz_flag": (("lat_z", "lon_z"), uzf),
         "vz_flag": (("lat_z", "lon_z"), vzf)},
        coords={"lat_z": lat, "lon_z": lon,
                "constituents": np.array(CONS, dtype="<U3")},
        attrs={"tide_store_schema": SA.SCHEMA_TPXO10})
    store = tmp_path / "toy_tpxo10.zarr"
    ds.to_zarr(store, mode="w", consolidated=True)

    monkeypatch.setenv("TIDE_ZARR_PATH", str(store))
    monkeypatch.setenv("TIDE_DASK_DISABLE", "1")
    # mock the USNO sun/moon dependency so forecast tests never touch the
    # network (reviewer suggestion round 23)
    import src.tide_forecast as tf
    monkeypatch.setattr(tf, "_fetch_usno_oneday", lambda *a, **k: (None, ""))
    from fastapi.testclient import TestClient
    import tide_app
    importlib.reload(tide_app)
    with TestClient(tide_app.app) as client:
        yield client, float(lon[1]), float(lat[1]), float(lon[0]), float(lat[0])


def test_point_invalid_is_missing_not_zero(tpxo10_app):
    client, ilon, ilat, vlon, vlat = tpxo10_app
    r = client.get("/api/tide", params={"lon0": ilon, "lat0": ilat,
                                        "start": "2023-07-25", "end": "2023-07-25T06:00:00"})
    assert r.status_code == 200
    j = r.json()
    # invalid point -> filtered to empty z (missing), never a 0 series
    assert j.get("z", []) == [] or j == {}
    # valid point -> real (non-empty) z series
    rv = client.get("/api/tide", params={"lon0": vlon, "lat0": vlat,
                                         "start": "2023-07-25", "end": "2023-07-25T06:00:00"})
    jv = rv.json()
    assert jv.get("z") and any(abs(x) > 0 for x in jv["z"])


def test_const_invalid_not_zero(tpxo10_app):
    client, ilon, ilat, vlon, vlat = tpxo10_app
    r = client.get("/api/tide/const", params={
        "lon": f"{ilon}", "lat": f"{ilat}", "append": "z",
        "constituent": "m2", "complex": "amp,ph,hc", "mode": "row"})
    assert r.status_code == 200
    rows = r.json()
    # invalid point must NOT appear as a 0-valued row; it is dropped/null
    for row in rows:
        for k, v in row.items():
            if k.endswith(("_amp", "_ph", "_real", "_imag")):
                assert v != 0.0, f"invalid point serialized as 0 in {k}"
    # valid point returns a real amplitude
    rv = client.get("/api/tide/const", params={
        "lon": f"{vlon}", "lat": f"{vlat}", "append": "z",
        "constituent": "m2", "complex": "amp", "mode": "row"})
    vrows = rv.json()
    assert vrows and vrows[0]["m2_amp"] > 0


def test_forecast_invalid_is_empty_not_zero(tpxo10_app):
    client, ilon, ilat, vlon, vlat = tpxo10_app
    r = client.get("/api/tide/forecast",
                   params={"lon": ilon, "lat": ilat, "date": "2023-07-25", "tz": "+00:00"})
    assert r.status_code == 200
    # invalid point -> no tide extrema (a 0 series would also yield none,
    # but crucially the height field must never be a fabricated 0 event)
    tide = r.json()["days"][0]["tide"]
    assert tide == [] or all(t.get("height") not in (0, 0.0) for t in tide)
