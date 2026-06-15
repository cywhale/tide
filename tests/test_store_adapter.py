"""Stage 3 store-adapter unit tests (spec D9 + §7.1).

Synthetic toy stores for BOTH schemas — no production data required.
Covers: get_zarr_path env behavior; fail-closed schema detection;
raw-read (mask_and_scale=False) requirement; hc unit/convention
equivalence between schemas; tpxo10 flag-2 masking; lazy selection
(materialize only the selected subset).
"""
import numpy as np
import numpy.ma as ma
import pytest
import xarray as xr

from src import store_adapter as SA

CONS = ["m2", "s2", "k1"]
NLAT, NLON, NC = 6, 8, len(CONS)


def _legacy_store(tmp_path):
    """TPXO9-shape store: z/u/v amp+ph on lon/lat, no schema attr."""
    rng = np.random.default_rng(1)
    lat = np.linspace(0, 5, NLAT)
    lon = np.linspace(100, 107, NLON)
    amp = {v: rng.uniform(0.1, 2.0, (NLAT, NLON, NC)) for v in "zuv"}
    ph = {v: rng.uniform(0, 360, (NLAT, NLON, NC)) for v in "zuv"}
    data = {}
    for v in "zuv":
        data[f"{v}_amp"] = (("lat", "lon", "constituents"), amp[v])
        data[f"{v}_ph"] = (("lat", "lon", "constituents"), ph[v])
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon,
                                  "constituents": np.array(CONS, dtype="<U3")})
    p = tmp_path / "legacy.zarr"
    ds.to_zarr(p, mode="w", consolidated=True)
    return p, amp, ph


def _tpxo10_store(tmp_path, with_flag2=True):
    """tpxo10-cgrid-v1-shape store: z_Re/z_Im + uz/vz on lon_z/lat_z."""
    rng = np.random.default_rng(2)
    lat = np.linspace(0, 5, NLAT)
    lon = np.linspace(100, 107, NLON)
    re = {v: rng.integers(-2000, 2000, (NLAT, NLON, NC)).astype(np.int32)
          for v in ("z",)}
    im = {v: rng.integers(-2000, 2000, (NLAT, NLON, NC)).astype(np.int32)
          for v in ("z",)}
    uz_re = rng.uniform(-1, 1, (NLAT, NLON, NC)).astype(np.float32)
    uz_im = rng.uniform(-1, 1, (NLAT, NLON, NC)).astype(np.float32)
    vz_re = rng.uniform(-1, 1, (NLAT, NLON, NC)).astype(np.float32)
    vz_im = rng.uniform(-1, 1, (NLAT, NLON, NC)).astype(np.float32)
    zf = np.zeros((NLAT, NLON), np.uint8)
    uzf = np.zeros((NLAT, NLON), np.uint8)
    vzf = np.zeros((NLAT, NLON), np.uint8)
    if with_flag2:
        zf[0, 0] = 2; uzf[1, 1] = 2; vzf[2, 2] = 2
    data = {
        "z_Re": (("lat_z", "lon_z", "constituents"), re["z"]),
        "z_Im": (("lat_z", "lon_z", "constituents"), im["z"]),
        "uz_Re": (("lat_z", "lon_z", "constituents"), uz_re),
        "uz_Im": (("lat_z", "lon_z", "constituents"), uz_im),
        "vz_Re": (("lat_z", "lon_z", "constituents"), vz_re),
        "vz_Im": (("lat_z", "lon_z", "constituents"), vz_im),
        "z_flag": (("lat_z", "lon_z"), zf),
        "uz_flag": (("lat_z", "lon_z"), uzf),
        "vz_flag": (("lat_z", "lon_z"), vzf),
    }
    ds = xr.Dataset(data, coords={"lat_z": lat, "lon_z": lon,
                                  "constituents": np.array(CONS, dtype="<U3")},
                    attrs={"tide_store_schema": SA.SCHEMA_TPXO10})
    p = tmp_path / "tpxo10.zarr"
    ds.to_zarr(p, mode="w", consolidated=True)
    return p, re["z"], im["z"], dict(uz=(uz_re, uz_im), vz=(vz_re, vz_im)), \
        dict(z=zf, uz=uzf, vz=vzf)


# ---------- get_zarr_path ----------

def test_get_zarr_path_default_and_env(monkeypatch):
    monkeypatch.delenv("TIDE_ZARR_PATH", raising=False)
    assert SA.get_zarr_path() == SA.DEFAULT_ZARR_PATH
    assert SA.get_zarr_path("x.zarr") == "x.zarr"
    monkeypatch.setenv("TIDE_ZARR_PATH", "/custom/store.zarr")
    assert SA.get_zarr_path() == "/custom/store.zarr"
    assert SA.get_zarr_path("ignored.zarr") == "/custom/store.zarr"


# ---------- fail-closed detection ----------

def test_detect_tpxo10(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(p))
    assert isinstance(a, SA.Tpxo10Adapter)
    assert a.lon_name == "lon_z" and a.lat_name == "lat_z"
    assert a.constituents == CONS


def test_detect_legacy_only_with_full_var_set(tmp_path):
    p, *_ = _legacy_store(tmp_path)
    a = SA.open_store(str(p))
    assert isinstance(a, SA.LegacyAdapter)
    assert a.lon_name == "lon" and a.lat_name == "lat"


def test_missing_attr_incomplete_legacy_aborts(tmp_path):
    ds = xr.Dataset({"z_amp": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0, 1], "lon": [0, 1],
                            "constituents": np.array(["m2"], dtype="<U3")})
    with pytest.raises(SA.StoreSchemaError, match="incomplete legacy"):
        SA.make_adapter(ds)


def test_unknown_schema_aborts(tmp_path):
    ds = xr.Dataset(coords={"constituents": np.array(["m2"], dtype="<U3")},
                    attrs={"tide_store_schema": "tpxo10-cgrid-v2-future"})
    with pytest.raises(SA.StoreSchemaError, match="unknown tide_store_schema"):
        SA.make_adapter(ds)


def test_tpxo10_attr_but_missing_vars_aborts(tmp_path):
    ds = xr.Dataset({"z_Re": (("lat_z", "lon_z"), np.zeros((2, 2)))},
                    coords={"lat_z": [0, 1], "lon_z": [0, 1],
                            "constituents": np.array(["m2"], dtype="<U3")},
                    attrs={"tide_store_schema": SA.SCHEMA_TPXO10})
    with pytest.raises(SA.StoreSchemaError, match="missing required vars"):
        SA.make_adapter(ds)


# ---------- raw read (mask_and_scale=False) ----------

def test_open_store_uses_raw_read(tmp_path):
    """A tpxo10 store with fill_value attrs must NOT be CF-decoded:
    z_Re stays int32, flag==2 cells are not NaN-masked away."""
    p, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(p))
    assert a.ds["z_Re"].dtype == np.int32
    assert a.ds["z_flag"].dtype == np.uint8
    assert int((a.ds["z_flag"].values == 2).sum()) == 1


# ---------- hc unit/convention ----------

def test_legacy_hc_matches_amp_ph_formula(tmp_path):
    p, amp, ph = _legacy_store(tmp_path)
    a = SA.open_store(str(p))
    sub = a.sel_point(amp_lon := a.lon[3], a.lat[2], tol=1.0)
    hc = a.hc(sub, "z")
    j, i = 2, 3
    expect = amp["z"][j, i] * np.exp(-1j * ph["z"][j, i] * np.pi / 180.0)
    assert np.allclose(np.asarray(hc), expect, rtol=1e-12)


def test_tpxo10_hc_units_z_metres_uv_cms(tmp_path):
    p, zre, zim, uvw, _ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    j, i = 2, 3
    hz = np.asarray(a.hc(sub, "z"))
    assert np.allclose(hz, 1e-3 * (zre[j, i] + 1j * zim[j, i]), rtol=1e-6)
    hu = np.asarray(a.hc(sub, "u"))
    ure, uim = uvw["uz"]
    assert np.allclose(hu, 100.0 * (ure[j, i] + 1j * uim[j, i]), rtol=1e-5)


def test_amp_ph_round_trips_to_hc(tmp_path):
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    amp, ph = a.amp_ph(sub, "z")
    hc = amp * np.exp(-1j * ph * np.pi / 180.0)
    assert np.allclose(hc, np.asarray(a.hc(sub, "z")), rtol=1e-9)


# ---------- flag-2 masking ----------

def test_tpxo10_flag2_masks_hc(tmp_path):
    p, *_ , flags = _tpxo10_store(tmp_path, with_flag2=True)
    a = SA.open_store(str(p))
    # uz_flag[1,1]==2 -> u hc fully masked at that node
    sub = a.sel_bbox(a.lon[0], a.lon[-1], a.lat[0], a.lat[-1])
    hu = a.hc(sub, "u")
    assert ma.is_masked(hu)
    assert bool(np.all(hu.mask[1, 1, :]))
    assert not bool(hu.mask[0, 0, :].any())  # a valid node stays unmasked


# ---------- lazy selection ----------

def test_selection_is_lazy(tmp_path):
    """sel_* must not materialize the whole store; the returned subset is
    still dask/lazy until .values is taken on the small selection."""
    p, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(p), chunks={})  # force dask-backed
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    var = sub["z_Re"]
    assert var.chunks is not None  # still lazy after selection
    assert var.size == NC  # selection reduced to one node x constituents


def test_subsample_and_coords(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(p))
    sub = a.sel_bbox(a.lon[0], a.lon[-1], a.lat[0], a.lat[-1])
    sm = a.subsample(sub, 2)
    glon, glat = a.coord_values(sm)
    assert len(glon) == len(range(0, NLON, 2))
    assert len(glat) == len(range(0, NLAT, 2))
