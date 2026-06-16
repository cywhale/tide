"""Stage 3 store-adapter unit tests (spec D9 + §7.1).

Synthetic toy stores for BOTH schemas — no production data required.
Covers: get_zarr_path env behavior (cwd-independent default); full
fail-closed schema/coord/dtype validation; reserved-kwarg guard;
raw-read requirement; hc unit/convention equivalence; tpxo10 flag-2
masking; vectorized multipoint selection (order/duplicate/tolerance);
dateline-wrap bbox coordinate-order parity; and selection-before-
materialization proven in DIRECT mode with a read-counting store.
"""
import numpy as np
import numpy.ma as ma
import pytest
import xarray as xr
import zarr

from src import store_adapter as SA

CONS = ["m2", "s2", "k1"]
NLAT, NLON, NC = 6, 8, len(CONS)


def _legacy_store(tmp_path, name="legacy.zarr"):
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
    p = tmp_path / name
    ds.to_zarr(p, mode="w", consolidated=True)
    return p, amp, ph


def _tpxo10_store(tmp_path, with_flag2=True, name="tpxo10.zarr",
                  lon=None, chunks=None):
    rng = np.random.default_rng(2)
    lat = np.linspace(0, 5, NLAT)
    if lon is None:
        lon = np.linspace(100, 107, NLON)
    zre = rng.integers(-2000, 2000, (NLAT, NLON, NC)).astype(np.int32)
    zim = rng.integers(-2000, 2000, (NLAT, NLON, NC)).astype(np.int32)
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
        "z_Re": (("lat_z", "lon_z", "constituents"), zre),
        "z_Im": (("lat_z", "lon_z", "constituents"), zim),
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
    enc = None
    if chunks:
        enc = {k: {"chunks": chunks if ds[k].ndim == 3 else chunks[:2]}
               for k in ds.data_vars}
    p = tmp_path / name
    ds.to_zarr(p, mode="w", consolidated=True, encoding=enc)
    return p, zre, zim, dict(uz=(uz_re, uz_im), vz=(vz_re, vz_im)), \
        dict(z=zf, uz=uzf, vz=vzf)


# ---------- get_zarr_path (F8) ----------

def test_get_zarr_path_default_is_repo_rooted(monkeypatch):
    monkeypatch.delenv("TIDE_ZARR_PATH", raising=False)
    p = SA.get_zarr_path()
    assert p.endswith("data/tpxo10.zarr")
    assert p.startswith("/")  # absolute (repo-root resolved, cwd-independent)


def test_get_zarr_path_env_verbatim(monkeypatch):
    monkeypatch.setenv("TIDE_ZARR_PATH", "/custom/store.zarr")
    assert SA.get_zarr_path() == "/custom/store.zarr"
    assert SA.get_zarr_path("ignored.zarr") == "/custom/store.zarr"


# ---------- reserved-kwarg guard (F2) ----------

def test_open_store_rejects_reserved_kwargs(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    for bad in ({"mask_and_scale": True}, {"chunks": {}}, {"decode_times": True}):
        with pytest.raises(ValueError, match="binding params|fixed by contract"):
            SA.open_store(str(p), **bad)
    with pytest.raises(ValueError, match="unsupported kwargs"):
        SA.open_store(str(p), engine="zarr")


# ---------- fail-closed detection (F1) ----------

def test_detect_tpxo10_and_legacy(tmp_path):
    pt, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(pt))
    assert isinstance(a, SA.Tpxo10Adapter) and a.lon_name == "lon_z"
    pl, *_ = _legacy_store(tmp_path)
    b = SA.open_store(str(pl))
    assert isinstance(b, SA.LegacyAdapter) and b.lon_name == "lon"


def test_tpxo10_missing_z_flag_aborts_at_startup(tmp_path):
    """The reviewer's exact case: dropping z_flag must fail at make_adapter,
    not later at query time."""
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds = ds.drop_vars("z_flag")
    with pytest.raises(SA.StoreSchemaError, match="missing variable.*z_flag"):
        SA.make_adapter(ds)


def test_legacy_missing_coord_aborts(tmp_path):
    p, *_ = _legacy_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds = ds.drop_vars("constituents")  # drop the coord, keep the dim
    with pytest.raises(SA.StoreSchemaError, match="missing coordinate"):
        SA.make_adapter(ds)


def test_dtype_kind_mismatch_aborts(tmp_path):
    """A CF-decoded tpxo10 store (z_Re float64 instead of int32) is rejected."""
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds["z_Re"] = ds["z_Re"].astype(np.float64)
    with pytest.raises(SA.StoreSchemaError, match="dtype.*int32"):
        SA.make_adapter(ds)


def test_tpxo10_transposed_dims_aborts(tmp_path):
    """z_Re with axis order (constituents, lat_z, lon_z) must be rejected
    (would silently misplace the constituent axis) — round 20 finding 1."""
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds["z_Re"] = ds["z_Re"].transpose("constituents", "lat_z", "lon_z")
    with pytest.raises(SA.StoreSchemaError, match="dims.*axis order"):
        SA.make_adapter(ds)


def test_tpxo10_flag_transposed_dims_aborts(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds["uz_flag"] = ds["uz_flag"].transpose("lon_z", "lat_z")
    with pytest.raises(SA.StoreSchemaError, match="dims"):
        SA.make_adapter(ds)


def test_tpxo10_wrong_float_width_aborts(tmp_path):
    """uz_Re as float64 (not float32) must be rejected (exact dtype)."""
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds["uz_Re"] = ds["uz_Re"].astype(np.float64)
    with pytest.raises(SA.StoreSchemaError, match="dtype.*float64"):
        SA.make_adapter(ds)


def test_non_monotone_coord_aborts(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    bad = ds["lat_z"].values.copy()
    bad[3] = bad[0] - 1.0  # break strict increase
    ds = ds.assign_coords(lat_z=bad)
    with pytest.raises(SA.StoreSchemaError, match="strictly increasing"):
        SA.make_adapter(ds)


def test_coord_on_foreign_dim_aborts(tmp_path):
    """A store whose lon axis is named 'x' (lon_z is a non-dimension
    coordinate) must be rejected at STARTUP, never surface as a sel()-time
    KeyError (round 21). Caught by the var-dim and/or coord-dim check."""
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    lonv = ds["lon_z"].values
    ds = ds.drop_vars("lon_z").rename_dims({"lon_z": "x"})
    ds = ds.assign_coords(lon_z=("x", lonv))
    with pytest.raises(SA.StoreSchemaError,
                       match="not a dimension coordinate|axis order"):
        SA.make_adapter(ds)


def test_coord_dim_check_in_isolation():
    """Directly exercise the coord dimension-coordinate check: a minimal
    legacy-shaped dataset whose lon coord rides a foreign dim 'x'."""
    n = 4
    z = np.zeros((n, n, 1))
    base = {f"{v}_{p}": (("lat", "x", "constituents"), z)
            for v in "zuv" for p in ("amp", "ph")}
    ds = xr.Dataset(base, coords={
        "lat": ("lat", np.arange(n, dtype=float)),
        "lon": ("x", np.arange(n, dtype=float)),   # foreign dim
        "constituents": ("constituents", np.array(["m2"], dtype="<U3"))})
    # var dims are (lat, x, constituents) != (lat, lon, constituents) -> the
    # var check fires; either way startup rejects (no late KeyError).
    with pytest.raises(SA.StoreSchemaError):
        SA.make_adapter(ds)


def test_duplicate_constituents_aborts(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    ds = ds.assign_coords(constituents=np.array(["m2", "m2", "k1"], dtype="<U3"))
    with pytest.raises(SA.StoreSchemaError, match="duplicates"):
        SA.make_adapter(ds)


def test_non_finite_coord_aborts(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    ds = xr.open_zarr(p, decode_times=False, mask_and_scale=False, chunks=None)
    bad = ds["lon_z"].values.copy()
    bad[2] = np.nan
    ds = ds.assign_coords(lon_z=bad)
    with pytest.raises(SA.StoreSchemaError, match="non-finite"):
        SA.make_adapter(ds)


def test_unknown_schema_aborts():
    ds = xr.Dataset(coords={"constituents": np.array(["m2"], dtype="<U3")},
                    attrs={"tide_store_schema": "tpxo10-cgrid-v2-future"})
    with pytest.raises(SA.StoreSchemaError, match="unknown tide_store_schema"):
        SA.make_adapter(ds)


# ---------- raw read ----------

def test_open_store_uses_raw_read(tmp_path):
    p, *_ = _tpxo10_store(tmp_path)
    a = SA.open_store(str(p))
    assert a.ds["z_Re"].dtype == np.int32 and a.ds["z_flag"].dtype == np.uint8
    assert int((a.ds["z_flag"].values == 2).sum()) == 1


# ---------- hc unit/convention ----------

def test_legacy_hc_matches_amp_ph_formula(tmp_path):
    p, amp, ph = _legacy_store(tmp_path)
    a = SA.open_store(str(p))
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    j, i = 2, 3
    expect = amp["z"][j, i] * np.exp(-1j * ph["z"][j, i] * np.pi / 180.0)
    assert np.allclose(np.asarray(a.hc(sub, "z")), expect, rtol=1e-12)


def test_tpxo10_hc_units_z_metres_uv_cms(tmp_path):
    p, zre, zim, uvw, _ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    j, i = 2, 3
    assert np.allclose(np.asarray(a.hc(sub, "z")),
                       1e-3 * (zre[j, i] + 1j * zim[j, i]), rtol=1e-6)
    ure, uim = uvw["uz"]
    assert np.allclose(np.asarray(a.hc(sub, "u")),
                       100.0 * (ure[j, i] + 1j * uim[j, i]), rtol=1e-5)


def test_amp_ph_carries_mask_for_flag2(tmp_path):
    """round 23 F1: amp_ph must carry the flag==2 mask so callers can fill
    NaN (not 0) at the boundary. np.ma.filled(.., nan) must be NaN — never
    the on-disk 0 — at the invalid node."""
    p, *_ = _tpxo10_store(tmp_path, with_flag2=True)  # uz_flag[1,1]==2
    a = SA.open_store(str(p))
    sub = a.sel_bbox(a.lon[0], a.lon[-1], a.lat[0], a.lat[-1])
    amp, ph = a.amp_ph(sub, "u")
    assert ma.is_masked(amp) and bool(np.all(amp.mask[1, 1, :]))
    filled = np.ma.filled(amp, np.nan)
    assert bool(np.all(np.isnan(filled[1, 1, :])))         # invalid -> NaN
    assert not np.isnan(filled[0, 0, :]).any()             # valid stays real


def test_amp_ph_round_trips_to_hc(tmp_path):
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    sub = a.sel_point(a.lon[3], a.lat[2], tol=1.0)
    amp, ph = a.amp_ph(sub, "z")
    assert np.allclose(amp * np.exp(-1j * ph * np.pi / 180.0),
                       np.asarray(a.hc(sub, "z")), rtol=1e-9)


def test_tpxo10_flag2_masks_hc(tmp_path):
    p, *_, _flags = _tpxo10_store(tmp_path, with_flag2=True)
    a = SA.open_store(str(p))
    sub = a.sel_bbox(a.lon[0], a.lon[-1], a.lat[0], a.lat[-1])
    hu = a.hc(sub, "u")
    assert ma.is_masked(hu) and bool(np.all(hu.mask[1, 1, :]))
    assert not bool(hu.mask[0, 0, :].any())


# ---------- vectorized multipoint (F3) ----------

def test_sel_points_order_and_duplicates(tmp_path):
    p, zre, zim, *_ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    # pick three points incl. a duplicate, in a non-monotone order
    idx = [(2, 5), (0, 1), (2, 5)]
    lons = [a.lon[i] for (_, i) in idx]
    lats = [a.lat[j] for (j, _) in idx]
    sub = a.sel_points(lons, lats, tol=1.0)
    assert sub.sizes["points"] == 3
    hc = np.asarray(a.hc(sub, "z"))
    for n, (j, i) in enumerate(idx):
        assert np.allclose(hc[n], 1e-3 * (zre[j, i] + 1j * zim[j, i]), rtol=1e-6)
    assert np.allclose(hc[0], hc[2])  # duplicate coords -> identical rows


def test_sel_points_out_of_tolerance_raises(tmp_path):
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False)
    a = SA.open_store(str(p))
    with pytest.raises(KeyError):
        a.sel_points([a.lon[0], 999.0], [a.lat[0], 999.0], tol=0.01)


# ---------- dateline-wrap bbox parity (F5) ----------

def test_sel_bbox_dateline_wrap_matches_manual_concat(tmp_path):
    # global-ish lon spanning the 0/360 seam
    lon = np.linspace(0.5, 359.5, NLON)
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False, name="wrap.zarr", lon=lon)
    a = SA.open_store(str(p))
    lon0, lon1 = lon[-2], lon[1]  # wraps: lon0 > lon1
    got = a.sel_bbox(lon0, lon1, a.lat[0], a.lat[-1])
    s1 = a.ds.sel(lon_z=slice(lon0, float(lon[-1])), lat_z=slice(a.lat[0], a.lat[-1]))
    s2 = a.ds.sel(lon_z=slice(float(lon[0]), lon1), lat_z=slice(a.lat[0], a.lat[-1]))
    manual = xr.concat([s1, s2], dim="lon_z")
    assert np.array_equal(got["lon_z"].values, manual["lon_z"].values)
    assert np.array_equal(np.asarray(a.hc(got, "z")),
                          np.asarray(a.hc(manual, "z")))


# ---------- selection-before-materialization in DIRECT mode (F4) ----------

class _CountingStore(zarr.DirectoryStore):
    """Records chunk-data read keys (excludes metadata keys)."""
    _META = (".zarray", ".zattrs", ".zgroup", ".zmetadata")

    def __init__(self, path):
        super().__init__(str(path))
        self.keys_read = []

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if not key.endswith(self._META):
            self.keys_read.append(key)
        return v


def _z_data_chunks(store):
    """Chunk keys read for the z hc data variables (excludes coordinate
    arrays, which nearest-neighbor selection legitimately scans)."""
    return {k for k in store.keys_read
            if k.split("/", 1)[0] in ("z_Re", "z_Im", "z_flag")}


def test_direct_mode_point_reads_only_selected_chunks(tmp_path):
    """D5 direct mode (chunks=None): a single-point query reads ONLY the
    data chunks covering that point — proven by exact chunk-key
    containment, not a loose count (round 20 finding 2)."""
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False, name="chunked.zarr",
                          chunks=(2, 2, NC))  # ceil(6/2)*ceil(8/2)=12 chunks/var
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    store.keys_read = []
    j, i = 2, 3
    sub = a.sel_point(a.lon[i], a.lat[j], tol=1.0)
    _ = np.asarray(a.hc(sub, "z"))
    lc, oc, cc = j // 2, i // 2, 0  # chunk coords of the point
    expected = {f"z_Re/{lc}.{oc}.{cc}", f"z_Im/{lc}.{oc}.{cc}",
                f"z_flag/{lc}.{oc}"}
    assert _z_data_chunks(store) == expected, _z_data_chunks(store)


def test_direct_mode_bbox_subset_reads_only_selected_chunks(tmp_path):
    p, *_ = _tpxo10_store(tmp_path, with_flag2=False, name="chunked2.zarr",
                          chunks=(2, 2, NC))
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    store.keys_read = []
    sub = a.sel_bbox(a.lon[0], a.lon[1], a.lat[0], a.lat[1])  # 2x2 corner chunk
    _ = np.asarray(a.hc(sub, "z"))
    expected = {"z_Re/0.0.0", "z_Im/0.0.0", "z_flag/0.0"}
    assert _z_data_chunks(store) == expected, _z_data_chunks(store)
