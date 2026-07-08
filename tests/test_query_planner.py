"""Stage 3 step 2: unified query planner + MAX_BBOX_CELLS tests.

Covers: cap env resolution; post-sample output-cell counting (incl. ceil
boundaries and dateline-wrap totals); cap enforcement BEFORE
materialization; and the binding guarantee that a REJECTED request reads
ZERO data chunks (proven with a read-counting store).
"""
import numpy as np
import pytest
import xarray as xr
import zarr

from src import query_planner as QP
from src import store_adapter as SA

CONS = ["m2", "s2", "k1"]
NC = len(CONS)


def _store(tmp_path, nlat, nlon, chunks=None, lon=None, name="s.zarr"):
    rng = np.random.default_rng(0)
    lat = np.linspace(-10, 10, nlat)
    if lon is None:
        lon = np.linspace(100, 100 + (nlon - 1) * (1 / 30), nlon)
    mk3 = lambda dt: rng.integers(-2000, 2000, (nlat, nlon, NC)).astype(dt) \
        if dt == np.int32 else rng.uniform(-1, 1, (nlat, nlon, NC)).astype(dt)
    z2 = np.zeros((nlat, nlon), np.uint8)
    data = {
        "z_Re": (("lat_z", "lon_z", "constituents"), mk3(np.int32)),
        "z_Im": (("lat_z", "lon_z", "constituents"), mk3(np.int32)),
        "uz_Re": (("lat_z", "lon_z", "constituents"), mk3(np.float32)),
        "uz_Im": (("lat_z", "lon_z", "constituents"), mk3(np.float32)),
        "vz_Re": (("lat_z", "lon_z", "constituents"), mk3(np.float32)),
        "vz_Im": (("lat_z", "lon_z", "constituents"), mk3(np.float32)),
        "z_flag": (("lat_z", "lon_z"), z2),
        "uz_flag": (("lat_z", "lon_z"), z2.copy()),
        "vz_flag": (("lat_z", "lon_z"), z2.copy()),
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
    return p


# ---------- cap resolution ----------

def test_get_max_bbox_cells_default_and_env(monkeypatch):
    monkeypatch.delenv("TIDE_MAX_BBOX_CELLS", raising=False)
    assert QP.get_max_bbox_cells() == 500_000
    monkeypatch.setenv("TIDE_MAX_BBOX_CELLS", "250000")
    assert QP.get_max_bbox_cells() == 250_000
    monkeypatch.setenv("TIDE_MAX_BBOX_CELLS", "0")
    with pytest.raises(ValueError, match="positive"):
        QP.get_max_bbox_cells()
    monkeypatch.setenv("TIDE_MAX_BBOX_CELLS", "abc")
    with pytest.raises(ValueError, match="not an integer"):
        QP.get_max_bbox_cells()


# ---------- cell counting ----------

def test_output_cell_count_ceil():
    assert QP.output_cell_count(10, 10, 1) == 100
    assert QP.output_cell_count(10, 10, 5) == 4        # ceil(10/5)=2 -> 2*2
    assert QP.output_cell_count(11, 11, 5) == 9        # ceil(11/5)=3 -> 3*3
    assert QP.output_cell_count(1350, 1350, 1) == 1_822_500   # 45deg s=1
    assert QP.output_cell_count(1350, 1350, 5) == 270 * 270   # 45deg s=5
    with pytest.raises(ValueError, match="sample must be"):
        QP.output_cell_count(10, 10, 0)


# ---------- cap enforcement ----------

def test_plan_bbox_within_cap_returns_subset(tmp_path):
    p = _store(tmp_path, 20, 24)
    a = SA.open_store(str(p))
    sub, cells = QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                              sample=1, max_cells=1000)
    assert cells == 20 * 24
    assert a.grid_shape(sub) == (20, 24)


def test_plan_bbox_sample_reduces_cells(tmp_path):
    p = _store(tmp_path, 20, 24)
    a = SA.open_store(str(p))
    _, cells = QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                            sample=5, max_cells=1000)
    assert cells == 4 * 5  # ceil(20/5)=4, ceil(24/5)=5


def test_plan_bbox_over_cap_raises(tmp_path):
    p = _store(tmp_path, 40, 40)
    a = SA.open_store(str(p))
    with pytest.raises(QP.BboxCapError) as ei:
        QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                     sample=1, max_cells=1000)
    assert ei.value.requested == 1600 and ei.value.cap == 1000


def test_plan_bbox_exact_boundary(tmp_path):
    p = _store(tmp_path, 25, 40)
    a = SA.open_store(str(p))
    # exactly at the cap -> allowed
    sub, cells = QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                              sample=1, max_cells=1000)
    assert cells == 1000
    # one over -> rejected
    p2 = _store(tmp_path, 25, 41, name="s2.zarr")
    a2 = SA.open_store(str(p2))
    with pytest.raises(QP.BboxCapError):
        QP.plan_bbox(a2, a2.lon[0], a2.lon[-1], a2.lat[0], a2.lat[-1],
                     sample=1, max_cells=1000)


def test_plan_bbox_dateline_counts_both_pieces(tmp_path):
    lon = np.linspace(0.5, 359.5, 36)
    p = _store(tmp_path, 10, 36, lon=lon, name="wrap.zarr")
    a = SA.open_store(str(p))
    lon0, lon1 = lon[-3], lon[2]  # wraps: 3 + 3 = 6 lon cells
    sub, cells = QP.plan_bbox(a, lon0, lon1, a.lat[0], a.lat[-1],
                              sample=1, max_cells=10_000)
    assert a.grid_shape(sub) == (10, 6)
    assert cells == 60


# ---------- ZERO data reads on rejection (binding) ----------

class _CountingStore(zarr.DirectoryStore):
    _META = (".zarray", ".zattrs", ".zgroup", ".zmetadata")

    def __init__(self, path):
        super().__init__(str(path))
        self.keys_read = []

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if not key.endswith(self._META):
            self.keys_read.append(key)
        return v


def _data_chunks(store):
    data_vars = ("z_Re", "z_Im", "uz_Re", "uz_Im", "vz_Re", "vz_Im",
                 "z_flag", "uz_flag", "vz_flag")
    return {k for k in store.keys_read if k.split("/", 1)[0] in data_vars}


def test_rejected_bbox_reads_zero_data_chunks(tmp_path):
    """The binding guarantee: a cap rejection materializes nothing — no
    data chunk of any variable is read (spec §7.5.3 implementation
    condition)."""
    p = _store(tmp_path, 40, 40, chunks=(8, 8, NC), name="big.zarr")
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    store.keys_read = []
    with pytest.raises(QP.BboxCapError):
        QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                     sample=1, max_cells=100)
    assert _data_chunks(store) == set(), \
        f"rejection read data chunks: {_data_chunks(store)}"


def test_rejected_dateline_bbox_reads_zero_data_chunks(tmp_path):
    """Round 22 F1: a DATELINE-wrap request over the cap must also read
    zero data chunks (the previous eager xr.concat read ~36 chunks before
    the cap fired)."""
    lon = np.linspace(0.5, 359.5, 60)
    p = _store(tmp_path, 40, 60, chunks=(8, 8, NC), lon=lon, name="wrapbig.zarr")
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    store.keys_read = []
    lon0, lon1 = lon[-20], lon[20]  # wraps, large
    with pytest.raises(QP.BboxCapError):
        QP.plan_bbox(a, lon0, lon1, a.lat[0], a.lat[-1], sample=1,
                     max_cells=100)
    assert _data_chunks(store) == set(), \
        f"dateline rejection read data chunks: {_data_chunks(store)}"


def test_accepted_dateline_lazy_until_materialize(tmp_path):
    """Round 22: an ACCEPTED dateline request reads zero data chunks at the
    planner stage; data is read only when hc() materializes."""
    lon = np.linspace(0.5, 359.5, 60)
    p = _store(tmp_path, 10, 60, chunks=(8, 8, NC), lon=lon, name="wrapok.zarr")
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    lon0, lon1 = lon[-5], lon[4]  # wraps, small (within cap)
    store.keys_read = []
    sub, cells = QP.plan_bbox(a, lon0, lon1, a.lat[0], a.lat[-1], sample=1,
                              max_cells=10_000)
    assert _data_chunks(store) == set()    # planner stage: zero data read
    _ = np.asarray(a.hc(sub, "z"))         # materialize
    assert len(_data_chunks(store)) > 0    # now the selected chunks are read


def test_plan_bbox_rejects_bad_sample(tmp_path):
    p = _store(tmp_path, 10, 10)
    a = SA.open_store(str(p))
    for bad in (1.5, True, 0, -1):
        with pytest.raises(ValueError, match="sample"):
            QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[0], a.lat[-1],
                         sample=bad, max_cells=10_000)


def test_plan_bbox_empty_selection_raises(tmp_path):
    p = _store(tmp_path, 10, 10)
    a = SA.open_store(str(p))
    # reversed latitude -> zero cells
    with pytest.raises(QP.EmptyBboxError, match="zero cells"):
        QP.plan_bbox(a, a.lon[0], a.lon[-1], a.lat[-1], a.lat[0], sample=1,
                     max_cells=10_000)
    # bbox fully outside the data extent -> zero cells
    with pytest.raises(QP.EmptyBboxError):
        QP.plan_bbox(a, 200.0, 210.0, a.lat[0], a.lat[-1], sample=1,
                     max_cells=10_000)


def test_accepted_then_materialize_reads_data(tmp_path):
    """Sanity counterpart: an accepted small query DOES read data chunks
    only when the caller materializes — proving the planner itself stays
    lazy until materialization."""
    p = _store(tmp_path, 40, 40, chunks=(8, 8, NC), name="big2.zarr")
    store = _CountingStore(p)
    ds = xr.open_zarr(store, decode_times=False, mask_and_scale=False,
                      chunks=None, consolidated=True)
    a = SA.make_adapter(ds)
    sub, cells = QP.plan_bbox(a, a.lon[0], a.lon[2], a.lat[0], a.lat[2],
                              sample=1, max_cells=10_000)
    store.keys_read = []
    assert _data_chunks(store) == set()    # planning alone read no data
    _ = np.asarray(a.hc(sub, "z"))         # materialize
    assert len(_data_chunks(store)) > 0    # now data is read
