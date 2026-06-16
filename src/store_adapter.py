"""v0.3.0 Stage 3 store adapter (spec D9).

Schema-keyed, fail-closed adapter presenting ONE uniform interface over
both store schemas so the runtime never branches on schema and a rollback
is a single `TIDE_ZARR_PATH` env change + restart:

  * legacy tpxo9   — `z_amp/z_ph/u_amp/u_ph/v_amp/v_ph` on `lon/lat`;
                     hc = amp * exp(-i*ph*pi/180)
  * tpxo10-cgrid-v1 — `z_Re/z_Im` + D12-centered `uz_*/vz_*` on the z-grid
                     (`lon_z/lat_z`); hc = scale * (Re + 1j*Im), masked
                     where the per-node flag == 2 (invalid)

Uniform interface (schema mapping ONLY — no resampling, no native-node
access, no runtime regridding; D12): coordinate names; scalar-point,
vectorized multipoint, and dateline-aware bbox selection; and
per-variable complex harmonic constants preserving the legacy units
(z in metres, u/v in cm/s), array shape, and phase convention, so the
downstream pyTMD prediction interface is unchanged. (TPXO10 values
differ from TPXO9 — different model generation; true rollback byte
parity is for the legacy store, proven by the G3 golden test.)

Binding contracts:
  * RAW READ (G2 raw-read contract): stores are opened with
    `mask_and_scale=False` — the tpxo10 fill_values are STRUCTURAL
    (flags fill=2, Re/Im fill=0), not CF missing-value sentinels.
  * DIRECT (D5 open-mode): `chunks=None` (no dask). Callers may NOT
    override the binding open params (reserved-kwarg guard).
  * FAIL CLOSED (D9): full required-variable + required-coordinate +
    discriminating-dtype validation at startup; a missing
    `tide_store_schema` attr is accepted as legacy ONLY if the full
    legacy contract holds; anything else aborts.
  * LAZY: selection happens before materialization — selection on the
    lazy dataset, `.values` only on the selected subset.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import numpy.ma as ma
import xarray as xr

SCHEMA_TPXO10 = "tpxo10-cgrid-v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZARR_RELPATH = "data/tpxo10.zarr"

# unit conventions feeding the legacy pyTMD path (preserved exactly):
#   z  -> metres ;  u/v -> cm/s
Z_SCALE = 1e-3        # tpxo10 z_Re/z_Im (mm) -> m
UV_SCALE = 100.0      # tpxo10 uz/vz (m/s)    -> cm/s
FLAG_INVALID = 2

# binding open params the caller may NEVER override (F2 / G2 raw-read + D5)
_RESERVED_OPEN_KWARGS = {"mask_and_scale", "chunks", "decode_times"}
_ALLOWED_OPEN_KWARGS = {"consolidated", "storage_options"}

# Structural contracts validated fail-closed at startup (round 20): each
# entry is var -> (expected dims tuple, expected dtype). dtype is an exact
# numpy type for tpxo10 (locks int32/float32/uint8) or a kind string for
# legacy float amp/ph (robust to float32/64). Dim TUPLES lock axis order
# so a transposed store (e.g. (constituents, lat, lon)) is rejected.
_LAT, _LON, _CON = "lat_z", "lon_z", "constituents"
TPXO10_VAR_SPEC = {
    "z_Re": ((_LAT, _LON, _CON), np.int32),
    "z_Im": ((_LAT, _LON, _CON), np.int32),
    "uz_Re": ((_LAT, _LON, _CON), np.float32),
    "uz_Im": ((_LAT, _LON, _CON), np.float32),
    "vz_Re": ((_LAT, _LON, _CON), np.float32),
    "vz_Im": ((_LAT, _LON, _CON), np.float32),
    "z_flag": ((_LAT, _LON), np.uint8),
    "uz_flag": ((_LAT, _LON), np.uint8),
    "vz_flag": ((_LAT, _LON), np.uint8),
}
TPXO10_REQUIRED_COORDS = (_LON, _LAT, _CON)
TPXO10_MONOTONE_COORDS = (_LON, _LAT)
LEGACY_VAR_SPEC = {
    f"{v}_{p}": (("lat", "lon", "constituents"), "f")
    for v in "zuv" for p in ("amp", "ph")
}
LEGACY_REQUIRED_COORDS = ("lon", "lat", "constituents")
LEGACY_MONOTONE_COORDS = ("lon", "lat")
# kept for back-compat references
LEGACY_REQUIRED_VARS = list(LEGACY_VAR_SPEC)
TPXO10_REQUIRED_VARS = list(TPXO10_VAR_SPEC)


class StoreSchemaError(RuntimeError):
    """Fail-closed schema detection error (D9)."""


def get_zarr_path(default: Optional[str] = None) -> str:
    """Single source of the runtime store path (spec Stage 3 store-path
    audit). `TIDE_ZARR_PATH` is used verbatim if set (absolute or
    caller-relative); otherwise the default resolves against the REPO
    ROOT, not the process cwd (F8)."""
    env = os.environ.get("TIDE_ZARR_PATH")
    if env:
        return env
    if default is not None:
        return default
    return str(_REPO_ROOT / DEFAULT_ZARR_RELPATH)


def open_store(path: Optional[str] = None, **open_kwargs) -> "StoreAdapter":
    """Open the store RAW + DIRECT and return a fail-closed adapter.
    Reserved binding params cannot be overridden (F2)."""
    bad = _RESERVED_OPEN_KWARGS & set(open_kwargs)
    if bad:
        raise ValueError(
            f"open_store: binding params {sorted(bad)} are fixed by contract "
            "(mask_and_scale=False, chunks=None) and cannot be overridden")
    extra = set(open_kwargs) - _ALLOWED_OPEN_KWARGS
    if extra:
        raise ValueError(f"open_store: unsupported kwargs {sorted(extra)}")
    path = path or get_zarr_path()
    ds = xr.open_zarr(path, decode_times=False, mask_and_scale=False,
                      chunks=None, **open_kwargs)
    return make_adapter(ds)


def _check_dtype(name, actual, expected, label):
    if isinstance(expected, str):  # kind check (legacy floats)
        if actual.kind != expected:
            raise StoreSchemaError(
                f"{label}: {name} dtype kind {actual.kind!r} != {expected!r} "
                "(store likely CF-decoded — open raw with mask_and_scale=False)")
    elif actual != np.dtype(expected):  # exact (tpxo10 int32/float32/uint8)
        raise StoreSchemaError(
            f"{label}: {name} dtype {actual} != expected {np.dtype(expected)} "
            "(store likely CF-decoded — open raw with mask_and_scale=False)")


def _validate(ds: xr.Dataset, var_spec, coords_, monotone, label: str) -> None:
    # variables: presence + exact dim-tuple (axis order) + dtype
    for name, (dims, dtype) in var_spec.items():
        if name not in ds.variables:
            raise StoreSchemaError(f"{label}: missing variable {name}")
        if tuple(ds[name].dims) != dims:
            raise StoreSchemaError(
                f"{label}: {name} dims {tuple(ds[name].dims)} != {dims} "
                "(axis order / transposition mismatch)")
        _check_dtype(name, ds[name].dtype, dtype, label)
    # coordinates: presence + DIMENSION coordinate (self-indexing 1-D, so
    # xarray can build an index for nearest selection — round 21). A 1-D
    # coord on a foreign dim (e.g. lon_z indexed by 'x') would pass a bare
    # ndim check yet KeyError at sel() time.
    for c in coords_:
        if c not in ds.coords and c not in ds.variables:
            raise StoreSchemaError(f"{label}: missing coordinate {c}")
        if tuple(ds[c].dims) != (c,):
            raise StoreSchemaError(
                f"{label}: coordinate {c} is not a dimension coordinate "
                f"(dims={tuple(ds[c].dims)}, expected ({c!r},))")
    # numeric coords strictly increasing + finite
    for c in monotone:
        vals = np.asarray(ds[c].values)
        if not np.all(np.isfinite(vals)):
            raise StoreSchemaError(f"{label}: coordinate {c} has non-finite values")
        if not np.all(np.diff(vals) > 0):
            raise StoreSchemaError(
                f"{label}: coordinate {c} is not strictly increasing")
    # constituents: non-empty and unique (must build a unique label index)
    cons = np.asarray(ds["constituents"].values)
    if cons.size == 0:
        raise StoreSchemaError(f"{label}: constituents coordinate is empty")
    if len(set(cons.tolist())) != cons.size:
        raise StoreSchemaError(f"{label}: constituents contains duplicates")


def make_adapter(ds: xr.Dataset) -> "StoreAdapter":
    schema = ds.attrs.get("tide_store_schema")
    if schema == SCHEMA_TPXO10:
        _validate(ds, TPXO10_VAR_SPEC, TPXO10_REQUIRED_COORDS,
                  TPXO10_MONOTONE_COORDS, f"schema {schema!r}")
        return Tpxo10Adapter(ds)
    if schema is None:
        _validate(ds, LEGACY_VAR_SPEC, LEGACY_REQUIRED_COORDS,
                  LEGACY_MONOTONE_COORDS,
                  "no tide_store_schema attr -> legacy candidate")
        return LegacyAdapter(ds)
    raise StoreSchemaError(f"unknown tide_store_schema {schema!r}")


def _amp_ph_from_hc(hc: np.ndarray):
    """Polar form (amp, phase[deg]) carrying hc's mask EXPLICITLY, so an
    invalid (flag==2) cell stays masked through `np.angle` (which does not
    reliably preserve the mask) — callers fill it to NaN at the
    serialization/prediction boundary, never to 0 (round 23 F1)."""
    mask = ma.getmaskarray(hc) if ma.isMaskedArray(hc) else None
    data = ma.getdata(hc)
    amp = np.abs(data)
    ph = np.rad2deg(-np.angle(data)) % 360.0
    if mask is not None:
        amp = ma.array(amp, mask=mask)
        ph = ma.array(ph, mask=mask)
    return amp, ph


class StoreAdapter:
    """Uniform interface base. Subclasses set coord names and implement
    `_hc_from_subset`. All selection methods operate on the lazy dataset
    and materialize only the selected subset."""

    schema: str = ""
    lon_name: str = "lon"
    lat_name: str = "lat"

    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.constituents = [str(c) for c in ds["constituents"].values]

    @property
    def lon(self) -> np.ndarray:
        return self.ds[self.lon_name].values

    @property
    def lat(self) -> np.ndarray:
        return self.ds[self.lat_name].values

    def select_constituents(self, ds: xr.Dataset, constituents) -> xr.Dataset:
        if constituents is None:
            return ds
        return ds.sel(constituents=list(constituents))

    # -- selection (lazy; no .values here) ---------------------------------
    def sel_point(self, lon: float, lat: float, tol: float,
                  constituents: Optional[Iterable] = None) -> xr.Dataset:
        ds = self.select_constituents(self.ds, constituents)
        return ds.sel({self.lon_name: lon, self.lat_name: lat},
                      method="nearest", tolerance=tol)

    def sel_points(self, lons: Sequence[float], lats: Sequence[float],
                   tol: float, constituents: Optional[Iterable] = None
                   ) -> xr.Dataset:
        """Vectorized point-paired (NOT Cartesian) nearest selection over a
        `points` dimension — the /api/tide/const access pattern (F3).
        Preserves point order; duplicate coordinates yield duplicate rows;
        any point beyond `tol` raises KeyError (xarray, == legacy)."""
        ds = self.select_constituents(self.ds, constituents)
        lon_idx = xr.DataArray(np.asarray(lons, dtype=float), dims="points")
        lat_idx = xr.DataArray(np.asarray(lats, dtype=float), dims="points")
        return ds.sel({self.lon_name: lon_idx, self.lat_name: lat_idx},
                      method="nearest", tolerance=tol)

    def bbox_indices(self, lon0: float, lon1: float, lat0: float, lat1: float,
                     sample: int = 1, halo: float = 0.0):
        """Resolve the post-halo, post-`sample` INTEGER index arrays
        (lat_idx, lon_idx) for a bbox from the COORDINATE arrays only — no
        data is touched, so the planner can count + cap before any
        selection. Inclusive bounds match xarray label slicing
        (searchsorted left/right). A dateline wrap (`lon0 > lon1`) yields a
        concatenated lon index [lon0..end] ++ [0..lon1] BEFORE the stride,
        so stride-across-seam matches the legacy concat-then-isel exactly
        (round 22 F1/F5)."""
        lat = self.lat
        lon = self.lon
        jlo = int(np.searchsorted(lat, lat0 - halo, side="left"))
        jhi = int(np.searchsorted(lat, lat1 + halo, side="right"))
        lat_idx = np.arange(jlo, jhi)[::sample]
        ilo = int(np.searchsorted(lon, lon0 - halo, side="left"))
        ihi = int(np.searchsorted(lon, lon1 + halo, side="right"))
        if lon0 <= lon1:
            lon_full = np.arange(ilo, ihi)
        else:  # wrap the 0/360 seam: [lon0..end] then [0..lon1]
            lon_full = np.concatenate([np.arange(ilo, len(lon)),
                                       np.arange(0, ihi)])
        return lat_idx, lon_full[::sample]

    def isel_grid(self, lat_idx, lon_idx,
                  constituents: Optional[Iterable] = None) -> xr.Dataset:
        """Orthogonal (outer) integer selection on the z-grid producing a
        (len(lat_idx), len(lon_idx)) map. Lazy until materialized; reads
        only the chunks covering the indices (incl. the wrap pieces)."""
        ds = self.select_constituents(self.ds, constituents)
        return ds.isel({self.lat_name: np.asarray(lat_idx),
                        self.lon_name: np.asarray(lon_idx)})

    def sel_bbox(self, lon0: float, lon1: float, lat0: float, lat1: float,
                 constituents: Optional[Iterable] = None, halo: float = 0.0
                 ) -> xr.Dataset:
        """Dateline-aware bbox selection (sample=1) via index isel — lazy,
        no eager concat (round 22). Coordinate order identical to the
        legacy two-slice/concat."""
        lat_idx, lon_idx = self.bbox_indices(lon0, lon1, lat0, lat1,
                                             sample=1, halo=halo)
        return self.isel_grid(lat_idx, lon_idx, constituents)

    def subsample(self, sub: xr.Dataset, step: int) -> xr.Dataset:
        return sub.isel({self.lon_name: slice(None, None, step),
                         self.lat_name: slice(None, None, step)})

    def coord_values(self, sub: xr.Dataset):
        return sub[self.lon_name].values, sub[self.lat_name].values

    def grid_shape(self, sub: xr.Dataset):
        return sub.sizes[self.lat_name], sub.sizes[self.lon_name]

    # -- uniform harmonic-constant accessors -------------------------------
    def _hc_from_subset(self, sub: xr.Dataset, var: str) -> np.ndarray:
        raise NotImplementedError

    def hc(self, sub: xr.Dataset, var: str) -> np.ndarray:
        """Complex harmonic constants for `var` ('z'|'u'|'v') over the
        already-selected subset, in legacy native units (z: m, u/v: cm/s).
        Returns a numpy (possibly masked) array; the constituent axis is
        last, spatial/point dims precede it as in the subset."""
        return self._hc_from_subset(sub, var)

    def amp_ph(self, sub: xr.Dataset, var: str):
        """(amplitude, phase[deg]) in the legacy convention, derived from
        hc — the form the /api/tide/const endpoint returns."""
        return _amp_ph_from_hc(self.hc(sub, var))


class LegacyAdapter(StoreAdapter):
    schema = "legacy-tpxo9"
    lon_name = "lon"
    lat_name = "lat"

    def _hc_from_subset(self, sub: xr.Dataset, var: str) -> np.ndarray:
        amp = np.asarray(sub[f"{var}_amp"].values)
        ph = np.asarray(sub[f"{var}_ph"].values)
        hc = amp * np.exp(-1j * ph * np.pi / 180.0)
        return ma.array(hc, mask=np.isnan(hc))


class Tpxo10Adapter(StoreAdapter):
    schema = SCHEMA_TPXO10
    lon_name = "lon_z"
    lat_name = "lat_z"

    # runtime reads ONLY z-grid vars (D12): z_Re/z_Im and the centered
    # uz/vz; native u/v nodes and hz/hu/hv are never touched here.
    _VARMAP = {
        "z": ("z_Re", "z_Im", "z_flag", Z_SCALE),
        "u": ("uz_Re", "uz_Im", "uz_flag", UV_SCALE),
        "v": ("vz_Re", "vz_Im", "vz_flag", UV_SCALE),
    }

    def _hc_from_subset(self, sub: xr.Dataset, var: str) -> np.ndarray:
        re_v, im_v, flag_v, scale = self._VARMAP[var]
        re = np.asarray(sub[re_v].values).astype(np.float64)
        im = np.asarray(sub[im_v].values).astype(np.float64)
        hc = scale * (re + 1j * im)
        flag = np.asarray(sub[flag_v].values)  # 2D / 1D-points (no cons axis)
        mask = np.isnan(hc)
        if hc.ndim == flag.ndim + 1:
            mask = mask | (flag == FLAG_INVALID)[..., np.newaxis]
        else:
            mask = mask | (flag == FLAG_INVALID)
        return ma.array(hc, mask=mask)
