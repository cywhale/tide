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
per-variable complex harmonic constants in the SAME native units the
legacy runtime produced (z in metres, u/v in cm/s), so downstream pyTMD
prediction is byte-for-byte the legacy path.

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

# (var, coord, dtype-kind) contracts validated fail-closed at startup
LEGACY_REQUIRED_VARS = ["z_amp", "z_ph", "u_amp", "u_ph", "v_amp", "v_ph"]
LEGACY_REQUIRED_COORDS = ["lon", "lat", "constituents"]
TPXO10_REQUIRED_VARS = ["z_Re", "z_Im", "uz_Re", "uz_Im", "vz_Re", "vz_Im",
                        "z_flag", "uz_flag", "vz_flag"]
TPXO10_REQUIRED_COORDS = ["lon_z", "lat_z", "constituents"]
# discriminating dtype checks (kind only — robust to platform int width)
TPXO10_DTYPE_KIND = {"z_Re": "i", "z_Im": "i", "z_flag": "u",
                     "uz_flag": "u", "vz_flag": "u"}
LEGACY_DTYPE_KIND = {"z_amp": "f", "z_ph": "f"}


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


def _validate(ds: xr.Dataset, vars_, coords_, dtype_kind, label: str) -> None:
    missing_v = [v for v in vars_ if v not in ds.variables]
    if missing_v:
        raise StoreSchemaError(f"{label}: missing variables {missing_v}")
    missing_c = [c for c in coords_ if c not in ds.coords and c not in ds.variables]
    if missing_c:
        raise StoreSchemaError(f"{label}: missing coordinates {missing_c}")
    for name, kind in dtype_kind.items():
        k = ds[name].dtype.kind
        if k != kind:
            raise StoreSchemaError(
                f"{label}: {name} dtype kind {k!r} != expected {kind!r} "
                "(store likely CF-decoded — open raw with mask_and_scale=False)")


def make_adapter(ds: xr.Dataset) -> "StoreAdapter":
    schema = ds.attrs.get("tide_store_schema")
    if schema == SCHEMA_TPXO10:
        _validate(ds, TPXO10_REQUIRED_VARS, TPXO10_REQUIRED_COORDS,
                  TPXO10_DTYPE_KIND, f"schema {schema!r}")
        return Tpxo10Adapter(ds)
    if schema is None:
        _validate(ds, LEGACY_REQUIRED_VARS, LEGACY_REQUIRED_COORDS,
                  LEGACY_DTYPE_KIND,
                  "no tide_store_schema attr -> legacy candidate")
        return LegacyAdapter(ds)
    raise StoreSchemaError(f"unknown tide_store_schema {schema!r}")


def _amp_ph_from_hc(hc: np.ndarray):
    amp = np.abs(hc)
    ph = np.rad2deg(-np.angle(hc)) % 360.0
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

    def sel_bbox(self, lon0: float, lon1: float, lat0: float, lat1: float,
                 constituents: Optional[Iterable] = None, halo: float = 0.0
                 ) -> xr.Dataset:
        """Rectangular bbox selection with an optional cell halo. When the
        longitude window wraps the 0/360 seam (`lon0 > lon1` in store
        coordinates), select the two pieces and concat along longitude in
        ascending-wrapped order — coordinate order identical to the legacy
        two-slice/concat (F5)."""
        ds = self.select_constituents(self.ds, constituents)
        lat_sl = slice(lat0 - halo, lat1 + halo)
        if lon0 <= lon1:
            return ds.sel({self.lon_name: slice(lon0 - halo, lon1 + halo),
                           self.lat_name: lat_sl})
        lon_max = float(self.ds[self.lon_name].values[-1])
        lon_min = float(self.ds[self.lon_name].values[0])
        s1 = ds.sel({self.lon_name: slice(lon0 - halo, lon_max),
                     self.lat_name: lat_sl})
        s2 = ds.sel({self.lon_name: slice(lon_min, lon1 + halo),
                     self.lat_name: lat_sl})
        return xr.concat([s1, s2], dim=self.lon_name)

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
