"""v0.3.0 Stage 3 store adapter (spec D9).

Schema-keyed, fail-closed adapter presenting ONE uniform interface over
both store schemas so the runtime never branches on schema and a rollback
is a single `TIDE_ZARR_PATH` env change + restart:

  * legacy tpxo9   — `z_amp/z_ph/u_amp/u_ph/v_amp/v_ph` on `lon/lat`;
                     hc = amp * exp(-i*ph*pi/180)
  * tpxo10-cgrid-v1 — `z_Re/z_Im` + D12-centered `uz_*/vz_*` on the z-grid
                     (`lon_z/lat_z`); hc = scale * (Re + 1j*Im), uz/vz
                     masked where the per-node flag == 2 (invalid)

Uniform interface (schema mapping ONLY — no resampling, no native-node
access, no runtime regridding; D12): coordinate names, point/bbox
selection, and per-variable complex harmonic constants in the SAME
native units the legacy runtime produced (z in metres, u/v in cm/s), so
downstream pyTMD prediction is byte-for-byte the legacy path.

Binding contracts:
  * RAW READ (G2 raw-read contract): stores are opened with
    `mask_and_scale=False` — the tpxo10 fill_values are STRUCTURAL
    (flags fill=2, Re/Im fill=0), not CF missing-value sentinels; CF
    decoding would silently int32->float64 and NaN-mask fill-equal cells.
  * FAIL CLOSED (D9): a missing `tide_store_schema` attr is accepted as
    legacy ONLY if the full legacy variable set is present; a recognized
    attr requires its full runtime read set; anything else aborts.
  * LAZY (D9): selection happens before materialization — point/bbox
    selection on lazy arrays, `.values` only on the selected subset; no
    eager whole-store hc construction.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import numpy.ma as ma
import xarray as xr

SCHEMA_TPXO10 = "tpxo10-cgrid-v1"
DEFAULT_ZARR_PATH = "data/tpxo10.zarr"

# unit conventions feeding the legacy pyTMD path (preserved exactly):
#   z  -> metres ;  u/v -> cm/s
Z_SCALE = 1e-3        # tpxo10 z_Re/z_Im (mm) -> m
UV_SCALE = 100.0      # tpxo10 uz/vz (m/s)    -> cm/s

LEGACY_REQUIRED = ["z_amp", "z_ph", "u_amp", "u_ph", "v_amp", "v_ph"]
TPXO10_REQUIRED = ["z_Re", "z_Im", "uz_Re", "uz_Im", "vz_Re", "vz_Im",
                   "uz_flag", "vz_flag"]
FLAG_INVALID = 2


class StoreSchemaError(RuntimeError):
    """Fail-closed schema detection error (D9)."""


def get_zarr_path(default: str = DEFAULT_ZARR_PATH) -> str:
    """Single source of the runtime store path (spec Stage 3 store-path
    audit): `TIDE_ZARR_PATH` env override, else the v0.3.0 default."""
    return os.environ.get("TIDE_ZARR_PATH", default)


def open_store(path: Optional[str] = None, **open_kwargs) -> "StoreAdapter":
    """Open the store RAW (mask_and_scale=False) and return a fail-closed
    schema-selected adapter."""
    path = path or get_zarr_path()
    kwargs = dict(decode_times=False, mask_and_scale=False, chunks=None)
    kwargs.update(open_kwargs)
    ds = xr.open_zarr(path, **kwargs)
    return make_adapter(ds)


def make_adapter(ds: xr.Dataset) -> "StoreAdapter":
    schema = ds.attrs.get("tide_store_schema")
    if schema == SCHEMA_TPXO10:
        missing = [v for v in TPXO10_REQUIRED if v not in ds.variables]
        if missing:
            raise StoreSchemaError(
                f"schema {schema!r} missing required vars {missing}")
        return Tpxo10Adapter(ds)
    if schema is None:
        missing = [v for v in LEGACY_REQUIRED if v not in ds.variables]
        if missing:
            raise StoreSchemaError(
                "no tide_store_schema attr and incomplete legacy variable "
                f"set (missing {missing}); refusing to guess")
        return LegacyAdapter(ds)
    raise StoreSchemaError(f"unknown tide_store_schema {schema!r}")


def _amp_ph_from_hc(hc: np.ndarray):
    """Polar form of hc in the legacy convention hc = amp*exp(-i*ph*pi/180):
    amp = |hc|, ph = (-angle(hc) in degrees) mod 360."""
    amp = np.abs(hc)
    ph = np.rad2deg(-np.angle(hc)) % 360.0
    return amp, ph


class StoreAdapter:
    """Uniform interface base. Subclasses set coord names and implement
    `_hc_from_subset`. All selection methods operate on lazy arrays and
    materialize only the selected subset."""

    schema: str = ""
    lon_name: str = "lon"
    lat_name: str = "lat"

    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.constituents = [str(c) for c in ds["constituents"].values]

    # -- coordinate access -------------------------------------------------
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

    def sel_bbox(self, lon0: float, lon1: float, lat0: float, lat1: float,
                 constituents: Optional[Iterable] = None) -> xr.Dataset:
        ds = self.select_constituents(self.ds, constituents)
        return ds.sel({self.lon_name: slice(lon0, lon1),
                       self.lat_name: slice(lat0, lat1)})

    def subsample(self, sub: xr.Dataset, step: int) -> xr.Dataset:
        return sub.isel({self.lon_name: slice(None, None, step),
                         self.lat_name: slice(None, None, step)})

    def coord_values(self, sub: xr.Dataset):
        return sub[self.lon_name].values, sub[self.lat_name].values

    # -- uniform harmonic-constant accessors -------------------------------
    def _hc_from_subset(self, sub: xr.Dataset, var: str) -> np.ndarray:
        raise NotImplementedError

    def hc(self, sub: xr.Dataset, var: str) -> np.ndarray:
        """Complex harmonic constants for `var` ('z'|'u'|'v') over the
        already-selected subset, in legacy native units (z: m, u/v: cm/s).
        Returns a numpy (possibly masked) array; spatial dims flattened by
        the caller as needed."""
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
        cph = -1j * ph * np.pi / 180.0
        hc = amp * np.exp(cph)
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
        # structural-invalid mask (flag == 2); flag is 2D (no constituent
        # axis) so broadcast across the trailing constituent dimension
        flag = np.asarray(sub[flag_v].values)
        mask = np.isnan(hc)
        if flag.shape and hc.ndim == flag.ndim + 1:
            mask = mask | (flag == FLAG_INVALID)[..., np.newaxis]
        elif flag.shape == hc.shape:
            mask = mask | (flag == FLAG_INVALID)
        else:
            mask = mask | (np.asarray(flag) == FLAG_INVALID)
        return ma.array(hc, mask=mask)
