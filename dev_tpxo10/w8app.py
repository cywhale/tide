"""W8 benchmark app (spec §7.5.3) — run inside the PRODUCTION env.

Reproduces the production /api/tide map path FAITHFULLY (review round 12,
finding 2): Zarr slice -> hc -> pyTMD 2.2.8 predict.map + infer_minor ->
`tide_to_output` (verbatim copy of tide_app.py:104, incl. meshgrid
flatten, full .tolist() before filtering, invalid-index sets,
per-variable list-comprehension filtering) -> jsonable_encoder ->
ORJSONResponse, with the production call-site arguments
(mode='map', absmax=10000.0).

Env:
  TIDE_BENCH_ZARR    store path
  TIDE_BENCH_OPEN    direct | dask | dist-auto | dist-native
  TIDE_BENCH_SCHEMA  tpxo10 (default) | legacy  (legacy = data/tpxo9.zarr
                     amp/ph schema, for the round-12 finding-4 comparison)
  TIDE_BENCH_SCHED   dask scheduler address (dist-* modes)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import numpy.ma as ma
import xarray as xr
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from pyTMD import predict

STATE: dict = {}


def open_store():
    schema = os.environ.get("TIDE_BENCH_SCHEMA", "tpxo10")
    default = (Path(__file__).resolve().parents[1] / "data" / "tpxo9.zarr"
               if schema == "legacy" else
               Path(__file__).resolve().parent / "stores" / "tpxo10_proto.zarr")
    path = os.environ.get("TIDE_BENCH_ZARR", str(default))
    mode = os.environ.get("TIDE_BENCH_OPEN", "direct")
    if mode.startswith("dist"):
        from dask.distributed import Client
        STATE["client"] = Client(os.environ["TIDE_BENCH_SCHED"])
    chunks = (None if mode == "direct"
              else ({} if mode == "dist-native" else "auto"))
    ds = xr.open_zarr(path, consolidated=True, decode_times=False, chunks=chunks)
    STATE.update(ds=ds, schema=schema, mode=mode,
                 cons=[str(c) for c in ds["constituents"].values])


@asynccontextmanager
async def lifespan(app: FastAPI):
    open_store()
    yield
    STATE["ds"].close()


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)


# ---- verbatim copy of production tide_app.py:104 tide_to_output ----------
def tide_to_output(tide, lon, lat, dtime, variables, mode="time", absmax=-1):
    if all(np.all(np.isnan(np.array(tide[var])) if isinstance(tide[var], list) else np.isnan(tide[var])) for var in variables):
        return {}

    longitude, latitude = np.meshgrid(lon, lat)
    longitude_flat = longitude.ravel()
    mask = longitude_flat > 180
    longitude_flat = np.where(mask, longitude_flat - 360, longitude_flat)
    latitude_flat = latitude.ravel().tolist()

    out_dict = {
        'longitude': longitude_flat.tolist(),
        'latitude': latitude_flat,
        'time': dtime if 'time' in mode else [dtime[0]]
    }

    if 'time' in mode:
        valid_indices = set(range(len(dtime)))
        var_rechk = ['time'] + variables
    else:
        valid_indices = set(range(len(longitude_flat)))
        var_rechk = ['longitude', 'latitude'] + variables

    invalid_indices = set()
    for var in variables:
        if var in tide:
            if var == 'z' and 'time' not in mode:
                var_data = tide[var] * 100.0
            else:
                var_data = tide[var]
            invalid_indices |= set(np.where(np.isnan(var_data) | np.isinf(var_data) | (var_data == 0) | (np.abs(var_data) > absmax))[0])
            out_dict[var] = var_data.tolist()

    valid_indices = sorted(valid_indices - invalid_indices)

    for var in var_rechk:
        out_dict[var] = [out_dict[var][i] for i in valid_indices]

    if 'truncate' in mode:
        out_dict['longitude'] = [round(x, 5) for x in out_dict['longitude']]
        out_dict['latitude'] = [round(x, 5) for x in out_dict['latitude']]
        for var in variables:
            if var in out_dict:
                out_dict[var] = [round(x, 3) if x is not None else None for x in out_dict[var]]

    return out_dict
# --------------------------------------------------------------------------


def _hc_flat(sub, var: str):
    """Flat (npts, ncons) masked hc per schema/variable, in the same units
    production feeds predict.map (z: m; u/v: cm/s)."""
    nc = len(STATE["cons"])
    if STATE["schema"] == "legacy":
        amp = np.asarray(sub[f"{var}_amp"].values).reshape(-1, nc)
        ph = np.asarray(sub[f"{var}_ph"].values).reshape(-1, nc)
        # legacy store units: z amp in m, u/v amp already in cm/s
        # (empirically confirmed by the T-C unit-scale detection)
        return ma.masked_invalid(amp * np.exp(-1j * np.deg2rad(ph)))
    names = {"z": ("z_Re", "z_Im", 1e-3), "u": ("uz_Re", "uz_Im", 100.0),
             "v": ("vz_Re", "vz_Im", 100.0)}[var]
    re_v, im_v, scale = names
    hc = ma.masked_invalid(
        scale * (np.asarray(sub[re_v].values).astype(np.float64).reshape(-1, nc)
                 + 1j * np.asarray(sub[im_v].values).astype(np.float64).reshape(-1, nc)))
    flag = np.asarray(sub["z_flag"].values).reshape(-1)
    return ma.array(hc, mask=ma.getmaskarray(hc) | (flag == 2)[:, np.newaxis])


@app.get("/bench/point")
def point(lon: float, lat: float, response: Response):
    ds = STATE["ds"]
    response.headers["X-Worker-PID"] = str(os.getpid())
    coords = (dict(lat=lat, lon=lon) if STATE["schema"] == "legacy"
              else dict(lat_z=lat, lon_z=lon))
    sel = ds.sel(**coords, method="nearest")
    if STATE["schema"] == "legacy":
        amp = np.asarray(sel["z_amp"].values)[np.newaxis, :]
        ph = np.asarray(sel["z_ph"].values)[np.newaxis, :]
        hc = ma.masked_invalid(amp * np.exp(-1j * np.deg2rad(ph)))
    else:
        hc = ma.masked_invalid(1e-3 * (
            np.asarray(sel["z_Re"].values)[np.newaxis, :].astype(np.float64)
            + 1j * np.asarray(sel["z_Im"].values)[np.newaxis, :]))
    t = 11000.0 + np.arange(0, 1.0, 1 / 144.0)
    tide = predict.time_series(t, hc, STATE["cons"], deltat=0.0,
                               corrections="netcdf")
    minor = predict.infer_minor(t, hc, STATE["cons"], deltat=0.0,
                                corrections="netcdf")
    z = np.round((tide.filled(np.nan) + minor.filled(0.0)) * 100.0, 3)
    return {"pid": os.getpid(), "z_cm": z.tolist()}


@app.get("/bench/map")
def bench_map(lon0: float, lon1: float, lat0: float, lat1: float,
              sample: int):
    ds = STATE["ds"]
    if STATE["schema"] == "legacy":
        sub = ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
        sub = sub.isel(lat=slice(0, None, sample), lon=slice(0, None, sample))
        lon_vals, lat_vals = sub["lon"].values, sub["lat"].values
    else:
        sub = ds.sel(lat_z=slice(lat0, lat1), lon_z=slice(lon0, lon1))
        sub = sub.isel(lat_z=slice(0, None, sample),
                       lon_z=slice(0, None, sample))
        lon_vals, lat_vals = sub["lon_z"].values, sub["lat_z"].values
    sub = sub.compute() if STATE["mode"] != "direct" else sub

    t0 = 11000.0
    tide = {}
    for var in ("z", "u", "v"):
        hc = _hc_flat(sub, var)
        m = predict.map(t0, hc, STATE["cons"], deltat=0.0, corrections="netcdf")
        minor = predict.infer_minor(t0, hc, STATE["cons"], deltat=0.0,
                                    corrections="netcdf")
        tide[var] = np.asarray(m.filled(np.nan) + minor.filled(0.0))

    out = tide_to_output(tide, lon_vals, lat_vals, ["2022-02-15T00:00:00"],
                         ["z", "u", "v"], "map", absmax=10000.0)
    resp = ORJSONResponse(content=jsonable_encoder(out))
    resp.headers["X-Worker-PID"] = str(os.getpid())
    return resp
