"""W8 benchmark app (spec §7.5.3) — run inside the PRODUCTION env.

A minimal FastAPI app reproducing the production /api/tide data path
against the tpxo10 prototype store: Zarr slice -> hc -> pyTMD 2.2.8
predict (map or time_series) + infer_minor -> orjson payload. Launched by
scripts/w8_harness.py via production gunicorn (2 uvicorn workers, NO
--reload).

Env:
  TIDE_BENCH_ZARR   store path (default dev_tpxo10/stores/tpxo10_proto.zarr)
  TIDE_BENCH_OPEN   direct | dask | dist-auto | dist-native
  TIDE_BENCH_SCHED  dask scheduler address (dist-* modes)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import numpy.ma as ma
import xarray as xr
from fastapi import FastAPI, Response
from fastapi.responses import ORJSONResponse
from pyTMD import predict

STATE: dict = {}


def open_store():
    path = os.environ.get(
        "TIDE_BENCH_ZARR",
        str(Path(__file__).resolve().parent / "stores" / "tpxo10_proto.zarr"))
    mode = os.environ.get("TIDE_BENCH_OPEN", "direct")
    if mode.startswith("dist"):
        from dask.distributed import Client
        STATE["client"] = Client(os.environ["TIDE_BENCH_SCHED"])
    if mode == "direct":
        chunks = None
    elif mode == "dist-native":
        chunks = {}  # store-native chunking (one dask block per zarr chunk)
    else:  # dask | dist-auto
        chunks = "auto"
    ds = xr.open_zarr(path, consolidated=True, decode_times=False, chunks=chunks)
    STATE["ds"] = ds
    STATE["cons"] = [str(c) for c in ds["constituents"].values]
    STATE["mode"] = mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    open_store()
    yield
    STATE["ds"].close()


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)


def _hc(values_re, values_im, scale):
    hc = scale * (values_re.astype(np.float64) + 1j * values_im.astype(np.float64))
    return ma.masked_invalid(hc)


def _predict_at(hc, t0=11000.0):
    tide = predict.map(t0, hc, STATE["cons"], deltat=0.0, corrections="netcdf")
    minor = predict.infer_minor(t0, hc, STATE["cons"], deltat=0.0,
                                corrections="netcdf")
    return np.asarray(tide.filled(np.nan) + minor.filled(0.0))


@app.get("/bench/point")
def point(lon: float, lat: float, response: Response):
    ds = STATE["ds"]
    response.headers["X-Worker-PID"] = str(os.getpid())
    sel = ds.sel(lat_z=lat, lon_z=lon, method="nearest")
    zre = np.asarray(sel["z_Re"].values)[np.newaxis, :]
    zim = np.asarray(sel["z_Im"].values)[np.newaxis, :]
    hc = _hc(zre, zim, 1e-3)
    t = 11000.0 + np.arange(0, 1.0, 1 / 144.0)  # 1 day, 10-min step
    tide = predict.time_series(t, hc, STATE["cons"], deltat=0.0,
                               corrections="netcdf")
    minor = predict.infer_minor(t, hc, STATE["cons"], deltat=0.0,
                                corrections="netcdf")
    z = np.round((tide.filled(np.nan) + minor.filled(0.0)) * 100.0, 3)
    return {"pid": os.getpid(), "z_cm": z.tolist()}


@app.get("/bench/map")
def bench_map(lon0: float, lon1: float, lat0: float, lat1: float,
              sample: int, response: Response):
    ds = STATE["ds"]
    response.headers["X-Worker-PID"] = str(os.getpid())
    sub = ds[["z_Re", "z_Im", "uz_Re", "uz_Im", "vz_Re", "vz_Im", "z_flag"]].sel(
        lat_z=slice(lat0, lat1), lon_z=slice(lon0, lon1))
    sub = sub.isel(lat_z=slice(0, None, sample), lon_z=slice(0, None, sample))
    sub = sub.compute() if STATE["mode"] != "direct" else sub
    flag = np.asarray(sub["z_flag"].values)
    shape = flag.shape
    out = {"pid": os.getpid(),
           "lon": np.round(sub["lon_z"].values, 5).tolist(),
           "lat": np.round(sub["lat_z"].values, 5).tolist()}
    for name, (re_v, im_v, scale, unit_factor) in {
        "z": ("z_Re", "z_Im", 1e-3, 100.0),    # m -> cm
        "u": ("uz_Re", "uz_Im", 1.0, 100.0),   # m/s -> cm/s
        "v": ("vz_Re", "vz_Im", 1.0, 100.0),
    }.items():
        hc = _hc(np.asarray(sub[re_v].values).reshape(-1, len(STATE["cons"])),
                 np.asarray(sub[im_v].values).reshape(-1, len(STATE["cons"])),
                 scale)
        hc = ma.array(hc, mask=ma.getmaskarray(hc)
                      | (flag.reshape(-1) == 2)[:, np.newaxis])
        vals = _predict_at(hc) * unit_factor
        out[name] = np.round(vals.reshape(shape), 3).tolist()
    return out
