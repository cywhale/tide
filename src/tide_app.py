import xarray as xr
import pandas as pd
import numpy as np
import polars as pl
from fastapi import FastAPI, Query, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
from pydantic import BaseModel, Field
from tempfile import NamedTemporaryFile
from datetime import date, datetime, timedelta
from model_utils import get_tide_time, get_tide_series, get_tide_map
import config
from dask.distributed import Client
client = Client('tcp://localhost:8786')


def generate_custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ODB Tide API",
        version="1.0.0",
        # description="ODB geoconv API schema",
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {
            "url": "https://eco.odb.ntu.edu.tw"
        }
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI(root_path="/tide", docs_url=None)


@app.get("/tide/swagger/openapi.json", include_in_schema=False)
async def custom_openapi():
    return JSONResponse(generate_custom_openapi())


@app.get("/tide/swagger", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/tide/swagger/openapi.json",
        title=app.title
    )

### Global variables: config.py ###


def to_gobal_lonlat(lon: float, lat: float) -> tuple:
    mlon = 180 if lon > 180 else (-180 if lon < -180 else lon)
    mlat = 90 if lat > 90 else (-90 if lat < -90 else lat)
    mlon = mlon + 360 if mlon < 0 else mlon
    return (mlon, mlat)


@app.on_event("startup")
async def startup():
    config.dz = xr.open_zarr('tpxo9.zarr', chunks='auto', decode_times=False)
    config.gridSz = 1/30
    config.timeLimit = 7
    config.LON_RANGE_LIMIT = 90
    config.LAT_RANGE_LIMIT = 90
    config.AREA_LIMIT = config.LON_RANGE_LIMIT * config.LAT_RANGE_LIMIT


@app.get("/tide")
async def get_tide(
    lon0: float = Query(...,
                        description="Minimum longitude, range: [-180, 180]"),
    lat0: float = Query(..., description="Minimum latitude, range: [-90, 90]"),
    lon1: Optional[float] = Query(
        None, description="Maximum longitude, range: [-180, 180]"),
    lat1: Optional[float] = Query(
        None, description="Maximum latitude, range: [-90, 90]"),
    start: Optional[date] = Query(
        None, description="Start datetime (UTC) of tide data to query. If none, current datetime is default"),
    end: Optional[date] = Query(
        None, description="End datetime (UTC) of tide data to query"),
    mode: Optional[str] = Query(
        None,
        description="Allow modes: row. Optional can be none"),
    append: Optional[str] = Query(
        None, description="Data fields to append, separated by commas. If none, 'h': tide level is default. Allowed fields: h, u, v")
):
    """
    Query tide from TPXO9-atlas-v5 model by longitude/latitude/date (in JSON).

    #### Usage
    * One-point tide level with time-span limitation (<= 7 days, hourly data): e.g. /tide?lon0=135&lat0=15&start=2023-07-25&end=2023-07-26T01:30:00.000Z
    * Get current in bounding-box <= 90x90 in degrees at one time moment(in ISOstring): e.g. /tide?lon0=135&lon1&=140&lat0=15&lat1=30&start=2023-07-25T01:30:00.000
    """

    if append is None:
        append = 'h'

    variables = list(set([var.strip() for var in append.split(
        ',') if var.strip() in ['h', 'u', 'v']]))
    if not variables:
        raise HTTPException(
            status_code=400, detail="Invalid variable(s). Allowed variables are 'h', 'u', 'n'.")
    variables.sort()  # in-place sort not return anything

    if start is None:
        start_date = pd.to_datetime(datetime.now())
    else:
        try:
            start_date = pd.to_datetime(start)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid start datetime format")

    if end is None:
        end_date = start_date
    else:
        try:
            end_date = pd.to_datetime(end)
            if (end_date - start_date).days > config.timeLimit:
                end_date = start_date + timedelta(days=config.timeLimit)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid end datetime format")

    if end_date < start_date:
        start_date, end_date = end_date, start_date

    tide_time, dtime = get_tide_time(start_date, end_date)

    try:
        orig_lon0, orig_lon1 = lon0, lon1
        lon0, lat0 = to_gobal_lonlat(lon0, lat0)

        if lon1 is None or lat1 is None or (orig_lon0 == orig_lon1 and lat0 == lat1):
            # Only one point, no date range limitation
            dsub = config.dz.sel(lon=slice(lon0-0.5*config.gridSz, lon0+0.5*config.gridSz-0.001),
                                 lat=slice(lat0-0.5*config.gridSz, lat0+0.5*config.gridSz-0.001))
            out = {}
            for var in variables:
                amp_var = f'{var}_amp'
                ph_var = f'{var}_ph'

                ts = get_tide_series(dsub[amp_var].isel(lon=0, lat=0).values,
                                     dsub[ph_var].isel(lon=0, lat=0).values,
                                     dsub.coords['constituents'].values, tide_time)
                out[var] = ts
        else:
            # Bounding box
            if lat1 < lat0:
                lat0, lat1 = lat1, lat0

            lon1, lat1 = to_gobal_lonlat(lon1, lat1)

            lon_range = abs(orig_lon1 - orig_lon0)
            lat_range = abs(lat1 - lat0)
            area_range = lon_range * lat_range

            if (lon_range > config.LON_RANGE_LIMIT and lat_range > config.LAT_RANGE_LIMIT) or (area_range > config.AREA_LIMIT):
                if orig_lon1 > orig_lon0:
                    orig_lon1 = orig_lon0 + \
                        config.LON_RANGE_LIMIT if lon_range > config.LON_RANGE_LIMIT else orig_lon1
                else:
                    orig_lon1 = orig_lon0 - \
                        config.LON_RANGE_LIMIT if lon_range > config.LON_RANGE_LIMIT else orig_lon1
                lat1 = lat0 + config.LAT_RANGE_LIMIT if lat_range > config.LAT_RANGE_LIMIT else lat1
                lon1 = orig_lon1
                lon1, lat1 = to_gobal_lonlat(lon1, lat1)

            if lon0 > lon1 and np.sign(orig_lon0) == np.sign(orig_lon1):
                # Swap if lon0 > lon1 but the same sign
                lon0, lon1 = lon1, lon0
                orig_lon0, orig_lon1 = orig_lon1, orig_lon0

            if np.sign(orig_lon0) != np.sign(orig_lon1):
                # Requested area crosses the zero meridian
                if orig_lon1 < 0:
                    # Swap if orig_lon1 < 0 and now 180 < lon1 < 360
                    lon0, lon1 = lon1, lon0
                    orig_lon0, orig_lon1 = orig_lon1, orig_lon0

                subset1 = config.dz.sel(
                    lon=slice(lon0, 360), lat=slice(lat0, lat1))
                subset2 = config.dz.sel(
                    lon=slice(0, lon1), lat=slice(lat0, lat1))
                dsub = xr.concat([subset1, subset2], dim='lon')
            else:
                # Requested area doesn't cross the zero meridian
                dsub = config.dz.sel(lon=slice(lon0, lon1),
                                     lat=slice(lat0, lat1))

            # if not single-pont mode, only allow one datetime moment
            tide_time = tide_time[0:1]
            dtime = dtime[0:1]
            out = get_tide_map(dsub, tide_time, format='netcdf',
                               type=variables, drop_dim=True)

        if mode is None or mode != 'row':
            return JSONResponse(content=jsonable_encoder(out))
        else:
            df = p

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
