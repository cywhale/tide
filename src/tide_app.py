import xarray as xr
import pandas as pd
import numpy as np
#import polars as pl
from fastapi import FastAPI, Query, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional #List
from pydantic import Field #BaseModel,
#from tempfile import NamedTemporaryFile
from datetime import date, datetime, timedelta
from src.model_utils import get_tide_time, get_tide_series, get_tide_map
import src.config as config
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


@app.get("/api/swagger/tide/openapi.json", include_in_schema=False)
async def custom_openapi():
    return JSONResponse(generate_custom_openapi())


@app.get("/api/swagger/tide", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/swagger/tide/openapi.json",
        title=app.title
    )


### Global variables: config.py ###


def to_global_lonlat(lon: float, lat: float) -> tuple:
    mlon = 180 if lon > 180 else (-180 if lon < -180 else lon)
    mlat = 90 if lat > 90 else (-90 if lat < -90 else lat)
    mlon = mlon + 360 if mlon < 0 else mlon
    return (mlon, mlat)


def tide_to_output(tide, lon, lat, dtime, variables, mode="time", absmax=-1): #, format="list"):
    # Check if tide is all NaN values
    if all(np.all(np.isnan(np.array(tide[var])) if isinstance(tide[var], list) else np.isnan(tide[var])) for var in variables): 
        # Return an empty JSON response
        #if format == 'list':
        return {}
        #return pl.DataFrame({})
    
    # Generate longitude and latitude grids
    longitude, latitude = np.meshgrid(lon, lat)

    # Flatten the longitude and latitude grids
    longitude_flat = longitude.ravel().tolist()
    latitude_flat = latitude.ravel().tolist()

    out_dict = {
        'longitude': longitude_flat,
        'latitude': latitude_flat,
        'time': dtime if mode == 'time' else [dtime[0]]
    }

    # Initialize a set to store valid indices
    if mode == 'time':
        valid_indices = set(range(len(dtime)))
        var_rechk = ['time'] + variables
    else:    
        valid_indices = set(range(len(longitude_flat)))
        var_rechk = ['longitude', 'latitude'] + variables

    # Iterate through variables
    invalid_indices = set()
    for var in variables:
        if var in tide:
            var_data = tide[var]

            # Find indices of NaN, -Inf, Inf, and 0 values
            invalid_indices |= set(np.where(np.isnan(var_data) | np.isinf(var_data) | (var_data == 0) | (np.abs(var_data) > absmax))[0])
            out_dict[var] = var_data.tolist()

    # Convert valid indices back to a sorted list
    valid_indices = sorted(valid_indices - invalid_indices)

    for var in var_rechk:
        # Filter var based on valid indices
        out_dict[var] = [out_dict[var][i] for i in valid_indices]

    return out_dict
    # Convert the dictionary to a Polars DataFrame
    # df = pl.DataFrame(out_dict)
    # return df


@app.on_event("startup")
async def startup():
    config.dz = xr.open_zarr('tpxo9.zarr', chunks='auto', decode_times=False)
    config.gridSz = 1/30
    config.timeLimit = 7
    config.LON_RANGE_LIMIT = 45
    config.LAT_RANGE_LIMIT = 45
    config.AREA_LIMIT = config.LON_RANGE_LIMIT * config.LAT_RANGE_LIMIT
    config.cons = config.dz.coords['constituents'].values


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
    sample: Optional[int] = Query(
        5, description="Re-sampling every N points(default 5)"),
    mode: Optional[str] = Query(
        None,
        description="Allowed modes: row. Optional can be none"),
    append: Optional[str] = Query(
        None, description="Data fields to append, separated by commas. If none, 'z': tide height is default. Allowed fields: z, u, v"),
    constituent: Optional[str] = Query(
        None,
        description="Allowed harmonic constituents are 'q1,o1,p1,k1,n2,m2,s1,s2,k2,m4,ms4,mn4,2n2,mf,mm'. If none, all 15 constituents will be included in evaluation. See also: https://www.tpxo.net/global")
):
    """
    Query tide from TPXO9-atlas-v5 model by longitude/latitude/date (in JSON).

    #### Usage
    * One-point tide height with time-span limitation (<= 7 days, hourly data): e.g. /tide?lon0=135&lat0=15&start=2023-07-25&end=2023-07-26T01:30:00.000Z
    * Get current in bounding-box <= 90x90 in degrees at one time moment(in ISOstring): e.g. /tide?lon0=135&lon1&=140&lat0=15&lat1=30&start=2023-07-25T01:30:00.000
    """

    if append is None:
        append = 'z'

    variables = list(set([var.strip() for var in append.split(
        ',') if var.strip() in ['z', 'u', 'v']]))
    if not variables:
        raise HTTPException(
            status_code=400, detail="Invalid variable(s). Allowed variables are 'z', 'u', 'v'")
    variables.sort()  # in-place sort not return anything

    if constituent is None:
        cons = config.cons
    else:
        cons = list(set([c.strip() for c in constituent.split(',') if c.strip() in config.cons]))
        if not cons:
            raise HTTPException(
                status_code=400, detail="Invalid constituents. Allowed constituents are 'q1','o1','p1','k1','n2','m2','s1','s2','k2','m4','ms4','mn4','2n2','mf','mm'")

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

    if sample is None or sample <= 0:
        sample = 5

    tide_time, dtime = get_tide_time(start_date, end_date)
    output_mode = 'time'

    try:
        orig_lon0, orig_lon1 = lon0, lon1
        lon0, lat0 = to_global_lonlat(lon0, lat0)

        if lon1 is None or lat1 is None or (orig_lon0 == orig_lon1 and lat0 == lat1) or (abs(lat1 - lat0) < config.gridSz and abs(orig_lon1 - orig_lon0) < config.gridSz):
            # Only one point, no date range limitation
            #dsub = config.dz.sel(lat=lat0, lon=lon0, constituents=cons, method='nearest')
            dsub = config.dz.sel(lon=slice(lon0, lon0+1.0*config.gridSz),
                                 lat=slice(lat0, lat0+1.0*config.gridSz),
                                 constituents=cons)            
            tide = {}
            for var in variables:
                amp_var = f'{var}_amp'
                ph_var = f'{var}_ph'

                ts = get_tide_series(dsub[amp_var].isel(lon=0, lat=0).values,
                                     dsub[ph_var].isel(lon=0, lat=0).values,
                                     cons, tide_time, format="netcdf", unit="cm", drop_mask=True)
                tide[var] = ts
        else:
            # Bounding box
            offset_lat = 0.0
            offset_lon = 0.0
            if lat1 == lat0 or abs(lat1 - lat0) < config.gridSz:
                offset_lat = 1.0 #1.0 means 1.0 * config.gridSz, not in degree
                #because the to_global_lonlat() not snap to grid, so must +1 gridSz to ensure have at least one point 
                #print("lat0, lat1 equal: ", lat0, lat1)

            if lat1 < lat0:
                lat0, lat1 = lat1, lat0

            if lon1 == lon0 or abs(lon1 - lon0) < config.gridSz:
                offset_lon = 1.0
                #print("lon0, lon1 equal: ", lon0, lon1)

            if lon1 < lon0:
                lon0, lon1 = lon1, lon0

            orig_lon0, orig_lon1 = lon0, lon1
            lon0, lat0 = to_global_lonlat(lon0, lat0)
            lon1, lat1 = to_global_lonlat(lon1, lat1)

            lon_range = abs(orig_lon1 - orig_lon0)
            lat_range = abs(lat1 - lat0)
            area_range = lon_range * lat_range

            if (lon_range > config.LON_RANGE_LIMIT and lat_range > config.LAT_RANGE_LIMIT) or (area_range > config.AREA_LIMIT):                           
                orig_lon1 = orig_lon0 + \
                    config.LON_RANGE_LIMIT if lon_range > config.LON_RANGE_LIMIT else orig_lon1
                lat1 = lat0 + config.LAT_RANGE_LIMIT if lat_range > config.LAT_RANGE_LIMIT else lat1
                lon1 = orig_lon1
                lon1, lat1 = to_global_lonlat(lon1, lat1)
          
            if np.sign(orig_lon0) != np.sign(orig_lon1):
                # Requested area crosses the zero meridian
                if orig_lon1 < 0:
                    # Swap if orig_lon1 < 0 and now 180 < lon1 < 360
                    lon0, lon1 = lon1, lon0
                    orig_lon0, orig_lon1 = orig_lon1, orig_lon0

                subset1 = config.dz.sel(
                    lon=slice(lon0, 360), lat=slice(lat0, lat1+offset_lat*config.gridSz), constituents=cons)
                subset2 = config.dz.sel(
                    lon=slice(0, lon1), lat=slice(lat0, lat1+offset_lat*config.gridSz), constituents=cons)
                ds1 = xr.concat([subset1, subset2], dim='lon')
            else:
                # Requested area doesn't cross the zero meridian
                ds1 = config.dz.sel(lon=slice(lon0, lon1+offset_lon*config.gridSz),
                                    lat=slice(lat0, lat1+offset_lat*config.gridSz),
                                    constituents=cons)
                
            dsub = ds1.isel(lon=slice(None, None, sample), lat=slice(None, None, sample))    
            # if not single-pont mode, only allow one datetime moment
            tide_time = tide_time[0:1]
            dtime = dtime[0:1]
            tide = get_tide_map(dsub, tide_time, format='netcdf', type=variables, drop_dim=True)
            output_mode = 'map'

        #if mode is None or mode != 'row':
        print(tide)
        out = tide_to_output(tide, dsub.coords['lon'].values, dsub.coords['lat'].values, dtime, variables, output_mode, absmax=10000.0) #, 'list')
        return JSONResponse(content=jsonable_encoder(out))
        #else:
        #    out = tide_to_output(tide, dsub.coords['lon'].values, dsub.coords['lat'].values, variables, 'dataframe')
        #    return JSONResponse(content=out.to_dicts())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
