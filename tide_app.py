import xarray as xr
import pandas as pd
import numpy as np
import polars as pl
from fastapi import FastAPI, status, Query, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from typing import Optional, List, Union
from pydantic import BaseModel
import requests
import json
from datetime import datetime, timedelta
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
        description=('Open API to query TPXO9-v5 global tide models, compiled by ODB. Reference: Egbert, Gary D., and Svetlana Y. Erofeeva. "Efficient inverse modeling of barotropic ocean tides." Journal of Atmospheric and Oceanic Technology 19.2 (2002): 183-204.\n' +
                     '* The tide model predictions provided by this API are for reference purposes only and are intended to serve as a preliminary resource, not to be considered as definitive for scientific research or risk assessment. Users should understand that no legal liability or responsibility is assumed by the provider of this API for any decisions made based on reliance on this data. Users should conduct their own independent analysis and verification before relying on the data.\n' +
                     '* 本API提供的模型預測數據僅供參考之用，旨在做為初步的資訊來源，而不應被視為科學研究或風險評估的決定性依據。使用者須理解，對於依賴這些數據所做出的任何決策，本API提供者不承擔任何法律責任或義務。使用者在依賴這些數據前，應進行獨立分析和驗證。\n' +
                     '* Parts of this API utilize functions provided by pyTMD (https://github.com/tsutterley/pyTMD). We acknowledge and thank the original authors for their contributions.'),
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {
            "url": "https://eco.odb.ntu.edu.tw"
        }
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# @app.on_event("startup")
# async def startup():
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.dz = xr.open_zarr('data/tpxo9.zarr', chunks='auto', decode_times=False)
    config.gridSz = 1/30
    config.timeLimit = 30
    config.LON_RANGE_LIMIT = 45
    config.LAT_RANGE_LIMIT = 45
    config.AREA_LIMIT = config.LON_RANGE_LIMIT * config.LAT_RANGE_LIMIT
    config.cons = config.dz.coords['constituents'].values
    yield
    # below code to execute when app is shutting down
    config.dz.close()


app = FastAPI(lifespan=lifespan, docs_url=None, default_response_class=ORJSONResponse)


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
    return mlon, mlat


def arr_global_lonlat(lon, lat):
    # Convert lon and lat to NumPy arrays if they are not already
    lon = np.array(lon)
    lat = np.array(lat)

    # Ensure lon and lat are within -180 to 180 and -90 to 90
    lon = np.clip(lon, -180, 180)
    lat = np.clip(lat, -90, 90)

    # Convert lon to the 0-360 range
    lon = np.where(lon < 0, lon + 360, lon)

    return lon, lat


def tide_to_output(tide, lon, lat, dtime, variables, mode="time", absmax=-1):
    # Check if tide is all NaN values
    if all(np.all(np.isnan(np.array(tide[var])) if isinstance(tide[var], list) else np.isnan(tide[var])) for var in variables):
        # Return an empty JSON response
        #if format == 'list':
        return {}
        #return pl.DataFrame({})

    # Generate longitude and latitude grids
    longitude, latitude = np.meshgrid(lon, lat)

    # Flatten the longitude and latitude grids
    longitude_flat = longitude.ravel()
    mask = longitude_flat > 180
    longitude_flat = np.where(mask, longitude_flat - 360, longitude_flat)
    latitude_flat = latitude.ravel().tolist()

    out_dict = {
        'longitude': longitude_flat.tolist(),
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


class TideResponse(BaseModel):
    longitude: float
    latitude: float
    time: str
    z: Optional[float]
    u: Optional[float]
    v: Optional[float]


@app.get("/api/tide", response_model=List[TideResponse], tags=["Tide"], summary="Query tide height and tidal current")
async def get_tide(
    lon0: float = Query(...,
                        description="Minimum longitude, range: [-180, 180]"),
    lat0: float = Query(..., description="Minimum latitude, range: [-90, 90]"),
    lon1: Optional[float] = Query(
        None, description="Maximum longitude, range: [-180, 180]"),
    lat1: Optional[float] = Query(
        None, description="Maximum latitude, range: [-90, 90]"),
    start: Optional[str] = Query(
        None, description="Start datetime (UTC) of tide data to query. If none, current datetime is default"),
    end: Optional[str] = Query(
        None, description="End datetime (UTC) of tide data to query"),
    sample: Optional[int] = Query(
        5, description="Re-sampling every N points(default 5)"),
    mode: Optional[str] = Query(
        None,
        description="Allowed modes: list. Optional can be none (default output is list). Multiple/special modes can be separated by comma."),
    tol: Optional[float] = Query(
        None,
        description="Tolerance for nearest method to locate points. Nearest method can explictly specified in mode as a special mode 'nearest', or by just giving tolerance value. Default tolerance is ±1/30 degree, and maximum is ±0.25 degree."),
    append: Optional[str] = Query(
        None, description="Data fields to append, separated by commas. If none, 'z': tide height is default. Allowed fields: z,u,v"),
    constituent: Optional[str] = Query(
        None,
        description="Allowed harmonic constituents are 'q1,o1,p1,k1,n2,m2,s1,s2,k2,m4,ms4,mn4,2n2,mf,mm'. If none, all 15 constituents will be included in evaluation. See also: https://www.tpxo.net/global")
):
    """
    Query tide from TPXO9-atlas-v5 model by longitude/latitude/date (in JSON).

    #### Usage
    * One-point tide height with time-span limitation (<= 30 days, hourly data): e.g. /tide?lon0=125&lat0=15&start=2023-07-25&end=2023-07-26T01:30:00.000
    * Get current in bounding-box <= 45x45 in degrees at one time moment(in ISOstring): e.g. /tide?lon0=125&lon1&=135&lat0=15&lat1=30&start=2023-07-25T01:30:00.000
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
    if mode is None:
        mode = 'list'

    try:
        if lon1 is None or lat1 is None or (lon0 == lon1 and lat0 == lat1) or (abs(lat1 - lat0) < config.gridSz and abs(lon1 - lon0) < config.gridSz):
            # Only one point, no date range limitation
            lon0, lat0 = to_global_lonlat(lon0, lat0)
            findNear = False
            if 'nearest' in mode:
                findNear = True

            if tol not in [np.nan, np.NaN, None] or findNear:
                findNear = True
                if tol in [np.nan, np.NaN, None] or tol <= 0:
                    tol = config.gridSz
                elif tol > 7.5*config.gridSz:
                    tol = 7.5*config.gridSz

            if findNear:
                dsub = config.dz.sel(lon=lon0, lat=lat0, method="nearest", tolerance=tol).sel(constituents=cons)
            else:
                dsub = config.dz.sel(lon=slice(lon0-0.5*config.gridSz, lon0+0.5*config.gridSz),
                                     lat=slice(lat0-0.5*config.gridSz, lat0+0.5*config.gridSz),
                                     constituents=cons)
            tide = {}
            for var in variables:
                amp_var = f'{var}_amp'
                ph_var = f'{var}_ph'

                if findNear:
                    ts = get_tide_series(dsub[amp_var].values, dsub[ph_var].values,
                                         cons, tide_time, format="netcdf", unit="cm", drop_mask=True)
                else:
                    ts = get_tide_series(dsub[amp_var].isel(lon=0, lat=0).values,
                                         dsub[ph_var].isel(lon=0, lat=0).values,
                                         cons, tide_time, format="netcdf", unit="cm", drop_mask=True)
                tide[var] = ts
        else:
            # Bounding box
            if lat1 < lat0:
                lat0, lat1 = lat1, lat0
            if lon1 < lon0:
                lon0, lon1 = lon1, lon0

            orig_lon0, orig_lon1 = lon0, lon1
            lon0, lat0 = to_global_lonlat(lon0, lat0)
            lon1, lat1 = to_global_lonlat(lon1, lat1)

            lon_range = abs(orig_lon1 - orig_lon0) #cannot use lon0, lon1 to evaluate range if cross-zero
            lat_range = abs(lat1 - lat0)
            area_range = lon_range * lat_range

            if (lon_range > config.LON_RANGE_LIMIT and lat_range > config.LAT_RANGE_LIMIT) or (area_range > config.AREA_LIMIT):
                orig_lon1 = orig_lon0 + \
                    config.LON_RANGE_LIMIT if lon_range > config.LON_RANGE_LIMIT else orig_lon1
                # print("Greater than range with lon, lat:", lon0, lat0, lon1, lat1, orig_lon0, orig_lon1)
                lat1 = lat0 + config.LAT_RANGE_LIMIT if lat_range > config.LAT_RANGE_LIMIT else lat1
                lon1 = orig_lon0 + config.LON_RANGE_LIMIT if lon_range > config.LON_RANGE_LIMIT else orig_lon1
                orig_lon1 = lon1
                lon1, lat1 = to_global_lonlat(lon1, lat1)

            if np.sign(orig_lon0) != np.sign(orig_lon1):
                # print("Cross-zero lon, lat:", lon0, lat0, lon1, lat1, orig_lon0, orig_lon1)
                # Requested area crosses the zero meridian
                # The following should not happen because lon1 < lon0 had been swapped aboving
                #if orig_lon1 < 0:
                #    # Swap if orig_lon1 < 0 and now 180 < lon1 < 360
                #    lon0, lon1 = lon1, lon0
                #    orig_lon0, orig_lon1 = orig_lon1, orig_lon0
                subset1 = config.dz.sel(
                    lon=slice(lon0-0.5*config.gridSz, 360),
                    lat=slice(lat0-0.5*config.gridSz, lat1+0.5*config.gridSz),
                    constituents=cons)
                subset2 = config.dz.sel(
                    lon=slice(0, lon1+0.5*config.gridSz),
                    lat=slice(lat0-0.5*config.gridSz, lat1+0.5*config.gridSz),
                    constituents=cons)
                ds1 = xr.concat([subset1, subset2], dim='lon')
            else:
                # Requested area doesn't cross the zero meridian
                # print("Current subsetting lon, lat:", lon0, lat0, lon1, lat1)
                ds1 = config.dz.sel(lon=slice(lon0-0.5*config.gridSz, lon1+0.5*config.gridSz),
                                    lat=slice(lat0-0.5*config.gridSz, lat1+0.5*config.gridSz),
                                    constituents=cons)

            dsub = ds1.isel(lon=slice(None, None, sample), lat=slice(None, None, sample))
            # if not single-pont mode, only allow one datetime moment
            tide_time = tide_time[0:1]
            dtime = dtime[0:1]
            tide = get_tide_map(dsub, tide_time, format='netcdf', type=variables, drop_dim=True)
            output_mode = 'map'

        #if mode is None or mode != 'row':
        #print(tide)
        out = tide_to_output(tide, dsub.coords['lon'].values, dsub.coords['lat'].values, dtime, variables, output_mode, absmax=10000.0) #, 'list')
        return ORJSONResponse(content=jsonable_encoder(out))
        #else:
        #    out = tide_to_output(tide, dsub.coords['lon'].values, dsub.coords['lat'].values, variables, 'dataframe')
        #    return ORJSONResponse(content=out.to_dicts())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def numarr_query_validator(qry):
    if ',' in qry:
        try:
            out = np.array([float(x.strip()) for x in qry.split(',')])
            return (out)
        except ValueError:
            return ("Format Error")
    else:
        try:
            out = np.array([float(qry.strip())])
            return (out)
        except ValueError:
            return ("Format Error")


#def custom_encoder(obj):
#    if isinstance(obj, np.ndarray):
#        return obj.tolist()  # Convert NumPy arrays to Python lists
#    elif isinstance(obj, np.generic):
#        return np.asscalar(obj)
#    else:
#        return obj
def custom_encoder(obj):
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj  # Basic types are already JSON serializable
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to Python lists
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    elif isinstance(obj, dict):
        # Recursively encode values in dictionaries
        return {key: custom_encoder(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively encode elements in lists or tuples
        return [custom_encoder(item) for item in obj]
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

""" ---- wide format ----
def const_to_output(data_dict): #, data_var='amp'):
    # Initialize an empty DataFrame with longitude and latitude columns
    #if data_var == 'amp':
    #    varx = 'amplitude'
    #elif data_var == 'ph':
    #    varx = 'phase'
    #else:
    #    varx = 'hc_real'
    columns = ['longitude', 'latitude']
    df = pd.DataFrame(columns=columns)

    # Iterate through data_dict and dynamically append columns based on keys
    for data in data_dict: #enumerate(data_dict): #[varx]):
        # Extract longitude and latitude
        longitude = data['longitude'] #[idx]
        latitude = data['latitude'] #[idx]
        row_dict = {'longitude': longitude, 'latitude': latitude}

        # Iterate through amp, ph, hc_real, and hc_imag keys in data_dict
        for key in data.keys():
            if key != 'longitude' and key != 'latitude':
                for constituent_type, value in data[key].items():
                    constituent, data_type = constituent_type.split('_')
                    col_name = f"{key}_{constituent}_{data_type}"
                    row_dict[col_name] = value

        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    return df
"""
def data_to_wide(df, mode):
    # df = df.dropna().reset_index(drop=True)
    # print(df['longitude'].apply(type).unique())
    if 'onlyOnePt' in mode:
        df['value'] = df['value'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    wide_format = df.pivot_table(index=['longitude', 'latitude', 'grid_lon', 'grid_lat', 'type'],
                                 columns=['constituents', 'variable'],
                                 values='value').reset_index()
    #if 'uppercase' in mode:
    #    col_names = ['longitude', 'latitude', 'type'] + \
    #                [f"{c[0].upper()}_{c[1]}"for c in wide_format.columns[3:]]
    #else:
    col_names = ['longitude', 'latitude', 'grid_lon', 'grid_lat', 'type'] + \
                [f"{c[0]}_{c[1]}"for c in wide_format.columns[5:]]

    wide_format.columns = col_names
    return wide_format

def const_to_output(data_dict, mode):
    data_list = []

    for data in data_dict:
        longitude = data['longitude']
        latitude = data['latitude']

        for key in data.keys():
            if key != 'longitude' and key != 'latitude':
                for constituent_type, value in data[key].items():
                    constituent, data_type = constituent_type.split('_')
                    row_dict = {
                        'longitude': longitude,
                        'latitude': latitude,
                        'variable': key,
                        'constituents': constituent,
                        'type': data_type,
                        'value': value  # Store the value
                    }
                    data_list.append(row_dict)

    df = pd.DataFrame(data_list)
    if 'long' in mode:
        return df

    return data_to_wide(df, mode)


def const_to_output_vec(data_dict, mode):
    data_list = []
    lon_values = data_dict['longitude']
    lat_values = data_dict['latitude']

    # Extract keys that are not 'longitude' or 'latitude'
    other_keys = [key for key in data_dict.keys() if key not in ['longitude', 'latitude', 'grid_lon', 'grid_lat']]

    for idx, (longitude, latitude) in enumerate(zip(lon_values, lat_values)):
        for key in other_keys:
            data_type, constituent, var_type = key.split('_')
            value = data_dict[key][idx]
            row_dict = {
                'longitude': longitude,
                'latitude': latitude,
                'grid_lon': data_dict['grid_lon'][idx],
                'grid_lat': data_dict['grid_lat'][idx],
                'variable': var_type,  # Extracting the variable type (amp, ph, etc.) from the key
                'constituents': constituent,
                'type': data_type,  # Extracting the type (u, v, etc.) from the key
                'value': value  # Store the value
            }
            data_list.append(row_dict)

    df = pd.DataFrame(data_list)
    if 'long' in mode:
        return df

    return data_to_wide(df, mode)


def get_constituent_vec(
        dsub, loni, lati, vars=['amp', 'ph'],
        constituent=['q1', 'o1', 'p1', 'k1', 'n2', 'm2', 's1', 's2', 'k2', 'm4', 'ms4', 'mn4', '2n2', 'mf', 'mm'],
        type=['u', 'v']):
    out = {'longitude': loni.tolist(),
           'latitude': lati.tolist(),
           'grid_lon': dsub['lon'].values.tolist(),
           'grid_lat': dsub['lat'].values.tolist()}

    for TYPE in type:
        amp_all = dsub[TYPE+'_amp'].values
        ph_all = dsub[TYPE+'_ph'].values
        for idx, const in enumerate(constituent):
            key = f"{TYPE}_{const}"
            amp = amp_all[..., idx]
            ph = ph_all[..., idx]
            cph = -1j * ph * np.pi / 180.0
            hc = amp * np.exp(cph)
            if 'amp' in vars:
                out[key+"_amp"] = amp.tolist()
            if 'ph' in vars:
                out[key+"_ph"] = ph.tolist()
            if 'hc' in vars:
                out[key+"_real"] = hc.real.tolist()
                out[key+"_imag"] = hc.imag.tolist()
    return out


def get_constituent(dz, lon, lat, vars=['amp', 'ph'],
                    constituent=['q1', 'o1', 'p1', 'k1', 'n2', 'm2', 's1', 's2', 'k2', 'm4', 'ms4', 'mn4', '2n2', 'mf', 'mm'],
                    type=['u', 'v']):
    # Note dz should be a filtered zarr dataset including only filtered constituents
    amplitudes = {}
    phase = {}
    imag = {}
    real = {}
    out = {}
    lon = lon-360 if lon > 180 else lon
    out['longitude'] = lon
    out['latitude'] = lat
    # Just debug if (lon == 120.1375 and lat == 23.61861):
    #                print("Find target: ", dz)
    # vars = list(set([var.strip() for var in mode.split(',') if var.strip() in ['amp', 'ph', 'hc']]))
    if not vars:
        vars = ['amp', 'ph']

    for TYPE in type:
        for const in constituent:
            key = f"{const}_{TYPE}"
            amp = dz[TYPE+'_amp'].sel(constituents=const).values
            amplitudes[key] = float(amp.ravel())
            ph = dz[TYPE+'_ph'].sel(constituents=const).values
            phase[key] = float(ph.ravel())
            cph = -1j * ph * np.pi / 180.0
            # Calculate constituent oscillation
            hc = amp * np.exp(cph)
            imag[key] = float(hc.imag.ravel())
            real[key] = float(hc.real.ravel())

    if 'amp' in vars:
        out["amp"] = amplitudes

    if 'ph' in vars:
        out["ph"] = phase

    if 'hc' in vars:
        out["real"] = real
        out["imag"] = imag

    return out


class ConstMinResponse(BaseModel):
    longitude: float
    latitude: float
    grid_lon: float
    grid_lat: float
    type: str


@app.get("/api/tide/const", response_model=List[Union[ConstMinResponse, dict]],
         tags=["Tide"], summary="Get harmonic constituents of TPXO9 model")
async def get_tide_const(
    lon: Optional[str] = Query(
            None,
            description="comma-separated longitude values. One of lon/lat and jsonsrc should be specified as longitude/latitude input.",
            example="122.36,122.47"),
    lat: Optional[str] = Query(
            None,
            description="comma-separated latitude values. One of lon/lat and jsonsrc should be specified as longitude/latitude input.",
            example="25.02,24.82"),
    mode: Optional[str] = Query(
        None,
        description="Allowed modes: list, object, row (dataframe in wide format; long-format dataframe is also available as a special mode 'long'). Optional can be none (default output is list). Multiple/special modes can be separated by comma."),
    tol: Optional[float] = Query(
        None,
        description="Tolerance for nearest method to locate points. Nearest method can explictly specified in mode as a special mode 'nearest', or by just giving tolerance value. Default tolerance is ±1/30 degree, and maximum is ±0.25 degree."),
    append: Optional[str] = Query(
        None, description="Data fields to append, separated by commas. If none, 'z': tide height is default. Allowed fields: z,u,v"),
    constituent: Optional[str] = Query(
        None,
        description="Allowed harmonic constituents are 'q1,o1,p1,k1,n2,m2,s1,s2,k2,m4,ms4,mn4,2n2,mf,mm'. If none, all 15 constituents will be included in evaluation. See also: https://www.tpxo.net/global"),
    complex: Optional[str] = Query(
        None, description="Harmonic complex constants for output, separated by commas. If none, 'amp,ph' is default. Allowed variables: amp, ph, hc, which means amplitude, phase, harmonic in complex (real, imag), respectively"),
    jsonsrc: Optional[str] = Query(
        None,
        description='Optional. A valid URL for JSON source or a JSON string that contains longitude and latitude keys with values in array.\n' +
                    'Example: {"longitude":[122.36,122.47,122.56,122.66],"latitude":[25.02,24.82,24.72,24.62]}')
):
    """
    Query harmonic constituents from TPXO9-atlas-v5 model by longitude/latitude.

    #### Usage
    * e.g. /tide/const?lon=122.36,122.47&lat=25.02,24.82&constituent=k1,m2,n2,o1,p1,s2&complex=amp,ph,hc&append=z,u,v
    """
    try:
        if jsonsrc:
            # Validate it's a URL
            try:
                json_resp = requests.get(jsonsrc)
                json_resp.raise_for_status()
                json_obj = json_resp.json()
            except:  # noqa: E722
                try:
                    json_obj = json.loads(jsonsrc)
                except:  # noqa: E722
                    raise ValueError("Input jsonsrc must be a valid URL or a JSON string.")

            # Validate the JSON has 'longitude' and 'latitude' keys
            # LonLat(**json_obj)
            loni = np.array(json_obj['longitude'])
            lati = np.array(json_obj['latitude'])
        else:
            if lon and lat:
                loni = numarr_query_validator(lon)
                lati = numarr_query_validator(lat)

                if isinstance(loni, str) or isinstance(lati, str):
                    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                        content=jsonable_encoder({"Error": "Check your input format should be comma-separated values"}))
            else:
                raise ValueError("Both 'lon' and 'lat' parameters must be provided, otherwise use 'jsonsrc' as input")

    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"Error": str(e)})
    except requests.HTTPError as e:
        return JSONResponse(status_code=e.response.status_code,
                            content={"Error": str(e)})

    if len(loni) != len(lati):
        # config.dz.close()
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content=jsonable_encoder({"Error": "Check your input of lon/lat should be in equal length"}))
    onlyOnePt = False
    if len(loni) == 1:
        onlyOnePt = True

    mlon, mlat = arr_global_lonlat(loni, lati) #to 0-360

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

    if complex is None:
        complex = 'amp,ph'

    if ',' in complex:
        pars = list(set([par.strip() for par in complex.split(
            ',') if par.strip() in ['amp', 'ph', 'hc']]))
    elif complex.strip() in ['amp', 'ph', 'hc']:
        pars=[complex.strip()]

    if not pars:
        pars = ['amp', 'ph']

    mode = 'list' if mode is None else mode.lower()

    if onlyOnePt:
        mode = mode + ',onlyOnePt'

    findNear = False
    if 'nearest' in mode:
        findNear = True

    if tol not in [np.nan, np.NaN, None] or findNear:
        findNear = True
        if tol in [np.nan, np.NaN, None] or tol <= 0:
            tol = config.gridSz
        elif tol > 7.5*config.gridSz:
            tol = 7.5*config.gridSz

    #pre-subsetting if bounding box within 45 x 45 degrees
    if not onlyOnePt:
        min_lon, max_lon = min(loni), max(loni)
        min_lat, max_lat = min(lati), max(lati)
        lon_rng = max_lon - min_lon
        lat_rng = max_lat - min_lat
        if (lon_rng > config.LON_RANGE_LIMIT and lat_rng > config.LAT_RANGE_LIMIT) or (
            lon_rng * lat_rng > config.AREA_LIMIT) or np.sign(min_lon) != np.sign(max_lon):
            # Note if sign is different, do pre-subset may cause error because we must use slice in ds.sel
            ds = config.dz.sel(constituents=cons)
        else:
            #if np.sign(min_lon) != np.sign(max_lon):
            #    min_lon, max_lon = min(mlon), max(mlon)
            #    subset1 = config.dz.sel(
            #            lon=slice(min_lon, 360),
            #            lat=slice(min_lat, max_lat+1.0*config.gridSz),
            #            constituents=cons)
            #    subset2 = config.dz.sel(
            #            lon=slice(0, max_lon+1.0*config.gridSz),
            #            lat=slice(min_lat, max_lat+1.0*config.gridSz),
            #            constituents=cons)
            #    ds = xr.concat([subset1, subset2], dim='lon')
            #else:
            min_lon, max_lon = min(mlon), max(mlon)
            ds = config.dz.sel(lon=slice(min_lon-0.5*config.gridSz, max_lon+0.5*config.gridSz),
                               lat=slice(min_lat-0.5*config.gridSz, max_lat+0.5*config.gridSz),
                               constituents=cons)
        #vectorized version
        # Create a multi-dimensional coordinate array for vectorized selection
        coords = xr.DataArray(np.arange(len(mlon)),
                    coords={'points_lon': ('points', mlon),
                            'points_lat': ('points', mlat)}, dims='points')
        if findNear:
            dsub = ds.sel(lon=coords.points_lon, lat=coords.points_lat, method="nearest", tolerance=tol)
        else:
            dsub = ds.sel(lon=coords.points_lon, lat=coords.points_lat, method="nearest", tolerance=0.5*config.gridSz)

    else:
        if findNear:
            dsub = config.dz.sel(lon=mlon[0], lat=mlat[0], method="nearest", tolerance=tol)
        else:
            dsub = config.dz.sel(lon=slice(mlon[0]-0.5*config.gridSz, mlon[0]+0.5*config.gridSz),
                                 lat=slice(mlat[0]-0.5*config.gridSz, mlat[0]+0.5*config.gridSz))

    out = get_constituent_vec(dsub, loni, lati, vars=pars, constituent=cons, type=variables)
    #nested-loop version
    #out = []
    #for lon0, lat0 in zip(mlon, mlat):
    #    if findNear:
    #        dsub = ds.sel(lon=lon0, lat=lat0, method="nearest", tolerance=tol)
    #    else:
    #        dsub = ds.sel(lon=slice(lon0, lon0+1.0*config.gridSz),
    #                      lat=slice(lat0, lat0+1.0*config.gridSz))
    #    #results = {}
    #    #results['longitude'] = lon0
    #    #results['latitude'] = lat0
    #    #results['grid_lon'] = dsub["lon"].values[0]
    #    #results['grid_lat'] = dsub["lat"].values[0]
    #    constants = get_constituent(dsub, lon0, lat0, vars=pars, constituent=cons, type=variables)
    #    #for key, value in constants.items():
    #    #    results[key] = value
    #    out.append(constants)
    # print("Test vec version:", out)

    if mode is not None and 'object' in mode:
        # Serialize the data to JSON
        # json_data = json.dumps(out_encoded)
        return ORJSONResponse(content=jsonable_encoder(custom_encoder(out)))

    dfout = const_to_output_vec(out, mode)
    #dfout = dfout.where(pd.notna(dfout), None)
    #print(dfout)
    if mode is not None and 'row' in mode:
        df1 = pl.from_pandas(dfout)
        return ORJSONResponse(content=df1.to_dicts())

    return ORJSONResponse(content=dfout.to_dict())
