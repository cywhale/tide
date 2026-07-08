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
from typing import Optional
import requests
import json
from datetime import datetime, timedelta
from src.model_utils import get_tide_time, get_tide_series, get_tide_map
import src.config as config
from src import store_adapter
from src.store_adapter import get_zarr_path
from src.query_planner import plan_bbox, get_max_bbox_cells, BboxCapError, EmptyBboxError
# from dask.distributed import Client
# client = Client('tcp://localhost:8786')
from src.dask_client_manager import get_dask_client, close_dask_client
from src.tide_forecast import forecast_router
client = get_dask_client("tideapi")


def generate_custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ODB Tide API",
        version="1.1.0",
        description=('Open API to query TPXO global tide models (TPXO10-atlas-v2), compiled by ODB. Reference: Egbert, Gary D., and Svetlana Y. Erofeeva. "Efficient inverse modeling of barotropic ocean tides." Journal of Atmospheric and Oceanic Technology 19.2 (2002): 183-204.\n' +
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
    # v0.3.0 Stage 3 (D9): open via the schema-keyed store adapter; the
    # store is resolved by TIDE_ZARR_PATH (default data/tpxo10.zarr) so a
    # rollback to data/tpxo9.zarr is a single env change + restart.
    config.adapter = store_adapter.open_store(get_zarr_path())
    config.dz = config.adapter.ds           # coords/metadata access
    config.gridSz = 1/30
    config.timeLimit = 30
    config.LON_RANGE_LIMIT = 45
    config.LAT_RANGE_LIMIT = 45
    config.AREA_LIMIT = config.LON_RANGE_LIMIT * config.LAT_RANGE_LIMIT
    config.cons = np.asarray(config.adapter.constituents)
    config.MAX_BBOX_CELLS = get_max_bbox_cells()
    yield
    # below code to execute when app is shutting down
    config.adapter.ds.close()
    close_dask_client("tideapi")


app = FastAPI(lifespan=lifespan, docs_url=None, default_response_class=ORJSONResponse)
app.include_router(forecast_router)


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
        'time': dtime if 'time' in mode else [dtime[0]]
    }

    # Initialize a set to store valid indices
    if 'time' in mode:
        valid_indices = set(range(len(dtime)))
        var_rechk = ['time'] + variables
    else:
        valid_indices = set(range(len(longitude_flat)))
        var_rechk = ['longitude', 'latitude'] + variables

    # Iterate through variables
    invalid_indices = set()
    for var in variables:
        if var in tide:
            if var == 'z' and 'time' not in mode:
                var_data = tide[var] * 100.0  # map z is m; API returns cm
            else:
                var_data = tide[var]

            # Find indices of NaN, -Inf, Inf, and 0 values
            invalid_indices |= set(np.where(np.isnan(var_data) | np.isinf(var_data) | (var_data == 0) | (np.abs(var_data) > absmax))[0])
            out_dict[var] = var_data.tolist()

    # Convert valid indices back to a sorted list
    valid_indices = sorted(valid_indices - invalid_indices)

    for var in var_rechk:
        # Filter var based on valid indices
        out_dict[var] = [out_dict[var][i] for i in valid_indices]

    # Apply flow_trunc truncation for output_mode="map" and special mode="flow_trunc"
    if 'truncate' in mode: # allow both 'time' and 'map' all has truncate mode
        # Truncate longitude and latitude to 5 decimal places
        out_dict['longitude'] = [round(x, 5) for x in out_dict['longitude']]
        out_dict['latitude'] = [round(x, 5) for x in out_dict['latitude']]

        # Truncate tide values to 3 decimal places
        for var in variables:
            if var in out_dict:
                out_dict[var] = [round(x, 3) if x is not None else None for x in out_dict[var]]

    return out_dict
    # Convert the dictionary to a Polars DataFrame
    # df = pl.DataFrame(out_dict)
    # return df


def _ordered_valid_tokens(text: str, allowed) -> list:
    """Parse comma-separated tokens, preserving order and de-duplicating."""
    out = []
    seen = set()
    allowed_set = set(allowed)
    for token in text.split(','):
        token = token.strip()
        if token in allowed_set and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def _openapi_example(value):
    return {"example": {"value": value}}


@app.get("/api/tide", tags=["Tide"], summary="Query tide height and tidal current")
async def get_tide(
    lon0: float = Query(...,
                        description="Minimum longitude, range: [-180, 180]",
                        openapi_examples=_openapi_example(-157.86453)),
    lat0: float = Query(..., description="Minimum latitude, range: [-90, 90]",
                        openapi_examples=_openapi_example(21.303333)),
    lon1: Optional[float] = Query(
        None, description="Maximum longitude for bbox/map queries, range: [-180, 180]",
        openapi_examples=_openapi_example(-157.6)),
    lat1: Optional[float] = Query(
        None, description="Maximum latitude for bbox/map queries, range: [-90, 90]",
        openapi_examples=_openapi_example(21.6)),
    start: Optional[str] = Query(
        None, description="Start datetime (UTC). If omitted, current datetime is used.",
        openapi_examples=_openapi_example("2023-07-25T00:00:00")),
    end: Optional[str] = Query(
        None, description="End datetime (UTC). Point time series are limited to 30 days.",
        openapi_examples=_openapi_example("2023-07-26T00:00:00")),
    sample: Optional[int] = Query(
        5,
        description="Stride for bbox/map output grid. Default 5. sample=1 returns every selected grid cell and may hit MAX_BBOX_CELLS=500000.",
        openapi_examples=_openapi_example(5)),
    mode: Optional[str] = Query(
        None,
        description="Optional comma-separated modes. `truncate` rounds lon/lat to 5 decimals and values to 3 decimals; `nearest` enables nearest-point tolerance behavior.",
        openapi_examples=_openapi_example("truncate")),
    tol: Optional[float] = Query(
        None,
        description="Nearest-point tolerance in degrees. Default 1/60 degree (half grid cell); maximum 0.25 degree."),
    append: Optional[str] = Query(
        None, description="Comma-separated fields. Default `z`. Allowed fields: z,u,v. Invalid/missing model cells are omitted; all-missing requests return {}.",
        openapi_examples=_openapi_example("z")),
    constituent: Optional[str] = Query(
        None,
        description="Comma-separated harmonic constituents. If omitted, all 15 constituents are used. Allowed: q1,o1,p1,k1,n2,m2,s1,s2,k2,m4,ms4,mn4,2n2,mf,mm. See also: https://www.tpxo.net/global",
        openapi_examples=_openapi_example("m2,k1"))
):
    """
    Query tide from the TPXO global tide model (TPXO10-atlas-v2 by default) by longitude/latitude/date (in JSON).

    #### Usage
    * One-point tide height (<= 30 days, hourly): `/api/tide?lon0=-157.86453&lat0=21.303333&start=2023-07-25&end=2023-07-26`
    * Small bbox map: `/api/tide?lon0=-158.2&lon1=-157.6&lat0=21.0&lat1=21.6&start=2023-07-25T00:00:00&sample=5`
    * Units: z tide height is cm; u and v tidal-current components are cm/s.
    * Large maps are capped at MAX_BBOX_CELLS=500000 after applying sample.
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
        cons = _ordered_valid_tokens(constituent, config.cons)
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

            if tol not in [np.nan, None] or findNear:
                findNear = True
                if tol in [np.nan, None] or tol <= 0:
                    tol = 0.5*config.gridSz
                elif tol > 7.5*config.gridSz:
                    tol = 7.5*config.gridSz
            else:
                tol = 0.5*config.gridSz

            # Handle the edge case for longitude 0
            # We found nearest 0 point (but > 0) may encounter index error in xarray
            # The grid in dataset is -4.06e-6 - 0.0333
            zero_nearest_pt = np.round(0.5*config.gridSz, 3) #0.017
            if lon0 >= 0 and lon0 < zero_nearest_pt:  # Consider values very close to 0 as 0
                lon0 = zero_nearest_pt

            # v0.3.0 Stage 3: nearest-point selection + per-variable hc via
            # the adapter; the prediction core (get_tide_series) is unchanged.
            try:
                dsub = config.adapter.sel_point(lon0, lat0, tol, constituents=cons)
            except KeyError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Requested point is outside the tide grid coverage.") from exc
            glon = np.atleast_1d(dsub[config.adapter.lon_name].values)
            glat = np.atleast_1d(dsub[config.adapter.lat_name].values)

            tide = {}
            for var in variables:
                unit = 'cm' if var == 'z' else ''
                amp, ph = config.adapter.amp_ph(dsub, var)
                # fill masked (TPXO10 flag==2 invalid) to NaN, NOT 0, so the
                # prediction core treats it as missing (round 23 F1).
                ts = get_tide_series(np.ma.filled(amp, np.nan),
                                     np.ma.filled(ph, np.nan),
                                     cons, tide_time, format="netcdf",
                                     unit=unit, drop_mask=True)
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

            # v0.3.0 Stage 3: the unified planner is the ONLY bbox/map path.
            # It resolves the post-halo/post-sample indices from coordinates
            # (dateline-wrap aware: lon0 > lon1 after the normalization above
            # means a cross-zero window), enforces MAX_BBOX_CELLS BEFORE any
            # materialization, and rejects an empty selection — both as
            # HTTP 400.
            try:
                dsub, _cells = plan_bbox(
                    config.adapter, lon0, lon1, lat0, lat1, sample,
                    constituents=cons, halo=0.5*config.gridSz,
                    max_cells=config.MAX_BBOX_CELLS)
            except (BboxCapError, EmptyBboxError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            glon, glat = config.adapter.coord_values(dsub)
            # if not single-pont mode, only allow one datetime moment
            tide_time = tide_time[0:1]
            dtime = dtime[0:1]
            tide = get_tide_map(config.adapter, dsub, tide_time,
                                format='netcdf', type=variables, drop_dim=True)
            output_mode = 'map'

        #if mode is None or mode != 'row':
        #print(tide)
        out = tide_to_output(tide, glon, glat, dtime, variables,
                             # allow both 'map' and 'time' mode can have truncate mode, 202504
                             output_mode + ',truncate' if 'truncate' in mode else output_mode,
                             # output_mode + ',truncate' if 'truncate' in mode and output_mode == 'map' else output_mode,
                             absmax=10000.0)
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
        adapter, dsub, loni, lati, vars=['amp', 'ph'],
        constituent=['q1', 'o1', 'p1', 'k1', 'n2', 'm2', 's1', 's2', 'k2', 'm4', 'ms4', 'mn4', '2n2', 'mf', 'mm'],
        type=['u', 'v']):
    # v0.3.0 Stage 3: amp/ph (and hc real/imag) come from the adapter in
    # the store's native units, so the response is schema-agnostic.
    out = {'longitude': loni.tolist(),
           'latitude': lati.tolist(),
           'grid_lon': np.asarray(dsub[adapter.lon_name].values).tolist(),
           'grid_lat': np.asarray(dsub[adapter.lat_name].values).tolist()}

    for TYPE in type:
        amp_all, ph_all = adapter.amp_ph(dsub, TYPE)   # (npoints, nc) masked
        hc_all = adapter.hc(dsub, TYPE) if 'hc' in vars else None
        for idx, const in enumerate(constituent):
            key = f"{TYPE}_{const}"
            # fill masked (TPXO10 flag==2 invalid) to NaN -> null in JSON,
            # never 0 (round 23 F1).
            if 'amp' in vars:
                out[key+"_amp"] = np.ma.filled(amp_all[..., idx], np.nan).tolist()
            if 'ph' in vars:
                out[key+"_ph"] = np.ma.filled(ph_all[..., idx], np.nan).tolist()
            if 'hc' in vars:
                hc = np.ma.filled(hc_all[..., idx], np.nan + 1j*np.nan)
                out[key+"_real"] = np.asarray(hc.real).tolist()
                out[key+"_imag"] = np.asarray(hc.imag).tolist()
    return out


def get_constituent(adapter, dsub, lon, lat, vars=['amp', 'ph'],
                    constituent=['q1', 'o1', 'p1', 'k1', 'n2', 'm2', 's1', 's2', 'k2', 'm4', 'ms4', 'mn4', '2n2', 'mf', 'mm'],
                    type=['u', 'v']):
    # v0.3.0 Stage 3: legacy single-point constituent helper (currently
    # unused by the endpoint, which uses get_constituent_vec); migrated to
    # the adapter so no schema-specific variable names remain. `dsub` is an
    # adapter point selection already filtered to `constituent`.
    amplitudes = {}
    phase = {}
    imag = {}
    real = {}
    out = {}
    lon = lon-360 if lon > 180 else lon
    out['longitude'] = lon
    out['latitude'] = lat
    if not vars:
        vars = ['amp', 'ph']

    for TYPE in type:
        amp_all, ph_all = adapter.amp_ph(dsub, TYPE)
        hc_all = adapter.hc(dsub, TYPE)
        for idx, const in enumerate(constituent):
            key = f"{const}_{TYPE}"
            amplitudes[key] = float(np.ma.filled(amp_all[..., idx], np.nan).ravel())
            phase[key] = float(np.ma.filled(ph_all[..., idx], np.nan).ravel())
            hc = np.ma.filled(hc_all[..., idx], np.nan + 1j*np.nan)
            imag[key] = float(np.asarray(hc.imag).ravel())
            real[key] = float(np.asarray(hc.real).ravel())

    if 'amp' in vars:
        out["amp"] = amplitudes

    if 'ph' in vars:
        out["ph"] = phase

    if 'hc' in vars:
        out["real"] = real
        out["imag"] = imag

    return out


@app.get("/api/tide/const", tags=["Tide"], summary="Get harmonic constituents of the TPXO model")
async def get_tide_const(
    lon: Optional[str] = Query(
            None,
            description="comma-separated longitude values. One of lon/lat and jsonsrc should be specified as longitude/latitude input.",
            openapi_examples=_openapi_example("-157.86453,-70.9137")),
    lat: Optional[str] = Query(
            None,
            description="comma-separated latitude values. One of lon/lat and jsonsrc should be specified as longitude/latitude input.",
            openapi_examples=_openapi_example("21.303333,41.6212")),
    mode: Optional[str] = Query(
        None,
        description="Optional modes: default/list returns a column-oriented object; row returns row records; long returns long-format records; object returns the raw column object.",
        openapi_examples=_openapi_example("row")),
    tol: Optional[float] = Query(
        None,
        description="Nearest-point tolerance in degrees. Default 1/60 degree (half grid cell); maximum 0.25 degree."),
    append: Optional[str] = Query(
        None, description="Comma-separated fields. Default `z`. Allowed fields: z,u,v. z constants are tide height; u/v constants are current components.",
        openapi_examples=_openapi_example("z,u,v")),
    constituent: Optional[str] = Query(
        None,
        description="Comma-separated harmonic constituents. If omitted, all 15 constituents are returned. Allowed: q1,o1,p1,k1,n2,m2,s1,s2,k2,m4,ms4,mn4,2n2,mf,mm. See also: https://www.tpxo.net/global",
        openapi_examples=_openapi_example("m2,k1")),
    complex: Optional[str] = Query(
        None, description="Comma-separated output components. Default amp,ph. Allowed: amp, ph, hc. hc returns real/imag columns.",
        openapi_examples=_openapi_example("amp,ph")),
    jsonsrc: Optional[str] = Query(
        None,
        description='Optional. A valid URL for JSON source or a JSON string that contains longitude and latitude keys with values in array.\n' +
                    'Example: {"longitude":[122.36,122.47,122.56,122.66],"latitude":[25.02,24.82,24.72,24.62]}')
):
    """
    Query harmonic constituents from the TPXO global tide model (TPXO10-atlas-v2 by default) by longitude/latitude.

    #### Usage
    * `/api/tide/const?lon=-157.86453,-70.9137&lat=21.303333,41.6212&constituent=m2,k1&complex=amp,ph&append=z,u,v&mode=row`
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

    # Replace any loni elements that are > 0 and < zero_nearest_pt with zero_nearest_pt
    zero_nearest_pt = np.round(0.5*config.gridSz, 3) #0.017            
    loni = np.where((loni > 0) & (loni < zero_nearest_pt), zero_nearest_pt, loni)

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
        cons = _ordered_valid_tokens(constituent, config.cons)
        if not cons:
            raise HTTPException(
                status_code=400, detail="Invalid constituents. Allowed constituents are 'q1','o1','p1','k1','n2','m2','s1','s2','k2','m4','ms4','mn4','2n2','mf','mm'")

    if complex is None:
        complex = 'amp,ph'

    pars = []
    if ',' in complex:
        pars = _ordered_valid_tokens(complex, ['amp', 'ph', 'hc'])
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

    if tol not in [np.nan, None] or findNear:
        findNear = True
        if tol in [np.nan, None] or tol <= 0:
            tol = 0.5*config.gridSz
        elif tol > 7.5*config.gridSz:
            tol = 7.5*config.gridSz
    else:
        tol = 0.5*config.gridSz


    # v0.3.0 Stage 3: vectorized point-paired (NOT Cartesian) nearest
    # selection via the adapter — the /api/tide/const access pattern. Used
    # for both single- and multi-point requests (this is NOT a bbox/map
    # query and is never routed through plan_bbox).
    try:
        dsub = config.adapter.sel_points(mlon, mlat, tol, constituents=cons)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail="One or more requested points are outside the tide grid "
                   "coverage (beyond the selection tolerance).") from exc

    out = get_constituent_vec(config.adapter, dsub, loni, lati,
                              vars=pars, constituent=cons, type=variables)

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
