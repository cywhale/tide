import numpy as np
import numpy.ma as ma
from scipy.interpolate import RectBivariateSpline
from pyTMD.io import model, GOT, ATLAS
from pyTMD import predict
from timescale.time import convert_calendar_dates, datetime_to_list
# from pyTMD.time import convert_calendar_dates, datetime_to_list after pyTMD 2.1.2
# from pyTMD.load_constituent import load_constituent #deprecated after pyTMD 2.1.2
import pyTMD.arguments
from datetime import timedelta


# Create model object
def get_tide_model(model_name, model_directory, model_format, compressed=False):
    return model(model_directory, format=model_format, compressed=compressed).elevation(model_name)


# Create current model object
def get_current_model(model_name, model_directory, model_format, compressed=False):
    return model(model_directory, format=model_format, compressed=compressed).current(model_name)


# Load and interpolate tidal constants
def get_tide_constants(lon, lat, tide_model, model_name, isCurrModel=False):
    if 'GOT' in model_name:
        constituents = GOT.read_constants(
            tide_model.model_file, compressed=tide_model.compressed)
        amp, ph = GOT.interpolate_constants(np.array([lon]), np.array([lat]),
                                            constituents, scale=tide_model.scale,
                                            method='spline', extrapolate=True)

    elif 'atlas' in model_name:
        scale = 1e-4 if isCurrModel else tide_model.scale
        constituents = ATLAS.read_constants(
            tide_model.grid_file, tide_model.model_file, type=tide_model.type, compressed=tide_model.compressed)
        amp, ph, D = ATLAS.interpolate_constants(
            np.atleast_1d(lon), np.atleast_1d(lat),
            constituents, type=tide_model.type, scale=scale, #tide_model.scale, #use 1e-4 before pyTMD new release
            method='spline', extrapolate=True)

    return constituents, amp, ph


def get_tide_time(start_date, end_date):
    # Generate a list of every hour between start and end date
    dtime = []
    tide_time = []
    while start_date <= end_date:
        dtime.append(start_date)
        dtlist = datetime_to_list(start_date)
        tide_time.append(convert_calendar_dates(
            dtlist[0], dtlist[1], dtlist[2], dtlist[3]))
        start_date += timedelta(hours=1)

    tide_time = np.array(tide_time)
    return tide_time, dtime


# calculate complex phase in radians for Euler's
def get_tide_series(amp, ph, c, tide_time, format="netcdf", unit="cm", drop_mask=False):
    try:
        cph = -1j * ph * np.pi / 180.0
        # calculate constituent oscillation
        hc = (amp * np.exp(cph))[np.newaxis, :]
        # Create a mask where values are NA # or 0 #modified v0.1.1 let it contribute 0, not NA
        mask = np.isnan(hc) # | (hc == 0)

        # Convert hc to a masked array
        hc = ma.array(hc, mask=mask)

        DELTAT = np.zeros_like(tide_time)

        # Predict tide
        tide = predict.time_series(
            tide_time, hc, c, deltat=DELTAT, corrections=format)
        minor = predict.infer_minor(
            tide_time, hc, c, deltat=DELTAT, corrections=format)
        tide.data[:] += minor.data[:]
        # convert to centimeters
        if unit=='cm':
            tide.data[:] *= 100.0

        if not drop_mask:
            return tide

        tide.data[tide.mask] = np.nan
        out = tide.data
        return out
    
    except AttributeError as e:
        # Catch specific error when tide or minor prediction fails
        raise ValueError(
            "Error in tide prediction. This may be due to ininappropriate parameters, e.g., insufficient constituents. "
            f"Details: {str(e)}"
        ) from e
    except Exception as e:
        # General error handling
        raise ValueError(
            "An unexpected error occurred during tide prediction. "
            f"Details: {str(e)}"
        ) from e

# modified from pyTMD.predict.time_series() to get each constituent, not summer up
def time_series_for_constituents(t, hc, constituents, deltat=0.0):
    nt = len(t)
    # load the nodal corrections
    # pu, pf, G = pyTMD.arguments(
    #     t + 48622.0, constituents, deltat=deltat, corrections='ATLAS') #changed after pyTMD 2.1.2
    # number of days between MJD and the tide epoch (1992-01-01T00:00:00)
    _mjd_tide = 48622.0
    pu, pf, G = pyTMD.arguments.arguments(t + _mjd_tide,
        constituents,
        deltat=deltat,
        corrections='ATLAS'
    )
    # allocate for output time series
    ht = np.ma.zeros((nt, len(constituents)))
    # for each constituent
    for k, c in enumerate(constituents):
        # if corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf'):
        # load parameters for each constituent
        # amp, ph, omega, alpha, species = load_constituent(c) #before pyTMD 2.1.1
        amp, ph, omega, alpha, species = pyTMD.arguments._constituent_parameters(c)
        th = omega * t * 86400.0 + ph + pu[:, k]
        # elif corrections in ('GOT', 'FES'):
        #   th = G[:, k] * np.pi / 180.0 + pu[:, k]
        ht[:, k] = pf[:, k] * hc.real[0, k] * \
            np.cos(th) - pf[:, k] * hc.imag[0, k] * np.sin(th)

    return ht


# Note dz is the data from Zarr
def get_tide_map(dz, tide_time, format='netcdf', type=['u', 'v'], drop_dim=False):
    DELTAT = np.zeros_like(tide_time)
    c = dz.coords['constituents'].values
    nx = dz.coords['lon'].size
    ny = dz.coords['lat'].size
    timelen = len(tide_time)
    tide = {}

    for TYPE in type:
        amp = dz[TYPE+'_amp'].values
        ph = dz[TYPE+'_ph'].values
        shpx = amp.shape
        ampx = amp.reshape((shpx[0] * shpx[1], shpx[2]))
        phx = ph.reshape((shpx[0] * shpx[1], shpx[2]))
        # calculate complex phase in radians for Euler's
        cph = -1j * phx * np.pi / 180.0
        # calculate constituent oscillation
        hc = ampx * np.exp(cph)
        # Create a mask where values are NA # or 0 #modified v0.1.1 let it contribute 0, not NA
        mask = np.isnan(hc) # | (hc == 0)
        # Convert hc to a masked array
        hc = ma.array(hc, mask=mask)  # mask=False

        if drop_dim:
            TIDE = predict.map(tide_time[0], hc, c,
                               deltat=DELTAT[0], corrections=format)
            MINOR = predict.infer_minor(
                tide_time[0], hc, c, deltat=DELTAT[0], corrections=format)
            tx = TIDE+MINOR
            tx.data[tx.mask] = np.nan
            tide[TYPE] = tx.data
        else:
            tide[TYPE] = np.ma.zeros((ny, nx, timelen))
            for hour in range(timelen):
                # print('Get tidal current in time: ', hour)
                # predict tidal elevations at time and infer minor corrections
                TIDE = predict.map(tide_time[hour], hc, c,
                                   deltat=DELTAT[hour], corrections=format)
                MINOR = predict.infer_minor(
                    tide_time[hour], hc, c, deltat=DELTAT[hour], corrections=format)
                # add major and minor components and reform grid
                # Reshape TIDE and MINOR to have the shape (ny, nx)
                # print("Before reshape, TIDE'shape")
                # print(TIDE.shape, MINOR.shape) #It's 1D ny*nx length!!
                tide[TYPE][:, :, hour] = np.reshape((TIDE+MINOR), (ny, nx))

    return tide


# Note dz is the data from Zarr and tide_time will get its first element
def get_current_map(x0, y0, x1, y1, dz, tide_time, mask_grid=5, normalize=True):
    grid_sz = 1/30
    dsub = dz.sel(lon=slice(x0-grid_sz, x1+grid_sz),
                  lat=slice(y0-grid_sz, y1+grid_sz))
    gtide = get_tide_map(dsub, tide_time[0:1])

    t = 0
    nx = dsub.coords['lon'].size
    ny = dsub.coords['lat'].size
    glon, glat = np.meshgrid(
        dsub.coords['lon'].values, dsub.coords['lat'].values)

    # Reshape u and v to 2D
    u0 = gtide['u'][:, :, t]
    v0 = gtide['v'][:, :, t]

    # Create a grid of indices for subsetting
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Calculate magnitude of the current
    magnitude = np.sqrt(u0**2 + v0**2)
    # Normalize the arrows to create a uniform arrow size across the plot
    if normalize:
        u = u0/magnitude
        v = v0/magnitude
    else:
        u = u0
        v = v0

    n = mask_grid
    mask = (X % n == 0) & (Y % n == 0)

    x = glon[mask]
    y = glat[mask]
    u = u[mask]
    v = v[mask]
    mag = magnitude[mask]

    return x, y, u, v, mag


# Ref/modified from pyTMD.interpolate.spline()
def spline_2d(
    lon_axis: np.ndarray,
    lat_axis: np.ndarray,
    data: np.ndarray,
    ilon: np.ndarray,
    ilat: np.ndarray,
    fill_value: float = None,
    dtype: str | np.dtype = np.float64,
    reducer=np.ceil,
    **kwargs
):
    # set default keyword arguments
    kwargs.setdefault('kx', 1)
    kwargs.setdefault('ky', 1)
    # verify that input data is masked array
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.array(data)
        data.mask = np.zeros_like(data, dtype=bool)
    # interpolate gridded data values to data
    # npts = len(ilon)
    # allocate to output interpolated data array
    # out = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    # out.mask = np.ones((npts), dtype=bool)

    # interpolate gridded data values to data
    nlat, nlon = ilon.shape
    # allocate to output interpolated data array
    out = np.ma.zeros((nlat, nlon), dtype=dtype, fill_value=fill_value)
    out.mask = np.ones((nlat, nlon), dtype=bool)

    if np.iscomplexobj(data):
        s1 = RectBivariateSpline(
            lon_axis, lat_axis, data.data.real.T, **kwargs)
        s2 = RectBivariateSpline(
            lon_axis, lat_axis, data.data.imag.T, **kwargs)
        s3 = RectBivariateSpline(lon_axis, lat_axis, data.mask.T, **kwargs)
        # evaluate the spline at input coordinates #data.data may not be writeable
        # data.data.real[:] = s1.ev(lon, lat)
        # data.data.imag[:] = s2.ev(lon, lat)
        # data.mask[:] = reducer(s3.ev(lon, lat)).astype(bool)
        real_data = s1.ev(ilon, ilat).copy()
        imag_data = s2.ev(ilon, ilat).copy()
        out = np.ma.array(data=real_data + 1j * imag_data,
                          mask=reducer(s3.ev(ilon, ilat)).astype(bool))

    else:
        s1 = RectBivariateSpline(lon_axis, lat_axis, data.data.T, **kwargs)
        s2 = RectBivariateSpline(lon_axis, lat_axis, data.mask.T, **kwargs)
        # evaluate the spline at input coordinates
        out.data[:] = s1.ev(ilon, ilat).astype(dtype)
        out.mask[:] = reducer(s2.ev(ilon, ilat)).astype(bool)

    # return interpolated values
    return out
