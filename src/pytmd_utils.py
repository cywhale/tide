# Utilties that pyTMD intenally used
# https://github.com/tsutterley/pyTMD/blob/main/pyTMD/io/ATLAS.py
import numpy as np
import netCDF4
import pathlib
import uuid
import gzip


def extend_array(input_array: np.ndarray, step_size: float):
    """
    Extends a longitude array

    Parameters
    ----------
    input_array: np.ndarray
        array to extend
    step_size: float
        step size between elements of array

    Returns
    -------
    temp: np.ndarray
        extended array
    """
    n = len(input_array)
    temp = np.zeros((n+2), dtype=input_array.dtype)
    # extended array [x-1,x0,...,xN,xN+1]
    temp[0] = input_array[0] - step_size
    temp[1:-1] = input_array[:]
    temp[-1] = input_array[-1] + step_size
    return temp


# PURPOSE: Extend a global matrix
def extend_matrix(input_matrix: np.ndarray):
    """
    Extends a global matrix

    Parameters
    ----------
    input_matrix: np.ndarray
        matrix to extend

    Returns
    -------
    temp: np.ndarray
        extended matrix
    """
    ny, nx = np.shape(input_matrix)
    temp = np.ma.zeros((ny, nx+2), dtype=input_matrix.dtype)
    temp[:, 0] = input_matrix[:, -1]
    temp[:, 1:-1] = input_matrix[:, :]
    temp[:, -1] = input_matrix[:, 0]
    return temp


# PURPOSE: read elevation file to extract real and imaginary components for
# constituent
def read_netcdf_elevation(
    input_file: str | pathlib.Path,
    **kwargs
):
    """
    Read elevation file to extract real and imaginary components for constituent

    Parameters
    ----------
    input_file: str or pathlib.Path
        input elevation file
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    h: np.ndarray
        tidal elevation
    con: str
        tidal constituent ID
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # read the netcdf format tide elevation file
    input_file = pathlib.Path(input_file).expanduser()
    # reading a combined global solution with localized solutions
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        fileID = netCDF4.Dataset(uuid.uuid4().hex, 'r', memory=f.read())
    else:
        fileID = netCDF4.Dataset(input_file, 'r')
    # constituent name
    con = fileID.variables['con'][:].tobytes().decode('utf8')
    # variable dimensions
    nx = fileID.dimensions['nx'].size
    ny = fileID.dimensions['ny'].size
    # real and imaginary components of elevation
    h = np.ma.zeros((ny, nx), dtype=np.complex64)
    h.mask = np.zeros((ny, nx), dtype=bool)
    h.data.real[:, :] = fileID.variables['hRe'][:, :].T
    h.data.imag[:, :] = fileID.variables['hIm'][:, :].T
    # close the file
    fileID.close()
    f.close() if kwargs['compressed'] else None
    # return the elevation and constituent
    return (h, con.strip())


# PURPOSE: read transport file to extract real and imaginary components for
# constituent
def read_netcdf_transport(
    input_file: str | pathlib.Path,
    variable: str,
    **kwargs
):
    """
    Read transport file to extract real and imaginary components for constituent

    Parameters
    ----------
    input_file: str or pathlib.Path
        input transport file
    variable: str
        Tidal variable to read

            - ``'u'``: horizontal transport velocities
            - ``'U'``: horizontal depth-averaged transport
            - ``'v'``: vertical transport velocities
            - ``'V'``: vertical depth-averaged transport

    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    tr: np.ndarray
        tidal transport
    con: str
        tidal constituent ID
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # read the netcdf format tide transport file
    input_file = pathlib.Path(input_file).expanduser()
    # reading a combined global solution with localized solutions
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        fileID = netCDF4.Dataset(uuid.uuid4().hex, 'r', memory=f.read())
    else:
        fileID = netCDF4.Dataset(input_file, 'r')
    # constituent name
    con = fileID.variables['con'][:].tobytes().decode('utf8')
    # variable dimensions
    nx = fileID.dimensions['nx'].size
    ny = fileID.dimensions['ny'].size
    # real and imaginary components of transport
    tr = np.ma.zeros((ny, nx), dtype=np.complex64)
    tr.mask = np.zeros((ny, nx), dtype=bool)
    if variable in ('U', 'u'):
        tr.data.real[:, :] = fileID.variables['uRe'][:, :].T
        tr.data.imag[:, :] = fileID.variables['uIm'][:, :].T
    elif variable in ('V', 'v'):
        tr.data.real[:, :] = fileID.variables['vRe'][:, :].T
        tr.data.imag[:, :] = fileID.variables['vIm'][:, :].T
    # close the file
    fileID.close()
    f.close() if kwargs['compressed'] else None
    # return the transport components and constituent
    return (tr, con.strip())


# PURPOSE: read grid file
def read_netcdf_grid(
    input_file: str | pathlib.Path,
    variable: str,
    **kwargs
):
    """
    Read grid file to extract model coordinates and bathymetry

    Parameters
    ----------
    input_file: str or pathlib.Path
        input grid file
    variable: str
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'U'``: horizontal depth-averaged transport
            - ``'v'``: vertical transport velocities
            - ``'V'``: vertical depth-averaged transport

    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    lon: np.ndarray
        longitudinal coordinates of input grid
    lat: np.ndarray
        latitudinal coordinates of input grid
    bathymetry: np.ndarray
        model bathymetry
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # read the netcdf format tide grid file
    input_file = pathlib.Path(input_file).expanduser()
    # reading a combined global solution with localized solutions
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        fileID = netCDF4.Dataset(uuid.uuid4().hex, 'r', memory=f.read())
    else:
        fileID = netCDF4.Dataset(input_file, 'r')
    # variable dimensions
    nx = fileID.dimensions['nx'].size
    ny = fileID.dimensions['ny'].size
    # allocate numpy masked array for bathymetry
    bathymetry = np.ma.zeros((ny, nx))
    # read bathymetry and coordinates for variable type
    if (variable == 'z'):
        # get bathymetry at nodes
        bathymetry.data[:, :] = fileID.variables['hz'][:, :].T
        # read latitude and longitude at z-nodes
        lon = fileID.variables['lon_z'][:].copy()
        lat = fileID.variables['lat_z'][:].copy()
    elif variable in ('U', 'u'):
        # get bathymetry at u-nodes
        bathymetry.data[:, :] = fileID.variables['hu'][:, :].T
        # read latitude and longitude at u-nodes
        lon = fileID.variables['lon_u'][:].copy()
        lat = fileID.variables['lat_u'][:].copy()
    elif variable in ('V', 'v'):
        # get bathymetry at v-nodes
        bathymetry.data[:, :] = fileID.variables['hv'][:, :].T
        # read latitude and longitude at v-nodes
        lon = fileID.variables['lon_v'][:].copy()
        lat = fileID.variables['lat_v'][:].copy()
    # set bathymetry mask
    bathymetry.mask = (bathymetry.data == 0.0)
    # close the grid file
    fileID.close()
    f.close() if kwargs['compressed'] else None
    return (lon, lat, bathymetry)
