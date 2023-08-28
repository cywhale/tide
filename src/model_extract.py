import numpy as np
import pathlib
from pyTMD.interpolate import extrapolate
from pyTMD.io import ATLAS
from src.pytmd_utils import *
from src.model_utils import spline_2d


# Use pyTMD extract_constants, basically work, but not solve small unit_conv = D/100 problem
def extract_ATLAS_pytmd(lon, lat, start_lon, end_lon, start_lat, end_lat,
                        tide_model, type, constituents, chunk_num):
    print('model type-chunk is: ', type, '-', chunk_num)
    if type in ['u', 'v']:
        model_files = tide_model.model_file[type]
    else:
        model_files = tide_model.model_file

    lon_chunk = lon[start_lon:end_lon]
    lat_chunk = lat[start_lat:end_lat]
    lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)

    # if type == 'z':
    # if constituents is None:
    #    constituents = ATLAS.read_constants(
    #        tide_model.grid_file, model_files, type=type, compressed=tide_model.compressed)

    # amp, ph, D = ATLAS.interpolate_constants(
    #    lon_grid.ravel(), lat_grid.ravel(),
    #    constituents, type=type, scale=tide_model.scale,
    #    method='spline', extrapolate=True)
    # else:
    amp, ph, D, c = ATLAS.extract_constants(
        lon_grid.ravel(), lat_grid.ravel(),
        tide_model.grid_file,
        model_files, type=type, method='spline',
        scale=tide_model.scale, compressed=tide_model.compressed)

    chunkx = end_lon - start_lon  # slicing is not include end_lon
    chunky = end_lat - start_lat
    amplitude = np.reshape(amp, (chunky, chunkx, len(c)))
    phase = np.reshape(ph, (chunky, chunkx, len(c)))
    return amplitude, phase, c


def extract_ATLAS_v2(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
                     tide_model, type, chunk_num, global_grid=False,
                     en_interpolate=False, interpolate_to=None, en_extrapolate=False,
                     shallowLimit=0):
    if type in ['u', 'v']:
        model_files = tide_model.model_file[type]
    else:
        model_files = tide_model.model_file

    compressed = tide_model.compressed
    scale = tide_model.scale
    type = tide_model.type if type is None else type
    print('model type-chunk is: ', type, '-', chunk_num)

    # number of constituents
    nc = len(model_files)
    # list of constituents
    constituents = []
    # hc = np.ma.zeros((ny, nx, nc), dtype=np.complex128)
    # amplitude and phase
    chunkx = end_lon - start_lon  # slicing is not include end_lon
    chunky = end_lat - start_lat
    ampl = np.ma.zeros((chunky, chunkx, nc))
    ampl.mask = np.zeros((chunky, chunkx, nc), dtype=bool)
    ph = np.ma.zeros((chunky, chunkx, nc))
    ph.mask = np.zeros((chunky, chunkx, nc), dtype=bool)
    Di = np.ma.zeros((chunky, chunkx))
    Di.mask = np.zeros((chunky, chunkx), dtype=bool)

    if interpolate_to is None:
        lon_chunk = lon[start_lon:end_lon]
        lat_chunk = lat[start_lat:end_lat]
        Di = np.ma.array(
            bathymetry.data[start_lat:end_lat, start_lon:end_lon],
            mask=bathymetry.mask[start_lat:end_lat, start_lon:end_lon])
    else:
        glon, glat, gbathy = interpolate_to
        if global_grid:
            lon_chunk = glon[(start_lon-1):(end_lon-1)]
            Di = np.ma.array(
                gbathy.data[start_lat:end_lat,
                            (start_lon-1):(end_lon-1)],
                mask=gbathy.mask[start_lat:end_lat,
                                 (start_lon-1):(end_lon-1)])
        else:
            lon_chunk = glon[start_lon:end_lon]
            Di = np.ma.array(
                gbathy.data[start_lat:end_lat, start_lon:end_lon],
                mask=gbathy.mask[start_lat:end_lat, start_lon:end_lon])
        lat_chunk = glat[start_lat:end_lat]

    # lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)
    print("Interpolate to write: ",
          lon_chunk[0], lon_chunk[-1], lat_chunk[0], lat_chunk[-1])

    if type in ['u', 'v']:
        if shallowLimit == 0:
            non_masked_zero = np.logical_and(
                Di.mask == False, Di.data == 0)
        else:
            non_masked_zero = np.logical_and(
                Di.mask == False, Di.data < shallowLimit)

        Di.mask |= non_masked_zero
        unit_conv = (Di.data[Di.mask == False]/100.0)
    else:
        unit_conv = 1

    # only when type is u or v, i.e en_interpolate = True, to the following to do unit_conv = D/100
    # https://github.com/tsutterley/pyTMD/blob/ba49b1f9d466e02104317adef7373c0d8e80e476/pyTMD/io/ATLAS.py#L257
    # I think Di is no need, The Du, Dv, Dz are slightly different because C-grid method that
    # use different u-node, v-node, z-node in one C-grid
    # so if we interpolate u-node's hc to z-node's position, We just use Dz, no need to do interpolation Du
    # Di = spline_2d(lon, lat, bathymetry, lon_grid, lat_grid, reducer=np.ceil, kx=1, ky=1)

    for i, model_file in enumerate(model_files):
        # check that model file is accessible
        model_file = pathlib.Path(model_file).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(str(model_file))
        if (type == 'z'):
            # read constituent from elevation file (hcx is in 5401 * 10800)
            hcx, cons = read_netcdf_elevation(
                model_file, compressed=compressed)
        elif type in ('U', 'u', 'V', 'v'):
            # read constituent from transport file
            hcx, cons = read_netcdf_transport(
                model_file, variable=type, compressed=compressed)

        # append constituent to list
        constituents.append(cons)
        # replace original values with extend matrices
        if global_grid:
            hcx = extend_matrix(hcx)

        if en_interpolate:
            lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)
            # lon_grid, lat_grid is in chunk, not in full range, default is 360*45 in degree, 10800*1351 in size
            hci = spline_2d(lon, lat, hcx, lon_grid, lat_grid,
                            reducer=np.ceil, kx=1, ky=1)

            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value
        else:
            hci = hcx[start_lat:end_lat, start_lon:end_lon]
            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value

        if en_extrapolate:
            print("extrapolate: ", hci.mask.shape)
            # so hci should be in lat_chunk * lon_chunk degree size
            invy, invx = np.nonzero(hci.mask)
            # replace invalid values with nan
            hcx.data[hcx.mask] = np.nan
            extra_coords = np.column_stack((lon_chunk[invx], lat_chunk[invy]))
            # extrapolate points within cutoff of valid model points
            hci[invy, invx] = extrapolate(
                lon, lat, hcx, extra_coords[:, 0], extra_coords[:, 1], dtype=hcx.dtype)

        ampx = np.ma.zeros((chunky, chunkx))
        # print("----min, max of unit_conv: ", np.min(unit_conv), np.max(unit_conv))
        # zeroidx = np.where(unit_conv == 0)
        # if len(zeroidx[0]) > 0: #np.where() function returns a tuple of arrays
        #    print("----Warning: may divide by zero----")
        #    print((Di.mask[hci.mask==False])[zeroidx])
        #    print((hci.mask[hci.mask==False])[zeroidx])
        ampx.data[Di.mask == False] = np.abs(
            hci.data[Di.mask == False])/unit_conv
        ampx.data[Di.mask] = 0
        ampl.data[:, :, i] = ampx.data
        ampl.mask[:, :, i] = np.copy(hci.mask)
        ph.data[:, :, i] = np.arctan2(-np.imag(hci.data), np.real(hci.data))
        ph.mask[:, :, i] = np.copy(hci.mask)

    # convert amplitude from input units to meters
    amplitude = ampl*scale
    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase[phase < 0] += 360.0
    return amplitude, phase, constituents
