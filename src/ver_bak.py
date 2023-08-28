import numpy as np
import pathlib
from pyTMD.interpolate import extrapolate
from src.pytmd_utils import *
from src.model_utils import spline_2d
from scipy.ndimage import label
from scipy.interpolate import griddata


# FUNCTION to fill NA in gridded data
def fill_gridded_nans(data):
    # Create coordinate arrays
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Indices of non-NaN values
    valid_idx = ~np.isnan(data)
    coords_non_nan = np.column_stack((X[valid_idx], Y[valid_idx]))

    # Interpolated NaN values
    data_no_nans = griddata(
        coords_non_nan, data[valid_idx], (X, Y), method='nearest')

    return data_no_nans


# FUNCTION to label blocks of NaNs
def label_blocks(array):
    labeled_array, num_features = label(np.isnan(array))
    return labeled_array


def extract_ATLAS_v1(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
                     tide_model, type=None, global_grid=False,
                     en_interpolate=False, interpolate_to=None):
    if type in ['u', 'v']:
        model_files = tide_model.model_file[type]
    else:
        model_files = tide_model.model_file

    compressed = tide_model.compressed
    scale = tide_model.scale
    type = tide_model.type if type is None else type
    print('model type is: ', type)

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

    if en_interpolate:
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

        lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)
        print("Interpolate to write: ",
              lon_chunk[0], lon_chunk[-1], lat_chunk[0], lat_chunk[-1])

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

        # print('----after extend-----')
        # print(hcx.mask.shape)
        hcx.mask[:, :] |= bathymetry.mask[:, :]

        if en_interpolate:
            lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)
            # lon_grid, lat_grid is in chunk, not in full range, default is 360*45 in degree, 10800*1351 in size
            hci = spline_2d(lon, lat, hcx, lon_grid, lat_grid,
                            reducer=np.ceil, kx=1, ky=1)

            # if interpolate_to is not None:
            #    if global_grid:
            #        hci.mask[:] = np.copy(gbathy.mask[start_lat:end_lat, (start_lon-1):(end_lon-1)])
            #    else:
            #        hci.mask[:] = np.copy(gbathy.mask[start_lat:end_lat, start_lon:end_lon])
            # else:
            #    hci.mask[:] |= np.copy(bathymetry.mask[start_lat:end_lat, start_lon:end_lon])

            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value
            # temporarily disable extropolate
            if False:
                print("extrapolate: ", hci.mask.shape)
                # so hci should be in lat_chunk * lon_chunk degree size
                invy, invx = np.nonzero(hci.mask)
                # replace invalid values with nan
                hcx.data[hcx.mask] = np.nan
                extra_coords = np.column_stack(
                    (lon_chunk[invx], lat_chunk[invy]))
                # extrapolate points within cutoff of valid model points
                hci[invy, invx] = extrapolate(
                    lon, lat, hcx, extra_coords[:, 0], extra_coords[:, 1], dtype=hcx.dtype)
        else:
            hci = hcx[start_lat:end_lat, start_lon:end_lon]
            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value

        if en_interpolate:
            ampx = np.ma.zeros((chunky, chunkx))
            non_masked_zero = np.logical_and(Di.mask == False, Di.data == 0)
            Di.mask |= non_masked_zero
            # non_masked_zero = np.logical_or(Di.mask == True, hci.mask == True)
            hci.mask |= non_masked_zero
            unit_conv = (Di.data[hci.mask == False]/100.0)
            # print("----min, max of unit_conv: ", np.min(unit_conv), np.max(unit_conv))
            zeroidx = np.where(unit_conv == 0)
            if len(zeroidx[0]) > 0:  # np.where() function returns a tuple of arrays
                print("----Warning: may divide by zero----")
                print((Di.mask[hci.mask == False])[zeroidx])
                print((hci.mask[hci.mask == False])[zeroidx])

            ampx.data[hci.mask == False] = np.abs(
                hci.data[hci.mask == False])/unit_conv
            ampx.data[hci.mask] = 0  # np.abs(hci.data[hci.mask])
            # print("----min, max of ampx: ", np.min(ampx.data), np.max(ampx.data))
            ampl.data[:, :, i] = ampx.data
        else:
            ampl.data[:, :, i] = np.abs(hci.data)  # /unit_conv, now is 1

        ampl.mask[:, :, i] = np.copy(hci.mask)
        ph.data[:, :, i] = np.arctan2(-np.imag(hci.data), np.real(hci.data))
        ph.mask[:, :, i] = np.copy(hci.mask)

    # convert amplitude from input units to meters
    amplitude = ampl*scale
    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase[phase < 0] += 360.0
    return amplitude, phase, constituents
