import numpy as np
import xarray as xr
import pandas as pd
import pathlib
import os
from src.model_utils import *
from src.model_plot import *
from src.pytmd_utils import *
# from pyTMD.io import read_netcdf_elevation, read_netcdf_transport, read_netcdf_grid #not work for read_netcdf_grid
from pyTMD.interpolate import extrapolate
import concurrent.futures

# Global variables
BATHY_gridfile = '/home/bioer/python/tide/data_src/TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc'
tpxo_model_directory = '/home/bioer/python/tide/data_src'
tpxo_model_format = 'netcdf'
tpxo_compressed = False
tpxo_model_name = 'TPXO9-atlas-v5'
# global_grid = True
xChunkSz = 45
yChunkSz = 45
grid_sz = 1/30
chunk_file = 'tpxo9_chunks.zarr'
maxWorkers = 4


def extract_ATLAS(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
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

            non_masked_zero = np.logical_and(Di.mask == False, Di.data == 0)
            print('---- non_masked_indices:', np.sum(non_masked_zero))
            Di.mask |= non_masked_zero
            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value
            print("extrapolate: ", hci.mask.shape)
            # so hci should be in lat_chunk * lon_chunk degree size
            invy, invx = np.nonzero(hci.mask)
            # replace invalid values with nan
            hcx.data[hcx.mask] = np.nan
            extra_coords = np.column_stack((lon_chunk[invx], lat_chunk[invy]))
            # extrapolate points within cutoff of valid model points
            hci[invy, invx] = extrapolate(
                lon, lat, hcx, extra_coords[:, 0], extra_coords[:, 1], dtype=hcx.dtype)
        else:
            hci = hcx[start_lat:end_lat, start_lon:end_lon]
            hci.mask[:] |= np.copy(Di.mask[:])
            hci.data[hci.mask] = hci.fill_value

        if en_interpolate:
            ampx = np.ma.zeros((chunky, chunkx))
            non_masked_zero = np.logical_or(Di.mask == True, hci.mask == True)
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


def save_to_zarr(amplitude, phase, constituents, amp_var, ph_var, lon, lat, output_file, group_name, mode='write_chunk'):
    # Check if the group exists in the Zarr file
    if mode == 'append_chunk':
        ds = xr.open_zarr(output_file, group=group_name)

        # Append new data variables
        ds[amp_var] = (['lat', 'lon', 'constituents'], amplitude)
        ds[ph_var] = (['lat', 'lon', 'constituents'], phase)
    else:
        # except ValueError:  # If group does not exist, create a new dataset
        ds = xr.Dataset({
            amp_var: (['lat', 'lon', 'constituents'], amplitude),
            ph_var: (['lat', 'lon', 'constituents'], phase)
        }, coords={
            'lon': lon,
            'lat': lat,
            'constituents': np.array(constituents, dtype=str)
        })

    ds.to_zarr(output_file, mode='a', group=group_name)


# Function to process each chunk
def process_chunk(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
                  tpxo_model, type, global_grid, en_interpolate, interpolate_to, chunk_num, amp_var, ph_var, chunk_file, mode):

    amp_chunk, ph_chunk, c = extract_ATLAS(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
                                           tpxo_model, type, global_grid, en_interpolate, interpolate_to)

    amp_chunk[amp_chunk.mask] = np.nan
    ph_chunk[ph_chunk.mask] = np.nan
    amp = amp_chunk.data
    ph = ph_chunk.data

    if en_interpolate and interpolate_to is not None:
        glon, glat, gbathy = interpolate_to
        if global_grid:
            lon_chunk = glon[(start_lon-1):(end_lon-1)]
        else:
            lon_chunk = glon[start_lon:end_lon]
        lat_chunk = glat[start_lat:end_lat]
    else:
        lon_chunk = lon[start_lon:end_lon]
        lat_chunk = lat[start_lat:end_lat]

    group_name = f"chunk_{chunk_num}"
    save_to_zarr(amp, ph, c, amp_var, ph_var, lon_chunk,
                 lat_chunk, chunk_file, group_name, mode)
    return chunk_num


def tpxo2zarr(lon, lat, bathymetry, amp_var, ph_var, tpxo_model,
              chunk_size_lon=45, chunk_size_lat=45, grid_sz=1/30,
              chunk_file='chunks.zarr', type=None, mode='write_chunk',
              global_grid=False, en_interpolate=False, interpolate_to=None):

    if (global_grid):
        lon_range = range(0, len(lon), int(chunk_size_lon/grid_sz))
        lon_range = list([x + 1 for x in lon_range])
    else:
        lon_range = list(range(0, len(lon), int(chunk_size_lon/grid_sz)))
        if len(lon) not in lon_range:  # Ensure last element is included if it's not already
            lon_range.append(len(lon))

    lat_range = range(0, len(lat), int(chunk_size_lat/grid_sz))
    # lat length is 5401 not 5400
    lat_range = list([x + 1 if x == 5400 else x for x in lat_range])
    if len(lat) not in lat_range:
        lat_range.append(len(lat))

    chunks = []

    for lat_idx in range(len(lat_range) - 1):
        for lon_idx in range(len(lon_range) - 1):
            start_lon = lon_range[lon_idx]
            end_lon = lon_range[lon_idx+1]
            start_lat, end_lat = lat_range[lat_idx], lat_range[lat_idx+1]
            chunks.append((lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry, tpxo_model,
                          type, global_grid, en_interpolate, interpolate_to, len(chunks), amp_var, ph_var, chunk_file, mode))

    processed_chunks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        for chunk_num in executor.map(process_chunk, *zip(*chunks)):
            processed_chunks.append(chunk_num)
    return len(processed_chunks)


def combine_chunk_zarr(lon_range, lat_range, chunk_file):
    num_lat_chunks = len(lat_range) - 1
    num_lon_chunks = len(lon_range) - 1

    for lat_idx in range(num_lat_chunks):
        for lon_idx in range(num_lon_chunks):
            chunk_label = f"chunk_{lat_idx * num_lon_chunks + lon_idx}"
            assert os.path.exists(os.path.join(
                chunk_file, chunk_label)), f"Chunk {chunk_label} is missing!"

    # Organize the chunks into a 2D matrix
    chunks_matrix = [[xr.open_zarr(chunk_file, group=f"chunk_{lat_idx * num_lon_chunks + lon_idx}")
                      for lon_idx in range(num_lon_chunks)]
                     for lat_idx in range(num_lat_chunks)]

    # Concatenate chunks
    concat_along_lon = [xr.concat(lon_chunks, dim='lon')
                        for lon_chunks in chunks_matrix]
    ds_combined = xr.concat(concat_along_lon, dim='lat')

    chunk_size = 338
    ds_rechunked = ds_combined.chunk(
        {'lat': chunk_size, 'lon': chunk_size, 'constituents': -1})
    ds_rechunked.to_zarr('tpxo9.zarr', mode='w', safe_chunks=False)


def main():
    lonz, latz, bathy_z = read_netcdf_grid(BATHY_gridfile, variable='z')
    lon_range = list(range(0, len(lonz), int(xChunkSz/grid_sz)))
    if len(lonz) not in lon_range:  # Ensure last element is included if it's not already
        lon_range.append(len(lonz))
    print(lon_range)

    lat_range = range(0, len(latz), int(yChunkSz/grid_sz))
    # lat length is 5401 not 5400
    lat_range = list([x + 1 if x == 5400 else x for x in lat_range])
    if len(latz) not in lat_range:
        lat_range.append(len(latz))
    print(lat_range)

    TYPE = ['u', 'v'] #'z',
    for type in TYPE:
        if type == 'z':
            tpxo_model = get_tide_model(
                tpxo_model_name, tpxo_model_directory, tpxo_model_format, tpxo_compressed)
            global_grid = False
            en_interpolate = False
        else:
            tpxo_model = get_current_model(
                tpxo_model_name, tpxo_model_directory, tpxo_model_format, tpxo_compressed)
            global_grid = True
            en_interpolate = True

        if global_grid:
            lonc, latc, bathy_c = read_netcdf_grid(
                BATHY_gridfile, variable=type)
            # Adjust longitude and lat values as per convention
            lonx = np.copy(lonc)
            dlon = lonx[1] - lonx[0]
            lonx = extend_array(lonx, dlon)
            bathy_x = extend_matrix(bathy_c)
            bathy_x.mask = (bathy_x.data == 0)  # create masks
            print('----Extend bathy: ', bathy_x.shape, ' for type: ', type)

            chunkx = tpxo2zarr(lonx, latc, bathy_x, type+'_amp', type+'_ph',
                               tpxo_model, xChunkSz, yChunkSz, grid_sz=grid_sz,
                               chunk_file=chunk_file, type=type, mode="append_chunk",
                               global_grid=global_grid,
                               en_interpolate=en_interpolate,
                               interpolate_to=(lonz, latz, bathy_z))
        else:
            print('----Original bathy: ', bathy_z.shape, ' for type: ', type)
            chunkx = tpxo2zarr(lonz, latz, bathy_z, 'h_amp', 'h_ph',  tpxo_model,
                               xChunkSz, yChunkSz, grid_sz=grid_sz, chunk_file=chunk_file,
                               type=type, mode='write_chunk', global_grid=False,
                               en_interpolate=False, interpolate_to=None)

        print('----Done for type: ', type, ' with chunk:', chunkx)
        combine_chunk_zarr(lon_range, lat_range, chunk_file)


if __name__ == "__main__":
    main()
