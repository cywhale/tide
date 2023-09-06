import numpy as np
import xarray as xr
import os
from src.model_utils import *
from src.model_plot import *
from src.pytmd_utils import *
# from pyTMD.io import read_netcdf_elevation, read_netcdf_transport, read_netcdf_grid #not work for read_netcdf_grid
# from pyTMD.interpolate import extrapolate
from src.model_extract import extract_ATLAS_v2
import concurrent.futures

# Global variables
BATHY_gridfile = '/home/bioer/python/tide/data_src/TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc'
tpxo_model_directory = '/home/bioer/python/tide/data_src'
tpxo_model_format = 'netcdf'
tpxo_compressed = False
tpxo_model_name = 'TPXO9-atlas-v5'
# global_grid = True
xChunkSz = 15
yChunkSz = 15
grid_sz = 1/30
chunk_file = 'tpxo9_chunks.zarr'
# if D < shallowLimit, then dont compute u,v which U/(D/100) can cause very large value.
ShallowLimit = 0
extract_method = 'extract_Atlas_v2'  # 'extract_pytmd'
enExtrapolate = True
# concurrent processors used
maxWorkers = 4


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
                  tpxo_model, type, global_grid, en_interpolate, interpolate_to,
                  constituents, chunk_num, amp_var, ph_var, chunk_file, mode):

    # if extract_method == 'extract_ATLAS_pytmd'
    # amp_chunk, ph_chunk, c = extract_ATLAS_pytmd(lon, lat, start_lon, end_lon, start_lat, end_lat,
    #                                             tpxo_model, type, constituents, chunk_num)
    amp_chunk, ph_chunk, c = extract_ATLAS_v2(lon, lat, start_lon, end_lon, start_lat, end_lat, bathymetry,
                                              tpxo_model, type, chunk_num, global_grid, en_interpolate, interpolate_to,
                                              en_extrapolate=enExtrapolate, shallowLimit=ShallowLimit)

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
    # c = ['q1', 'o1', 'p1', 'k1', 'n2', 'm2', 's1',
    #     's2', 'k2', 'm4', 'ms4', 'mn4', '2n2', 'mf', 'mm']
    save_to_zarr(amp, ph, c, amp_var, ph_var, lon_chunk,
                 lat_chunk, chunk_file, group_name, mode)
    return chunk_num


def tpxo2zarr(lon, lat, bathymetry, amp_var, ph_var, tpxo_model,
              chunk_size_lon=45, chunk_size_lat=45, grid_sz=1/30,
              chunk_file='chunks.zarr', type=None, mode='write_chunk',
              global_grid=False, en_interpolate=False, interpolate_to=None,
              constituents=None):

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
                          type, global_grid, en_interpolate, interpolate_to, constituents,
                          len(chunks), amp_var, ph_var, chunk_file, mode))

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

    # 'z', 'u' #seems have some bugs, that u -> v cannot work
    # 'z', 'u', #not it's parallel unsure bug, 'u', 'v' cannot run sequently
    # Note that if you specify 'v' only, modify the elif type == 'u' to else
    TYPE = ['v']
    for type in TYPE:
        if type == 'z':
            tpxo_model = get_tide_model(
                tpxo_model_name, tpxo_model_directory, tpxo_model_format, tpxo_compressed)
            global_grid = False
            en_interpolate = False
        else:  # Note that if continuously run u and v, no need change tpxo_model in memory
            # elif type == 'u':
            tpxo_model = get_current_model(
                tpxo_model_name, tpxo_model_directory, tpxo_model_format, tpxo_compressed)
            # used if not etract_ATLAS_v1,v2
            global_grid = True if extract_method == 'extract_Atlas_v2' else False
            en_interpolate = True if extract_method == 'extract_Atlas_v2' else False

        print('model type is: ', type)
        if type in ['u', 'v']:
            model_files = tpxo_model.model_file[type]
            mode = "append_chunk"
        else:
            model_files = tpxo_model.model_file
            mode = 'write_chunk'

        constituents = None
        # ATLAS.read_constants(
        #    tpxo_model.grid_file, model_files, type=type, compressed=tpxo_model.compressed)

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
                               chunk_file=chunk_file, type=type, mode=mode,
                               global_grid=global_grid,
                               en_interpolate=en_interpolate,
                               interpolate_to=(lonz, latz, bathy_z),
                               constituents=constituents)
        else:
            print('----Original bathy: ', bathy_z.shape, ' for type: ', type)
            chunkx = tpxo2zarr(lonz, latz, bathy_z, type+'_amp', type+'_ph',  tpxo_model,
                               xChunkSz, yChunkSz, grid_sz=grid_sz, chunk_file=chunk_file,
                               type=type, mode=mode, global_grid=False,
                               en_interpolate=False, interpolate_to=None,
                               constituents=constituents)

        print('----Done for type: ', type, ' with chunk:', chunkx)
        combine_chunk_zarr(lon_range, lat_range, chunk_file)


if __name__ == "__main__":
    main()
