import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
# from dask import delayed, compute
# from dask.distributed import get_client, default_client, LocalCluster, Client
from pyTMD.io import ATLAS
from src.model_utils import get_current_model
from src.pytmd_utils import read_netcdf_grid
from numcodecs import MsgPack
import zarr
import logging

logging.basicConfig(level=logging.INFO)

maxWorkers = 6
# client = Client('tcp://localhost:8786')
# try:
#    client = get_client()
#    client.shutdown()
#    client.close()
# except ValueError:
#    # No existing Dask client
#    pass
# client = Client(n_workers=4, dashboard_address=':8798', scheduler_port=0)

BATHY_gridfile = '/home/bioer/python/tide/data_src/TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc'
tpxo_model_directory = '/home/bioer/python/tide/data_src'
tpxo_model_format = 'netcdf'
tpxo_compressed = False
tpxo_model_name = 'TPXO9-atlas-v5'
tpxo_model = get_current_model(
    tpxo_model_name, tpxo_model_directory, tpxo_model_format, tpxo_compressed)


def recompute_na_points(coord, lonz, latz, bathy_mask, bathy_data):
    ilat_idx, ilon_idx = coord

    # If it's a land point or depth <= 0
    if bathy_mask[ilat_idx, ilon_idx] or bathy_data[ilat_idx, ilon_idx] <= 0.0:
        return None

    ilon = lonz[ilon_idx]
    ilat = latz[ilat_idx]

    results = {}
    for var_type in ['u', 'v']:
        scale = 1e-4 #tpxo_model.scale 
        amp, ph, _, _ = ATLAS.extract_constants(
            np.atleast_1d(ilon), np.atleast_1d(ilat),
            tpxo_model.grid_file,
            tpxo_model.model_file[var_type], type=var_type, method='spline',
            scale=scale, compressed=tpxo_model.compressed
        )
        results[f"{var_type}_amp"] = amp
        results[f"{var_type}_ph"] = ph

    return results


def process_chunk(cluster, lonz, latz, tpxo_model, var_type, cluster_idx, cluster_num):
    start_lat, end_lat, start_lon, end_lon = cluster
    lon_chunk = lonz[start_lon:end_lon]
    lat_chunk = latz[start_lat:end_lat]
    lon_grid, lat_grid = np.meshgrid(lon_chunk, lat_chunk)
    scale = 1e-4 ## replace tpxo_model.scale before pyTMD new release

    amp, ph, D, c = ATLAS.extract_constants(
        lon_grid.ravel(), lat_grid.ravel(),
        tpxo_model.grid_file,
        tpxo_model.model_file[var_type], type=var_type, method='spline',
        scale=scale, compressed=tpxo_model.compressed, extrapolate=True)

    # reshape back amp and ph
    amp = amp.reshape((end_lat - start_lat, end_lon - start_lon, -1))
    ph = ph.reshape((end_lat - start_lat, end_lon - start_lon, -1))
    print(f"Cluster index: {cluster_idx}/{cluster_num} for variable: {var_type} wtih scale: {scale}")

    return (cluster_idx, cluster_num, start_lat, end_lat, start_lon, end_lon, var_type, amp, ph)


def filter_and_form_clusters(coords_to_recompute, neighborx=1, neighbory=4):
    coords_to_recompute = set(coords_to_recompute)  # Ensure it's a set for efficient removal
    clusters = []

    while coords_to_recompute:
        ilat_idx, ilon_idx = next(iter(coords_to_recompute))  # Take one coord from the set without removing it

        neighbors = [(ilat_idx + dlat, ilon_idx + dlon) for dlat in range(neighbory) for dlon in range(neighborx)]

        if all(neighbor in coords_to_recompute for neighbor in neighbors):
            start_lat, start_lon = min(neighbors, key=lambda x: (x[0], x[1]))
            end_lat, end_lon = max(neighbors, key=lambda x: (x[0], x[1]))
            # +1 because we want to include the last point when slicing
            clusters.append((start_lat, end_lat + 1, start_lon, end_lon + 1))

            for neighbor in neighbors:
                coords_to_recompute.discard(neighbor)  # Remove these neighbors from further consideration
        else:
            coords_to_recompute.discard((ilat_idx, ilon_idx))  # Remove the current point if not all its neighbors are in the set

    return clusters


def replace_na_from_second_dataset(input_file, lonz, latz, bathy_mask, bathy_data):
    ds = xr.open_zarr(input_file)

    # Check for NaNs in the four variables
    coords_to_recompute = set()
    for var in ['u_amp', 'v_amp']:
        print("Now process var to find na: ", var)
        nan_locs = np.argwhere(np.isnan(ds[var].values).any(axis=-1))
        for loc in nan_locs:
            ilat_idx, ilon_idx = loc
            if not bathy_mask[ilat_idx, ilon_idx] and bathy_data[ilat_idx, ilon_idx] > 0.0:
                coords_to_recompute.add((ilat_idx, ilon_idx))

    total_points = len(coords_to_recompute)
    print(f"Total points to process: {total_points}")

    # Filter coordinates and form 5x5 clusters
    # clusters = filter_and_form_clusters(coords_to_recompute)
    clusters = filter_and_form_clusters(coords_to_recompute, neighborx=1, neighbory=4)

    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futures_list = []
        total_clusters = len(clusters)
        for idx, chunk in enumerate(clusters):
            for var_type in ['u', 'v']:
                future = executor.submit(process_chunk, chunk, lonz, latz, tpxo_model, var_type, idx, total_clusters)
                futures_list.append(future)

        for future in as_completed(futures_list):
            idx_processed, cluster_num, start_lat, end_lat, start_lon, end_lon, var_type_processed, amp, ph = future.result()
            print(f"Processed cluster index: {idx_processed}/{cluster_num} for variable: {var_type_processed}")

            # Refill zarr dataset based on the variable type
            if var_type_processed == 'u':
                ds['u_amp'][start_lat:end_lat, start_lon:end_lon, :] = amp
                ds['u_ph'][start_lat:end_lat, start_lon:end_lon, :] = ph
            else:
                ds['v_amp'][start_lat:end_lat, start_lon:end_lon, :] = amp
                ds['v_ph'][start_lat:end_lat, start_lon:end_lon, :] = ph    

    # If there are still individual points left, you might want to process them separately
    # remaining_points = coords_to_recompute - filtered_coords
    # if remaining_points:
    #    print("Remaining points at final: ", len(remaining_points))
    #    #pass
    return ds


def main():
    input_file = "../data/tpxo9.zarr"
    output_file = "../data/tpxo9_fillna05.zarr"
    lonz, latz, bathy_z = read_netcdf_grid(BATHY_gridfile, variable='z')
    ds = replace_na_from_second_dataset(
        input_file, lonz, latz, bathy_z.mask, bathy_z.data)

    # Save the updated dataset but may too large to overload memory
    print("Start to re-write zarr dataset...")
    # ds.to_zarr(input_file, mode='w')
    # ds.close()
    store = zarr.open(output_file, mode='w')

    # Save dimension data
    for dim_name in ['lat', 'lon', 'constituents']:
        data = ds[dim_name].values
        chunks = ds[dim_name].encoding.get('chunks', ds[dim_name].shape)
        dtype = ds[dim_name].dtype
    
        # Check if dtype is object and handle accordingly
        if dtype == object:
            codec = MsgPack()
            arr = store.array(dim_name, data=data, chunks=chunks, dtype=dtype, object_codec=codec)
        else:
            arr = store.array(dim_name, data=data, chunks=chunks, dtype=dtype)
    
        arr.attrs['_ARRAY_DIMENSIONS'] = [dim_name]

    # Assuming chunking over lat and lon as in your example
    chunk_size = 338

    # Create placeholder arrays with `_ARRAY_DIMENSIONS` attribute
    for var_name in ds.data_vars:
        shape = ds[var_name].shape
        dtype = ds[var_name].dtype
        chunks = (chunk_size, chunk_size, ds['constituents'].shape[0])  # Assuming 3D data with constituents as the third dimension
        arr = store.empty(var_name, shape=shape, dtype=dtype, chunks=chunks)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lat', 'lon', 'constituents']

    # Write data in chunks
    for i in range(0, len(ds['lat']), chunk_size):
        for j in range(0, len(ds['lon']), chunk_size):

            # Extract chunk from dataset
            ds_chunk = ds.isel(lat=slice(i, i+chunk_size), lon=slice(j, j+chunk_size))

            # Write chunk to appropriate location in Zarr store
            for var_name, variable in ds_chunk.data_vars.items():
                store[var_name][i:i+chunk_size, j:j+chunk_size, :] = variable.values

if __name__ == '__main__':
    main()
