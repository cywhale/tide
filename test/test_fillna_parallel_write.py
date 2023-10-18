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
import time
from datetime import datetime

# client = Client('tcp://localhost:8786')
# try:
#    client = get_client()
#    client.shutdown()
#    client.close()
# except ValueError:
#    # No existing Dask client
#    pass
# client = Client(n_workers=4, dashboard_address=':8798', scheduler_port=0)

## Global variables
maxWorkers = 4

def log_elapsed_time(start_time, work=""):
    et = time.time()
    end_time = datetime.fromtimestamp(et)
    elapsed_time = et - start_time
    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'{work} DONE at: {end_time}') 
    print(f'Elapsed time: {int(hours)} hr, {int(minutes)} min, {int(seconds)} sec')


def save_chunk_to_zarr(store, ds, i, j, chunk_size):
    # Extract chunk from dataset
    ds_chunk = ds.isel(lat=slice(i, i+chunk_size), lon=slice(j, j+chunk_size))
    # Write chunk to appropriate location in Zarr store
    for var_name, variable in ds_chunk.data_vars.items():
        store[var_name][i:i+chunk_size, j:j+chunk_size, :] = variable.values


# This will be your main function to save in parallel
def save_dataset_parallel(ds, store, chunk_size, max_workers=4):
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(ds['lat']), chunk_size):
            for j in range(0, len(ds['lon']), chunk_size):
                tasks.append(executor.submit(save_chunk_to_zarr, store, ds, i, j, chunk_size))
        # Ensure all tasks have finished
        for future in as_completed(tasks):
            future.result()


def main():
    st = time.time()
    start_time = datetime.fromtimestamp(st)
    print("Fill_NA Main process start: ", start_time)
    input_file = "../data/tpxo9.zarr"
    output_file = "./tmp_tpxo9.zarr"

    # Save the updated dataset but may too large to overload memory
    ds = xr.open_zarr(input_file, chunks='auto', decode_times=False, consolidated=True) 

    if True:
        print("Start to re-write zarr dataset...: ", start_time)
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

        chunk_size = 338
        # Create placeholder arrays with `_ARRAY_DIMENSIONS` attribute
        for var_name in ds.data_vars:
            shape = ds[var_name].shape
            dtype = ds[var_name].dtype
            chunks = (chunk_size, chunk_size, ds['constituents'].shape[0])  # Assuming 3D data with constituents as the third dimension
            arr = store.empty(var_name, shape=shape, dtype=dtype, chunks=chunks)
            arr.attrs['_ARRAY_DIMENSIONS'] = ['lat', 'lon', 'constituents']

        # Assuming chunking over lat and lon as in your example
        save_dataset_parallel(ds, store, chunk_size=chunk_size, max_workers=maxWorkers)

    log_elapsed_time(st, "2. Save result to output_file")

    # merge zarr_fillna_savefile.py
    st = time.time()
    print("Save ok!! Now consolidate_metadata")
    zarr.convenience.consolidate_metadata(output_file)

if __name__ == '__main__':
    main()
