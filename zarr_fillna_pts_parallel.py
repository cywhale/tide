import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
# from dask import delayed, compute
# from dask.distributed import get_client, default_client, LocalCluster, Client
from pyTMD.io import ATLAS
from src.model_utils import get_current_model
from src.pytmd_utils import read_netcdf_grid

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
        amp, ph, _, _ = ATLAS.extract_constants(
            np.atleast_1d(ilon), np.atleast_1d(ilat),
            tpxo_model.grid_file,
            tpxo_model.model_file[var_type], type=var_type, method='spline',
            scale=tpxo_model.scale, compressed=tpxo_model.compressed
        )
        results[f"{var_type}_amp"] = amp
        results[f"{var_type}_ph"] = ph

    return results

# @delayed


def replace_na_from_second_dataset(input_file, lonz, latz, bathy_mask, bathy_data):
    ds = xr.open_zarr(input_file)

    # Check for NaNs in the four variables
    coords_to_recompute = set()
    variables = ['u_amp', 'v_amp', 'u_ph', 'v_ph']

    for var in variables:
        print("Now process var to find na: ", var)
        nan_locs = np.argwhere(np.isnan(ds[var].values))
        for loc in nan_locs:
            ilat_idx, ilon_idx, _ = loc
            if not bathy_mask[ilat_idx, ilon_idx] and bathy_data[ilat_idx, ilon_idx] > 0.0:
                coords_to_recompute.add((ilat_idx, ilon_idx))

    # Parallelize the re-computation using ProcessPoolExecutor
    total_points = len(coords_to_recompute)
    print(f"Total points to process: {total_points}")

    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futures = [executor.submit(recompute_na_points, coord, lonz,
                                   latz, bathy_mask, bathy_data) for coord in coords_to_recompute]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"Processed {i}/{total_points}")
            results.append(future.result())

    # Replace NaN values in the dataset with recomputed values
    for i, coord in enumerate(coords_to_recompute):
        ilat_idx, ilon_idx = coord
        for var in results[i]:
            if results[i] is not None:
                ds[var][ilat_idx, ilon_idx, :] = results[i][var]

    # Save the updated dataset
    ds.to_zarr(input_file, mode='w')


def main():
    input_file = "tpxo9.zarr"
    lonz, latz, bathy_z = read_netcdf_grid(BATHY_gridfile, variable='z')
    replace_na_from_second_dataset(
        input_file, lonz, latz, bathy_z.mask, bathy_z.data)


if __name__ == '__main__':
    main()
