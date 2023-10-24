import xarray as xr
import numpy as np
import zarr

All_NA_CONDITION = False

def resave_fillna_dataset(method1, method2):
#   with xr.open_zarr(method1) as ds1, xr.open_zarr(method2) as ds2:
    with xr.open_zarr(method1) as ds1, xr.open_zarr(method2, chunks='auto', decode_times=False, consolidated=True) as ds2:
        # For u_amp, u_ph, v_amp, v_ph
        ds2['lat'].values[2700] = 0
        ds2 = ds2.sortby('lat')
        ds2['constituents'].encoding = {'dtype': 'str'}
        # Rechunk the dataset for uniform chunks
        ds2 = ds2.chunk({'lat': 113, 'lon': 113, 'constituents': 8})

        variables = ['u_amp', 'u_ph', 'v_amp', 'v_ph']
        for var in variables:
            # Check where ds1 is NaN and ds2 is not NaN
            if All_NA_CONDITION:
                condition_all_nan = np.isnan(ds1[var]).all(dim='constituents')
                condition_not_all_nan = ~np.isnan(ds2[var]).all(dim='constituents')
                condition = condition_all_nan & condition_not_all_nan
            else:
                condition = np.isnan(ds1[var]).any(dim='constituents') & ~np.isnan(ds2[var]).any(dim='constituents')

            ds1[var] = xr.where(condition, ds2[var], ds1[var])

    # Save the corrected dataset
    ds1.to_zarr('../data/tpxo9.zarr')


if __name__ == "__main__":
    method1_file = "../data/tpxo9_fillna12.zarr" #"tpxo9_method1.zarr" #All_NA_CONDITION set False 
    method2_file = "../data/tpxo9_fillna13.zarr" #"tpxo9_method2.zarr"
    zarr.convenience.consolidate_metadata(method2_file)
    resave_fillna_dataset(method1_file, method2_file)
