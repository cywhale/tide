import xarray as xr
import numpy as np


def replace_na_from_second_dataset(method1, method2):
    with xr.open_zarr(method1) as ds1, xr.open_zarr(method2) as ds2:
        # For u_amp, u_ph, v_amp, v_ph
        variables = ['u_amp', 'u_ph', 'v_amp', 'v_ph']
        for var in variables:
            # Check where ds1 is NaN and ds2 is not NaN
            condition = np.isnan(ds1[var]) & ~np.isnan(ds2[var])
            ds1[var] = xr.where(condition, ds2[var], ds1[var])

    # Save the corrected dataset
    ds1.to_zarr('tpxo9.zarr')


if __name__ == "__main__":
    method1_file = "tpxo9_method1.zarr"
    method2_file = "tpxo9_method2.zarr"
    replace_na_from_second_dataset(method1_file, method2_file)
