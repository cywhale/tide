import xarray as xr
import numpy as np
import zarr

All_NA_CONDITION = False
ForceUseExact = True

def resave_fillna_dataset(method1, method2):
    useNear = False
    chunk_size = {'lat': 113, 'lon': 113, 'constituents': 8}

    # with xr.open_zarr(method1) as ds1, xr.open_zarr(method2) as ds2:
    with xr.open_zarr(method1) as ds1, xr.open_zarr(method2, chunks='auto', decode_times=False, consolidated=True) as ds2:
        # For u_amp, u_ph, v_amp, v_ph
        ds2['lat'].values[2700] = 0
        ds2 = ds2.sortby('lat')
        ds2['constituents'].encoding = {'dtype': 'str'}

        print("Pre-check size equality:")
        eq1 = ds1['lat'].equals(ds2['lat'])
        print(ds1['lat'].size, ds2['lat'].size)
        eq2 = np.all(ds1['lat'].values == ds2['lat'].values)
        print(eq1, eq2)
        if not ForceUseExact and (not eq1 or not eq2):
            useNear = True

        if useNear:
            print("Warning: Use nearest method, not exactly match join!!")
            print("Please check the input ds1 file should be a valid old tpxo0.zarr")
            return
            #ds2 = ds2.assign_coords(lat=ds1['lat'])

            #lat_values = ds2['lat'].values
            #is_lat_monotonic = np.all(np.diff(lat_values) > 0)
            #print("Is lat monotonic?", is_lat_monotonic)
            #print(ds2['lat'].values[2698:2702])
            #if not is_lat_monotonic:
            #    ds2['lat'].values[2700] = 0
            #    ds2 = ds2.sortby('lat')

        # Rechunk ds2 to ensure uniform chunk sizes
        ds2 = ds2.chunk(chunk_size)

        variables = ['u_amp', 'u_ph', 'v_amp', 'v_ph']
        for var in variables:
            # Check where ds1 is NaN and ds2 is not NaN
            if All_NA_CONDITION:
                condition_all_nan = np.isnan(ds1[var]).all(dim='constituents')
                condition_not_all_nan = ~np.isnan(ds2[var]).all(dim='constituents')
                condition = condition_all_nan & condition_not_all_nan
            else:
                condition = np.isnan(ds1[var]).any(dim='constituents') & ~np.isnan(ds2[var]).any(dim='constituents')

            # Update values based on condition
            ds1[var] = xr.where(condition, ds2[var], ds1[var])

        # Explicitly set chunk sizes in encoding
        # if useNear:
        #    for var_name in ds1.data_vars:
        #        ds1[var_name].encoding['chunks'] = (113, 113, 8) #chunk_size

        #    # Rechunk ds1 if necessary
        #    ds1 = ds1.chunk(chunk_size)

    # Save the corrected dataset
    # if useNear:
    #    ds1.to_zarr('../data/tpxo9.zarr', compute=True)
    # else:
    ds1.to_zarr('../data/tpxo9.zarr')


if __name__ == "__main__":
    method1_file = "../data/tpxo9_bak44.zarr" #"tpxo9_method1.zarr" #All_NA_CONDITION set False
    method2_file = "../data/tpxo9_new.zarr" #"tpxo9_method2.zarr"
    zarr.convenience.consolidate_metadata(method2_file)
    resave_fillna_dataset(method1_file, method2_file)
