import xarray as xr
import polars as pl
import numpy as np
from datetime import datetime
from src.model_utils import get_tide_time, get_tide_map
from src.model_plot import plot_current


def mock_data(lon, lat, variables):
    ny, nx = len(lat), len(lon)
    out = {}
    if 'h' in variables:
        # Random data for demonstration
        out['h'] = np.random.rand(ny, nx).ravel()
    if 'u' in variables:
        # Random data for demonstration
        test_u = np.random.rand(ny, nx)
        print('---- test_u in ny, nx shape ----')
        print(test_u)
        out['u'] = test_u.ravel()
    if 'v' in variables:
        # Random data for demonstration
        out['v'] = np.random.rand(ny, nx).ravel()
    return out


def tide2polars(tide_data, lon, lat, variables):
    glon, glat = np.meshgrid(lon, lat)

    # Flatten the longitude and latitude grids
    lonx = glon.ravel()
    latx = glat.ravel()

    # Create dictionary to hold results
    out_dict = {
        'longitude': lonx,
        'latitude': latx,
    }

    # Flatten and add to dictionary the variables that are present
    for var in variables:
        if var in tide_data:
            out_dict[var] = np.ravel(tide_data[var])

    # Convert the dictionary to a Polars DataFrame
    df = pl.DataFrame(out_dict)
    return df


# Mock Test
lon = np.array([1, 2, 3])
lat = np.array([4, 5])
variables = ['u', 'v']
testdt = mock_data(lon, lat, variables)
print(testdt)
df_test = tide2polars(testdt, lon, lat, variables)
print('----test_out df----')
print(df_test)


dz = xr.open_zarr('src/tpxo9.zarr', chunks='auto', decode_times=False)

x0, y0, x1, y1 = 118.0, 20.0, 129.75, 31.25
grid_sz = 1/30
dsub = dz.sel(lon=slice(x0-grid_sz, x1+grid_sz),
              lat=slice(y0-grid_sz, y1+grid_sz))

start_date = datetime(2023, 7, 25)
end_date = datetime(2023, 7, 28)
tide_time, dtime = get_tide_time(start_date, end_date)
variables = ['u', 'v']
tide_curr = get_tide_map(dsub, tide_time, type=variables, drop_dim=True)

tidf = tide2polars(
    tide_curr, dsub.coords['lon'].values, dsub.coords['lat'].values, variables)
print('----tide df----')
print(tidf)
