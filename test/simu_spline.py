import numpy as np
import numpy.ma as ma
import xarray as xr
from src.model_utils import spline_2d, fill_gridded_nans
import unittest


def mock_data():
    # Create a mock 8x8 data grid
    # Random values for the real part
    real_part = np.round(10*np.random.rand(8, 8), 0)
    # Random values for the imaginary part
    imag_part = np.round(10*np.random.rand(8, 8), 0)
    idata = real_part + 1j * imag_part

    # Mask values
    idata[2, 3] = 0
    idata[5, 6] = np.nan
    idata[7, 4] = 0

    # Create a mask for zeros and NaNs
    mask = np.logical_or(idata == 0, np.isnan(idata))
    return np.ma.array(idata, mask=mask)


def test_spline_2d():
    idata = fill_gridded_nans(mock_data().data)
    print(idata)
    print('-----simulate interpolation-----')

    # Mock 8x8 grid for lon and lat
    lon, lat = np.linspace(0, 1, 8), np.linspace(0, 1, 8)

    # Mock 4x4 region within the 8x8 grid for ilon and ilat
    ilon, ilat = np.linspace(0.2, 0.6, 4), np.linspace(0.2, 0.6, 4)
    ilon_grid, ilat_grid = np.meshgrid(ilon, ilat)

    # Get the interpolated data
    interpolated_data = spline_2d(lon, lat, idata, ilon_grid, ilat_grid)

    # Assertions or evaluations can go here
    # For now, we'll just print the shape of the result and its mask
    print(interpolated_data.data)  # Expected (4, 4)
    print(interpolated_data.mask)


# test_spline_2d()

class TestSplineInterpolation(unittest.TestCase):

    def setUp(self):
        # 8x8 grids for lon and lat
        self.lon_axis = np.linspace(0, 7, 8)
        self.lat_axis = np.linspace(0, 7, 8)

        # Interpolation grids (4x4)
        self.ilon, self.ilat = np.meshgrid(
            np.linspace(2, 5, 9), np.linspace(2, 5, 9))

    def generate_fixed_pattern(self):
        # Generate a complex array with real and imaginary parts being integers.
        data = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, np.nan, 9, 10],  # Note the 6 being marked True
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],   # Note the 6 -> NA being marked True
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 0, 1, 2, 3]
        ])
        # imag_part = real_part - 1    # Just an example, you can set your own pattern
        data = fill_gridded_nans(data)  # real_part + 1j * imag_part

        # Mask specific positions
        masked_data = np.ma.array(data, mask=False)
        masked_data.mask[2, 3] = True
        masked_data.mask[4, 5] = True
        return masked_data

    def generate_random_pattern(self):
        # Generate a random complex array with values rounded to 2 decimals
        # Multiplied by 10 for variety
        real_part = np.round(np.random.rand(8, 8) * 10, 2)
        imag_part = np.round(np.random.rand(8, 8) * 10, 2)
        data = real_part + 1j * imag_part

        # Mask specific positions
        masked_data = np.ma.array(data, mask=False)
        masked_data.mask[2, 3] = True
        masked_data.mask[5, 6] = True
        return masked_data

    def test_extrapolate(self):
        data = self.generate_fixed_pattern()
        print(data)
        ds = xr.Dataset({
            "z": (['lat', 'lon'], data),
        }, coords={
            'lon': self.lon_axis,
            'lat': self.lat_axis
        })

        lon_test = np.linspace(0.25, 7.25, 9)
        lat_test = np.linspace(0.25, 7.25, 9)
        ds_interp = ds.interp(lon=lon_test, lat=lat_test, kwargs={
                              "fill_value": "extrapolate"})

        result = ds_interp["z"].values
        print('----Test 0.0 -----')
        print(result)
        self.assertIsNotNone(result)

    def test_fixed_pattern(self):
        data = self.generate_fixed_pattern()
        result = spline_2d(self.lon_axis, self.lat_axis,
                           data, self.ilon, self.ilat)
        print(data)
        # print(self.ilon)
        # print(self.ilat)
        print('----Test 1 -----')
        print(result)
        self.assertIsNotNone(result)  # Check if result is not None
        # Add more assertions based on expected results or behaviors

    def test_random_pattern(self):
        data = self.generate_random_pattern()
        result = spline_2d(self.lon_axis, self.lat_axis,
                           data, self.ilon, self.ilat)
        print('----Test 2 -----')
        # np.set_printoptions(formatter={'complex_kind': '{:.2f}'.format})
        print(np.around(result, 2))
        self.assertIsNotNone(result)  # Check if result is not None
        # Add more assertions based on expected results or behaviors


if __name__ == "__main__":
    unittest.main()
