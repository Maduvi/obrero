from obrero import io

import os.path

import numpy as np
import xarray as xr
import pandas as pd


def read_example_data(file_name):
    path = os.path.join(os.path.dirname(__file__), 'data', file_name)
    return io.read_nc(path, 'var')


def create_2d_data(num=1):
    data = np.tile(num, (5, 5))
    coords = {'latitude': np.arange(5), 'longitude': np.arange(5)}
    da = xr.DataArray(data, coords=coords,
                      dims=('longitude', 'latitude'),
                      name = 'var')
    return da


def create_3d_data(num=1, periods=24, freq='M'):
    data = np.tile(num, (periods, 5, 5))
    dates = pd.date_range('2000-01-01', periods=periods, freq=freq)
    da = xr.DataArray(data,
                      coords={'time': dates, 'latitude': np.arange(5),
                              'longitude': np.arange(5)},
                      dims=('time', 'latitude', 'longitude'),
                      name = 'var')
    return da


def create_4d_data(num=1, periods=24, freq='M'):
    data = np.tile(num, (periods, 5, 5, 5))
    dates = pd.date_range('2000-01-01', periods=periods, freq=freq)
    da = xr.DataArray(data,
                      coords={'time': dates,
                              'level': np.arange(5),
                              'latitude': np.arange(5),
                              'longitude': np.arange(5)},
                      dims=('time', 'level', 'latitude', 'longitude'))
    return da


def test_read_nc():
    read_example_data('example.nc')


def test_save_nc(tmpdir):
    data = read_example_data('example.nc')
    io.save_nc(data, str(tmpdir) + 'test.nc', check=False)
