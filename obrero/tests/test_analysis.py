import numpy as np

from obrero import analysis

from numpy.testing import assert_allclose

from .test_io import *

def test_get_climatology():
    da = create_3d_data(1)
    clim = analysis.get_climatology(da)
    actual = clim.values
    expected = np.tile(1, (12, 5, 5))
    assert_allclose(actual, expected)


def test_get_season_means():
    da = create_3d_data(1)
    smean = analysis.get_season_means(da)
    actual = smean.values
    expected = np.tile(1, (4, 5, 5))
    assert_allclose(actual, expected)


def test_get_season_series():
    da = create_3d_data(1)
    series = analysis.get_season_series(da)
    actual = series.values
    expected = np.tile(1, (9, 5, 5))
    assert_allclose(actual, expected)


def test_get_wind_speed():
    u = create_3d_data(2)
    v = create_3d_data(2)
    spd = analysis.get_wind_speed(u, v)
    actual = spd.values
    expected = np.sqrt(8)
    assert_allclose(actual, expected)


def test_add_climatology():
    da = create_3d_data(1)
    clim = create_3d_data(1, periods=12)
    new = analysis.add_climatology(da, clim)
    actual = new.values
    expected = np.tile(2, (24, 5, 5))
    assert_allclose(actual, expected)


def test_get_significant_diff():
    da1 = create_3d_data(1)
    da2 = create_3d_data(2)
    diff = analysis.get_significant_diff(da1, da2)
    actual = diff.values
    expected = -1
    assert_allclose(actual, expected)


def test_get_zonal_means():
    da = create_3d_data(1)
    zm = analysis.get_zonal_mean(da)
    actual = zm.values
    expected = np.tile(1, (24, 5))
    assert_allclose(actual, expected)
