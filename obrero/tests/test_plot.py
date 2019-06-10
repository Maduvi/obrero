import numpy as np

from obrero import plot

from .test_io import *

from numpy.testing import assert_allclose


def test_create_save_dir(tmpdir):
    plot.create_save_dir(str(tmpdir) + 'test')


def test_corner_coords():
    data = create_2d_data(1)
    lat = data.latitude.values
    lon = data.longitude.values
    actual = plot.corner_coords(lat, lon)
    expected = np.tile(np.arange(-0.5, 5, 1), (2, 1))
    assert_allclose(actual, expected)


def test_save_func(tmpdir):
    plot.save_func(str(tmpdir) + 'test.png', False)


def test_create_clev():
    data = create_2d_data(1)
    actual = plot.create_clev(data, minv=-1, maxv=1, nlevels=3)
    expected = np.array([-1, 0, 1.])
    assert_allclose(actual, expected)


def test_plot_global_contour():
    data = create_2d_data(1)
    plot.plot_global_contour(data, minv=0, maxv=1)


def test_get_cyclic_values():
    data = create_2d_data(1)
    actual, dumm = plot.get_cyclic_values(data)
    expected = np.ones((5, 6))
    assert_allclose(actual, expected)
    
