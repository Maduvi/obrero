import warnings

import numpy as np
import pandas as pd

from obrero import mcwd

from .test_io import *

from numpy.testing import assert_allclose


def test_get_saturation_index():
    precip = create_3d_data(1, periods=12)
    evap = create_2d_data(0)
    actual = mcwd.get_saturation_index(precip, evap)
    expected = 0
    assert_allclose(actual, expected)


def test_accumulate_cwd():
    precip = create_3d_data(1, periods=12)
    evap = create_2d_data(0)
    sat_index = mcwd.get_saturation_index(precip, evap)
    actual = mcwd.accumulate_cwd(precip, evap, sat_index)
    expected = 0
    assert_allclose(actual, expected)


def test_get_mcwd():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        precip = create_3d_data(1, periods=12)
        evap = create_2d_data(0)
        actual = mcwd.get_mcwd(precip, evap)
        expected = 0
        assert_allclose(actual.values, expected)


def test_mcwd_composite_map():
    map_exp = create_3d_data(1)
    mcwd_exp = create_3d_data(0)
    map_ctl = create_3d_data(1)
    mcwd_ctl = create_3d_data(0)
    actual, dumm = mcwd.mcwd_composite_map(mcwd_exp, map_exp, mcwd_ctl,
                                           map_ctl)
    expected = np.nan
    assert_allclose(actual.values, expected)


def test_plot_mcwd_composite(tmpdir):
    composite = create_2d_data(1.5)
    mcwd.plot_mcwd_composite(composite, save=str(tmpdir) + 'test.png')


def test_plot_malhi(tmpdir):
    data = dict(lon=[1., 2.], lat=[1., 2.], ctl_map=[1., 2.],
                exp_map=[1., 2.], ctl_mcwd=[1., 2.],
                exp_mcwd=[1., 2.], comp=[0.5, 1.5])
    table = pd.DataFrame(data)
    table.index.name = 'id'
    mcwd.plot_malhi(table, save=str(tmpdir) + 'test.png')


def test_panel_plot_malhi(tmpdir):
    data = dict(lon=[1., 2.], lat=[1., 2.], ctl_map=[1., 2.],
                exp_map=[1., 2.], ctl_mcwd=[1., 2.],
                exp_mcwd=[1., 2.], comp=[0.5, 1.5])
    table = pd.DataFrame(data)
    table.index.name = 'id'
    mcwd.panel_plot_malhi(table, save=str(tmpdir) + 'test.png')
    
