from obrero import cal

import numpy as np
import pandas as pd
import cftime

from numpy.testing import assert_equal


def test_leap_year():
    actual = cal._leap_year(2020, calendar='standard')
    expected = True
    assert actual == expected


def test_get_dpm():
    drange = pd.date_range('2001-01-01', periods=5, freq='M')
    actual = cal.get_dpm(drange, calendar='standard')
    expected = [31, 28, 31, 30, 31]
    assert_equal(actual, expected)


def test_get_month_name():
    actual = cal.get_month_name(1, names='1L')
    expected = 'J'
    assert actual == expected


def test_get_dates_pandas():
    dates = np.arange('2001-01', '2001-05', dtype='datetime64[M]')
    actual = cal.get_dates(dates)
    expected = pd.date_range('2001-01-01', periods=4, freq='MS')
    assert_equal(actual.values, expected.values)


def test_get_dates_cftime():
    dates = cftime.num2date([0, 1, 2, 3], 'months since 2001-01-01',
                            calendar = '360_day')
    actual = cal.get_dates(dates)
    expected = pd.date_range('2001-01-01', periods=4, freq='MS')
    assert_equal(actual.values, expected.values)
