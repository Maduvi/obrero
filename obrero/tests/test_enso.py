from obrero import io
from obrero import enso

from .test_io import *

from numpy.testing import assert_allclose

def test_get_oni():
    oni = enso.get_oni(years=1980)
    actual = oni[0]
    expected = 0.6
    assert actual == expected


def test_enso_finder():
    oni = enso.get_oni()
    ens = enso.enso_finder(oni)
    actual = ens.loc[1997][-1]
    expected = 1
    assert actual == expected


def test_amplify_with_enso(tmpdir):
    data = create_3d_data(1)
    oni = enso.get_oni(years=range(2000, 2002))
    amp = enso.amplify_with_enso(data, oni, 2)
    actual = amp.values[0]
    expected = 2
    assert_allclose(actual, expected)
    

def test_get_enso_phases():
    data = create_3d_data(1)
    nina, = enso.get_enso_phases(data, nino=False,neutral=False)
    assert nina.time.size == 15
