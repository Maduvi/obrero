import warnings

import numpy as np
import xarray as xr
from scipy.special import legendre

from . import utils


def get_rectreg(data, bounds):
    """Returns data set only for rectangular region.

    This can be used to selec a rectangular region of interest in a
    data array.

    Parameters
    ----------
    data: xarray.DataArray
        It must have named coordinates `latitude` and `longitude`.
    bounds: tuple or list
        Bounds must have the sequence: [x0, x1, y0, y1], using x for
        longitudes and y for latitudes.

    Returns
    -------
    xarray.DataArray sliced to desired bounds.
    """  # noqa

    # check bounds
    utils.check_bounds(bounds)

    # unpack bounds
    x0, x1, y0, y1 = bounds

    # get region
    reg = data.sel(latitude=slice(y0, y1),
                   longitude=slice(x0, x1))

    # wheter dataset or dataarray
    if isinstance(data, xr.core.dataset.Dataset):
        for r in reg:
            if 0 in reg[r].shape:
                msg = ('empty values in some coordinate: check' +
                       ' for inverted latitudes or longitudes')
                warnings.warn(msg)
    else:
        if 0 in reg.shape:
            msg = ('empty values in some coordinate: check' +
                   ' for inverted latitudes or longitudes')
            warnings.warn(msg)
    return reg


def coord_shift180(lon):
    """Enforce coordinate longiude to range from -180 to 180.

    Sometimes longitudes are 0-360. This simple function will subtract
    360 from those that are above 180 to get the more user-friendly 
    [-180, 180], since slicing is a lot easier.

    Parameters
    ----------
    lon: numpy.ndarray
        Array of longitudes with values in range [0, 360].
    
    Returns
    -------
    numpy.ndarray with longitudes in range [-180, 180].
    """  # noqa

    nlon = len(lon)

    # change those above 180.0 to negative
    for i in range(nlon):
        x = lon[i]
        if x > 180:
            lon[i] = x - 360

    return lon


def coslat_weights(data):
    """Weights for area averages based on cosine of latitude.

    This is a very widely used technique to obtain weights.

    Parameters
    ----------
    data: xarray.DataArray
        This array must have named coordinate `latitude`.
    
    Returns
    -------
    numpy.ndarray with 1 dimensions (latitude) and weights for each
    latitude.

    Note
    ----
    Use these weights after having done a zonal mean. To be able to
    multiply every latitude with its weight.
    """  # noqa

    # get latitudes
    lat = np.array(data.latitude.values)

    # convert latbnds and lonbds
    latrad = np.deg2rad(lat)

    # weights
    weights = np.cos(latrad)

    return weights


def gauss_weights(data):
    """Weights for area average using Legendre polynomial.

    Compute Gaussian weights solving Legendre polynomial.
    Use only when having a global domain (in latitude).

    Parameters
    ----------
    data: xarray.DataArray
        This array must have named coordinate `latitude`. Latitude
        should be global when solving these polynomials.
    
    Returns
    -------
    numpy.ndarray with 1 dimensions (latitude) and weights for each
    latitude.

    Note
    ----
    Use these weights after having done a zonal mean. To be able to
    multiply every latitude with its weight.
    """  # noqa

    # get number of latitudes
    nlat = data.latitude.size

    # get Legendre polynomial weights
    weights = legendre(nlat).weights[:, 1]

    return weights


def area_average(data, method='area'):
    """Get area average using weights.

    This function can be used to obtain time series from spatially 2D
    data with latitude and longitude coordinates.

    Parameters
    ----------
    data: xarray.DataArray
        Input array must have named coordinates `latitude` and
        `longitude`.
    method: str, optional
        Which method is used to obtain weights for
        latitudes. Available options are:
            
            'area': weights based on area.
            'coslat': weights based on cosine of latitude.
            'gauss': weights based on Legendre polynomials.
    
    Returns
    -------
    xarray.DataArray in which the 2 spatial dimensions have been
    collapsed into a weighted average. So if input had a time
    coordinate, now this returned array is a time series.
    """  # noqa

    # get weights
    if method == 'area':
        weights = area_weights(data)
    elif method == 'coslat':
        weights = coslat_weights(data)
    elif method == 'gauss':
        weights = gauss_weights(data)
    else:
        msg = 'unknown method for weights'
        raise ValueError(msg)

    # average all longitudes for each latitude (zonal mean)
    zm = data.mean(dim='longitude', keep_attrs=True)

    # weight latitudes
    wlat_mean = zm.copy()
    wlat_mean.values = (np.array(zm.values) * weights) / weights.sum()

    # add all weighted latitudes
    area_mean = wlat_mean.sum(dim='latitude', keep_attrs=True)

    return area_mean


def get_bounds_indices(data, bounds):
    """Get indices of bounds in latitudes and longitudes.

    These indices are necessary to be able to choose desired slices of
    data. Wheter longitudes or latitudes are inverted, this function
    checks that indices go from smaller to larger.

    Parameters
    ----------
    data: xarray.DataArray
        Input array must have named `latitude` and `longitude` coordinates.
    bounds: tuple or list
        Bounds must have the sequence: [x0, x1, y0, y1], using x for
        longitudes and y for latitudes.

    Returns
    -------
    Tuple object with indices for every bound value so:

    (i_x0, i_x1, j_y0, j_y1)

    using x for longitudes and y for latitudes.
    """  # noqa

    # check bounds
    utils.check_bounds(bounds)

    # extract coordinates
    lat = np.array(data.latitude.values)
    lon = np.array(data.longitude.values)

    # unpack bounds
    x0, x1, y0, y1 = bounds

    # bounds indice: minimum abs difference (0 at bound)
    lon0 = np.argmin(np.abs(lon - x0))
    lon1 = np.argmin(np.abs(lon - x1))
    lat0 = np.argmin(np.abs(lat - y0))
    lat1 = np.argmin(np.abs(lat - y1))

    # make sure in the right order
    if lon0 > lon1:
        old = lon0
        lon0 = lon1
        lon1 = old

    if lat0 > lat1:
        old = lat0
        lat0 = lat1
        lat1 = old

    return (lon0, lon1, lat0, lat1)


def cells_area(data):
    """Compute area for each grid cell.

    Computes latitude-longitude rectangle areas as explained by Doctor
    Rick in: http://mathforum.org/library/drmath/view/63767.html. Here
    the spherical radius of Earth is used as 6.3675e6 m.

    Parameters
    ----------
    data: xarray.DataArray
        This array must have named coordinates `latitude` and
        `longitude`.
    
    Returns
    -------
    numpy.ndarray with shape (nlat, mlon) containning area  of each
    grid cell. 
    """  # noqa

    # get latitudes and longitude
    lat = np.array(data.latitude.values)
    lon = np.array(data.longitude.values)

    # get ther sizes
    nlat = data.latitude.size
    mlon = data.longitude.size

    # check monotonic increase or decrease
    utils.check_monotonic(lat)
    utils.check_monotonic(lon)

    # get distances x and y using any 2 coords
    dx = abs(lon[1] - lon[0]) / 2.0
    dy = abs(lat[1] - lat[0]) / 2.0

    # guess bounds for each gridpoint
    lonbnds = []
    latbnds = []
    for x in lon:
        lonbnds.append([x + dx, x - dx])
    for y in lat:
        latbnds.append([y + dy, y - dy])

    # earth radius in meters
    R = 6.3675e6
    R2 = R ** 2

    # convert latbnds and lonbds
    latrad = np.deg2rad(latbnds)
    lonrad = np.deg2rad(lonbnds)

    # get areas using Doctor Rick formula
    areas = np.zeros((nlat, mlon))
    for i in range(nlat):
        yb = latrad[i]
        for j in range(mlon):
            xb = lonrad[j]
            s = R2 * (np.sin(yb[1]) - np.sin(yb[0])) * (xb[1] - xb[0])
            areas[i, j] = abs(s)

    return areas


def area_weights(data):
    """Weights for area averages based on areas.

    Computes weights for averaging based on latitude-longitude
    rectangle areas as explained by Doctor Rick in: http://mathforum
    .org/library/drmath/view/63767.html. Here the spherical radius of
    Earth is used as 6.3675e6 m.

    Parameters
    ----------
    data: xarray.DataArray
        This array must have named coordinates `latitude` and
        `longitude`.
    
    Returns
    -------
    numpy.ndarray with 1 dimension (latitude) and weights for each
    latitude.

    Note
    ----
    Use these weights after having done a zonal mean. To be able to
    multiply every latitude with its weight.
    """  # noqa

    # get cells area
    areas = cells_area(data)
    
    # weights are areas divided by total area
    weights = areas / areas.sum()
    latweights = np.sum(weights, axis=1)

    return latweights


def dashift180(dataarray):
    """
    Shift data array to -180,180 longitude when it comes in 0-360 flavour.
    """

    # get hemispheres
    right = dataarray.sel(longitude=slice(0, 180))
    left = dataarray.sel(longitude=slice(181, 360))

    # get longitude coordinate and shift it
    lon = dataarray.longitude.values
    shift = coord_shift180(lon)
    slon = np.concatenate([shift[shift < 0], shift[shift >= 0]])

    # change coordinate
    dataarray.longitude.values = slon

    # join hemispheres
    dataarray = xr.concat([left, right], dim='longitude')

    return dataarray
