import warnings

import numpy as np
import xarray as xr
from scipy.stats import ttest_ind

from . import cal


def get_climatology(data):
    """Compute 12-month climatological annual cycle.

    First we group all values by month and then we take the mean for
    every month.

    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        Input must have a `time` coordinate and must be an xarray
        object.

    Returns
    -------
    xarray.DataArray or xarray.Dataset which contains 12 time steps in
    which the first time is January and the last is December. We do
    not keep actual dates in the time index. 
    """  # noqa

    # compute mean grouping by month
    mongr = data.groupby('time.month')
    clim = mongr.mean(dim='time', keep_attrs=True)

    # rename month coordinate to time for compatibility
    clim = clim.rename({'month': 'time'})

    return clim


def get_anomalies(data, clim=None, return_clim=False):
    """Compute anomalies for dataset and its climatology.

    This function works only for monthly data. Climatology can be
    provided or will be computed. Anomalies are the values resulting
    from subtracting the mean values from a time series.

    Parameters
    ----------
    data: xarray.DataArray
         Input must be monthly datam and have a named `time` coorinate.
    clim: xarray.DataArray, optional
         It must contain only 12 time steps and have a named `time`
         coordinate. 
    return_clim: bool, optional
         In case user wants climatology also as output.

    Returns
    -------
    xarray.DataArray containing the time series of anomalies. It will
    also return the climatology as an xarray.DataArray if desired.
    """  # noqa

    if clim is not None:
        try:
            if clim.time.size != 12:
                msg = 'climatology must have 12 time steps'
                raise TypeError(msg)
        except AttributeError:
            msg = 'climatology must hava a named \'time\' dimension'
            raise TypeError(msg)
    else:
        clim = get_climatology(data)

    ntim = data.time.size
    nyear = ntim // 12
    ndim = len(data.dims)

    # tile climatology to this size
    if ndim == 2:
        tclim = np.tile(clim.values, (nyear, 1))
    elif ndim == 3:
        tclim = np.tile(clim.values, (nyear, 1, 1))
    elif ndim == 4:
        tclim = np.tile(clim.values, (nyear, 1, 1, 1))
    else:
        msg = 'only 2D, 3D or 4D data allowed'
        raise TypeError(msg)

    # compute anomalies
    anom = data.copy()
    anom.values = np.array(data.values) - tclim

    if return_clim is False:
        return anom
    else:
        return anom, clim


def get_season_means(data, calendar='360_day'):
    """Get season means: DJF, MAM, JJA and SON.

    Simply compute the mean of every three-month slices grouping
    December, January and February (DJF), March, April and May (MAM),
    July, June and August (JJA) and September, October and November
    (SON).

    Parameters
    ----------
    data: xarray.DataArray
        Input can be a time series of any length, but must have a
        named `time` coordinate.
    calendar: str, optional
        To be able to compute weights based on month's length, a
        calendar must be provided. Here it is assumed that data comes
        from a 360 days calendar, which means weights are not being
        actually used. Other options include: `standard`, `leap`,
        `365_day`, and `366_day`.

    Returns
    -------
    xarray.DataArray with 4 time steps, one for every season. It
    returns an object with a named `season` coordinate with string
    values to be able to select each season.
    """  # noqa

    # compute weights based on number of days in month
    mlen = xr.DataArray(
        cal.get_dpm(data.time.to_index(), calendar=calendar),
        coords=[data.time],
        name='month_length')
    seagr = mlen.groupby('time.season')
    wgt = seagr / seagr.sum()

    # season means
    seagr = (data * wgt).groupby('time.season')
    smean = seagr.sum(dim='time', keep_attrs=True)

    return smean


def get_season_series(data, calendar='360_day', season=None):
    """Get season means time series for all seasons or single one.
    
    Take the mean of every trimester for the whole time series. Output
    will have as many time steps as there are years times four. A
    single season can be picked in which output will have as many time
    steps as number of years.

    Parameters
    ----------
    data: xarray.DataArray
        Input must have a named `time` coordinate.
    calendar: str, optional
        To be able to compute weights based on month's length, a
        calendar must be provided. Here it is assumed that data comes
        from a 360 days calendar, which means weights are not being
        actually used. Other options include: `standard`, `leap`,
        `365_day`, and `366_day`.
    season: str, optional
        User can choose a single season: `DJF`, `MAM`, `JJA` or `SON`.
    
    Returns
    -------
    xarray.DataArray with a `season` index as well as a `time`
    index. The former will only contain the season name series,
    whereas the latter contains timestamps for values.
    """  # noqa

    # create index for seasons using timestamps
    time = data.time.values
    dindex = cal.get_dates(time)
    dindex.name = 'time'
    tstamp = dindex.to_period(freq='Q-NOV').to_timestamp(how='E')

    # assgin new coord
    data = data.assign_coords(tsamp=tstamp)

    # compute weights based on number of days in month
    mlen = xr.DataArray(
        cal.get_dpm(data.time.to_index(), calendar=calendar),
        coords=[data.time],
        name='month_length')
    mlen = mlen.assign_coords(tstamp=tstamp)
    mlgr = mlen.groupby('tstamp')
    wgt = mlgr / mlgr.sum()

    # get season series
    seagr = (data * wgt).groupby('tstamp')
    series = seagr.sum(dim='time', keep_attrs=True)

    # rename season to time for compatibility
    series = series.rename({'tstamp': 'time'})
    # series.time.encoding = dataarray.time.encoding

    # add season index (DJF, MAM, JJA, SON)
    sindex = series.time.dt.season.to_index()
    series = series.assign_coords(season=sindex)

    if season is not None:
        return series.sel(time=series.season == season)
    else:
        return series


def get_wind_speed(u, v):
    """Compute wind speed scalar field from U and V winds.

    This scalar field is obtained using the formula:

        speed =   square_root( u^2 + v^2 )

    where `u` is the zonal wind vector and `v` is the meridional wind
    vector.

    Parameters
    ----------
    u: xarray.DataArray
    v: xarray.DataArray

    Returns
    -------
    xarray.DataArray with wind speed.
    """  # noqa

    # inherit attributes from u
    spd = u.copy()

    # change some names
    spd.name = 'spd'
    spd.attrs['long_name'] = 'wind speed'
    spd.attrs['standard_name'] = 'wind_speed'

    # compute wind speed
    spd.values = np.sqrt(u.values ** 2 + v.values ** 2)

    return spd


def add_climatology(data, clim):
    """Add 12-month climatology to a data array with more times.

    Suppose you have anomalies data and you want to add back its
    climatology to it. In this sense, this function does the opposite
    of `get_anomalies`. Though in this case there is no way to obtain
    the climatology so it has to be provided.

    Parameters
    ----------
    data: xarray.DataArray
        Input must have a named `time` coordinate.
    clim: xarray.DataArray
        The climatology must have the same spatial dimensions as
        `data`. Naturally, the time dimension can differ. The values 
        of this array will be replicated as many times as `data` has.

    Returns
    -------
    xarray.DataArray with both fields added.
    """  # noqa

    # make sure shapes are correct
    ddims = len(data.dims)
    cdims = len(clim.dims)

    if ddims != cdims:
        msg = 'both data arrays must have same dimensions'
        raise ValueError(msg)

    # get number of years in dataarray
    years = np.unique(data.time.dt.year)
    nyear = years.size

    # get tiled shape
    tshape = np.ones(ddims, dtype=int)
    tshape[0] = nyear

    # create tiled climatology
    tclim = np.tile(clim.values, tshape)

    # add climatology to data array
    new = data.copy()
    new.values = np.array(data.values) + tclim

    return new


def get_significant_diff(adata, bdata, alpha=0.05, eqvar=True,
                         fill_value=np.nan, diff=True,
                         means_too=False):
    """Perform a student's t-test (or Welch's if eqvar = False) to
    test the hypothesis:

        H0: datasets have equal mean
        H1: datasets do not have equal mean

    Using a significance level defined by `alpha`. This function will
    first compute the means of both datasets and then compute the
    difference subtracting `bdata` from `adata`, so order
    matters. When difference is not significant (p-value > alpha), the
    function masks those values in the differences array. A fill value
    can be specified for the masking operation. Also if the user wants
    the calculated mean values as output, it is possible. The
    right-most spatial dimensions of both data sets must match. This
    means they can have different time sizes, though this is
    discouraged. If input data are masked arrays, `nan` values will be
    ignored. 

    Parameters
    ----------
    adata: xarray.DataArray
        Input must have a named `time` dimension and same spatial
        dimensions as `bdata`.
    bdata: xarray.DataArray
        Input must have a named `time` dimension and same spatial
        dimensions as `adata`.
    alpha: float, optional
        Significance level for the test. Default is 0.05.
    eqvar: bool, optional
        Equal or different variance. Whether this is a student's
        t-test or a Welch's t-test. Default is True, so the former.
    fill_value: float, optional
        For the masking operation. Default is numpy.nan.
    diff: bool, optional
        If the user wants, instead of differences, one can obtain the
        masked original values from `adata`. Default is True, so
        output the differences array.
    means_too: bool, optional
        Whether to output a tuple with both means as well:
       (differences, adata mean, bdata mean). Default is False, so
        only differences array is the output.
    
    Returns
    -------
    xarray.DataArray with significant differences between both
    datasets. It can also output the calculated means for each of
    them, or the original values of `adata` but masked.
    """  # noqa

    # ignore scipy warnings (masked arrays)
    msg1 = 'invalid value encountered in less_equal'
    msg2 = 'invalid value encountered in less'
    msg3 = 'invalid value encountered in greater'
    msg4 = 'Mean of empty slice'
    warnings.filterwarnings('ignore', message=msg1)
    warnings.filterwarnings('ignore', message=msg2)
    warnings.filterwarnings('ignore', message=msg3)
    warnings.filterwarnings('ignore', message=msg4)

    # check shapes
    if adata.shape[-2:] != bdata.shape[-2:]:
        msg = '2 letf-most dimensions of arrays must match'
        raise ValueError(msg)

    # first get means difference
    amean = adata.mean(dim='time', keep_attrs=True)
    bmean = bdata.mean(dim='time', keep_attrs=True)
    difference = np.array(amean.values) - np.array(bmean.values)

    # test statistical significance of differences
    tval, pval = ttest_ind(adata.values, bdata.values,
                           equal_var=eqvar, nan_policy='omit')

    # reject null hypothesis (equal means)
    psig = pval < alpha

    # whether output differences or data array
    if diff is True:
        values = difference
    else:
        values = np.array(amean.values)

    # mask non significant
    maskval = np.ma.masked_array(values, psig == False,  # noqa E712
                                 fill_value=fill_value)

    # return array
    new = amean.copy()
    new.values = maskval

    if means_too is True:
        return new, amean, bmean
    else:
        return new


def get_zonal_means(data, time_mean=False):
    """Take the average values along longitude axis.

    For every latitude we will average all longitudes. This is known
    as a zonal mean.

    Parameters
    ----------
    data: xarray.DataArray
        Input must have a named `longitude` coordinate. If `time_mea`
        is True, then it also must have a named `time` coordinate.
    time_mean: bool, optional
        Whether to take the mean along th time axis as well. Default
        is False.

    Returns
    -------
    xarray.DataArray with zonal means (no `longitude` coordinate).
    """  # noqa

    # sometimes things are masked, ignore
    msg = 'Mean of empty slice'
    warnings.filterwarnings('ignore', message=msg)

    # compute zonal mean
    zm = data.mean(dim='longitude', keep_attrs=True)

    if time_mean is True:
        zm = zm.mean(dim='time', keep_attrs=True)

    return zm


def _zonal_mpsi(V, PS, P, LAT):
    """Compute zonal mean meridional stream function like NCL."""

    # dimensions
    klev, nlat, mlon = V.shape

    # constants and params
    G = 9.80616    # gravity [m s-1]
    A = 6.37122e6  # earth spherical radius [m]
    PI = np.pi
    RAD = PI / 180
    CON = 2.0 * PI * A / G

    # minimum and maximum pressures
    PTOP = 500
    PBOT = 100500

    # check p values
    if np.min(P) < PTOP:
        print(np.min(P))
        msg = ('pressures in pressure array cannot be below 500' +
               ' Pa. Check units.')
        raise ValueError(msg)

    # latitude values from array and cosine
    COSLAT = np.cos(LAT * RAD)

    if np.min(PS) < PTOP:
        msg = 'surface pressure cannot be below 500 Pa. Check units.'
        raise ValueError(msg)

    # create empty arrays with extended levels
    ptmp = np.zeros(2 * klev + 1)
    vvprof = np.zeros(2 * klev + 1)
    dp = np.zeros(2 * klev + 1)

    # create empty arrays with same sizes
    vtmp = np.zeros((klev, nlat, mlon))
    zonal_mpsi = np.zeros((klev, nlat))

    # counter
    count = 0

    # fill extended pressure array top levels (k=0 is bottom)
    for k in range(1, 2 * klev, 2):
        ptmp[k] = P[count]
        count += 1

    # fill extended pressure array bottom levels with neighbors mean
    for k in range(2, 2 * klev - 1, 2):
        ptmp[k] = (ptmp[k + 1] + ptmp[k - 1]) * 0.5

    # assign bottom and top values
    ptmp[0] = PTOP
    ptmp[2 * klev] = PBOT

    # compute pressure differences
    for k in range(1, 2 * klev):
        dp[k] = ptmp[k + 1] - ptmp[k - 1]

    # mask those with greater pressure than surface
    for m in range(mlon):
        for n in range(nlat):
            for k in range(klev):
                if P[k] > PS[n, m]:
                    vtmp[k, n, m] = np.nan
                else:
                    vtmp[k, n, m] = V[k, n, m]

    # zonal mean
    vbar = np.nanmean(vtmp, axis=2)

    # now compute mpsi for each latitude
    for n in range(nlat):

        C = CON * COSLAT[n]

        # reset start to 0
        ptmp[0] = 0

        # replace all for nan values again
        for k in range(1, 2 * klev):
            ptmp[k] = np.nan

        # make v of all bottom levels 0
        for k in range(0, 2 * klev + 1, 2):
            vvprof[k] = 0

        # get zonal mean v for all top levels (except last one)
        count = 0
        for k in range(1, 2 * klev, 2):
            vvprof[k] = vbar[count, n]
            count += 1

        # accumulate vvprof (INTEGRAL sum) at bottom levels
        for k in range(1, 2 * klev, 2):
            kflag = k
            ptmp[k + 1] = ptmp[k - 1] - C * vvprof[k] * dp[k]

        # make 0 at the last level (bottom/surface)
        ptmp[kflag + 1] = -ptmp[kflag-1]

        # average accumulated values at data levels
        for k in range(1, kflag, 2):
            ptmp[k] = (ptmp[k + 1] + ptmp[k - 1]) * 0.5

        # fill zonal_mpsi array (the minus sign is a convention)
        count = 0
        for k in range(1, 2 * klev, 2):
            if not np.isnan(ptmp[k]):
                zonal_mpsi[count, n] = -ptmp[k]
            else:
                zonal_mpsi[count, n] = ptmp[k]
            count += 1

    return zonal_mpsi


def ncl_zonal_mpsi(v, ps, p):
    """This function computes zonal mean meridional stream function.
    

    This is an almost exact copy of the NCL built in function
    `zonal_mpsi`, which is originally written in Fortran and can be
    found in Github here:

        https://github.com/yyr/ncl/blob/master/ni/src/
            lib/nfpfort/zon_mpsi.f

    Since we have read NCL's open letter about stopping development,
    we have now started working with Python and we needed a reliable
    way to calculate this quantity. Maybe there is a better pythonic
    way of doing this, but for now this will do. So big thanks to NCL
    folks, we have been using their software extensively.

    Parameters
    ----------
    v: xarray.DataArray
        This is the zonal wind (eastward wind) in units of m s-1.
        This can be a 3D (lev, lat, lon) or 4D (time, lev, lat, lon)
        array. It must have named coordinate `latitude`.
        Levels coordinate must go from top-to-bottom, i.e. air
        pressure must be increasing.
    ps: xarray.DataArray
        This is the surface pressure field in units of Pa. It must be
        a 2D array (lat, lon), which shape must match `v`.
    p: numpy.ndarray
        This is an array with pressure values. It should be the same
        pressure values in the `v` levels coordinate. We request the
        user to input this array because it must be in units of Pa,
        and there is no easy way we can guess the units from the
        levels coordinate in `v` to convert them to Pa. As well as in
        `v`, these pressure values must go from top-to-bottom,
        i.e. air pressure must be increasing.

    Returns
    -------
    xarray.DataArray with zonal mean meridional stream function. It
    will have the same `levels` and `latitude` values as input
    `v`. Units are kg s-1.
    """  # noqa

    # get dimensions
    if len(v.dims) == 4:
        ntim, klev, nlat, mlon = v.shape
    elif len(v.dims) == 3:
        ntim = None
        klev, nlat, mlon = v.shape
    else:
        msg = 'Input zonal wind must be 3 or 4 dimensional'
        raise ValueError(msg)

    # get latitude values
    lat = v.latitude.values

    if ntim is not None:

        zmpsi = np.zeros((ntim, klev, nlat))

        for t in range(ntim):
            zmpsi[t] = _zonal_mpsi(v.values[t], ps.values, p, lat)
    else:
        zmpsi = _zonal_mpsi(v.values, ps.values, p, lat)

    # create xarray
    xzmpsi = v.copy()
    xzmpsi.values = zmpsi
    xzmpsi.name = 'zonal_mpsi'
    xzmpsi.attrs['long_name'] = 'Zonal Mean Meridional Stream Function'
    xzmpsi.attrs['standard_name'] = 'zonal_mpsi'
    
    return xzmpsi
