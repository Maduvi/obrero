import os
import pkg_resources

import numpy as np
import xarray as xr
import pandas as pd

from obrero import io
from obrero import utils
from obrero import spatial
from obrero import analysis

# path where stored oni data
DATA_PATH = pkg_resources.resource_filename('obrero', 'data/')


def get_oni(data=None, years=slice(None)):
    """Compute ONI from data or using a given xarray. The given
    xarray must be a sea surface temperature (SST) data set for it to
    make sense. Compare results of the data one to: 

        https://origin.cpc.ncep.noaa.gov/products/
            analysis_monitoring/ensostuff/ONI_v5.php

    The user can specify a range of years of interest. This function
    uses data from NOAA's Climate Prediction Center obtained from the
    URL:

        https://origin.cpc.ncep.noaa.gov/products/
            analysis_monitoring/ensostuff/detrend.nino34.ascii.txt
    
    It contains temperature anomalies in the Nino 3.4 region since
    1950. ONI is calculated taking the 3-month running mean of the
    anomalies. An ONI over +0.5  indicates El Nino conditions, whereas
    below -0.5 indicates La Nina conditions.

    Parameters
    ----------
    data: xarray.DataArray
        It can be any xarray but it should contain SST values as
        variable in either Celsius or Kelvin. It must have a named
        `time` coordinate.
    years: sequence, range, list, optional
        If not set, all values from 1950 will be used. You can set the
        number of years you desire using the `range` function.

    Returns
    -------
    pandas.core.frame.DataFrame object containing the ONI values for
    every 3-month period. The index in the DataFrame are the years and
    the column names are the 1-letter months of every 3-month group.
    """  # noqa

    # columns header for ONI
    CHDR = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS',
            'ASO', 'SON', 'OND', 'NDJ']

    # check if file was given or else use txt
    if data is None:
        foni = os.path.join(DATA_PATH, 'detrend.nino34.ascii.csv')
        anom = pd.read_csv(foni)
    else:
        # get nino34 region
        nino = spatial.get_rectreg(data, [190, 240, 5, -5])

        # get years
        YR = data.time.dt.year.values

        # get anomalies
        annino = analysis.get_anomalies(nino)

        # now average them to time series
        avnino = spatial.area_average(annino, method='area')

        # extract anomalies data and make pd object
        anom = pd.DataFrame({'ANOM': avnino.values.flatten(),
                             'YR': YR})

    # create 3-month running using pandas
    oni = anom.ANOM.rolling(3, min_periods=2).mean()

    # ignore first value which is nan
    oni = oni[1:]

    # get number of years (plus 1 to accomodate last)
    ndat = len(oni)
    ny = ndat // 12
    res = ndat % 12

    # missing months in final year (if needed)
    if res != 0:
        ny = ny + 1
        missmon = 12 - res

        # add nan to finish last year
        oni = oni.append(pd.Series([np.nan] * missmon),
                         ignore_index=True)

        # create list of years for indexing
        ly = range(anom.YR.iloc[0], anom.YR.iloc[-1] + 1)
    else:
        ly = range(anom.YR.iloc[0], anom.YR.iloc[-1])

    # reshape panda series to be ny x 12
    onir = np.reshape(oni.values, (ny, 12))

    # create data frame for oni
    dfoni = pd.DataFrame(onir.round(1), index=ly, columns=CHDR)

    return dfoni.loc[years]


def enso_finder(dfoni):
    """From an ONI dataframe, this function returns an array with
    1 for El Nino, -1 for La Nina and 0 for neutral years. It has been
    established that if ONI persists over 0.5 or below -0.5 for 5
    consecutive months it is an El Nino or a La Nina respectively.

    Parameters
    ----------
    dfoni: pandas.core.frame.DataFrame
        Input object should have been created with function
        `get_oni`. IT should contain ONI values in that very same table
        format with column names and years index.

    Returns
    -------
    pandas.core.frame.DataFrame with the same structure as the input
    data frame but with values only 0, -1 and 1.
    """  # noqa

    # shape and number of values
    nrow, ncol = dfoni.shape
    nval = dfoni.size

    # flatten array to get time series
    onits = dfoni.values.flatten()

    # sliding window
    window = 5

    # enso
    enso = np.zeros(nval)

    # lets count how many in every window
    for i in range(nval - window + 1):

        # get slice
        slc = onits[i:i + window]

        # counters
        c1, c2 = (0, 0)

        # accumulate counters
        for x in slc:
            if x >= 0.5:
                c1 += 1
            elif x <= -0.5:
                c2 += 1

        # if 5 ninos, then for all put a 1 in array
        if c1 == 5:
            for j in range(i, i + window):
                if enso[j] == 0:
                    enso[j] = 1

        # if 5 ninas, then for all put a -1 in array
        if c2 == 5:
            for j in range(i, i + window):
                if enso[j] == 0:
                    enso[j] = -1

    # go back to original shape
    enso = enso.reshape((nrow, ncol))

    # create new dataframe
    dfenso = pd.DataFrame(enso, columns=dfoni.columns,
                          index=dfoni.index)

    return dfenso


def amplify_with_enso(data, oni, factor, bounds=None):
    """This function will amplify values in a dataset following ENSO 
    phases defined by ONI. If a region is not given it will amplify
    values all values in data. ONI must have same time span as
    xarray dataarray.

    Parameters
    ----------
    data: xarray.DataArray
        Any data set with name coordinate `time` that will be
        amplified in a bounded region following ENSO phases.
    oni: pandas.core.frame.DataFrame
        Data frame with ONI values computed by function `get_oni`.
    factor: float
        Number of times to amplify anomalies in the data set provided.
    bounds: list, optional
        This list should contain bounds in the order:
            
            [longitude0, longitude1, latitude0, latitude1]

        If None provided, everything will be amplified.
    
    Returns
    -------
    xarray.DataArray with amplified values in the desired region.
    """  # noqa

    # find enso wiht given oni
    enso = enso_finder(oni)

    # flatten oni DataFrame array
    phase = enso.values.flatten()

    # get number of months
    nphase = len(phase)

    # get time size in dataarray
    ntim = data.time.size

    if nphase != ntim:
        msg = ('number of months in ONI must be equal to those in' +
               ' data array')
        raise ValueError(msg)

    if bounds is not None:

        # check bounds
        utils.check_bounds(bounds)

        i0, i1, j0, j1 = spatial.get_bounds_indices(data, bounds)
        xslice = slice(i0, i1)
        yslice = slice(j0, j1)
    else:
        xslice = slice(None)
        yslice = slice(None)

    # try to change name if possible
    try:
        data.long_name = data.long_name + ' amplified'
    except AttributeError:
        pass

    # amplify region
    for t in range(ntim):

        # get phase
        p = phase[t]

        # only if nino or nina according phase
        if p == 1.0 or p == -1.0:
            old_values = np.array(data.values[t, yslice, xslice])
            new_values = factor * old_values
            data.values[t, yslice, xslice] = new_values

    return data


def get_enso_phases(data, nino=True, nina=True, neutral=True,
                    calendar='360_day', save=None):
    """Create arrays for different ENSO phases from a given dataset. 

    An ENSO phase always starts at June of year 0 and lasts until May
    of next year (+1). This function uses lists that contain the
    starting years of different ENSO phases: warm El Niño phase, cold
    La Niña phase and neutral. These have been identified using the
    ONI index from the Climate Prediction Center of NOAA. The users can
    choose which of the phases they want.

    Parameters
    ----------
    data: xarray.DataArray or xarray.DataSet
        This array should have a named `time` coordinate and a known
        calendar to be able to choose the years correctly.
    nino: bool, optional
        Whether the users want to get the years in which there were El
        Niño conditions. Default is True.
    nina: bool, optional
        Whether the users want to get the years in which there were La
        Niña conditions. Default is True.
    neutral: bool, optional
        Whether the users want to get the years in which there were
        neutral conditions. Default is True. 
    calendar: str, optional
        This option is to support 360 days calendars which do not have
        day May 31st. Default is set to 360 days calendars, so the
        last day used is May 30th, but the user can change this if
        their input uses a different calendar and include that last
        single day. 
    save: bool or str, optional
        Whether to output netCDF files which every phase. If this is a
        boolean, all output files will be named `output_phase.nc`,
        otherwise the users can specify a string base name for all
        files. 

    Returns
    -------
    list object containing desired xarray.DataArrays in the order: 

        [el nino, la nina, neutral]

   if all three are selected or only two of them.
    """  # noqa

    # initial year of phases
    inino = [1979, 1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004,
             2006, 2009]
    inina = [1983, 1984, 1988, 1995, 1998, 1999, 2000, 2005, 2007,
             2008]
    ineut = [1980, 1981, 1985, 1989, 1990, 1992, 1993, 1996,
             2001, 2003]

    # list for all phases
    phases = []

    # create empy lists
    lnino, lnina, lneut = [], [], []

    if nino is True:

        for iyear in inino:

            # create date strings
            eyear = iyear + 1
            idate = '%04i-06-01' % iyear

            if calendar != '360_day':
                edate = '%04i-05-31' % eyear
            else:
                edate = '%04i-05-30' % eyear

            lnino.append(data.sel(time=slice(idate, edate)))

        # create dataset
        dnino = xr.concat(lnino, dim='time')

        # save dataset or not
        if save is not None:
            enc = data.time.encoding
            dnino.time.encoding = enc
            if save is True:
                io.save_nc(dnino, 'output_nino.nc')
            elif save is False:
                pass
            else:
                io.save_nc(dnino, save + '_nino.nc')

        phases.append(dnino)

    if nina is True:

        for iyear in inina:

            # create date strings
            eyear = iyear + 1
            idate = '%04i-06-01' % iyear

            if calendar != '360_day':
                edate = '%04i-05-31' % eyear
            else:
                edate = '%04i-05-30' % eyear

            lnina.append(data.sel(time=slice(idate, edate)))

        # create dataset
        dnina = xr.concat(lnina, dim='time')

        # save dataset or not
        if save is not None:
            enc = data.time.encoding
            dnina.time.encoding = enc
            if save is True:
                io.save_nc(dnina, 'output_nina.nc')
            elif save is False:
                pass
            else:
                io.save_nc(dnina, save + '_nina.nc')

        phases.append(dnina)

    if neutral is True:

        for iyear in ineut:

            # create date strings
            eyear = iyear + 1
            idate = '%04i-06-01' % iyear

            if calendar != '360_day':
                edate = '%04i-05-31' % eyear
            else:
                edate = '%04i-05-30' % eyear

            lneut.append(data.sel(time=slice(idate, edate)))

        # create dataset
        dneutral = xr.concat(lneut, dim='time')

        # save dataset or not
        if save is not None:
            enc = data.time.encoding
            dnino.time.encoding = enc
            if save is True:
                io.save_nc(dneutral, 'output_neutral.nc')
            elif save is False:
                pass
            else:
                io.save_nc(dneutral, save + '_neutral.nc')

        phases.append(dneutral)

    return phases
