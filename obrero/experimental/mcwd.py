import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
from matplotlib.lines import Line2D

from .plasim_t21 import COORD_REGION1, REG_MARK1
from .plasim_t21 import COORD_REGION2, REG_MARK2

from obrero import io
from obrero import analysis
from obrero import utils
from obrero import plot as oplot

def mcwd_composite_map(mcwd_exp, map_exp, mcwd_ctl, map_ctl,
                       save=None, save_table=None):
    """
    Create 2D composite array in which numbers indicate how different
    are an experiment simulation (EXP) and its control simulation
    (CTL) in terms of maximum climatological water deficit (MCWD) and
    mean annual precipitation (MAP). In this case 'much' means a
    deviation (+/-) greater than 10% of the value of CTL. This map is
    explained in greater detail in Duque et al. (2019).This function
    also creates a pandas dataframe containing values of MAP, MCWD,
    coordinates and composite map for both simulations.

    Parameters
    ----------
    mcwd_exp: xarray.DataArray
        This data array contains MCWD values in units of mm for the
        experiment simulation. It can be obtained with function
       `get_mcwd` in this module.
    map_exp: xarray.DataArray
        This data array contains MAP values in units of mm year-1 for
        the experiment simulation.
    mcwd_ctl: xarray.DataArray
        This data array contains MCWD values in units of mm for the
        control simulation. It can be obtained with function
       `get_mcwd` in this module.
    map_ctl: xarray.DataArray
        This data array contains MAP values in units of mm year-1 for
        the control simulation.
    save: bool or str, optional
        This can be a boolean flag to create a netCDF file with the
        composite map, in which case the file will be named
        `output_composite.nc`, or a string with a specific name for
        the file.
    save_table: bool or str, optional
        This can be a boolean flag to create a CSV file with the
        values of the composite map, along with coordinates and MCWD,
        and MAP values for both simulations. In this case the file
        will be named `table.csv`. Or you can use a string with a
       specific name for this file.

    Returns
    -------
    A tuple of the form:

        (xarray.DataArray, pandas.DataFrame)

    The xarray contains the values of the composite map and the pandas
    data frame contains the values of MAP, MCWD, coordinates and
    composite map for both simulations.
    """  # noqa

    # get dimensions
    ntim, nlat, mlon = mcwd_exp.shape

    # get significant differences for both
    d1 = analysis.get_significant_diff(map_exp, map_ctl, eqvar=False)
    d2 = analysis.get_significant_diff(mcwd_exp, mcwd_ctl, eqvar=False)

    # get mean values
    pr_ctl = map_ctl.mean(dim='time', keep_attrs=True)
    pr_exp = map_exp.mean(dim='time', keep_attrs=True)
    wd_ctl = mcwd_ctl.mean(dim='time', keep_attrs=True)
    wd_exp = mcwd_exp.mean(dim='time', keep_attrs=True)

    # creating composite map
    comp = np.zeros((nlat, mlon))

    for i in range(nlat):
        for j in range(mlon):

            # get siginificant differences of pr (1) and mcwd (2)
            sd1 = -d1.values[i, j]  # if <0 exp rains less, so make positive
            sd2 = d2.values[i, j]
            c10pr = 0.1 * pr_ctl.values[i, j]
            c10wd = 0.1 * wd_ctl.values[i, j]

            # big if (filter no data)
            if not np.isnan(sd1) and not np.isnan(sd2):
                # pr much worse    wd much worse
                if sd1 > c10pr and sd2 < c10wd:
                    comp[i, j] = 0.5
                # pr much worse      wd worse but not much
                elif sd1 > c10pr and sd2 > c10wd and sd2 < 0:
                    comp[i, j] = 1.5
                # pr worse but not much       wd much worse
                elif sd1 < c10pr and sd1 > 0 and sd2 < c10wd:
                    comp[i, j] = 2.5
                # pr worse but not much       wd worse but not much
                elif sd1 < c10pr and sd1 > 0 and sd2 > c10wd and sd2 < 0:
                    comp[i, j] = 3.5
                # pr better but not much           wd better but not much
                elif -sd1 < c10pr and -sd1 > 0 and sd2 < -c10wd and sd2 > 0:
                    comp[i, j] = 4.5
                # pr much better       wd much better
                elif -sd1 > c10pr and sd2 > -c10wd:
                    comp[i, j] = 5.5
                else:
                    comp[i, j] = np.nan
            else:
                comp[i, j] = np.nan

    # get coordinates
    lat = np.array(map_ctl.latitude.values)
    lon = np.array(map_ctl.longitude.values)

    # create new dataarray
    xarr = xr.DataArray(comp, coords={'latitude': lat, 'longitude': lon},
                        dims=('latitude', 'longitude'),
                        name='composite')
    xarr.attrs['long_name'] = 'Composite MAP and MCWD'
    xarr.attrs['units'] = '1'

    # create pandas dataframe with values of variables
    x, y, p1, p2, w1, w2, cm = [], [], [], [], [], [], []

    for i in range(nlat):
        for j in range(mlon):
            c = comp[i, j]
            if not np.isnan(c):
                # ignore antarctica
                if lat[i] > -60.0:
                    x.append(lon[j])
                    y.append(lat[i])
                    p1.append(pr_ctl.values[i, j])
                    p2.append(pr_exp.values[i, j])
                    w1.append(wd_ctl.values[i, j])
                    w2.append(wd_exp.values[i, j])
                    cm.append(c)

    # create data dictionary
    data = dict(lon=x, lat=y, ctl_map=p1, exp_map=p2, ctl_mcwd=w1,
                exp_mcwd=w2, comp=cm)
    table = pd.DataFrame(data)
    table.index.name = 'id'

    # whether create netcdf file
    if save is not None:
        if save is True:
            io.save_nc(xarr, 'output_composite.nc')
        elif save is False:
            pass
        else:
            io.save_nc(xarr, save)

    # whether get pandas dataframe
    if save_table is not None:
        if save_table is True:
            table.to_csv('table.csv', float_format='%8.1f')
            print('Created table \'table.csv\'')
        elif save_table is False:
            pass
        else:
            table.to_csv(save_table, float_format='%8.1f')
            print('Created table \'' + save_table + '\'')

    return xarr, table


def get_mcwd(precipitation, evaporation, save=None):
    """
    Compute Maximum Climatological Water Deficit as described by Malhi
    et al. (2009). `evaporation` is a reference time mean map (2D) 
    that describes average energy requirements in every gridpoint. If
    you want to use a fixed value (as in Malhi et al. (2009)) you will
    need to create an array with the same spatial extent as the
    `precipitation` array.

    Parameters
    ----------
    precipitation: xarray.DataArray
        Input `precipitation` should be monthly. Units should be mm 
        month-1. 
    evaporation: xarray.DataArray
        This is a 2D map with only latitude and longitude as
        coordinates. It should be the time mean of evaporation and
        units should be mm month-1.
    save: bool or str, optional
        This can be a boolean flag to create a netCDF file with the
        composite map, in which case the file will be named
        `output_mcwd.nc`, or a string with a specific name for
        the file.

    Returns
    -------
    xarray.DataArray with the values of MCWD.
    """  # noqa

    # make sure evaporation is 2D
    if 'time' in evaporation.dims:
        msg = 'evaporation must be a 2D array (time mean)'
        raise ValueError(msg)

    if 'units' in precipitation.attrs:
        if precipitation.attrs['units'] != 'mm month-1':
            msg = 'units were not identified as mm month-1 but' + \
                'will continue anyways'
            warnings.warn(msg)
    else:
        msg = 'units were not identified as mm month-1 but' + \
            'will continue anyways'
        warnings.warn(msg)

    # get dimensions
    ntim, nlat, mlon = precipitation.shape

    if ntim % 12 != 0:
        msg = 'input precipitation has incomplete years but' + \
            ' will continue anyways'
        warnings.warn(msg)

    # number of years
    nyear = len(np.unique(precipitation.time.dt.year.values))

    # create array to fill with maximums
    mcwd = np.zeros((nyear, nlat, mlon))

    # keep dates in list
    dates = []

    # loop every 12 months
    for i in range(0, ntim, 12):

        # current year
        y = i // 12

        # get date for this year
        dates.append(precipitation.time.values[i + 11])

        # subset precipitation for this year only
        P = precipitation[i:i + 12]

        # get indices for saturation
        saturation_id = get_saturation_index(P, evaporation)

        # accumulating cwd
        cwd = accumulate_cwd(P, evaporation, saturation_id)

        # save minimum value in array (most negative)
        mcwd[y] = np.min(cwd, axis=0)

    # get coordinates
    lat = np.array(precipitation.latitude.values)
    lon = np.array(precipitation.longitude.values)

    # create new dataarray
    xarr = xr.DataArray(mcwd, coords={'time': dates,
                                      'latitude': lat,
                                      'longitude': lon},
                        dims=('time', 'latitude', 'longitude'),
                        name='mcwd')
    xarr.attrs['long_name'] = 'Maximum Climatological Water Deficit'
    xarr.attrs['units'] = 'mm'

    # encoding for time processing
    enc = precipitation.time.encoding
    xarr.time.encoding = enc

    # whether create netcdf file
    if save is not None:
        if save is True:
            io.save_nc(xarr, 'output_mcwd.nc')
        elif save is False:
            pass
        else:
            io.save_nc(xarr, save)

    return xarr


def accumulate_cwd(precipitation, evaporation, saturation_index):
    """
    Create Climatological Water Deficit (CWD) array and accumulate
    deficit (P - E) as described in Malhi et al. (2009). Index must
    have been obtained using function `get_saturation_index`. Though
    code does not impose it, it is best if input is 12-month data.

    Parameters
    ----------
    precipitation: xarray.DataArray
        Input `precipitation` should be monthly. Units should be mm 
        month-1. 
    evaporation: xarray.DataArray
        This is a 2D map with only latitude and longitude as
        coordinates. It should be the time mean of evaporation and
        units should be mm month-1.
    saturation_index: numpy.ndarray
        This array contains the indices of the months in which
        evaporation is less than precipitation and we can assume
        saturation of the soil. It should have been obtained using
        function `get_saturation_index`.
    
    Returns
    -------
    numpy.ndarray with the CWD for a 12-month period.
    """  # noqa

    # make sure evaporation is 2D
    if 'time' in evaporation.dims:
        msg = 'evaporation must be a 2D array (time mean)'
        raise ValueError(msg)

    # get dimensions
    ntim, nlat, mlon = precipitation.shape

    # get values
    P = np.array(precipitation.values)
    E = np.array(evaporation.values)

    # intialize cwd as all nan
    cwd = np.zeros((ntim, nlat, mlon))

    # traverse all gripoints
    for i in range(nlat):
        for j in range(mlon):

            # at this month cwd must be 0 (saturation)
            start = saturation_index[i, j]

            # accumulate for those starting in January
            if not np.isnan(start) and start == 0:

                # set to 0 once again after every loop
                cwd[int(start), i, j] = 0.0

                # loop from February to December
                for t in range(1, 12):

                    wd = cwd[t - 1, i, j] + P[t, i, j] - E[i, j]

                    # if deficit not negative, make 0 (no deficit)
                    if wd <= 0:
                        cwd[t, i, j] = wd
                    else:
                        cwd[t, i, j] = 0.0
            # accumulate for those starting after January
            elif not np.isnan(start):

                # set to 0 once again after every loop
                cwd[int(start), i, j] = 0.0

                # loop from start (any month but Jan) to December
                for t in range(int(start) + 1, 12):

                    wd = cwd[t - 1, i, j] + P[t, i, j] - E[i, j]

                    # if deficit not negative, make 0 (no deficit)
                    if wd <= 0:
                        cwd[t, i, j] = wd
                    else:
                        cwd[t, i, j] = 0.0

                # now loop from January to start - 1
                for t in range(int(start)):
                    if t == 0:

                        # closure: first uses last from previous loop
                        wd = cwd[11, i, j] + P[t, i, j] - E[i, j]

                        # if deficit not negative, make 0 (no deficit)
                        if wd <= 0:
                            cwd[t, i, j] = wd
                        else:
                            cwd[t, i, j] = 0.0
                    else:

                        wd = cwd[t - 1, i, j] + P[t, i, j] - E[i, j]

                        # if deficit not negative, make 0 (no deficit)
                        if wd <= 0:
                            cwd[t, i, j] = wd
                        else:
                            cwd[t, i, j] = 0.0
            else:
                # if there is no start, make nan
                cwd[:, i, j] = np.nan

    return cwd


def get_saturation_index(precipitation, evaporation):
    """
    To compute MCWD we need months in which P > E so we can assume
    saturation in the soil. This function will find months in which
    this happens. It initially guesses that it happens when max(P),
    but then checks if this is true and fixes it. Though code does
    not impose it, it is best if input is 12-month data.

    Parameters
    ----------
    precipitation: xarray.DataArray
        Input `precipitation` should be monthly. Units should be mm 
        month-1. 
    evaporation: xarray.DataArray
        This is a 2D map with only latitude and longitude as
        coordinates. It should be the time mean of evaporation and
        units should be mm month-1.

    Returns
    -------
    numpy.ndarray with the indices of months in which saturation is
    attained (P > E).
    """  # noqa

    # make sure evaporation is 2D
    if 'time' in evaporation.dims:
        msg = 'evaporation must be a 2D array (time mean)'
        raise ValueError(msg)

    # get dimensions
    ntim, nlat, mlon = precipitation.shape

    # get values
    P = np.array(precipitation.values)
    E = np.array(evaporation.values)

    # our first guess is that indices should be those of max_P
    # it has to be float to include nan values
    satmon_id = np.array(np.argmax(P, axis=0), dtype=float)

    # now lets check P > E or change index
    for i in range(nlat):
        for j in range(mlon):
            max_id = satmon_id[i, j]
            p = P[int(max_id), i, j]
            e = E[i, j]
            r = p - e

            if not np.isnan(r):
                # if max(P) not > E, use first whichin P > E
                if r < 0:
                    for t in range(ntim):
                        p = P[t, i, j]
                        e = E[i, j]
                        r = p - e

                        if not np.isnan(r):
                            if r >= 0:
                                satmon_id[i, j] = t
                            else:
                                satmon_id[i, j] = np.nan
                        else:
                            satmon_id[i, j] = np.nan
            else:
                satmon_id[i, j] = np.nan

    return satmon_id


def plot_mcwd_composite(composite, wmm=100, hmm=80, axes=None,
                        proj=None, lon0=0, save=None,
                        transparent=False):
    """Plot results from `mcwd_composite_map`.

    This is a custom map that combines significant differences in both
    mean annual precipitation (MAP) and maximum climatological water
    deficit (MCWD). To do this we first find significant
    differences. Then we ask whether this significan differences are
    above 10% of a control value. Based on this, each grid cell is
    assigned either numpy.nan or a value. Here we define how to plot
    those values in such composite map.

    Parameters
    ----------
    composite: xarray.DataArray
        This must be the output of function `mcwd_composite_map`.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 100 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 80 mm.
    axes: cartopy.mpl.geoaxes.GeoAxes, optional
        This axes should have some projection from Cartopy. If it is
        not provided, brand new axes will be created with default
        projection `proj`.
    proj: cartopy.crs.Projection, optional
        Map projection to be used to create the axes if not
        provided. By default we use the Mollweide projection.
    lon0: float, optional
        Central longitude to create the projection in the axes. By
        default the central longitudes is the Greenwich meridian. This
        argument is only meaningfull for the default projection which 
        is Mollweide. Otherwise it is unused.
    save: bool or str, optional
        This can be a boolean flag to create a PDF file with the
        plotted map, in which case the file will be named
        `output.pdf`, or a string with a specific name for the file.
        Default is only show the plot.
    transparent: bool, optional
        If `save` is True or some str object with a name, this keyword
        controls how the background is plotted. If True, then
        background will be transparent. This is useful if the image is
        to be used in slideshows. Default is False.

    Returns
    -------
    matplotlib.axes.Axes with plot attached.    
    """  # noqa

    # plot settings
    oplot.plot_settings()

    # get cyclic values and coords
    cval, clon = oplot.get_cyclic_values(composite)
    lat = composite.latitude.values

    # get projection if none given
    if proj is None:
        proj = oplot.moll(central_longitude=lon0)

        # in case we plot a single plot
    if axes is None:
        oplot.plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = oplot.plt.axes(projection=proj)
        maximize = 1
    else:
        maximize = 0

    # colormap
    # cmap = oplot.ListedColormap(['FireBrick', 'Chocolate', 'Orange',
    #                              'Yellow', 'YellowGreen', 'DarkGreen'])
    cmap_base = matplotlib.cm.get_cmap('BrBG')
    cmap = oplot.ListedColormap([cmap_base(x) for x
                                 in [0., 0.2, 0.28, 0.38, 0.8, 1.]])

    # levels
    lev = range(7)

    # normalize to number of colors
    cnorm = oplot.BoundaryNorm(lev, ncolors=cmap.N, clip=True)

    # add shorelines
    axes.coastlines(resolution='110m')

    # set global
    axes.set_global()

    # add gridlines
    oplot.add_gridlines(axes)

    # fix coords
    corlon, corlat = oplot.corner_coords(clon, lat)

    # plot map
    fmap = axes.pcolor(corlon, corlat, cval, cmap=cmap,
                       norm=cnorm, transform=oplot.pcar())

    # add colorbar
    cb = oplot.plt.colorbar(fmap, orientation='horizontal', pad=0.15,
                            shrink=0.8, ax=axes, extend='both')

    # colorbar ticks and labels
    fsize = 7
    bottom = ['much\ngreater', 'greater', 'much\ngreater',
              'greater', 'less', 'much\nless']
    top = ['much\nless', 'much\nless', 'less', 'less', 'greater',
           'much\ngreater']
    cb.set_ticks(np.arange(0.5, 6.5, 1))
    cb.ax.tick_params(top=True)
    cb.ax.set_xticklabels(bottom, multialignment='center',
                          fontsize=fsize)

    # add top ticklabels with custom text
    tickpos = cb.ax.get_xticks()
    for i, x in enumerate(tickpos):
        cb.ax.text(x, 1.8, top[i], fontsize=fsize, ha='center',
                   multialignment='center')

    # add titles at top and bottom
    cb.ax.text(-0.05, 2, r' \textbf{MAP}', fontsize=fsize,
               ha='center', va='bottom')
    cb.ax.text(-0.07, -2, r'\textbf{MCWD}', fontsize=fsize,
               ha='center', va='bottom')

    # maximize if only one
    if maximize == 1:
        oplot.plt.tight_layout()

        # savefig if provided name
        oplot.save_func(save, transparent)

    return axes


def plot_malhi(table_data, wmm=90, hmm=90, names=['CTL', 'EXP'],
               axes=None, ylim=[0, 4000], title='', bounds=None,
               legend=True, ylabel=r'MAP (mm year$^{-1}$)',
               xlabel=r'MCWD (mm)', save=None, transparent=False):
    """Plot output CSV file from `mcwd_composite_map`.

    This function will plot the table in comma separated values (CSV)
    format that the function `mcwd_composite_map` creates. It will
    read it as a dataframe and plot it, or you can go ahead and give the
    data frame directly. Since `mcwd_composite_map` only considers 2
    experiments, one control and one experimental simulation, this
    plot will do the same.
    
    Parameters
    ----------
    table_data: str or pandas.DataFrame
        If this is a string, it must be the name of the CSV file that
        `mcwd_composite_map` creates. But it can also be the
        pandas.DataFrame directly, without need of creating file.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 90 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
    names: list
        List with only 2 str names for two experiments. One of them is
        usually control. Default is ['CTL', 'EXP'].
    axes: matplotlib.axes.Axes, optional
        If this is going to be part of another bigger figure, a
        subplot axes can be provided for this plot to attach the ONI
        plot.
    ylim: list, optional
        List object with two float values to define the limits in the
        y axis. Default is [0, 4000].
    title: str, optional
        Center top title if desired. Default is empty.
    bounds: tuple or list, optional
        Bounds must have the sequence: [x0, x1, y0, y1], using x for
        longitudes and y for latitudes. Default is None.
    legend: bool, optional
        Whether to show legend or not.
    xlabel: str, optional
        Title for the x axis. Default is 'MCWD (mm)'.
    ylabel: str, optional
        Title for the y axis. Default is 'MAP (mm year-1)'.
    save: bool or str, optional
        This can be a boolean flag to create a PDF file with the
        plotted map, in which case the file will be named
        `output.pdf`, or a string with a specific name for the file.
        Default is only show the plot.
    transparent: bool, optional
        If `save` is True or some str object with a name, this keyword
        controls how the background is plotted. If True, then
        background will be transparent. This is useful if the image is
        to be used in slideshows. Default is False.

    Returns
    -------
    matplotlib.axes.Axes with plot attached.    
    """  # noqa

    # plot settings
    oplot.plot_settings()

    # check if df or text file
    if isinstance(table_data, str):
        table = pd.read_csv(table_data, index_col=0)
    elif isinstance(table_data, pd.core.frame.DataFrame):
        table = table_data

    # check it was the output from composite_map
    colnames = ['lon', 'lat', 'ctl_map', 'exp_map', 'ctl_mcwd',
                'exp_mcwd', 'comp']
    for x in colnames:
        if x in table.columns:
            pass
        else:
            msg = 'table missing \'' + x + '\' column'
            raise ValueError(msg)

    nrows = table.index.size

    # in case we plot a single plot
    if axes is None:
        oplot.plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = oplot.plt.axes()
        maximize = 1
    else:
        maximize = 0

    # colors dictionary
    cmap_base = matplotlib.cm.get_cmap('BrBG')
    cmap = [cmap_base(x) for x in [0., 0.2, 0.28, 0.38, 0.8, 1.]]
    cdict = {'0.5': cmap[0], '1.5': cmap[1], '2.5': cmap[2],
             '3.5': cmap[3], '4.5': cmap[4], '5.5': cmap[5]}

    # custom legend lines
    regions = []
    markers = []
    custom_names = ['Distance to ' + names[0]]
    custom_lines = [Line2D([0], [0], color='black', lw=0.5, alpha=0.5)]

    # to guess xlim later
    xplotted = []

    for i in range(nrows):
        row = table.loc[i]
        lat = row.lat
        lon = row.lon

        # get marker based on region
        key = '(%8.5f, %g)' % (lat, lon)
        reg = COORD_REGION1[key]
        mark = REG_MARK1[reg]

        if reg not in regions:
            regions.append(reg)

        if mark not in markers:
            markers.append(mark)

        # fix lon to be -180 - 180
        if lon > 180:
            lon = lon - 360

        x = [row.ctl_mcwd, row.exp_mcwd]
        y = [row.ctl_map, row.exp_map]
        c = cdict[str(row.comp)]

        if bounds is not None:

            # check bounds
            utils.check_bounds(bounds)

            # unpack bounds
            x0, x1, y0, y1 = bounds

            if (lon > x0) and (lon <= x1):
                if (lat > y0) and (lat <= y1):
                    axes.plot(x, y, color=c)
                    axes.plot(row.exp_mcwd, row.exp_map, '^',
                              color=c)

                    xplotted.extend(x)
        else:
            axes.plot(x, y, color=c, linewidth=0.5, alpha=0.5)
            axes.plot(row.exp_mcwd, row.exp_map, mark, color=c, ms=2)
            xplotted.extend(x)

    # guess min xlim
    if xplotted != []:
        positive_minx = abs(min(xplotted))
        positive_minx -= positive_minx % -100
        xlim = [-positive_minx, 0]
        axes.set_xlim(xlim)

    # plot settings
    axes.set_xlabel(xlabel)
    axes.set_ylabel(oplot.replace_minus(ylabel))
    axes.set_ylim(ylim)
    axes.set_title(title)
    axes.xaxis.set_major_formatter(oplot.FuncFormatter(oplot.no_hyphen))

    if legend is True:

        for r in regions:
            custom_names.append(r)

        for m in markers:
            custom_lines.append(Line2D([0], [0], linestyle='',
                                       marker=m, color='black', ms=2))
        axes.legend(custom_lines, custom_names, loc=2, ncol=3, fontsize=6)

    # maximize plot if only one
    if maximize == 1:
        oplot.plt.tight_layout()

        # savefig if provided name
        oplot.save_func(save, transparent)

    return axes


def panel_plot_malhi(table_data, wmm=180, hmm=120, names=['CTL', 'EXP'],
                     save=None, transparent=False):
    """Panels version of `plot_malhi`. 

    Here we separate several regions we have defined in COORD_REGION2
    within the plasim_t21.py module.

        REGION  LOCATION
        ----------------------------------------------
        R1: North America
        R2: Central America and Northern South America
        R3: Central South America (Amazonia)
        R4: Eurasia
        R5: East Africa
        R6: Australia

    Parameters
    ----------
    See `plot_malhi` parameters.

    Returns
    -------
    matplotlib.figure.Figure object with panel plots.
    """  # noqa

    # settings
    oplot.plot_settings()

    # check if df or text file
    if isinstance(table_data, str):
        table = pd.read_csv(table_data, index_col=0)
    elif isinstance(table_data, pd.core.frame.DataFrame):
        table = table_data

    # check it was the output from composite_map
    colnames = ['lon', 'lat', 'ctl_map', 'exp_map', 'ctl_mcwd',
                'exp_mcwd', 'comp']
    for x in colnames:
        if x in table.columns:
            pass
        else:
            msg = 'table missing \'' + x + '\' column'
            raise ValueError(msg)

    nrows = table.index.size

    # colors dictionary
    cmap_base = matplotlib.cm.get_cmap('BrBG')
    cmap = [cmap_base(x) for x in [0., 0.2, 0.28, 0.38, 0.8, 1.]]    
    cnorm = oplot.BoundaryNorm(range(7), ncolors=6, clip=True)
    cdict = {'0.5': cmap[0], '1.5': cmap[1], '2.5': cmap[2],
             '3.5': cmap[3], '4.5': cmap[4], '5.5': cmap[5]}
    
    # create figure
    fig = oplot.plt.figure(figsize=(wmm / 25.4, hmm / 25.4))

    # create axes for continents
    axes = fig.subplots(2, 3, sharey=True)

    # to guess xlim later
    xp1, xp2, xp3, xp4, xp5, xp6 = [], [], [], [], [], []

    for i in range(nrows):
        row = table.loc[i]
        lat = row.lat
        lon = row.lon

        # get marker based on region
        key = '(%8.5f, %g)' % (lat, lon)
        reg = COORD_REGION2[key]
        mark = REG_MARK2[reg]

        x = [row.ctl_mcwd, row.exp_mcwd]
        y = [row.ctl_map, row.exp_map]
        c = cdict[str(row.comp)]

        if reg == 'R1':
            axes[0, 0].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[0, 0].plot(x[1], y[1], mark, color=c, ms=2)
            xp1.extend(x)
        elif reg == 'R2':
            axes[0, 1].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[0, 1].plot(x[1], y[1], mark, color=c, ms=2)
            xp2.extend(x)
        elif reg == 'R3':
            axes[0, 2].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[0, 2].plot(x[1], y[1], mark, color=c, ms=2)
            xp3.extend(x)
        elif reg == 'R4':
            axes[1, 0].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[1, 0].plot(x[1], y[1], mark, color=c, ms=2)
            xp4.extend(x)
        elif reg == 'R5':
            axes[1, 1].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[1, 1].plot(x[1], y[1], mark, color=c, ms=2)
            xp5.extend(x)
        elif reg == 'R6':
            axes[1, 2].plot(x, y, color=c, linewidth=1, alpha=0.5)
            axes[1, 2].plot(x[1], y[1], mark, color=c, ms=2)
            xp6.extend(x)
        else:
            pass
            
    # plot settings
    for ax in axes[1, :]:
        ax.set_xlabel('MCWD (mm)')
        
    for ax in axes[:, 0]:
        ax.set_ylabel(oplot.replace_minus('MAP (mm year$^{-1}$)'))

    # ylim and change hyphen
    for ax in axes.flatten():
        ax.set_ylim([0, 3500])
        ax.xaxis.set_major_formatter(oplot.FuncFormatter(oplot.no_hyphen))

    # fix xlim
    for (x, ax) in zip([xp1, xp2, xp3, xp4, xp5, xp6], axes.flatten()):
        positive_minx = abs(min(x))
        positive_minx -= positive_minx % -100
        xlim = [-positive_minx, 0]
        ax.set_xlim(xlim)

    # titles
    axes[0, 0].set_title('(a) North America')
    axes[0, 1].set_title('(b) Central America/Northern South America')
    axes[0, 2].set_title('(c) Central South America')
    axes[1, 0].set_title('(d) Eurasia')
    axes[1, 1].set_title('(e) South East Africa')
    axes[1, 2].set_title('(f) Australia')

    # legend
    custom_names = ['Distance to ' + names[0], names[1]]
    custom_lines = [Line2D([0], [0], color='black', lw=1,
                           alpha=0.5),
                    Line2D([0], [0], linestyle='',
                           marker='^', color='black', ms=2)]
    axes[0, 0].legend(custom_lines, custom_names, loc=2, fontsize=6)
    axes[0, 1].legend(custom_lines, custom_names, loc=4, fontsize=6)
    axes[0, 2].legend(custom_lines, custom_names, loc=4, fontsize=6)
    axes[1, 0].legend(custom_lines, custom_names, loc=3, fontsize=6)
    axes[1, 1].legend(custom_lines, custom_names, loc=2, fontsize=6)
    axes[1, 2].legend(custom_lines, custom_names, loc=2, fontsize=6)

    # maximize
    oplot.plt.tight_layout()

    # save if given filename
    oplot.save_func(save, transparent)

    return fig

def plot_regions(composite, bnds, regname, xloc, yloc, wmm=30, hmm=30,
                 axes=None, proj=None, lon0=0, save=None,
                 transparent=False):
    
    # plot settings
    oplot.plot_settings()

    # copy data
    compcopy = composite.copy()

    # select region
    x0, x1, y0, y1 = bnds
    reg = compcopy.sel(latitude=slice(y0, y1),
                       longitude=slice(x0, x1))

    # get coords
    lon = reg.longitude.values
    lat = reg.latitude.values

    # shape
    nlat, mlon = reg.shape
    
    for i in range(nlat):
        ii = lat[i]
        for j in range(mlon):
            jj = lon[j]
            
            # get marker based on region
            key = '(%8.5f, %g)' % (ii, jj)

            try:
                region = COORD_REGION2[key]
            except:
                region = 'OTHER'

            if region != regname:
                reg.values[i, j] = np.nan

    # get projection if none given
    if proj is None:
        proj = oplot.pcar(central_longitude=lon0)

        # in case we plot a single plot
    if axes is None:
        oplot.plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = oplot.plt.axes(projection=proj)
        maximize = 1
    else:
        maximize = 0

    # colormap
    cmap_base = matplotlib.cm.get_cmap('BrBG')
    cmap = oplot.ListedColormap([cmap_base(x) for x
                                 in [0., 0.2, 0.28, 0.38, 0.8, 1.]])

    # levels
    lev = range(7)

    # normalize to number of colors
    cnorm = oplot.BoundaryNorm(lev, ncolors=cmap.N, clip=True)

    # add shorelines
    axes.coastlines(resolution='110m')

    # set global
    axes.set_extent(bnds)

    # add gridlines
    oplot.pcar_gridliner(axes, bnds, xloc, yloc)

    # fix coords
    corlon, corlat = oplot.corner_coords(lon, lat)

    # plot map
    fmap = axes.pcolor(corlon, corlat, reg.values, cmap=cmap,
                       norm=cnorm, transform=oplot.pcar())

    # maximize if only one
    if maximize == 1:
        oplot.plt.tight_layout()

        # savefig if provided name
        oplot.save_func(save, transparent)

    return axes
