import re
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.animation as animation

from cartopy.crs import Mollweide as moll
from cartopy.crs import PlateCarree as pcar
from cartopy.crs import Orthographic as ort

from . import cal
from . import utils

# avoid warning about registering pandas converters
register_matplotlib_converters()


def _get_pshape(nplots):
    """Get subplots shape based on number of plots.

    This is a function to be used by `panel_global_contour` to get the
    number of rows and columns for subplots.

    Parameters
    ----------
    nplots: int
        Total number of data arrays to be plotted.

    Returns
    -------
    Tuple of integer numbers:

        (number_rows, number_columns)
    """  # noqa

    # shape of plots
    if nplots <= 3:
        nrows, ncols = (1, nplots)
    elif nplots == 4:
        nrows, ncols = (2, 2)
    elif nplots == 5:
        nrows, ncols = (2, 3)
    elif nplots == 6:
        nrows, ncols = (2, 3)
    elif nplots == 7 or nplots == 8:
        nrows, ncols = (4, 2)
    elif nplots == 9:
        nrows, ncols = (3, 3)
    elif nplots == 12:
        nrows, ncols = (4, 3)
    else:
        msg = ('too many plots: no default grid structure to ' +
               'handle subplots')
        raise IndexError(msg)
    return (nrows, ncols)


def _no_hyphen(x, pos):
    """Replace long hyphen LaTeX uses by smaller minus sign."""  # noqa
    rv = format(x)

    if mpl.rcParams['text.usetex']:
        rv = re.sub('$-$', r'\mhyphen', rv)

    return rv


def plot_settings():
    """Helper to set up things we like."""

    # plot settings
    mpl.rc('savefig', dpi=300)
    mpl.rc('figure', autolayout=False)
    mpl.rc('font', family='sans-serif')
    mpl.rc('font', size=7)
    mpl.rc('axes', titlesize=8)
    mpl.rc('axes', labelsize=7)
    mpl.rc('legend', fontsize=7)
    mpl.rc('axes', unicode_minus=False)
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', unicode=True)
    mpl.rc('text.latex', preamble=[r'\usepackage{helvet}',
                                   r'\usepackage{sansmath}',
                                   r'\usepackage{subdepth}',
                                   r'\usepackage{type1cm}',
                                   r'\usepackage{gensymb}',
                                   r'\sansmath'])


def add_gridlines(axes):
    """Helper to add same gridlines to maps."""
    gl = axes.gridlines(crs=pcar(), linewidth=1, linestyle='--',
                        alpha=0.5, color='black')

    # specify gridlines locations
    gl.xlocator = FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = FixedLocator([-90, -60, -30, 0, 30, 60, 90])


def create_save_dir(save_dir):
    """Helper to create save directory for figures."""

    # get current working directory
    cwd = os.getcwd()

    # if save path specified
    if save_dir is not None:
        save_dir = str(save_dir)
        save_path = os.path.join(cwd, save_dir)
        if os.path.isdir(save_path):
            msg = ('save directory already exists: pictures ' +
                   'may be overwritten')
            warnings.warn(msg)
        else:
            try:
                os.mkdir(save_path)
            except OSError:
                msg = ('could not create directory to save' +
                       ' pictures: check name')
                raise OSError(msg)
        return save_path


def corner_coords(lon, lat):
    """Obtain corner coordinates from midpoint coordinates.

    `pcolormesh` from matplotlib expects corner coordinates. However,
    if we have netCDF data, we most likely have midpoint coordinates. 
    To obtain the right coordinates for matplotlib's function, we
    have to interpolate. Thanks to: Baird Langenbrunner.

    Parameters
    ----------
    lon: numpy.ndarray
        Longitude values array.
    lat: numpy.ndarray
        Latitude values array.

    Returns
    -------
    Tuple object with two numpy.ndarrays:

        (corner_lon_array, corner_lat_array)
    """  # noqa

    # create e(x)tended arrays
    xlon = np.zeros(lon.size + 2)
    xlat = np.zeros(lat.size + 2)

    # fill internal
    xlon[1:-1] = lon
    xlat[1:-1] = lat

    # get neighbour differences
    londiff = np.diff(lon)
    latdiff = np.diff(lat)

    # fill in extra endpoints
    xlon[0] = lon[0] - londiff[0]
    xlon[-1] = lon[-1] + londiff[-1]
    xlat[0] = lat[0] - latdiff[0]
    xlat[-1] = lat[-1] + latdiff[-1]

    # get new neighbour differences
    xlondiff = np.diff(xlon)
    xlatdiff = np.diff(xlat)

    # calculate the corners
    corner_lon = xlon[:-1] + 0.5 * xlondiff
    corner_lat = xlat[:-1] + 0.5 * xlatdiff

    return (corner_lon, corner_lat)


def save_func(save, transparent):
    """Function to control saving of plots.

    Parameters
    ----------
    save: bool or str
        This can be a boolean flag to create a PDF file with the
        plotted map, in which case the file will be named
        `output.pdf`, or a string with a specific name for the file.
    transparent: bool
        If `save` is True or some str object with a name, this keyword
        controls how the background is plotted. If True, then
        background will be transparent. This is useful if the image is
        to be used in slideshows.
    """  # noqa

    # savefig if provided name
    if save is not None:
        if save is True:
            plt.savefig('output.pdf')
            print('Created file: \'output.pdf\'')
        elif save is False:
            plt.show()
        else:
            plt.savefig(save, transparent=transparent)
            print('Created file: \'' + save + '\'')
            plt.close()
    else:
        plt.show()


def create_clev(data, minv=None, maxv=None, nlevels=None):
    """Create contour levels.

    Helper function to save some coding. Simply returns a range with
    contour levels whether minv, maxv, or nlevels have been specified or
    not. If they are not specified then we use the values in the data
    array to get minimum and maximum values and aim for 10 contour
    levels. 

    Parameters
    ----------
    data: xarray.DataArray 
        Data that the user wants to plot. From this array maximum and
        minimum values will be obtained if `minv` and `maxv` keywords
        are not used.
    minv: float, optional
        Minimum value for contour levels. If None, minimum value in
        `data` will be used. 
    maxv: float, optional
        Maximum value for contour levels. If None, maximum value in
        `data` will be used. 
    nlevevls: int, optional
        Number of contour levels desired. If None, 10 will be used.

    Returns
    -------
    numpy.ndarray with contour levels.
    """  # noqa

    # get min if not given
    if minv is None:
        minv = data.min().values
    else:
        minv = minv

    # get max if not given
    if maxv is None:
        maxv = data.max().values
    else:
        maxv = maxv

    # get nlevels if not given
    if nlevels is None:
        nlevels = 10
    else:
        nlevels = nlevels

    # create contour levels
    clevels = np.round(np.linspace(minv, maxv, nlevels), 2)

    return clevels


def get_cyclic_values(data):
    """Add another column to data array with same values as first but
    with an additional longitude coordinate, i.e. make it cyclic.
    This returns a numpy.ndarray array of values, not another xarray.

    Parameters
    ----------
    data: xarray.DataArray
        Input must have named `longitude` and `latitude` coordinates.

    Returns
    -------
    Tuple object with the following contents:

        (new_values, new_longitude_values)

    Both of these objects ar numpy.ndarray. If input had shape lat. x
    lon. 32 x 64, then `new_values` has shape 32 x 65, and
    `new_longitude_values` has a size of 65.

    Note
    ----
    Longitudes will be cyclic only if data are cyclic. Otherwise the
    last coordinate value will not be the same as first.
    """  # noqa

    # get values and longitudes
    val = data.values
    lon = data.longitude.values

    # get shape and add 1 more in right most coord
    shp = list(data.shape)
    shp[-1] += 1

    # create new empty arrays with extra column
    newval = np.zeros(shp)
    newlon = np.zeros(shp[-1])

    # copy all values except last column
    newval[..., :-1] = val
    newlon[:-1] = lon

    # repeat same values as first
    newval[..., -1] = val[..., 0]

    # get horizontal distance
    xhres = np.mean(np.diff(lon))

    # new lon is add distance to last item
    newlon[-1] = lon[-1] + xhres

    # if lon[-1] > 180.0:
    #     newlon[-1] = 360.0
    # else:
    #     newlon[-1] = 180.0

    return (newval, newlon)


def plot_global_contour(data, method='filled', cm='jet', axes=None,
                        proj=moll(), lon0=0, extend='neither',
                        levels=None, minv=None, maxv=None,
                        nlevels=None, cbstring=None, title='',
                        cticks=None, name=None):
    """Plot a filled countour global map of a single dataset.

    This function will simply use matplotlib and cartopy
    tools to plot a global map of filled countour values with the
    specified range of levels. It can return a new axis or attach to a
    given one. Resolution of gridlines is 110m and gridlines have only
    five longitudes every 90deg and seven latitudes every 30deg.

    Parameters
    ----------
    data: xarray.DataArray
        Input must be 2D. It must have named coordinates `latitude`
        and `longitude`.
    method: str, optional
        It can be either 'filled' to use matplotlib's `contourf`
        function, or 'mesh' which uses matplotlib's `pcolormesh`.
        Default is to plot filled contours.
    cm: str, optional
        Colormap to be used. By default is Jet colormap.
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
    extend: str, optional
        Whether to have pointing arrows at the ends of the
        colorbar. It can be 'neither', to not use arrows but have 
        blunt ends, 'max' to only have an arrow at the maximum end of
        the colorbar, 'min', and 'both'.
    levels: list or numpy.ndarray, optional
        Contour levels can be specified with this keyword. Otherwise
        you can let this function create the contour levels using
        keywords `minv`, `maxv`, and `nlevels`.
    minv: float, optional
        Minimum value for contour levels in colorbar. If not set, the
        minimum value found in the data array will be used.
    maxv: float, optional
        Maximum value for contour levels in colorbar. If not set, the
        maximum value found in the data array will be used.
    nlevels: int, optional
        Number of levels for contour coloring. If not used, default is
        10 contour levels.
    cbstring: str, optional
        Title for the colorbar. If none provided, it will try to use
        `units` attribute in data array if is defined.
    title: str, optional
        Center top title if desired. Default is empty.
    cticks: list or numpy.ndarray, optional
        These ticks are for the colorbar, in case you want particular
        ones. By default the function will try to use the "best ones",
        choosing values every 2 contour levels.
    name: str, optional
        Whether to place a small rectangular label in the bottom right
        corner of the map plot with a name for it.

    Returns
    -------
    Tuple object with the following content:

        (cartopy.mpl.geoaxes.GeoAxes, matplotlib.pyplot.colorbar)

    This function returns the colorbar handle in case there are
    necessary modifications for it. Like in functions `plot_landsea`
    and `plot_glacier`.
    """  # noqa

    # plot settings
    plot_settings()

    # check it is 2D
    if len(data.dims) > 2:
        msg = 'we can only handle 2D plots'
        raise TypeError(msg)

    # get cyclic values and coords
    cval, clon = get_cyclic_values(data)
    lat = data.latitude.values

    # get contour levels
    if levels is None:
        levels = create_clev(data, minv, maxv, nlevels)

    # choose projection if given lon0
    if lon0 != 0.0 and proj == moll():
        proj = moll(central_longitude=lon0)
    elif lon0 != 0.0:
        msg = 'lon0 argument only works if no projection provided'
        warnings.warn(msg)

    # in case we plot a single plot
    if axes is None:
        axes = plt.axes(projection=proj)

    # guess ticks for colorbar
    if cticks is None:
        cticks = levels[1:-1:2]

    if method == 'filled':
        # plot filled countour with specs
        fmap = axes.contourf(clon, lat, cval, levels=levels, cmap=cm,
                             transform=pcar(), extend=extend)
        cb = plt.colorbar(fmap, orientation='horizontal', pad=0.05,
                          format=FuncFormatter(_no_hyphen),
                          shrink=0.75, ax=axes, ticks=cticks)
    elif method == 'mesh':

        # fix coords
        corlon, corlat = corner_coords(clon, lat)

        # plot grid cells with specs
        cmap = get_cmap(cm, len(levels))
        cnorm = BoundaryNorm(levels, cmap.N)
        fmap = axes.pcolormesh(corlon, corlat, cval, cmap=cmap,
                               norm=cnorm, transform=pcar())
        cb = plt.colorbar(fmap, orientation='horizontal', pad=0.05,
                          format=FuncFormatter(_no_hyphen),
                          shrink=0.75, ax=axes, extend=extend,
                          ticks=cticks)
    else:
        msg = 'method can only be \'filled\' or \'mesh\''
        raise ValueError(msg)

    # add shorelines
    axes.coastlines(resolution='110m')

    # set global
    axes.set_global()

    # add gridlines
    add_gridlines(axes)

    # add colorbar title
    if cbstring is None:
        try:
            cbstring = data.units
        except AttributeError:
            pass
    cb.set_label(cbstring)

    # add plot title
    axes.set_title(title)

    if name is not None:
        # add text with name
        props = dict(boxstyle='square', facecolor='White',
                     edgecolor='Black', linewidth=0.75)

        # place a text box in upper left in axes coords
        axes.text(0.90, 0.06, name, transform=axes.transAxes,
                  fontsize=6, verticalalignment='center',
                  bbox=props)

    return (axes, cb)


def plot_landsea(land_mask, method='mesh', wmm=180, hmm=90, axes=None,
                 title='', proj=moll(), save=None, transparent=False):
    """Function specific to plot land binary mask.

    Since this is a special kind of plot in which there is only a
    solid color for land, it is handled different to other plots.

    Parameters
    ----------
    land_mask: xarra.DataArray
        This array contains binary values: 0 = ocean and 1 = land.
    method: str, optional
        It can be either 'filled' to use matplotlib's `contourf`
        function, or 'mesh' which uses matplotlib's `pcolormesh`.
        Default is to plot filled contours.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
    axes: cartopy.mpl.geoaxes.GeoAxes, optional
        This axes should have some projection from Cartopy. If it is
        not provided, brand new axes will be created with default
        projection `proj`.
    title: str, optional
        Center top title if desired. Default is empty.
    proj: cartopy.crs.Projection, optional
        Map projection to be used to create the axes if not
        provided. By default we use the Mollweide projection.
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
    cartopy.mpl.geoaxes.GeoAxes with the plotted map.
    """  # noqa

    # in case we plot a single plot
    if axes is None:
        plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = plt.axes(projection=proj)
        maximize = 1
    else:
        maximize = 0

    # specifications
    sp = dict(method=method, cm=ListedColormap(['ForestGreen']),
              cticks=[0.5], levels=[0, 1], cbstring='',
              title=title, axes=axes)

    axes, cb = plot_global_contour(land_mask, **sp)
    cb.ax.set_xticklabels(['Land'])

    # maximize plot if only one
    if maximize == 1:
        plt.tight_layout()

        # savefig if provided name
        save_func(save, transparent)

    return axes


def plot_glacier(glacier, method='mesh', wmm=180, hmm=90, save=None,
                 title='', proj=moll(), transparent=False, axes=None):
    """Function specific to plot land glacier mask.

    Since this is a special kind of plot in which there is only two
    solid colors: for glacier and for land, it is handled different to
    other plots. 

    Parameters
    ----------
    glacier: xarra.DataArray
        This array contains binary values: 0 = non-glacier and 1 =
        glacier.
    method: str, optional
        It can be either 'filled' to use matplotlib's `contourf`
        function, or 'mesh' which uses matplotlib's `pcolormesh`.
        Default is to plot filled contours.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
    save: bool or str, optional
        This can be a boolean flag to create a PDF file with the
        plotted map, in which case the file will be named
        `output.pdf`, or a string with a specific name for the file.
        Default is only show the plot.
    title: str, optional
        Center top title if desired. Default is empty.
    proj: cartopy.crs.Projection, optional
        Map projection to be used to create the axes if not
        provided. By default we use the Mollweide projection.
    transparent: bool, optional
        If `save` is True or some str object with a name, this keyword
        controls how the background is plotted. If True, then
        background will be transparent. This is useful if the image is
        to be used in slideshows. Default is False.
    axes: cartopy.mpl.geoaxes.GeoAxes, optional
        This axes should have some projection from Cartopy. If it is
        not provided, brand new axes will be created with default
        projection `proj`.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes with the plotted map.
    """  # noqa

    # in case we plot a single plot
    if axes is None:
        plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = plt.axes(projection=proj)
        maximize = 1
    else:
        maximize = 0

    # specifications
    sp = dict(method='mesh', levels=[0, 0.5, 1], cticks=[0.25, 0.75],
              cm=ListedColormap(['Aqua', 'Magenta']), cbstring='',
              title=title, axes=axes)

    axes, cb = plot_global_contour(glacier, **sp)
    cb.ax.set_xticklabels(['Non-glacier', 'Glacier'])

    # maximize plot if only one
    if maximize == 1:
        plt.tight_layout()

        # savefig if provided name
        save_func(save, transparent)

    return axes


def bbox_linecoords(bounds, xhres, yhres):
    """Returns coordinates to draw squared line.

    To highlight some region in a global map, this function uses given
    bounds, and x-direction and y-direction grid resolutions to obtain
    coordinates to plot the square that the bounds create.

    Parameters
    ----------
    bounds: tuple or list
        Bounds must have the sequence: [x0, x1, y0, y1], using x for
        longitudes and y for latitudes.
    xhres: float
        Horizontal grid resolution in the x direction.
    yhres: float
        Horizontal grid resolution in the y direction.

    Returns
    -------
    Tuple with two lists of coordinates that together draw a square:

        (xcoords_list, ycoords_list)

    Using `matplotlib.pyplot.plot(xcoords, ycoords) ` one can plot
    this.
    """  # noqa

    # check bounds
    utils.check_bounds(bounds)

    # get bounds
    x0, x1, y0, y1 = bounds

    # get xcoords (at center gridpoint)
    xcoord = [x0, x1, x1, x0, x0]

    if y1 >= y0:
        ycoord = [y1, y1, y0, y0, y1]
    else:
        ycoord = [y0, y0, y1, y1, y0]

    # distance to center point (regular grid)
    dx = xhres / 2.0
    dy = yhres / 2.0

    # estimate center of things
    clon = np.mean([x0, x1])
    clat = np.mean([y0, y1])

    # expand center to cell bounds
    totcoord = len(xcoord)
    for i in range(totcoord):
        if xcoord[i] > clon:
            xcoord[i] += dx
        else:
            xcoord[i] -= dx

        if ycoord[i] > clat:
            ycoord[i] += dy
        else:
            ycoord[i] -= dy

    return (xcoord, ycoord)


def panel_global_contour(dlist, slist, wmm=180, hmm=90,
                         save=None, transparent=False):
    """Panels version of plot_global_contour.

    This function takes in a list of data arrays and a list of
    specifications for each one of them and plots them using different
    subplots, which are distributed depending on the number of total
    plots. 

    Parameters
    ----------
    dlist: list of xarray.DataArray
        Right now the user can provide up to 12 data arrays in a list
        to be plotted together in panels.
    slist: list of dict objects of specifications
        Each array in `dlist` must have a dictionary of options or
        specifications in this list. See `plot_global_contour` for
        possible specifications.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
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
    matplotlib.figure.Figure object with panel plots.
    """  # noqa

    # plot settings
    plot_settings()

    # check number of dataset equals specifications
    ndat = len(dlist)
    nspc = len(slist)

    if ndat != nspc:
        msg = 'more/less specifications than datasets'
        raise ValueError(msg)

    # create plot
    fig = plt.figure(figsize=(wmm / 25.4, hmm / 25.4))

    # get better shape for these many datasets
    nrows, ncols = _get_pshape(ndat)

    # actually plot
    for p in range(ndat):

        # get data and specifications
        data = dlist[p]
        spec = slist[p]

        # remove any axes specification just in case
        try:
            del spec['axes']
        except KeyError:
            pass

        # see if projection was provided
        try:
            proj = spec['proj']
        except KeyError:
            # see if lon0 was provided at least
            try:
                lon0 = spec['lon0']
                proj = moll(central_longitude=lon0)
            except KeyError:
                proj = moll()

        # create axes
        ax = plt.subplot(nrows, ncols, p + 1, projection=proj)

        # plot
        plot_global_contour(data, axes=ax, **spec)

    # maximize output
    plt.tight_layout()

    # savefig if provided name
    save_func(save, transparent)

    return fig


def plot_ancy(data, spec, month, wmm=180, hmm=90, save=None,
              names='3L'):
    """Plot 12 maps, one for each month in a single-year file.

    This is intended to plot all months in a climatology data array
    that has 12 time steps. The resulting figure will have 4 rows and
    3 columns.

    Parameters
    ----------
    data: xarray.DataArray
        This array must have 12 time steps. As well as named
        coordinates `time`, `latitude` and `longitude`.
    spec: dict
        Dictionary with options to be passed to `plot_global_contour`
        function. 
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
    save: bool or str, optional
        This can be a boolean flag to create a PDF file with the
        plotted map, in which case the file will be named
        `output.pdf`, or a string with a specific name for the file.
        Default is only show the plot.
    names: str, optional
        Format of month names in every plot. Available options are:
        '3L' which is the default and shows January as Jan, '1L' would be
        J, and 'F' would be the full name.

    Returns
    -------
    matplotlib.figure.Figure object with panel plots.
    """  # noqa

    # check it is an annual cycle
    ntim = data.time.size

    if ntim != 12:
        msg = 'input has to be a 12-month file'
        raise ValueError(msg)

    # check if list or single
    if isinstance(month, list) or isinstance(month, range):

        # create lists of data and specs
        dlist = []
        slist = []

        for m in month:
            # add simply data in that time
            dlist.append(data[m - 1])

            # get month name
            mon = cal.get_month_name(m, names)

            # add name to specs
            newspec = dict(spec)
            newspec['name'] = mon

            # append to slist
            slist.append(newspec)

        # plot panel plot
        fig = panel_global_contour(dlist, slist, wmm, hmm, save)
    else:
        # get month data
        datmon = data[month - 1]

        # get month name
        mon = cal.get_month_name(month, names)

        # add to specs
        spec['name'] = mon

        # plot
        fig = panel_global_contour([datmon], [spec], wmm, hmm, save)

    return fig


def ortho_rotation(data, spec, wpx=1920, hpx=1080, lon0=0, dg=1,
                   save_dir=None, dpi=300):
    """Plot data in an Orthographic projection and rotating.

    This function is a different way to look at data. It could be used
    to make videos in which the planet is spinning while displaying
    data.

    Parameters
    ----------
    data: xarray.DataArray
        Input array can have as many time steps as wanted. It must
        have named `time` coordinate.
    spec: dict
        Dictionary with options to be passed to `plot_global_contour`
        function. All same parameters for that function are allowed,
        except the `axes` keyword, since we will set up an
        Orthographic projecion here.
    wpx: int, optional
        Width in pixels for the images. Default is 1920 px.
    hpx: int, optional
        Height in pixels for the images. Default is 1080 px.
    lon0: float, optional
        Initial longitude at which to start rotating every time step.
        Default is Greenwich meridian.
    dg: float, optional
        Degrees step to advance rotation. The maximum possible value is
        dg = 360 which means no rotation at all. The slowest possible is
        dg = 1. Default is 1.
    save_dir: str, optional
        If the user wants to save all plotted frames in a folder, they
        can set this keyword to a folder name and figures will be
        stored there. Otherwise figures will not be saved. Default is
        not save plots.
    dpi: int, optional
        Dots per inch for every frame. Default is 300.
    """  # noqa

    # get number of times
    ntim = data.time.size

    # create figure instance
    fig = plt.figure(figsize=(wpx / dpi, hpx / dpi))

    # rotation settings
    nmpr = int(360 / dg)

    # guess number of maps
    totm = nmpr * ntim

    # counter for names
    c = 1

    # create save directory
    save_path = create_save_dir(save_dir)

    # rotate for every time in file
    for t in range(ntim):

        # get data this time if multiple
        if ntim > 1:
            da = data[t]
        else:
            da = data

        # lets rotate
        for r in range(nmpr):
            # create projection
            proj = ort(central_longitude=lon0)
            spec['proj'] = proj

            # add date to title
            date = cal.get_dates(da.time.values)
            spec['title'] = date.strftime('%Y-%b')

            # plot
            plot_global_contour(da, **spec)

            # savefig if provided name
            if save_dir is not None:
                img = os.path.join(save_path, "rotate_%08d.png" % c)
                plt.savefig(img, dpi=300)
                plt.close(fig)
                sys.stdout.write('Plotting progress: %d%%  \r' %
                                 (100 * c/totm))
                sys.stdout.flush()
                # update counter
                c += 1
            else:
                plt.pause(0.05)

            # update lon0
            if lon0 > 0.0:
                lon0 = lon0 - dg
            else:
                lon0 = 360.0


def oni_lineplot(data, names=None, colors=None, styles=None,
                 wmm=175, hmm=70, cm='jet', dpi=300,
                 xlabel='Year', ylabel=r'($^{\circ}$C)',
                 title=r'Oceanic Niño Index', ylim=[-2.5, 2.5],
                 axes=None, save=None, transparent=False):
    """Plot lines from ONI data frames.

    xy-plot with lines where the x axis is the year and the y axis is
    the values of the Oceanic Nino Index (ONI). This plot adds
    horizontal lines to divide the plot at -0.5 (blue), 0 (gray) and
    0.5 (red).

    Parameters
    ----------
    data: pandas.DataFrame or list of pandas.DataFrame
        This should be ONI data. It can be a single data frame or
        several in a list. This function uses the output from function
        `analysis.get_oni`.
    names: list of str, optional
        If several data frames provided in `data`, then this should be
        a list of strings with the names of each of them in the same
        order. Default is 'EXP#'.
    colors: list of named colors, optional
        If several data frames provided in `data`, then this should be
        a list of named colors for each of them in the same
        order. These colors will be used for the lines' style. 
    styles: list of matplotlib style directives, optional
        If several data frames provided in `data`, then this should be
        a list of matplotlib style directives for the style of each
        line. Default is '-' for all lines.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 175 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 70 mm.
    cm: str, optional
        Colormap to be used. By default is Jet colormap.
    dpi: int, optional
        Dots per inch for every frame. Default is 300.
    xlabel: str, optional
        Title for the x axis. Default is 'Year'.
    ylabel: str, optional
        Title for the y axis. Default is '(oC)'.
    title: str, optional
        Center top title if desired. Default is empty.
    ylim: list, optional
        List object with two float values to define the limits in the
        y axis. Default is [-2.5, 2.5].
    axes: matplotlib.axes.Axes, optional
        If this is going to be part of another bigger figure, a
        subplot axes can be provided for this plot to attach the ONI
        plot. Default is create its own axes.
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
    plot_settings()

    if isinstance(data, list):
        ndata = len(data)
        iyear = data[0].index[0]
        eyear = data[0].index[-1]
    else:
        ndata = 1
        iyear = data.index[0]
        eyear = data.index[-1]

    # create dates range for x-axis
    idate = '%04i-01-01' % iyear
    edate = '%04i-12-31' % eyear
    dates = pd.date_range(idate, edate, freq='M')

    # arbitrary names if none given
    if names is None:
        names = ['EXP%d' % x for x in range(ndata)]

    # colors from cmap if not given
    if colors is None:
        cmap = plt.get_cmap(cm)
        index = np.linspace(0, 1., ndata)
        colors = cmap(index)

    # same style if not given
    if styles is None:
        styles = ['-'] * ndata

    # in case we plot a single plot
    if axes is None:
        plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = plt.axes()
        maximize = 1
    else:
        maximize = 0

    # plot line(s)
    if ndata > 1:
        for i in range(ndata):
            y = data[i].values.flatten()
            plt.plot(dates, y, styles[0], color=colors[i])
    else:
        y = data.values.flatten()
        plt.plot(dates, y, styles[0], color=colors[0])

    # add horizontal lines
    plt.axhline(y=0, linestyle='-', color='gray')
    plt.axhline(y=0.5, linestyle='--', color='r')
    plt.axhline(y=-0.5, linestyle='--', color='b')

    # add titling
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(names)

    # set ylims
    plt.ylim(ylim)

    # maximize plot if only one
    if maximize == 1:
        plt.tight_layout()

        # savefig if provided name
        save_func(save, transparent)

    return axes


def plot_zonal_mean(data, style=None, axes=None, xticks=None,
                    xticklabels=None, xlim=[-90, 90], ylim=None,
                    ylabel='', xlabel='', xminor=None, title='',
                    wmm=80, hmm=80, save=None, transparent=False):
    """Plot zonal means.

    A zonal mean is a mean performed along the longitude axis, so one
    is left with a single value for each latitude. This function is
    intended for a time mean of zonal means, so there is only a single
    dimension which is latitude. In this type of plot the x axis has
    latitudes. So this is simply a line plot but with some specific
    formatting. 

    Parameters
    ----------
    data: xarray.DataArray or list of xarray.DataArray
        Input data can be a list of several xarray arrays. They all
        must have only 1 dimension: latitude, which must be a named
        coordinate as well.
    style: dict or list of dict, optional
        Here the user can input a style dictionary using
        matplotlib.pyplot.plot keywords. If input data is a list, then
        this also must be a list and their size must match. Default is
        None.
    axes: matplotlib.axes.Axes, optional
        If this is going to be part of another bigger figure, a
        subplot axes can be provided for this plot to attach the ONI
        plot. Default is create its own axes.
    xticks: list or numpy.ndarray, optional
        In case the user wants different latitudes in the x
        axis. Default is [-80, -40, 0, 40, 80].
    xticklabels: list, optional
        List of strings to be used if the user has set different
        `xticks`. It really only makes sense to use this setting along
        with xticks, but it is not coded this way in case the user
        wants to change the default xticks labels.
    xlim: list, optional
        To specify the limits in the x axis. Default is [-90, 90].
    ylim: list, optional
        To specify the limits in the y axis. Default is set it to go
        from the 90% of minimum value to 110% of the maximum value so
        there is some nice 'padding'.
    xlabel: str, optional
        Title for the x axis. Default is empty.
    ylabel: str, optional
        Title for the y axis. Default is empty.
    xminor: list or numpy.ndarray, optional
        Position of minor x tick marks. Default is [-60, -20, 20, 60].
    title: str, optional
        Center top title if desired. Default is empty.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 80 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 80 mm.
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
    plot_settings()

    if isinstance(data, list):
        ndata = len(data)

        if style is not None:
            nstyle = len(style)

            if nstyle != ndata:
                msg = 'number of data items must match styles given'
                raise ValueError(msg)
    else:
        data = [data]
        ndata = 1

        if style is not None:
            style = [style]

    # in case we plot a single plot
    if axes is None:
        plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = plt.axes()
        maximize = 1
    else:
        maximize = 0

    # plot each line
    for i in range(ndata):

        # unpack values
        y = np.array(data[i].values)
        x = np.array(data[i].latitude.values)

        # line plot
        if style is not None:
            axes.plot(x, y, **style[i])
        else:
            axes.plot(x, y)

    # xtick marks
    if xticks is None:
        xticks = [-80, -40, 0, 40, 80]

    # xtick labelsize
    if xticklabels is None:
        xticklabels = [r'80$^{\circ}$S', r'40$^{\circ}$S',
                       r'0$^{\circ}$', r'40$^{\circ}$N',
                       r'80$^{\circ}$N']

    # xminor marks
    if xminor is None:
        xminor = [-60, -20, 20, 60]

    # get extreme values
    if ylim is None:
        minv = np.nanmin(data)
        maxv = np.nanmax(data)
        ylim = [0.9 * minv, 1.1 * maxv]

    # settings
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.xaxis.set_minor_locator(FixedLocator(xminor))

    if ndata is not None:
        axes.legend()

    # maximize plot if only one
    if maximize == 1:
        plt.tight_layout()

        # savefig if provided name
        save_func(save, transparent)

    return axes


def panel_zonal_mean(dlist, slist=None, wmm=180, hmm=90, save=None,
                     transparent=False):
    """Panels version of `plot_zonal_mean`.

    Parameters
    ----------
    dlist: list of xarray.DataArray
        All data arrays must be 1 dimensional with a named coordinate
        `latitude`. 
    slist: list of dict objects of specifications, optional
        Each array in `dlist` must have a dictionary of options or
        specifications in this list. See `plot_zonal_mean` for
        possible specifications.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
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
    matplotlib.figure.Figure object with panel plots.
    """  # noqa

    # plot settings
    plot_settings()

    # check number of dataset equals specifications
    ndat = len(dlist)

    if slist is not None:
        nspc = len(slist)

        if ndat != nspc:
            msg = 'more/less specifications than datasets'
            raise ValueError(msg)

    # create plot
    fig = plt.figure(figsize=(wmm / 25.4, hmm / 25.4))

    # get better shape for these many datasets
    nrows, ncols = _get_pshape(ndat)

    # get minimum and maximum values
    minv = np.nanmin(dlist)
    maxv = np.nanmax(dlist)

    # actually plot
    for p in range(ndat):

        # get data
        data = dlist[p]

        # create axes
        ax = plt.subplot(nrows, ncols, p + 1)

        # plot
        if slist is not None:
            spec = slist[p]

            if 'ylim' in spec:
                ylim = spec['ylim']
                del spec['ylim']
            else:
                ylim = [0.9 * minv, 1.1 * maxv]

            # remove any axes specification just in case
            try:
                del spec['axes']
            except KeyError:
                pass

            plot_zonal_mean(data, axes=ax, ylim=ylim, **spec)
        else:
            plot_zonal_mean(data, axes=ax, ylim=ylim)

    # maximize output
    plt.tight_layout()

    # savefig if provided name
    save_func(save, transparent)

    return fig


def animate_global_contour(data,  method='filled',
                           wmm=80, hmm=65, proj=moll(),
                           lon0=0, extend='neither', cm='jet',
                           levels=None, minv=None, maxv=None,
                           nlevels=None, cbstring=None,
                           cticks=None, name=None):
    """Create an animation for a Jupyter Notebook.

    This function is intended to be used within Jupyter Notebooks to
    include an animation of the contents of an array in a
    cell. Outside the scope of a Notebook, i.e. in a script, this
    function can be used to create files such as MP4 videos or
    GIFs. It returns an animation object from matplotlib which has the
    method `.save()` to be able to create videos or GIFs. Arguments
    are very similar to those of `plot_global_contour`. It also has
    some sizing options in case the user wants to save images.

    Parameters
    ----------
    data: xarray.DataArray
        Input must be 2D. It must have named coordinates `latitude`
        and `longitude`.
    method: str, optional
        It can be either 'filled' to use matplotlib's `contourf`
        function, or 'mesh' which uses matplotlib's `pcolormesh`.
        Default is to plot filled contours.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 80 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 65 mm.
    proj: cartopy.crs.Projection, optional
        Map projection to be used to create the axes if not
        provided. By default we use the Mollweide projection.
    lon0: float, optional
        Central longitude to create the projection in the axes. By
        default the central longitudes is the Greenwich meridian. This
        argument is only meaningfull for the default projection which 
        is Mollweide. Otherwise it is unused.
    extend: str, optional
        Whether to have pointing arrows at the ends of the
        colorbar. It can be 'neither', to not use arrows but have 
        blunt ends, 'max' to only have an arrow at the maximum end of
        the colorbar, 'min', and 'both'.
    cm: str, optional
        Colormap to be used. By default is Jet colormap.
    levels: list or numpy.ndarray, optional
        Contour levels can be specified with this keyword. Otherwise
        you can let this function create the contour levels using
        keywords `minv`, `maxv`, and `nlevels`.
    minv: float, optional
        Minimum value for contour levels in colorbar. If not set, the
        minimum value found in the data array will be used.
    maxv: float, optional
        Maximum value for contour levels in colorbar. If not set, the
        maximum value found in the data array will be used.
    nlevels: int, optional
        Number of levels for contour coloring. If not used, default is
        10 contour levels.
    cbstring: str, optional
        Title for the colorbar. If none provided, it will try to use
        `units` attribute in data array if is defined.
    cticks: list or numpy.ndarray, optional
        These ticks are for the colorbar, in case you want particular
        ones. By default the function will try to use the "best ones",
        choosing values every 2 contour levels.
    name: str, optional
        Whether to place a small rectangular label in the bottom right
        corner of the map plot with a name for it.

    Returns
    -------
    A matplotlib.animation.FuncAnimation object that has the method
    `.save()` available.
    """  # noqa

    # plot settings
    plot_settings()
    mpl.rcParams["animation.html"] = "html5"

    # get contour levels
    if levels is None:
        levels = create_clev(data, minv, maxv, nlevels)

    # choose projection if given lon0
    if lon0 != 0.0 and proj == moll():
        proj = moll(central_longitude=lon0)
    elif lon0 != 0.0:
        msg = 'lon0 argument only works if no projection provided'
        warnings.warn(msg)

    # get dates
    dates = cal.get_dates(data.time.values)

    # get cyclical values and coords
    cval, clon = get_cyclic_values(data)
    lat = data.latitude.values

    # guess ticks for colorbar
    if cticks is None:
        cticks = levels[1:-1:2]

    # guess colorbar title if none
    if cbstring is None:
        try:
            cbstring = data.units
        except AttributeError:
            cbstring = ''

    def init():
        """Needed only to include colorbar in animation."""
        if method == 'filled':
            # plot filled countour with specs
            fmap = axes.contourf(clon, lat, cval[0], levels=levels,
                                 cmap=cm, transform=pcar(),
                                 extend=extend)
            cb = fig.colorbar(fmap, orientation='horizontal', pad=0.05,
                              format=FuncFormatter(_no_hyphen),
                              shrink=0.75, ticks=cticks)
        elif method == 'mesh':
            # fix coords
            corlon, corlat = corner_coords(clon, lat)

            # plot grid cells with specs
            cmap = get_cmap(cm, len(levels))
            cnorm = BoundaryNorm(levels, cmap.N)
            fmap = axes.pcolormesh(corlon, corlat, cval[0], cmap=cmap,
                                   norm=cnorm, transform=pcar())
            cb = fig.colorbar(fmap, orientation='horizontal', pad=0.05,
                              format=FuncFormatter(_no_hyphen),
                              shrink=0.75, extend=extend,
                              ticks=cticks)
        else:
            msg = 'method can only be \'filled\' or \'mesh\''
            raise ValueError(msg)

        # add plot title
        dstr = dates[0].strftime('%Y-%b')
        title = r'\texttt{' + dstr + r'}'
        axes.set_title(title)

        # add colorbar title
        cb.set_label(cbstring)

        # maximize
        fig.tight_layout()

        return (axes.cla(),)

    def animate(i):
        """Create iterable of artists for animation."""

        # clear axes
        axes.cla()

        # add shorelines
        axes.coastlines(resolution='110m')

        # set global
        axes.set_global()

        # add gridlines
        add_gridlines(axes)

        if method == 'filled':
            axes.contourf(clon, lat, cval[i], levels=levels, cmap=cm,
                          transform=pcar(), extend=extend)
        elif method == 'mesh':
            # fix coords
            corlon, corlat = corner_coords(clon, lat)

            # plot grid cells with specs
            cmap = get_cmap(cm, len(levels))
            cnorm = BoundaryNorm(levels, cmap.N)
            axes.pcolormesh(corlon, corlat, cval[i], cmap=cmap,
                            norm=cnorm, transform=pcar())
        else:
            msg = 'method can only be \'filled\' or \'mesh\''
            raise ValueError(msg)

        # add plot title
        dstr = dates[i].strftime('%Y-%b')
        title = r'\texttt{' + dstr + r'}'
        axes.set_title(title)

    # create figure and subplot
    fig = plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
    axes = fig.add_subplot(111, projection=proj)

    # animation object
    nfr = data.time.size
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nfr, repeat=False)

    plt.close()

    return anim


def plot_pressure_latitude(data, method='filled', axes=None, wmm=80,
                           hmm=80, levels=None, minv=None, maxv=None,
                           nlevels=None, cm='jet', extend='neither',
                           cticks=None, xlim=[-90, 90], xticks=None,
                           xticklabels=None, xminor=None,
                           regular_axis=False, title='',
                           cbstring=None, save=None,
                           transparent=False):
    """Create a pressure versus latitude contour plot.

    Level units must be in hPa.

    Parameters
    ----------
    data: xarray.DataArray
        Input must have two dimensions: level and latitude.
    method: str, optional
        It can be either 'filled' to use matplotlib's `contourf`
        function, or 'mesh' which uses matplotlib's `pcolormesh`.
        Default is to plot filled contours.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 80 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 80 mm.
    levels: list or numpy.ndarray, optional
        Contour levels can be specified with this keyword. Otherwise
        you can let this function create the contour levels using
        keywords `minv`, `maxv`, and `nlevels`.
    minv: float, optional
        Minimum value for contour levels in colorbar. If not set, the
        minimum value found in the data array will be used.
    maxv: float, optional
        Maximum value for contour levels in colorbar. If not set, the
        maximum value found in the data array will be used.
    nlevels: int, optional
        Number of levels for contour coloring. If not used, default is
        10 contour levels.
    cm: str, optional
        Colormap to be used. By default is Jet colormap.
    extend: str, optional
        Whether to have pointing arrows at the ends of the
        colorbar. It can be 'neither', to not use arrows but have 
        blunt ends, 'max' to only have an arrow at the maximum end of
        the colorbar, 'min', and 'both'.
    cticks: list or numpy.ndarray, optional
        These ticks are for the colorbar, in case you want particular
        ones. By default the function will try to use the "best ones",
        choosing values every 2 contour levels.
    xlim: list, optional
        To specify the limits in the x axis. Default is [-90, 90].
    xticks: list or numpy.ndarray, optional
        In case the user wants different latitudes in the x
        axis. Default is [-80, -40, 0, 40, 80].
    xticklabels: list, optional
        List of strings to be used if the user has set different
        `xticks`. It really only makes sense to use this setting along
        with xticks, but it is not coded this way in case the user
        wants to change the default xticks labels.
    xminor: list or numpy.ndarray, optional
        Position of minor x tick marks. Default is [-60, -20, 20, 60].
    regular_axis: bool
        Y axis can be regular or logarithmic. Pressure levels have
        different altitudes, which irregularly spaced (unlike pressure
        values). A trick is to use the logarithm of pressure values to
        space the Y axis unevenly. Default is not to use a regular
        axis but the logarithmic.
    title: str, optional
        Center top title if desired. Default is empty.
    cbstring: str, optional
        Title for the colorbar. If none provided, it will try to use
        `units` attribute in data array if is defined.
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
    plot_settings()

    # get coordinates
    lev = data.level.values
    lat = data.latitude.values

    # get contour levels
    if levels is None:
        levels = create_clev(data, minv, maxv, nlevels)

    # guess ticks for colorbar
    if cticks is None:
        cticks = levels[1:-1:2]

    # in case we plot a single plot
    if axes is None:
        plt.figure(figsize=(wmm / 25.4, hmm / 25.4))
        axes = plt.axes()
        maximize = 1
    else:
        maximize = 0

    # xtick marks
    if xticks is None:
        xticks = [-80, -40, 0, 40, 80]

    # xtick labelsize
    if xticklabels is None:
        xticklabels = [r'80$^{\circ}$S', r'40$^{\circ}$S',
                       r'0$^{\circ}$', r'40$^{\circ}$N',
                       r'80$^{\circ}$N']

    # xminor marks
    if xminor is None:
        xminor = [-60, -20, 20, 60]

    # create unevenly spaced axis
    if regular_axis is False:
        lev = np.log10(lev)

    if method == 'filled':
        # plot filled countour with specs
        fmap = axes.contourf(lat, lev, data.values, levels=levels, cmap=cm,
                             extend=extend)
        cb = plt.colorbar(fmap, orientation='horizontal', pad=0.10,
                          format=FuncFormatter(_no_hyphen),
                          shrink=0.75, ax=axes, ticks=cticks)
    elif method == 'mesh':

        # fix coords
        corlev, corlat = corner_coords(lev, lat)

        # plot grid cells with specs
        cmap = get_cmap(cm, len(levels))
        cnorm = BoundaryNorm(levels, cmap.N)
        fmap = axes.pcolormesh(corlat, corlev, data.values, cmap=cmap,
                               norm=cnorm)
        cb = plt.colorbar(fmap, orientation='horizontal', pad=0.05,
                          format=FuncFormatter(_no_hyphen),
                          shrink=0.75, ax=axes, extend=extend,
                          ticks=cticks)
    else:
        msg = 'method can only be \'filled\' or \'mesh\''
        raise ValueError(msg)

    # x axis settings
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels)
    axes.set_xlim(xlim)
    axes.xaxis.set_minor_locator(FixedLocator(xminor))

    # set y ticks to specific values
    if regular_axis is False:
        yval = [850, 500, 200, 100]
        yticks = np.log10(yval)
        yticklabels = [str(x) for x in yval]
        axes.set_yticks(yticks)
        axes.set_yticklabels(yticklabels)

    # invert axis
    axes.invert_yaxis()

    # labels
    axes.set_ylabel(r'Pressure (hPa)')

    # add colorbar title
    if cbstring is None:
        try:
            cbstring = data.units
        except AttributeError:
            pass
    cb.set_label(cbstring)

    # add plot title
    axes.set_title(title)

    # maximize plot if only one
    if maximize == 1:
        plt.tight_layout()

        # savefig if provided name
        save_func(save, transparent)

    return axes


def panel_pressure_latitude(dlist, slist=None, wmm=180, hmm=90,
                            save=None, transparent=False):
    """Panels version of `plot_pressure_latitude`.

    Parameters
    ----------
    dlist: list of xarray.DataArray
        All data arrays must be 2 dimensional with named coordinates
        `latitude` and `level`. 
    slist: list of dict objects of specifications, optional
        Each array in `dlist` must have a dictionary of options or
        specifications in this list. See `plot_pressure_latitude` for
        possible specifications.
    wmm: float, optional
        Width of the figure to plot in units of mm. Default is 180 mm.
    hmm: float, optional
        Height of the figure to plot in units of mm. Default is 90 mm.
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
    matplotlib.figure.Figure object with panel plots.
    """  # noqa

    # settings
    plot_settings()

    # check number of dataset equals specifications
    ndat = len(dlist)

    if slist is not None:
        nspc = len(slist)

        if ndat != nspc:
            msg = 'more/less specifications than datasets'
            raise ValueError(msg)

    # create plot
    fig = plt.figure(figsize=(wmm / 25.4, hmm / 25.4))

    # get better shape for these many datasets
    nrows, ncols = _get_pshape(ndat)

    # actually plot
    for p in range(ndat):

        # get data
        data = dlist[p]

        # create axes
        ax = plt.subplot(nrows, ncols, p + 1)

        # plot
        if slist is not None:
            spec = slist[p]

            # remove any axes specification just in case
            try:
                del spec['axes']
            except KeyError:
                pass

            plot_pressure_latitude(data, axes=ax, **spec)
        else:
            plot_pressure_latitude(data, axes=ax)

    # maximize output
    plt.tight_layout()

    # savefig if provided name
    save_func(save, transparent)

    return fig
