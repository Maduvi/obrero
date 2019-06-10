import os
import sys
import pkg_resources

import numpy as np

from matplotlib.image import imread

import obrero.cal as ocal
import obrero.plot as oplot
import obrero.experimental.enso as oenso

# path where stored logo
DATA_PATH = pkg_resources.resource_filename('obrero', 'data/')


def _add_text_axes(axes, text):
    """Use a given axes to place given text."""

    txt = axes.text(0.5, 0.5, text, ha='center', va='center')
    axes.axis('off')

    return txt


def _latex_authoring(title, author, affil, email):
    """Creates a text object with LaTeX code to include in plots
    made with `video_udea`.
    """  # noqa

    texmsg = []

    # lets build it
    texmsg.append(r'\begin{center}')

    # title
    if isinstance(title, list):
        for t in title:
            texmsg.append(t + r'\\')
    else:
        texmsg.append(title + r'\\')

    # a bit of space
    texmsg.append(r'\vspace{1em}')

    # authors
    if isinstance(author, list):
        for a in author:
            texmsg.append(r'\tiny{' + a + r'}\\')
    else:
        texmsg.append(r'\tiny{' + author + r'}\\')

    # authors
    if isinstance(affil, list):
        for a in affil:
            texmsg.append(r'\tiny{' + a + r'}\\')
    else:
        texmsg.append(r'\tiny{' + affil + r'}\\')

    # email
    if isinstance(email, list):
        for e in email:
            texmsg.append(r'\tiny{' + e + r'}')
    else:
        texmsg.append(r'\tiny{' + email + r'}')

    # finish
    texmsg.append(r'\end{center}')

    # join
    latext = ' '.join(texmsg)

    return latext


def video_udea(dlist, slist, bbox, title, author, affil, email,
               rotate, wpx=1920, hpx=1080, dpi=300, lon0=0, dg=1,
               save_dir=None, smooth=False, winds=None, xhres=None):
    """Create video made for ExpoIngenieria 2018.

    A very specific format was used to produce this video and to keep
    it we created this function. It can only be used to produce such
    video. In this case we need for sets of data arrays: a variable to
    be plotted in an Orthographic projection rotating every `dg`
    degrees, two lines of time series area average over a region to be
    plotted and compared in an xy-plot, and sea surface temperature
    (SST) values to include the ONI time series. The user can also
    input horizontal wind fields U and V to have vectors plotted on
    top of contours.

    Parameters
    ----------
    dlist: list of xarray.DataArray
        This list must have the following order:

            [variable_for_contours, first_time_series,
             second_time_series, sst_array]

        The first variable will be plotted in a rotating Orthographic
        projection. The time series will be plotted together in an
        xy-plot. And the SST array will be used to plot also an ONI
        index axes.
    slist: list of dict objects of specifications
        This list must contain three dict objects: one for the contour
        plot, one for the time series plot and one for the ONI index
        plot. So the list must be:
    
            [specifications_contours, specifications_time_series,
             specifications_oni_index]

        For the specifications of the contours see keywords of
        function `plot_global_contour`, except keyword `axes`. For the
        time series specifications see keywords of the function
        `averages_video_udea`. And for the ONI plot see keywords in
        the `oni_video_udea` function.
    bbox: list of list objects
        This is a list of two list objects which have corner
        coordinates to plot a squared region: [xcorners,
        ycorners]. This in case the user wants to highlight a squared
        region somewhere in the Orthographic projection map. This
        object can be obatined using function `bbox_linecoords`.
    title: str or list of str
        Title to be placed in a text-only axes. Input for
        `_latex_authoring`. If multiple lines it should be a list of
        str in which each str is a single line.
    author: str or list of str
        Author information to be placed in a text-only axes. Input for
        `_latex_authoring`. If multiple lines it should be a list of
        str in which each str is a single line.
    affil: str or list of str
        Affiliation information of author to be placed in a text-only
        axes. Input for `_latex_authoring`. If multiple lines it
        should be a list of str in which each str is a single line.
    email: str or list of str
        Author e-mail information to be placed in a text-only
        axes. Input for `_latex_authoring`. If multiple lines it
        should be a list of str in which each str is a single line.
    rotate: list
        In this list the user can specify when to rotate the
        projection. To do this the user must use dates in the format:
        'YYYY-MMM', using 3 letters for the month. So for example if:

            rotate = ['1997-Jun', '1998-Dec']

        It means that the Orthographic projection will rotate for
        those two months only, in spite of the data arrays having more
        time steps.
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
    smooth: bool, optional
        Use this boolean flag to choose whether to smooth the time
        series or not. The smoothing will be done using a rolling mean
        every 3-time steps, so if it is monthly data, the user will
        actually be plotting 3-monthly rolling averages. Default is
        False.
    winds: list of xarray.DataArray, optional
        If the user has U and V winds data and wants to put vectors on
        top of the contours in the Orthographic projection plot, then
        they must use this option for input winds like so:
    
            winds = [u, v]
        
        For this to work the user must also use the `xhres` keyword
        because the function needs the resolution of the grid in the x
        direction to be able to avoid plotting vectors out of the
        projection bounds. 
    xhres: float, optional
        Grid resolution in the x direction. This keyword is only used
        if `winds` is being used, in which case it is a mandatory
        argument.
    """  # noqa

    # unpack data and specifications
    vmap, vline1, vline2, sst = dlist
    spec1, spec2, spec3 = slist

    # check if wind wanted and given
    if winds is not None:
        u, v = winds

        if xhres is None:
            msg = ('if you want wind you must specify horizontal ' +
                   'horizontal x resolution with \'xhres\' keyword')
            raise ValueError(msg)

        # only lats in between will have wind
        w_ymin = 4
        w_ymax = 28

        # longitudes will have wind
        wlon = 9

        # get longitudes as x
        x = u.longitude.values
        y = u.latitude.values
        mlon = x.size

    # smooth area averages if wanted
    if smooth is True:
        vline1 = (vline1.rolling(time=3, min_periods=2)
                  .mean(keep_attrs=True))
        vline2 = (vline2.rolling(time=3, min_periods=2)
                  .mean(keep_attrs=True))

    # get number of times
    ntim = vmap.time.size

    # get oni series from exp
    oni = oenso.oni(sst).values.flatten()

    # authoring message
    msg = _latex_authoring(title, author, affil, email)

    # get dates
    dates = ocal.get_dates(vmap.time.values)

    # guess number of maps
    nmpr = int(360 / dg)
    nrots = len(rotate)
    totm = (ntim - nrots) + nrots * nmpr

    # counter for names
    c = 1

    # create save directory
    save_path = oplot.create_save_dir(save_dir)

    # step every time
    for t in range(ntim):

        # rotate only for specified dates
        dstr = dates[t].strftime('%Y-%b')

        if dstr in rotate:
            rotation = True
            nrot = nmpr  # number of maps per rotation
        else:
            rotation = False
            nrot = 1

        if winds is not None:
            clon = x[(x >= lon0 - xhres / 2) & (x < lon0 + xhres / 2)]
            idx = np.where(x == clon)[0][0]

        # rotate or not
        for i in range(nrot):

            # create figure instance
            fig = oplot.plt.figure(1, figsize=(wpx / dpi, hpx / dpi))

            # projection
            prj = oplot.ort(central_longitude=lon0)

            # create axes for all
            ax1 = oplot.plt.subplot2grid((3, 6), (0, 0), colspan=3,
                                         rowspan=3, projection=prj)
            ax2 = oplot.plt.subplot2grid((3, 6), (0, 3), colspan=3)
            ax3 = oplot.plt.subplot2grid((3, 6), (1, 3), colspan=3)
            ax4 = oplot.plt.subplot2grid((3, 6), (2, 3), colspan=2)
            ax5 = oplot.plt.subplot2grid((3, 6), (2, 5))

            # add axes and title to specifications
            spec1['axes'] = ax1
            spec1['title'] = r'\texttt{' + dstr + r'}'

            # plot
            oplot.plot_global_contour(vmap[t], **spec1)

            # add wind arrows if given
            if winds is not None:

                # get winds
                U = u[t].values
                V = v[t].values

                # get longitude range indexes
                if (idx + wlon) < mlon:
                    xrang = np.arange(idx - wlon, idx + wlon + 1,
                                      dtype=int)
                else:
                    xrang = np.arange(idx - mlon - wlon, idx - mlon
                                      + wlon + 1, dtype=int)

                # select those to plot
                xx = x[xrang]
                yy = y[w_ymin:w_ymax]
                uu = U[w_ymin:w_ymax, xrang]
                vv = V[w_ymin:w_ymax, xrang]

                # add arrows
                quiv = ax1.quiver(xx, yy, uu, vv, pivot='middle',
                                  transform=oplot.pcar(),
                                  scale_units='inches',
                                  scale=8500 / 25.4)

                # add key
                ax1.quiverkey(quiv, 0.9, 0.1, 20, r'20 km h$^{-1}$',
                              labelpos='S', angle=180)

            # bounding box
            ax1.plot(bbox[0], bbox[1], '-', linewidth=1,
                     color='black', transform=oplot.pcar())

            # plot averages
            averages_video_udea(dates[:t + 1], vline1.values[:t + 1],
                                vline2.values[:t + 1], ax2, **spec2)

            # plot oni
            oni_video_udea(dates[:t + 1], oni[:t + 1], ax3, **spec3)

            # add message
            _add_text_axes(ax4, msg)

            # add logo
            udea_logo(ax5)

            # maximize plot
            oplot.plt.tight_layout()

            # savefig if provided name
            if save_dir is not None:
                img = os.path.join(save_path, "rotate_%08d.png" % c)
                oplot.plt.savefig(img, dpi=dpi)
                oplot.plt.close(fig)
                sys.stdout.write('Plotting progress: %d%%  \r' %
                                 (100 * c/totm))
                sys.stdout.flush()
                # update counter
                c += 1
            else:
                oplot.plt.pause(0.05)

            # update lon0
            if rotation is True:
                if lon0 > 0.0:
                    lon0 = lon0 - dg
                else:
                    lon0 = 360.0

            # update clon if winds and get ist index
            if winds is not None:
                if idx <= mlon - 1:
                    clon = x[(x >= lon0 - xhres / 2.0) &
                             (x < lon0 + xhres / 2.0)]
                    try:
                        idx = np.where(x == clon)[0][0]
                    except IndexError:
                        idx = 0
                else:
                    idx = 0


def oni_video_udea(dates, oni, axes, xticks=None, xlim=None,
                   ylim=[-3, 3], title='ONI', color='black',
                   xlabel=r'Year', ylabel=r'($^{\circ}$\,C)'):
    """Plot ONI time series for UdeA video.

    In the video there will be an axes with ONI values. This function
    will take care of it.

    Parameters
    ----------
    dates: pandas.DatetimeIndex
        These are the x axis values. Matplotlib will interpret them as
        dates and format them as such.
    oni: numpy.ndarray
        This is a time series. It should be obtained flattening the
        values of the data frame that the function `enso.get_oni`
        creates.
    axes: matplotlib.axes.Axes
        Generally created using `figure.add_subplot()`. Since this
        plot is to be appended to a larger picture, the axes must be
        created outside this function and used as input.
    xticks: list or numpy.ndarray, optional
        This controls the tick marks in the x axis. Default is to put
        a tick from the second year until the end every 2 years.
    xlim: list, optional
        Limits in the x axis. The user can choose the limit dates in
        this axis. Default is to use the first and last items in
        `dates`.
    ylim: list, optional
        Limits in the y axis. Default is [-3, 3].
    title: str, optional
        Centered title. Default is 'ONI'.
    xlabel: str, optional
        Title in the x axis. Default is 'Year'.
    ylabel: str, optional
        Title in the y axis. Default is '(oC)'.

    Returns
    -------
    matplotlib.axes.Axes with plot attached.
    """  # noqa

    # get ticks
    if xticks is None:
        xticks = dates[12::48]

    # get xlim
    if xlim is None:
        xlim = [dates[0], dates[-1]]

    # get colors for line plots
    cm = oplot.plt.get_cmap('bwr')
    cred = cm(cm.N)
    cblue = cm(0)

    # plot last as point
    point = oni[-1]

    if point > 0.5:
        cpoint = cred
    elif point < -0.5:
        cpoint = cblue
    else:
        cpoint = 'black'

    # line plot
    axes.plot(dates, oni, linewidth=1, color=color)
    axes.plot(dates[-1], point, 'o', color=cpoint, ms=2)

    # axes lims
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    # set ticks
    axes.set_xticks(xticks)

    # horizonatl lines
    axes.axhline(y=0, linestyle='--', alpha=0.5, linewidth=1,
                 color='black')
    axes.axhline(y=0.5, linestyle='--', alpha=0.5, linewidth=1,
                 color=cred)
    axes.axhline(y=-0.5, linestyle='--', alpha=0.5, linewidth=1,
                 color=cblue)

    # titling
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    return axes


def averages_video_udea(dates, dlist, axes, names=['Exp1', 'Exp2'],
                        colors=['black', 'DodgerBlue'], xticks=None,
                        xlim=None, ylim=[-3, 3], title='',
                        xlabel=r'Year', ylabel=''):
    """Plot area average time series of variable for UdeA video.

    In the video there will be axes with time series of some variable
    for two different data sets averaged spatially. This function will
    take care of it.

    Parameters
    ----------
    dates: pandas.DatetimeIndex
        These are the x axis values. Matplotlib will interpret them as
        dates and format them as such.
    dlist: list of numpy.ndarrays
        Only two arrays are supported. These should be time series of
        area averages  for some variable.
    axes: matplotlib.axes.Axes
        Generally created using `figure.add_subplot()`. Since this
        plot is to be appended to a larger picture, the axes must be
        created outside this function and used as input.
    names: list of str, optional
        Names to be shown in the legend. They must have the same order
        as the data in `dlist`. Default is ['Exp1', 'Exp2']. They will
        always be converted to upper case.
    colors: list of named colors, optional
        Colors for each line. They must have the same order as the
        data in `dlist`. Default is ['black', 'DodgerBlue']
    xticks: list or numpy.ndarray, optional
        This controls the tick marks in the x axis. Default is to put
        a tick from the second year until the end every 2 years.
    xlim: list of datetime objects, optional
        Limits in the x axis. The user can choose the limit dates in
        this axis. Default is to use the first and last items in
        `dates`.
    ylim: list of float, optional
        Limits in the y axis. Default is [-3, 3].
    title: str, optional
        Centered title. Default is empty.
    xlabel: str, optional
        Title in the x axis. Default is 'Year'.
    ylabel: str, optional
        Title in the y axis. Default is empty.

    Returns
    -------
    matplotlib.axes.Axes with plot attached.
    """  # noqa

    # get ticks
    if xticks is None:
        xticks = dates[12::48]

    # get xlim
    if xlim is None:
        xlim = [dates[0], dates[-1]]

    # unpack data
    av1, av2 = dlist

    # points
    point1 = av1[-1]
    point2 = av2[-1]

    # line plot for land
    axes.plot(dates, av1, linewidth=1, color=colors[0],
              label=names[0].upper())
    axes.plot(dates, av2, linewidth=1, color=colors[1],
              label=names[1].upper())
    axes.plot(dates[-1], point1, 'o', color=colors[0], ms=2)
    axes.plot(dates[-1], point2, 'o', color=colors[1], ms=2)

    # set lims
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    axes.set_xticks(xticks)

    # horizonatl lines
    axes.axhline(y=0, linestyle='--', alpha=0.5, linewidth=1,
                 color='black')

    # titling
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    axes.legend(ncol=2)

    return axes


def udea_logo(axes):
    """Add Universidad de Antioquia logo to given axes.

    For some plots it is nice to put the logo of the school. This
    function was specifically created to be used in `video_udea`
    function but might be used elsewhere.

    Paramaters
    ----------
    axes: matplotlib.axes.Axes
        Generally created using `figure.add_subplot()`.

    Returns
    -------
    matplotlib.axes.Axes with logo attached.
    """

    # university logo
    logo = imread(DATA_PATH + 'logo-udea_240px.png')

    # logo
    plog = axes.imshow(logo)
    axes.axis('off')

    return plog
