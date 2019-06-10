import os
import warnings
import subprocess

import numpy as np


def check_monotonic(coord):
    """Check coordinate increases or decreases monotonically.

    Most of the times it is important that coordinate values are
    ordered.
    
    Parameters
    ----------
    coord: xarray.DataArray or numpy.ndarray
        This should be one of the coordinates in an xarray.Datarray.

    Returns
    -------
    None. Simply it will fail if coordinate is not monotonic.
    """  # noqa

    # assume not
    mono = False

    # check coord increase monotonic
    if np.all(coord[1:] <= coord[:-1]):
        mono = True

    # check coord decrease monotonic
    if np.all(coord[1:] >= coord[:-1]):
        mono = True

    if mono is True:
        pass
    else:
        msg = 'coordinate not monotonic'
        raise ValueError(msg)


def mask_land_ocean(data, land_mask, ocean=False):
    """Mask land or ocean values using a land binary mask.

    Parameters
    ----------
    data: xarray.DataArray
        This input array can only have one of 2, 3 or 4
        dimensions. All spatial dimensions should coincide with those
        of the land binary mask.
    land_mask: xarray.DataArray
        This array must have the same spatial extent as the input
        data. Though it can have different times or levels. It can be
        binary or not, because internally it will make sure of
        it. Sometimes these masks actually contain a range of values
        from 0 to 1.
    ocean: bool, optional
        Whether the user wants to mask land or ocean values. Default
        is to mask ocean values (False).

    Returns
    -------
    xarray.Datarray same as input data but with masked values in
    either land or ocean.
    """  # noqa

    # remove numpy warning regarding nan_policy
    msg = 'Mean of empty slice'
    warnings.filterwarnings('ignore', message=msg)

    # get number of dimensions of both data arrays
    ndim_ds = len(data.dims)
    ndim_lm = len(land_mask.dims)

    # get dimensions of dataset
    if ndim_ds == 2:
        ntim = None
        nlat, mlon = data.shape
    elif ndim_ds == 3:
        ntim, nlat, mlon = data.shape
    elif ndim_ds == 4:
        ntim, nlev, nlat, mlon = data.shape
    else:
        msg = 'only 2, 3 or 4 dimensions allowed for data set'
        raise TypeError(msg)

    # get dimensions of land mask
    if ndim_lm == 2:
        lntim = None
        lnlat, lmlon = land_mask.shape
    elif ndim_lm == 3:
        lntim, lnlat, lmlon = land_mask.shape
    else:
        msg = 'only 2 or 3 dimensions allowed for land mask'
        raise TypeError(msg)

    # make sure dims agree
    if nlat != lnlat or mlon != lmlon:
        msg = 'spatial coordinates do not agree'
        raise ValueError(msg)

    # get a single land mask if many
    if lntim is not None or lntim == 1:
        land_mask = land_mask[0]

    # convert mask to binary if not already
    land_mask = binary_mask(land_mask)

    # create mask 1 (land) = True, 0 (ocean) = False
    mask = land_mask.values == 1

    # tile mask to number of times
    if ndim_ds == 2:
        tmask = mask
    elif ndim_ds == 3:
        tmask = np.tile(mask, (ntim, 1, 1))
    else:
        tmask = np.tile(mask, (ntim, 1, 1, 1))

    # create masked array
    values = np.array(data.values)

    if ocean is True:
        maskval = np.ma.masked_array(values, tmask)
    else:
        maskval = np.ma.masked_array(values, tmask is False)

    # replace values
    newdata = data.copy()
    newdata.values = maskval

    return newdata


def binary_mask(mask, thres=0.5):
    """Make sure values are either 0 or 1 in a binary mask.

    Sometimes masks do not have only 0 and 1 values, but also
    intermediate numbers. This function makes sure there are only 0
    and 1 values in the mask setting a threshold value.

    Parameters
    ----------
    mask: xarray.DataArray
        Input array must be 2D.
    thres: float, optional
        Threshold value above which a gridpoint will be considered
        land. Default is 0.5.

    Returns
    -------
    xarray.DataArray same as input mask but with only values 0 and 1.
    """

    if len(mask.dims) != 2:
        msg = 'mask must be 2D with lat and lon'
        raise ValueError(msg)

    # get shape
    lnlat, lmlon = mask.shape

    # convert mask to binary if not already
    for i in range(lnlat):
        for j in range(lmlon):
            val = mask.values[i, j]
            if val >= thres:
                mask.values[i, j] = 1
            else:
                mask.values[i, j] = 0
    mask = mask.astype(np.int64)

    return mask


def ffmpeg_make_video(path, base_name, out_name=None, frate=25):
    """Create videos using FFMpeg.
    
    This function makes a call to FFMpeg which must be installed
    in the host machine, otherwise it fails.

    Parameters
    ----------
    path: str
        Location in the host machine where images are stored to be
        grouped in video.
    base_name: str (formatted)
        It is important that all images are named with the same
        base name and an incremental number. This option should be
        something like: 'rotate_%06i.png' in which all figures are
        named the same but with a 6-digit integer incrementing.
    out_name: str, optional
        Name of the output video. Default is 'output.mp4'.
    frate: int, optional
        Frame rate. In case the user wants to control the speed of the
        videos.  Default is 25 fps.
    """  # noqa

    # check ffmpeg is installed
    status, result = subprocess.getstatusoutput('ffmpeg -version')

    if status != 0:
        msg = 'you need to have ffmpeg command installed'
        raise SystemError(msg)

    # build ffmpeg command
    cmd = ['ffmpeg', '-i']

    # append input name
    cmd.append(path + '/' + base_name)

    if frate != 25:
        cmd.extend(['-framerate', frate])

    # some options
    cmd.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y'])

    # output name
    if out_name is None:
        out_name = 'output.mp4'
    cmd.append(out_name)

    # check if exists already
    if os.path.isfile(out_name):
        ans = input("File \'" + out_name +
                    "\' exists. Overwrite? [y]/n: ")
        if (ans.strip() == '') or (ans.strip() == 'y'):
            # execute command
            subprocess.call(cmd)
            print("Done. Re-created file: \'" + out_name + "\'")
        else:
            print("Nothing done.")
    else:
        # execute command
        subprocess.call(cmd)
        print("Created file: \'" + out_name + "\'")


def mask_below(data, thres):
    """Mask values below a constant.

    Return data array with masked values below given threshold. We use
    absolute values in order to also discard small negative values.

    Parameters
    ----------
    data: xarray.DataArray
        Input array.
    thres: float
        Value below which values in input array will be masked.

    Returns
    -------
    xarray.DataArray with masked values.
    """  # noqa

    # get values
    values = np.array(data.values)

    # create a mask
    mask = abs(values) < thres
    mask_array = np.ma.masked_array(values, mask)

    # reassign values
    newdata = data.copy()
    newdata.values = mask_array

    return data


def check_bounds(bounds):
    """Check bounds are correct.

    Many functions use bounds as keyword argument, so this function
    checks they are usable.

    Parameters
    ----------
    bounds: list
        It should contain 4 items in the following order:
        
            [lon0, lon1, lat0, lat1]

    Returns
    -------
    None. Simply fails if bounds are not proper.
    """  # noqa

    if not isinstance(bounds, list):
        msg = 'bounds must be a list Python object'
        raise TypeError(msg)

    if len(bounds) != 4:
        msg = ('bounds must have for coordinate values: [lon0, ' +
               'lon1, lat0, lat1]')
        raise ValueError(msg)


def get_bounds(data):
    """Given a data array it returns bounding longitude and latitude.

    Simple way to obtain first and last values in coordinates
    `latitude` and `longitude` of a given data array. It also
    estimates horizontal resolution in the x direction and y
    direction. 

    Parameters
    ----------
    data: xarray.DataArray
        Input array must have named `latitude` and `longitude`
        coordinates.
    
    Returns
    -------
    A tuple object with three things:

        (bounds, x_horizontal_resolution, y_horizontal_resolution)

    Resolutions are simply the mean differences between values. If it
    is a regular grid, the mean should be the same as any
    difference. Bounds have the following order:

        [lon0, lon1, lat0, lat1]
    """  # noqa

    # get coordinates
    lat = data.latitude.values
    lon = data.longitude.values

    # get estimates for horizontal resolutions
    xhres = abs(np.diff(lon).mean())
    yhres = abs(np.diff(lat).mean())

    return ([lon[0], lon[-1], lat[0], lat[-1]]), xhres, yhres
