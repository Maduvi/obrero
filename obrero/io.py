import os
import cf_units
import xarray as xr


def _convert_units(self, new_units):
    """Convert units using UDUNITS2 through cf_units interface."""
    old_units = cf_units.Unit(self.units)
    old_values = self.values
    self.values = old_units.convert(old_values, new_units)
    self.attrs['units'] = new_units


def _get_variable(data, variable):
    """Get a variable with error handling. Returns DataArray."""
    try:
        xarr = data[variable]
    except KeyError:
        raise KeyError('variable \'' + variable +
                       '\' not found') from None
    return xarr


def _force_coord_names(dataset):
    """Enforce coordinate names to be standard."""

    if 'lat' in dataset.dims:
        dataset = dataset.rename({'lat': 'latitude'})

    if 'lon' in dataset.dims:
        dataset = dataset.rename({'lon': 'longitude'})

    if 'lev' in dataset.dims:
        dataset = dataset.rename({'lev': 'level'})

    return dataset


def read_nc(file_name, variable=None):
    """Open a netCDF file as an xarray object.

    Requires xarray. Basically is the same as `open_dataset` from
    xarray but it additionally forces spatial coordinates to be named
    `latitude` and `longitude`.
    
    Parameters 
    ----------
    file_name: str 
        A path to a netCDF file.
    variable: str or list of str, optional
        Either a string which is the name of a variable defined in the
        netCDF file or a list of them. If not used, load all variables
        in the file.

    Returns
    -------
    xarray.Dataset or xarray.DataArray if single variable specified.

    """  # noqa

    ds = xr.open_dataset(file_name)

    if variable is not None:

        # check if it is a list of variables
        if isinstance(variable, list):
            da = []
            for v in variable:
                da.append(_get_variable(ds, v))

            ds = xr.merge(da)
        else:
            ds = _get_variable(ds, variable)

    # enforce coordinate names
    ds = _force_coord_names(ds)

    # add convert units method as bound
    xr.core.dataarray.DataArray.convert_units = _convert_units

    return ds


def save_nc(data, file_name, check=True):
    """Save an xarray object as netCDF asking for confirmation.

    This is simply a wrapper around the `.to_netcdf()` method of
    xarray calsses, to be able to avoid overwriting files.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
         Objects from xarray classes that support the conversion to
         netCDF method.
    file_name: str
         Path or file string to create.
    check: bool
         Whether to check if file will be overwritten. Default is to
         check (True)
    """  # noqa

    msg1 = 'File \'' + file_name + '\' exists. Overwrite? [y]/n: '
    msg2 = 'Done. Re-created file: \'' + file_name + '\''
    msg3 = 'Created file: \'' + file_name + '\''

    if check is True:
        if os.path.exists(file_name):
            a = input(msg1)
            if (a.strip() == '') or (a.strip() == 'y'):
                os.remove(file_name)
                data.to_netcdf(file_name)
                print(msg2)
            else:
                print("Nothing done.")
        else:
            data.to_netcdf(file_name)
            print(msg3)
    else:
        data.to_netcdf(file_name)
        print(msg3)
