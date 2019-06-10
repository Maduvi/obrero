import cftime
import datetime
import numpy as np
import pandas as pd

# month names and abbreviations
_MONTH_FULL_NAMES = ('January',
                     'February',
                     'March',
                     'April',
                     'May',
                     'June',
                     'July',
                     'August',
                     'September',
                     'October',
                     'November',
                     'December')
_MON1L = [x[0] for x in _MONTH_FULL_NAMES]
_MON3L = [x[:3] for x in _MONTH_FULL_NAMES]

# dictionaries with names
DMON_FNAME = {i: _MONTH_FULL_NAMES[i - 1] for i in range(1, 12 + 1)}
DMON1L = {i: _MON1L[i - 1] for i in range(1, 12 + 1)}
DMON3L = {i: _MON3L[i - 1] for i in range(1, 12 + 1)}

# days per month (including 0 to have index 1 as month 1)
_REG = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_LEAP = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_D360 = (0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30)

# calendar names
_STD_CALENDARS = ('standard', 'gregorian', 'proleptic_gregorian',
                  'julian')
_ALL_CALENDARS = {'noleap': _REG,
                  '365_day': _REG,
                  'standard': _REG,
                  'gregorian': _REG,
                  'proleptic_gregorian': _REG,
                  'all_leap': _LEAP,
                  '366_day': _LEAP,
                  '360_day': _D360}


def _leap_year(year, calendar='360_day'):
    """Determine if year is a leap year. Thanks to Joe Hamman from the
    xarray docs for his nice blog entry.

    Parameters
    ----------
    year: int
        Year to be tested with modulo operator and conditionals to
        find out whether it is a leap year or not.
    calendar: str, optional
        Every calendar has a known set of rules for a leap year.

    Returns
    -------
    bool with truth value for leap year or not.
    """  # noqa

    # assume false
    leap = False

    if (calendar in _STD_CALENDARS) and (year % 4 == 0):
        leap = True
        if calendar == 'proleptic_gregorian':
            if (year % 100 == 0) and (year % 400 != 0):
                leap = False
        elif calendar in ['standard', 'gregorian']:
            if (year % 100 == 0) and (year % 400 != 0) and (year < 1583):
                leap = False

    return leap


def get_dpm(time, calendar='360_day'):
    """Return an array of days per month corresponding to the months 
    provided in `time`. Thanks to Joe Hamman from the xarray docs for 
    his nice blog entry.

    Parameters
    ----------
    time: DatetimeIndex
        This can be the `time` coordinate in an xarray. It must have
        attributes `.month` and `.year`.
    calendar: str, optional
        Every calendar has different number of days per month. Here
        default is a 360 days calendar in which every month has the same
        30 days (including February). Other options are: standard,
        gregorian, proleptic_gregorian, julian, 365_day, and 366_day.

    Returns
    -------
    numpy.ndarray containing the length of every month in the `time`
    index provided. This is intended to be used later in weighting
    averages. Not all time steps should be averaged the same if they
    are monthly means because they come from different length of days
    (depending on the calendar type).
    """  # noqa

    month_length = np.zeros(len(time), dtype=int)

    # get days in this calendar
    calendar_days = _ALL_CALENDARS[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = calendar_days[month]
        if _leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def get_month_name(month, names):
    """It returns a month's name.

    Names can be full as in `January`, one-lettered as in `J` or
    three-lettered as in `Jan`.

    Parameters
    ----------
    month: int
        It must be in range [1, 12], where 1 is January and 12 is
        December.
    names: str
        It can be `3L` for three-lettered month, `1L` for one-lettered
        or `F` for full name.

    Returns
    -------
    str which is the name of the month provided.
    """  # noqa

    # make sure its integer
    try:
        moni = int(month)
    except ValueError:
        msg = 'input must be an integer value'
        raise ValueError(msg)

    # make sure it's a month index
    if moni > 12 or moni < 1:
        msg = 'input month index must lie in [1, 12]'
        raise ValueError(msg)

    # get month name
    if names == '3L':
        mname = DMON3L[moni]
    elif names == '1L':
        mname = DMON1L[moni]
    elif names == 'F':
        mname = DMON_FNAME[moni]
    else:
        msg = 'unrecognized month name type'
        raise ValueError(msg)

    return mname


def get_dates(time):
    """Obtain pandas datetime objects from date objects.

    Sometimes there are problems with 360 days calendars and cftime
    units in pandas. This function uses `dates_from_360cal` to
    circumvent this problem when pandas can't be used directly to
    generate a DatetimeIndex.
    
    Parameters
    ----------
    time: single date object or numpy.ndarray of date objects
        From the `time` coordinate in an xarray, this variable is the
        values array, i.e. xarray.time.values. Objects can be of type
        numpy.datetime64 or cftime._cftime.Datetime360Day.

    Returns
    -------
    DatetimeIndex object.
    """  # noqa

    # in case it is not an array
    try:
        single = time[0]
    except IndexError:
        single = time

    # choose how to get the index
    if isinstance(single, np.datetime64):
        return pd.to_datetime(time)
    elif isinstance(single, cftime.Datetime360Day):
        return dates_from_360cal(time)
    else:
        msg = 'unrecognized date format'
        raise ValueError(msg)


def dates_from_360cal(time):
    """Convert numpy.datetime64 values in 360 calendar format.
    
    This is because 360 calendar cftime objects are problematic, so we
    will use datetime module to re-create all dates using the
    available data.

    Parameters
    ----------
    time: single or numpy.ndarray of cftime._cftime.Datetime360Day

    Returns
    -------
    DatetimeIndex object.
    """  # noqa

    # get all dates as strings
    dates = []

    for d in time:
        dstr = '%0.4i-%0.2i-%0.2i' % (d.year, d.month, d.day)
        date = datetime.datetime.strptime(dstr, '%Y-%m-%d')
        dates.append(date)

    return pd.to_datetime(dates)
