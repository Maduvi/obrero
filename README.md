# obrero

obrero is a set of Python 3.x functions to ease production of high-quality plots in climate modeling experiments. It helps our coding to be less cluttered. Scripts that used to have over 100 lines, become 20 lines long using obrero functions. Under the hood it is doing all kinds of things with [xarray](http://xarray.pydata.org/en/stable/) and [numpy](http://www.numpy.org/) objects. It also makes use of great [pandas](https://pandas.pydata.org/), [scipy](https://www.scipy.org/) and [cf_units](https://github.com/SciTools/cf-units) libraries for dealing with dates and other complex tasks like hypothesis testing. All plotting is done using [matplotlib](https://matplotlib.org/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/). Currently obrero is being used in studies that involve the El Niño--Southern Oscillation (ENSO) and maximum climatological water deficit (MCWD) as presented in Malhi et al. (2009). So there are two experimental modules for each of these subjects. This is all just an academic excercise.

## Modules
Basic modules include input/output functions, calendar related, spatial and time averaging and processing and some other utilities. As well as a plotting module. All of these are very basic and are just to ease are scripting. There are two very interesting modules however in the experimental section:

... **Video UdeA**: for an event we prepared some nice videos of some data in an Orthographic projection as the planet spins. Some of the videos are still [up](https://www.youtube.com/watch?v=7csgoDidIlY&t=115s), though now they have changed from that version. Maybe later we will upload more videos. These videos come with the logo of Universidad de Antioquia, the place where we work. This module then has all the functions required to create such videos.

... **ENSO**: this module has some functions that can aid in studying ENSO variability in climate modeling. It uses data from [NOAA's Climate Prediction Center](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt) to be able to make [this](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php) very same table that allows us to choose different ENSO phases. So part of this module deals with computing the Oceanic Niño Index (ONI). It also has functions to amplify sea surface temperature (SST) anomalies so that ONI's become greater.

... **MCWD**: in our studies we use a similar approach to Aragao et al. (2007) and Malhi et al. (2009), so this module is ready to compute MCWD. It also plots changes in MCWD and mean annual precipitation (MAP) and has a unique composite map that we have described in Duque et al. (2019).

## Tests
There is a very small and limited test suite in this package. It is barely a suite and it barely tests anything. It is more of a learning excercise.

## Data
This little package comes with two non-Python files. The first is an ASCII file with anomalies of SST in the Niño 3.4 region available online [here]((https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt). But also there is an image file that is the logo of the university where we work. It is available online [here](https://upload.wikimedia.org/wikipedia/commons/f/fb/Escudo-UdeA.svg).

## References
Malhi, Y., Aragão, L. E., Galbraith, D., Huntingford, C., Fisher, R., Zelazowski, P., ... & Meir, P. (2009). Exploring the likelihood and mechanism of a climate-change-induced dieback of the Amazon rainforest. Proceedings of the National Academy of Sciences, 106(49), 20610-20615.

Aragao, L. E. O., Malhi, Y., Roman‐Cuesta, R. M., Saatchi, S., Anderson, L. O., & Shimabukuro, Y. E. (2007). Spatial patterns and fire response of recent Amazonian droughts. Geophysical Research Letters, 34(7).

Duque et al. (2019). In submission.