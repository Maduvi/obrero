import setuptools

deps = ['numpy', 'matplotlib', 'xarray', 'pandas', 'scipy', 'cf_units',
        'cartopy']

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "obrero",
    version = "0.0.1",
    author = "Mateo Duque-Villegas",
    author_email = "mateo.duquev@udea.edu.co",
    description = ("Set of Python procedures to ease pre- and" +
                   " post-processing stages of climate modeling" +
                   " experiments"),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/pypa/sampleproject",
    packages = setuptools.find_packages(),
    install_requires = deps,
    include_package_data = True,
    license = "LICENSE.txt",
) 
