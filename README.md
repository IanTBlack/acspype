# acspype README

You can read the manuscript accompanying this software in [DIY Oceanography](https://tos.org/oceanography/article/acspype-an-open-source-python-package-for-ac-s-data-acquisition-and-processing).

![](https://github.com/IanTBlack/acspype/blob/main/dev_tools/_images/ooi_optaa_nsif.jpg?raw=true)
*Image from work supported by the U.S. National Science Foundation Ocean Observatories Initiative.*


acspype provides functions for reading [Sea-Bird Scientific ACS](https://www.seabird.com/ac-s-spectral-absorption-and-attenuation-sensor/product?id=60762467715) data over RS232 and for performing advanced processing with ACS data.

ACS data are sometimes difficult to work with, particularly for new users without strong optics backgrounds that desire to use the data for empirically derived data products like chlorophyll-a and particulate organic carbon. Some of the target audiences for this package include technicians looking to integrate an ACS into an existing data acquisition system, researchers looking to process prior collected ACS data, and data scientists looking to redistribute ACS data in different formats.
This package attempts to simplify the process of ingesting and processing ACS data so that users can more quickly get to the science application in their research. Complete descriptions of the code can be found in the [GitHub Repository](https://github.com/IanTBlack/acspype) or the [Documentation](https://iantblack.github.io/acspype/) website

## Quickstart Examples
A set of examples can be found in the [acspype GitHub Repository](https://github.com/IanTBlack/acspype/tree/main/examples).


## Installation
This package is available on the [Python Package Index (PyPI)](https://pypi.org/project/acspype/) and can be installed via pip:

`pip install acspype`

<br>


The package is also available on [GitHub](https://github.com/IanTBlack/acspype) and can be forked/cloned for development purposes:

`git clone https://github.com/IanTBlack/acspype.git`


<br>


## Issues, Discussions, and Contributions
If you experience an issue related to acspype, please use the [GitHub Issues](https://github.com/IanTBlack/acspype/issues) page to report it.

If you would like to propose functionality or discuss best practices, please start a discussion on the [GitHub Discussions](https://github.com/IanTBlack/acspype/discussions) page.

Contributions to acspype are encouraged via GitHub [pull request](https://github.com/IanTBlack/acspype/pulls). 
acspype will be updated on the PyPI as patches and major changes are implemented.


## **CAUTION**
### Equipment
If you are using this package to acquire data from an ACS over serial, it is your responsibility to ensure that you are 
using the appropriate equipment (power supply, cable, etc.). Please consult Sea-Bird Scientific for more information about the relevant equipment and
power requirements needed to acquire data from the ACS.

If looking for a USB-to-RS232 adapter, the [FTDI US232R-100-BULK](https://ftdichip.com/products/us232r-100-bulk/) is recommended.


### Serial Processing Limitations
Not all processing steps offered in acspype can be done in the same thread as the serial data acquisition, particularly the more advanced processing stages that use Xarray to handle wavelength indexes. 
Some steps require linear and cubic interpolation, which may take too much time to perform in between packets. 
Processing up through the temperature-salinity corrected absorption (a_mts) and attenuation (c_mts) can typically be done in the same thread in real-time, but requires accessing ancillary data from another source.
If ancillary data is not immediately available, then processing up through measured absorption (a_m) and attenuation (c_m) is more appropriate.


## Additional Information
Additional information about the ACS, the manuals referenced in construction in this package, and the recommended processing and QAQC test can be found in the [info](https://github.com/IanTBlack/acspype/tree/main/info) directory of the GitHub repository or on the acspype [Code Documentation](https://iantblack.github.io/acspype/) website.

### Processing Steps and QAQC
Recommended processing steps and QAQC tests are described in [PROCESSING.md](https://github.com/IanTBlack/acspype/blob/main/info/PROCESSING.md) document.

### Naming Conventions
Although *acspype* tries not enforce strict naming conventions for coordinates, dimensions, and variables, it is strongly recommended that users follow the naming conventions in [NAMING_CONVENTIONS.md](https://github.com/IanTBlack/acspype/blob/main/info/NAMING_CONVENTIONS.md).
The names put forth in this document attempt to combine [CF Guidelines](https://cfconventions.org/Data/cf-standard-names/docs/guidelines.html) and conventions commonly used in literature that utilizes ACS data.

### Reference Material
A small list of ACS, absorption, and attenuation related literature is provided in [REFERENCE_MATERIALS.md](https://github.com/IanTBlack/acspype/blob/main/info/REFERENCE_MATERIAL.md).


## What does acspype need from you (and Sea-Bird Scientific)?
In addition to the physical sensor and a test cable, one who deploys an ACS will also receive a device file (ACS-XXX.dev) and a temperature-salinity coefficient dependence file (TS4.cor) from Sea-Bird Scientific.

The device file is a text file that contains pure water offsets, delta T values, and the wavelength bins for your specific ACS.
No two device files are alike, making it critical to keep track of this information in order to perform corrections and compute advanced products.

The TS4.cor file is a text file that contains empirically derived temperature and salinity correction coefficients needed to correct absorption and attenuation data for changes in temperature and salinity.
These coefficients have been derived and interpolated to 0.1 nm bins from the original data presented in [Sullivan et al. 2006](https://doi.org/10.1364/AO.45.005294). 
The TS4.cor file received from Sea-Bird Scientific is not unique to the ACS. [This file has been converted and stored within acspype as a dictionary for convenience](https://github.com/IanTBlack/acspype/blob/main/acspype/ts4cor.py).

## What does this package **not** do?
This package does not provide any functionality for logging data from the ACS. The file or database type is entirely left up to you, the user.
For file-based logging, [SQLite](https://docs.python.org/3/library/sqlite3.html) is recommended, which can be used to store data within multiple tables in a single file. 
For application where concurrency is required, [PostgreSQL](https://www.psycopg.org/docs/) is a good option. It provides functionality for handling arrays, which are how ACS data are best represented.
The ACS produces a significant amount of data, so logging to text (.txt, .csv) or netCDF (.nc) files may best be done as hourly or daily files to prevent excessive memory usage.

This package does not remove data at any processing stage. In some processing stages, data may be returned as NaN, which could be an expected behaviour for poor quality data. 
For example, if the reference counts for absorption or attenuation are zero, the uncorrected absorption or attenuation at that wavelength bin will become Inf or NaN, because a division by zero occurs within the log calculation. Instances of NaN or Inf should be treated as suspect quality data and be further inspected by the user.
Because this package heavily relies on Xarray, users should use [Xarray methods for selecting data](https://docs.xarray.dev/en/latest/user-guide/indexing.html). 
