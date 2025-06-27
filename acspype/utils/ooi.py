"""
This module contains helper functions for accessing and working with OOI ACS data.
The OOI has named all their ACS sensors OPTAA.
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import re
import requests
from scipy.interpolate import make_interp_spline
import shutil
import xarray as xr


def reformat_ooi_optaa(ds: xr.Dataset) -> xr.Dataset:
    """
    Reformat an OOI OPTAA dataset to make it more compatible with acspype.

    :param ds: An OOI OPTAA dataset.
    :return: A reformatted and renamed OOI OPTAA dataset.
    """

    # Rename Wavelength dimensions
    ds = ds.rename({'wavelength_a': 'a_wavelength', 'wavelength_c': 'c_wavelength'})
    a_wvl = np.unique(ds.a_wavelength)
    c_wvl = np.unique(ds.c_wavelength)
    nwvls = int(np.unique(ds.num_wavelengths))
    ds = ds.drop_vars(['a_wavelength', 'c_wavelength', 'num_wavelengths'], errors='ignore')

    # Pull out lat/lon.
    # No OOI OPTAA dataset is currently mobile, so assign a static latitude and longitude to the entire dataset.
    lat = np.unique(ds.lat)
    lon = np.unique(ds.lon)
    ds = ds.drop_vars(['lat', 'lon'], errors='ignore')

    # Pull out deployment variable to assign as a dimension later.
    dep = np.unique(ds.deployment)
    ds = ds.drop_vars(['deployment'], errors='ignore')

    # Reformat to use time instead of obs.
    ds = ds.swap_dims({'obs': 'time'})
    ds = ds.drop_vars(['obs'], errors='ignore')

    # Create a new dataset with the appropriate dimensions.
    nds = xr.Dataset()
    nds = nds.assign_coords({'time': ds.time, 'a_wavelength': a_wvl, 'c_wavelength': c_wvl})
    for var in ds.data_vars:
        if var in ['a_signal_counts', 'a_reference_counts', 'optical_absorption']:
            nds[var] = (['time', 'a_wavelength'], ds[var].data)
        elif var in ['c_signal_counts', 'c_reference_counts', 'beam_attenuation']:
            nds[var] = (['time', 'c_wavelength'], ds[var].data)
        else:
            nds[var] = (['time'], ds[var].data)

    # Rename variables to match _acspype.
    mapper = {'external_temp_raw': 'raw_external_temperature',
              'c_signal_counts': 'c_signal',
              'a_signal_counts': 'a_signal',
              'int_ctd_pressure': 'sea_water_pressure',
              'a_reference_counts': 'a_reference',
              'driver_timestamp': 'driver_timestamp',
              'id': 'uuid',
              'provenance': 'provenance_uuid',
              'internal_timestamp': 'internal_timestamp',
              'beam_attenuation': 'beam_attenuation',
              'profiler_timestamp': 'profiler_timestamp',
              'internal_temp_raw': 'raw_internal_temperature',
              'ingestion_timestamp': 'ingestion_timestamp',
              'c_reference_dark_counts': 'c_reference_dark',
              'port_timestamp': 'port_timestamp',
              'optical_absorption': 'optical_absorption',
              'sea_water_practical_salinity': 'sea_water_practical_salinity',
              'on_seconds': 'elapsed_time',
              'elapsed_run_time': 'elapsed_time',
              'c_signal_dark_counts': 'c_signal_dark',
              'pressure_counts': 'raw_pressure',
              'preferred_timestamp': 'preferred_timestamp',
              'a_signal_dark_counts': 'a_signal_dark',
              'a_reference_dark_counts': 'a_reference_dark',
              'c_reference_counts': 'c_reference',
              'suspect_timestamp': 'suspect_timestamp',
              'sea_water_temperature': 'sea_water_temperature'}


    for key, value in mapper.items():
        try:
            nds = nds.rename({key: value})
            nds[value].attrs = ds[key].attrs
        except:
            continue

    qvars = [v for v in ds.data_vars if 'qartod' in v]
    for qvar in qvars:
        nds[qvar] = ds[qvar]

    # Update elapsed_time if it arrives with units of seconds.
    if nds.elapsed_time.attrs['units'] == 's':
        nds['elapsed_time'] = nds.elapsed_time * 1000
        nds['elapsed_time'].attrs['units'] = 'ms'

    # Reassign lat and lon dimensions.
    nds = nds.expand_dims({'latitude': lat,
                           'longitude': lon,
                           'deployment': dep})

    # Reassign attributes
    for attr in sorted(ds.attrs):
        if ds.attrs[attr] in ['', '[]']:
            continue
        else:
            nds.attrs[attr] = ds.attrs[attr]
    nds.attrs['number_of_output_wavelengths'] = nwvls

    # Drop confusing variables or variables we will reprocess to.
    vars_to_drop = ['driver_timestamp', 'uuid', 'provenance_uuid', 'internal_timestamp', 'ingestion_timestamp',
                    'profiler_timestamp', 'port_timestamp', 'preferred_timestamp', 'suspect_timestamp', 'raw_pressure',
                    'optical_absorption', 'beam_attenuation']
    nds = nds.drop_vars(vars_to_drop, errors='ignore')

    # Reorder variables alphabetically for convenience.
    nds = nds[sorted(nds.data_vars)]

    return nds


def _build_acpype_dev_obj(response_data, start) -> object:
    """
    Build an ACSDev-like object for processing OOI OPTAA data with _acspype.

    :param response_data: The response from an UID request to the OOI API.
    :param start: The start time of dataset, used to find the most recent calibration.
    :return: An ACSDev-like class object.
    """

    cals = response_data['calibration']
    cal_data = {}
    for cal in cals:
        if cal['name'] == 'CC_acwo':
            cname = 'a_offset'
        elif cal['name'] == 'CC_ccwo':
            cname = 'c_offset'
        elif cal['name'] == 'CC_tcarray':
            cname = 'c_delta_t'
        elif cal['name'] == 'CC_taarray':
            cname = 'a_delta_t'
        elif cal['name'] == 'CC_tcal':
            cname = 'tcal'
        elif cal['name'] == 'CC_awlngth':
            cname = 'a_wavelength'
        elif cal['name'] == 'CC_cwlngth':
            cname = 'c_wavelength'
        elif cal['name'] == 'CC_tbins':
            cname = 'tbins'
        else:
            raise ValueError(f'Unknown calibration value name: {cal["name"]}')
        subcals = cal['calData']
        matching_cals = {}
        for subcal in subcals:
            source = subcal['dataSource']
            dt_str = re.findall(r'_(\d{8})_', source)[0]
            dt = datetime.strptime(dt_str, '%Y%m%d')
            if dt <= start:
                matching_cals[dt] = subcal['value']

        cal_dates = list(matching_cals.keys())
        diffs = [start - cd for cd in cal_dates]
        matching_dt = cal_dates[np.argmin(diffs)]
        cal_data[cname] = matching_cals[matching_dt]

    class Dev:
        # Simulate an ACSDev object.

        a_wavelength = cal_data['a_wavelength']
        c_wavelength = cal_data['c_wavelength']
        tbin = cal_data['tbins']
        a_offset = cal_data['a_offset']
        c_offset = cal_data['c_offset']
        a_delta_t = cal_data['a_delta_t']
        c_delta_t = cal_data['c_delta_t']
        tcal = cal_data['tcal']
        path_length = 0.25

        # func_a_delta_t = interpolate.interp1d(cal_data['tbins'], cal_data['a_delta_t'], axis=1)
        # func_c_delta_t = interpolate.interp1d(cal_data['tbins'], cal_data['c_delta_t'], axis=1)
        func_a_delta_t = make_interp_spline(cal_data['tbins'], cal_data['a_delta_t'], k=1, axis=1)
        func_c_delta_t = make_interp_spline(cal_data['tbins'], cal_data['c_delta_t'], k=1, axis=1)

    return Dev


def get_ooi_optaa_cal(ds: xr.Dataset) -> object:
    """
    Find an OOI OPTAA calibration for a given dataset.

    :param ds: The OOI OPTAA dataset. It must contain data from a single deployment.
        It is recommended to not change the format of the file until after running this function.
    :return: A ACSDev-like object containing the calibration data.
    """

    start = pd.to_datetime(ds.time.min().values)
    asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587/asset'
    params = {'uid': ds.attrs['AssetUniqueID']}
    response = requests.get(asset_url, params=params)
    if response.status_code == requests.codes.ok:
        if 'aintenance' in response.text:  # Check for (m)aintenance message
            raise ConnectionError('OOI API is currently down for maintenance.')
        else:
            data = response.json()
            dev = _build_acpype_dev_obj(data, start)
            return dev


def download_and_load_goldcopy(thredds_fileserver_url: str,
                               save_dir: str = 'ooi_data/') -> xr.Dataset:
    """
    Download and load a dataset from the OOI THREDDS GoldCopy catalog.
    Sometimes data accessed through the OpenDAP URL is incomplete, so downloading is recommended.

    :param thredds_fileserver_url: A THREDDS GoldCopy URL. Must contain fileServer, not dodsC.
    :param save_dir: The directory to save the dataset to.
    :return: The xarray dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(thredds_fileserver_url)
    fp = os.path.join(save_dir, filename)
    if os.path.isfile(fp):
        ds = xr.open_dataset(fp)
        return ds
    else:
        with requests.get(thredds_fileserver_url, stream=True) as req:
            with open(fp, 'wb') as fileobj:
                shutil.copyfileobj(req.raw, fileobj)
        if os.path.isfile(fp):
            ds = xr.open_dataset(fp)
            return ds
        else:
            raise FileNotFoundError
