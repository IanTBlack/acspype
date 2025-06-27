"""
This module contains functions that are used to process data from the ACS in real-time or as a batch process.

Most functions are intended to be used with scalar values/1D arrays (e.g. data produced in real-time) or
xarray DataArrays (e.g. mass processing of an ACS deployment), or both.
Scalar/1D inputs are intended to be used in the processing of a single packet or spectrum from the ACS.
xr.DataArray implementation is intended to be used in the processing of multiple spectra already in an xarray.Dataset.
For computations that are not immediately builtin to xarray,the xarray.apply_ufunc function is used to
apply the function that is denoted with a prepended underscore.

It is strongly recommended that your xarray.Dataset contain the coordinates/dimensions of
    time, a_wavelength, c_wavelength

For a list of recommended coordinate/variable names for ACS data products, please see
https://github.com/IanTBlack/acspype/blob/main/info/NAMING_CONVENTIONS.md
"""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
import xarray as xr


def _convert_sn_hexdec(sn_int: int) -> str:
    """
    Converts an integer representing an ACS serial number to the hexadecimal representation.
    Application to an array of serial numbers requires this function to be vectorized. See convert_sn_hexdec.

    :param sn_int: An integer representing the ACS serial number.
    :return: The hexadecimal representation of the serial number, not really as readable as the integer representation.
    """

    sn_hexdec = hex(sn_int)
    return sn_hexdec


def convert_sn_hexdec(sn_int: int | xr.DataArray) -> str | xr.DataArray:
    """
    Converts an integer representing an ACS serial number to the hexadecimal representation.
    The function vectorizes _convert_sn_hexdec for xr.DataArray inputs.

    :param sn_int: An integer or xr.DataArray of integers representing the ACS serial number.
    :return: The hexadecimal representation of the serial number, not really as readable as the integer representation.
    """

    if not isinstance(sn_int, xr.DataArray):
        sn_hexdec = _convert_sn_hexdec(sn_int)
    else:
        sn_hexdec = xr.apply_ufunc(_convert_sn_hexdec, sn_int,
                                   input_core_dims=[[]],
                                   output_core_dims=[[]],
                                   vectorize=True)

        # Assign attributes if output is a xr.DataArray.
        sn_hexdec.attrs['ancillary_variables'] = sn_int.name
    return sn_hexdec


def _convert_sn_str(sn_int: int) -> str:
    """
    Converts an integer representing an ACS serial number to a string representation.
    Application to an array of serial numbers requires this function to be vectorized. See convert_sn_str.

    :param sn_int: An integer representing the ACS serial number.
    :return: The string representation of the serial number, zero-padded for consistent string length into the future.
    :note: The string serial number should match that of the serial number sticker on the ACS.
    """

    sn_hexdec = convert_sn_hexdec(sn_int)
    sn_str = 'ACS-' + str(int(sn_hexdec[-6:], 16)).zfill(5)  # sn string becomes ACS-XXXXX. Zero padded to 5 digits.
    return sn_str


def convert_sn_str(sn_int: int | xr.DataArray) -> str | xr.DataArray:
    """
    Converts an integer representing an ACS serial number to a string representation.
    This function vectorizes _convert_sn_str for xr.DataArray inputs.

    :param sn_int: An integer or xr.DataArray of integers representing the ACS serial number.
    :return: The string representation of the serial number.
    """

    if not isinstance(sn_int, xr.DataArray):
        return _convert_sn_str(sn_int)
    else:
        sn_str = xr.apply_ufunc(_convert_sn_str, sn_int,
                                input_core_dims=[[]],
                                output_core_dims=[[]],
                                vectorize=True)

        # Assign attributes if output is an xr.DataArray.
        sn_str.attrs['ancillary_variables'] = sn_int.name
        return sn_str


def compute_internal_temperature(counts: int | xr.DataArray) -> float | xr.DataArray:
    """
    Compute the internal temperature of the ACS from the counts of the internal temperature sensor.

    :param counts: Values of the internal temperature sensor, in counts.
    :return: A float value if the input was a scalar counts value or an xr.DataArray of the internal temperature
        if input as an xr.DataArray. Units of return are degrees Celsius.
    """

    a = 0.00093135
    b = 0.000221631
    c = 0.000000125741
    d = 273.15
    volts = 5 * counts / 65535
    resistance = 10000 * volts / (4.516 - volts)
    internal_temperature = 1 / (a + b * np.log(resistance) + c * np.log(resistance) ** 3) - d
    if not isinstance(counts, xr.DataArray):
        internal_temperature = float(internal_temperature)

    # Assign attributes if output is an xr.DataArray.
    if isinstance(internal_temperature, xr.DataArray):
        internal_temperature.attrs['ancillary_variables'] = counts.name
    return internal_temperature


def compute_external_temperature(counts: int | xr.DataArray) -> float | xr.DataArray:
    """
    Compute the external temperature of the ACS from the counts of the external temperature sensor.
    Not used in processing if the sensor is not placed in-situ or if there is a better _source that is indicative
    of the temperature of the water sampled by the ACS.

    :param counts: The counts of the external temperature sensor.
    :return: The external temperature in degrees Celsius.
    """

    a = -7.1023317e-13
    b = 7.09341920e-08
    c = -3.87065673e-03
    d = 95.8241397
    external_temperature = a * counts ** 3 + b * counts ** 2 + c * counts + d
    if not isinstance(counts, xr.DataArray):
        external_temperature = float(external_temperature)

    # Assign attributes if output is an xr.DataArray.
    if isinstance(external_temperature, xr.DataArray):
        external_temperature.attrs['ancillary_variables'] = counts.name
    return external_temperature


def compute_uncorrected(signal_counts: tuple[int, ...] | NDArray[float] | xr.DataArray,
                        reference_counts: tuple[int, ...] | NDArray[float] | xr.DataArray,
                        path_length: float = 0.25) -> NDArray[float] | xr.DataArray:
    """
    Compute the uncorrected coefficient from the signal and reference counts.

    :param signal_counts: The absorption or attenuation channel signal counts.
    :param reference_counts: The absorption or attenuation channel reference counts.
    :param path_length: The path length of the ACS in meters. Default is 0.25m, but it is recommended to use the path
        length in the device file.
    :return: Uncorrected absorption or attenuation coefficient in m^-1.
    """

    if not isinstance(signal_counts, xr.DataArray):
        signal_counts = np.array(signal_counts)
        reference_counts = np.array(reference_counts)
    uncorr = (1 / path_length) * np.log(signal_counts / reference_counts)

    # Assign attributes if output is an xr.DataArray.
    if isinstance(uncorr, xr.DataArray):
        uncorr.attrs['ancillary_variables'] = [signal_counts.name, reference_counts.name]
        uncorr.attrs['path_length'] = path_length
    return uncorr


def compute_measured(uncorrected: NDArray[float] | xr.DataArray,
                     internal_temperature: float | xr.DataArray,
                     offset: NDArray[float],
                     func_delta_t: Callable) -> NDArray[float] | xr.DataArray:
    """
    Compute the measured absorption or attenuation coefficient from the uncorrected coefficient
    and the internal temperature.

    :param uncorrected: The uncorrected absorption or attenuation coefficient.
        Typically computed via compute_uncorrected().
    :param internal_temperature: The internal temperature of the ACS in degrees Celsius.
        Typically computed via compute_internal_temperature().
    :param offset: The coefficient offset from the device file.
    :param func_delta_t: The interpolation function for correcting for internal temperature variation.
        This function is built by default when using an ACSDev object.
    :return: Measured absorption or attenuation coefficient in m^-1, corrected for internal temperature variation.
    """

    if not isinstance(uncorrected, xr.DataArray):
        measured = (offset - uncorrected) - func_delta_t(internal_temperature)
    else:
        # Something weird happens when using BSpline, so the transposition is necessary to get the right shape.
        measured = (offset - uncorrected) - func_delta_t(internal_temperature).T

    if isinstance(measured, xr.DataArray):
        # Assign attributes if output is an xr.DataArray.

        measured.attrs['ancillary_variables'] = [uncorrected.name, internal_temperature.name]
        measured.attrs['func_delta_t'] = 'scipy.interpolate.make_interp_spline'
        measured.attrs['func_delta_t_k_value'] = '1'
    return measured


def find_discontinuity_index(a_wavelength: NDArray[float],
                             c_wavelength: NDArray[float],
                             min_wvl: float = 535.0, max_wvl: float = 600.0) -> int:
    """
    Find the wavelength index of the discontinuity in the absorption and attenuation spectra. The discontinuity is
    in the range of 535-600nm.

    :param a_wavelength: The absorption wavelengths of the ACS.
    :param c_wavelength: The attenuation wavelengths of the ACS.
    :param min_wvl: Default is 535 nm. Can be adjusted if the discontinuity is not in the expected range.
    :param max_wvl: Default is 600 nm. Can be adjusted if the discontinuity is not in the expected range.
    :return: The index of the discontinuity in the absorption and attenuation spectra.
    """

    a_wvl = np.where((a_wavelength < min_wvl) | (a_wavelength > max_wvl), np.nan, a_wavelength)
    c_wvl = np.where((c_wavelength < min_wvl) | (c_wavelength > max_wvl), np.nan, c_wavelength)
    discontinuity_index = int(np.nanargmin(np.diff(a_wvl) + np.diff(c_wvl)))
    return discontinuity_index


def _compute_discontinuity_offset(measured: NDArray[float],
                                  wavelength: NDArray[float],
                                  disc_idx: int) -> float:
    """
    Compute the discontinuity offset for a measured coefficient.
    compute_discontinuity_offset provides vectorization for this function.

    :param measured: Absorption or attenuation coefficient.
    :param wavelength: The wavelengths of the absorption or attenuation spectra.
    :param disc_idx: The index of the discontinuity in the absorption and attenuation spectra.
    :return: The offset of the discontinuity in the absorption and attenuation spectra as an integer.
    """
    x = wavelength[disc_idx - 2:disc_idx + 1]
    y = measured[disc_idx - 2:disc_idx + 1]
    if np.any(np.isinf(y)) or np.any(np.isnan(y)):
        return np.nan
    else:
        cubic_spline = CubicSpline(x, y)
        interp = cubic_spline(wavelength[disc_idx + 1], extrapolate=True)
        offset = float(np.round((interp - measured[disc_idx + 1]), 5))
        return offset


def compute_discontinuity_offset(measured: NDArray[float] | xr.DataArray,
                                 wavelength: NDArray[float] | xr.DataArray,
                                 disc_idx: int,
                                 wavelength_dim: str) -> float | xr.DataArray:
    """
    Compute the discontinuity offset for a measured coefficient.

    :param measured: Absorption or attenuation coefficient.
    :param wavelength: The wavelengths of the absorption or attenuation spectra.
    :param disc_idx: The index of the discontinuity in the absorption and attenuation spectra.
    :param wavelength_dim: The name of the wavelength dimension if inputting an xarray DataArray.
    :return: The discontinuity offset of the absorption and attenuation spectra as a float.
    """

    if not isinstance(measured, xr.DataArray):
        disc_off = _compute_discontinuity_offset(measured, wavelength, disc_idx)
    else:
        disc_off = xr.apply_ufunc(_compute_discontinuity_offset, measured,
                                  kwargs={'wavelength': measured[wavelength_dim].values, 'disc_idx': disc_idx},
                                  input_core_dims=[[wavelength_dim]],
                                  output_core_dims=[[]],
                                  vectorize=True)

        # Assign attributes if output is an xr.DataArray.
        disc_off.attrs['ancillary_variables'] = measured.name
        disc_off.attrs['wavelength_dimension'] = wavelength_dim
        disc_off.attrs['discontinuity_index'] = disc_idx
    return disc_off


def _apply_discontinuity_offset(measured: NDArray[float],
                                disc_off: float,
                                disc_idx: int) -> NDArray[float]:
    """
    Apply a discontinuity offset to a measured coefficient.
    Note: This function is not vectorized. The input measured array is copied to prevent memory conflicts.

    :param measured: Measured absorption or attenuation coefficient.
    :param disc_off: The discontinuity offset of the absorption and attenuation spectra.
    :param disc_idx: The index of the discontinuity in the absorption and attenuation spectra.
    :return: The spectrum with the offset applied.
    """

    _measured = np.copy(measured)
    _measured[disc_idx + 1:] = _measured[disc_idx + 1:] + disc_off
    return _measured


def apply_discontinuity_offset(measured: NDArray[float] | xr.DataArray,
                               disc_off: float | NDArray[float],
                               disc_idx: int, wavelength_dim: str) -> NDArray[float] | xr.DataArray:
    """
    Apply a discontinuity offset to a measured coefficient. Vectorized version of _apply_discontinuity_offset, where
    applicable.

    :param measured: Measured absorption or attenuation coefficient.
    :param disc_off: The discontinuity offset of the absorption and attenuation spectra.
    :param disc_idx: The index of the discontinuity in the absorption and attenuation spectra.
    :param wavelength_dim: The name of the wavelength dimension if inputting an xarray DataArray.
    :return: The spectrum with the offset applied.
    """
    if not isinstance(measured, xr.DataArray):
        disc_applied = _apply_discontinuity_offset(measured, disc_off, disc_idx)
    else:
        disc_applied = xr.apply_ufunc(_apply_discontinuity_offset, measured, disc_off,
                                      kwargs={'disc_idx': disc_idx},
                                      input_core_dims=[[wavelength_dim], []],
                                      output_core_dims=[[wavelength_dim]],
                                      vectorize=True)

        # Assign attributes if output is an xr.DataArray.
        disc_applied.attrs['ancillary_variables'] = measured.name
        disc_applied.attrs['wavelength_dimension'] = wavelength_dim
        disc_applied.attrs['discontinuity_index'] = disc_idx
        disc_applied.attrs['discontinuity_offset_applied'] = 'True'
    return disc_applied


def discontinuity_correction(measured: xr.DataArray,
                             discontinuity_index: int,
                             wavelength_dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the discontinuity offset and apply it to the measured coefficient.
    NOTE: This function only works with xarray DataArrays.

    :param measured: The measured absorption or attenuation coefficient.
    :param discontinuity_index: The index of the discontinuity in the absorption and attenuation spectra.
    :param wavelength_dim: The wavelength dimension of the measured coefficient if inputting an xarray DataArray.
    :return: A tuple of xr.DataArrays (corrected measured coefficient, offset)
    """

    disc_offset = compute_discontinuity_offset(measured, measured[wavelength_dim].values,
                                               discontinuity_index, wavelength_dim)
    disc_applied = apply_discontinuity_offset(measured, disc_offset,
                                              discontinuity_index, wavelength_dim)

    disc_applied.attrs['discontinuity_corrected'] = 'True'
    return disc_applied, disc_offset


def ts_correction(measured: NDArray[float] | xr.DataArray,
                  temperature: float | xr.DataArray,
                  salinity: float | xr.DataArray,
                  psi_temperature: NDArray[float] | xr.DataArray,
                  psi_salinity: NDArray[float] | xr.DataArray,
                  tcal: float) -> NDArray[float] | xr.DataArray:
    """
    Correct the measured absorption or attenuation coefficient for temperature and salinity.
    This function works on singletons and xarray DataArrays.

    :param measured: The measured absorption or attenuation coefficient.
    :param temperature: The temperature of the water in degrees Celsius.
    :param salinity: The salinity of the water in PSU.
    :param psi_temperature: The temperature coefficients from ACSTSCor,
        subset by wavelength equivalent to the corresponding measured coefficient wavelengths.
    :param psi_salinity: The temperature coefficients from ACSTSCor,
        subset by wavelength equivalent to the corresponding measured coefficient wavelengths.
    :param tcal: The tcal value from the device file.
    :return: TS-corrected absorption or attenuation coefficient in m^-1.
    """

    if not isinstance(measured, xr.DataArray):
        dt = temperature - tcal
        psi_t = psi_temperature
        psi_s = psi_salinity
        s = salinity
        mts = measured - ((psi_t * dt) + (psi_s * s))
    else:
        dt, psi_t = np.meshgrid(temperature - tcal, psi_temperature)
        s, psi_s = np.meshgrid(salinity, psi_salinity)
        mts = measured - ((psi_t.T * dt.T) + (psi_s.T * s.T))

    if isinstance(mts, xr.DataArray):
        # Assign attributes if output is an xr.DataArray.
        mts.attrs['ancillary_variables'] = [measured.name, temperature.name, salinity.name, psi_temperature.name,
                                            psi_salinity.name]
        mts.attrs['tcal'] = tcal
    return mts


def zero_shift_correction(mts: NDArray[float] | xr.DataArray) -> NDArray[float] | xr.DataArray:
    """
    Shift values between [-0.005, 0] to 0. This function works both for 1D arrays and xarray DataArrays.

    :param mts: The TS-corrected absorption or attenuation coefficient
    :return: The shifted TS-corrected absorption or attenuation coefficient.
    """

    if not isinstance(mts, xr.DataArray):
        mts = np.where((mts >= -0.005) & (mts < 0), 0, mts)
    else:
        mts = mts.where((mts > 0) | (mts <= -0.005), 0)

        # Assign attributes if output is an xr.DataArray.
        mts.attrs['zero_shifted'] = "True"
    return mts


def interpolate_common_wavelengths(ds: xr.Dataset,
                                   a_wavelength_dim: str = 'a_wavelength',
                                   c_wavelength_dim: str = 'c_wavelength',
                                   new_wavelength_dim: str = 'wavelength',
                                   wavelength_range: list or str = 'infer', step: int = 1, ) -> xr.Dataset:
    """
    This function interpolates the absorption and attenuation spectra to a common wavelength range and step size.
    Only works on xarray Datasets.
    Applies interpolation to all variables on the a_wavelength and c_wavelength dimensions.

    :param ds: The input dataset with two wavelength dimensions, one for absorption and the other for attenuation.
    :param a_wavelength_dim: The name of the absorption wavelength dimension.
    :param c_wavelength_dim: The name of the attenuation wavelength dimension.
    :param new_wavelength_dim: The name of the new wavelength dimension to be created.
        Default is 'wavelength', which is the recommended name for the common wavelength dimension.
    :param wavelength_range: The wavelength range to interpolate along.
        Default is 'infer', which will use the minimums and maximums of the absorption and attenuation wavelengths to
        identify a common range for both channels.
    :param step: The wavelength step size to interpolate to. Default is 1 nm.
    :return: A dataset interpolated to the common wavelength range and step size. If there are variables that you do not
        want to interpolate along (e.g. a_signal, a_reference, c_signal, c_reference), it is recommended that you subset
        before interpolation.
    """

    if wavelength_range == 'infer':
        min_wvl = np.ceil(max(ds[a_wavelength_dim].min(), ds[c_wavelength_dim].min()))  # Max of a and c wvl mins.
        max_wvl = np.floor(min(ds[a_wavelength_dim].max(), ds[c_wavelength_dim].max()))  # Min of a and c wvl maxs.
    else:
        min_wvl, max_wvl = wavelength_range
    wvls = np.arange(min_wvl, max_wvl, step)
    cds = ds.interp({a_wavelength_dim: wvls, c_wavelength_dim: wvls})  # Interpolate to step size.
    cds = cds.reset_index([a_wavelength_dim,
                           c_wavelength_dim], drop=True).assign_coords(wavelength=wvls).rename(
        {a_wavelength_dim: new_wavelength_dim,
         c_wavelength_dim: new_wavelength_dim})

    cds.attrs['wavelength_range'] = [float(min_wvl), float(max_wvl)]
    cds.attrs['interpolation_step'] = step
    return cds


def baseline_scattering_correction(a_mts: NDArray[float] | xr.DataArray,
                                   reference_a: float | xr.DataArray) -> NDArray[float] | xr.DataArray:
    """
    Perform baseline scattering correction on the absorption coefficient.
    This is the baseline method from Zaneveld et al. (1994).

    :param a_mts: The TS-corrected absorption coefficient across all wavelengths.
    :param reference_a: The TS-corrected absorption coefficient at a reference wavelength.
    :return: The baseline scattering-corrected absorption coefficient.
    """

    scatcorr = a_mts - reference_a

    if isinstance(scatcorr, xr.DataArray):  # Assign attributes if output is an xarray.DataArray.
        scatcorr.attrs['ancillary_variables'] = a_mts.name
        scatcorr.attrs['scattering_correction_method'] = 'Baseline, Zaneveld et al. (1994)'
        scatcorr.attrs['scattering_correction_method_doi'] = 'https://doi.org/10.1117/12.190095'
    return scatcorr


def fixed_scattering_correction(a_mts: NDArray[float] | xr.DataArray,
                                c_mts: NDArray[float] | xr.DataArray,
                                epsilon: float = 0.14) -> NDArray[float] | xr.DataArray:
    """
    Perform fixed scattering correction on the absorption coefficient.
    This is the fixed correction method from Pegau et al. 2003.

    :param a_mts: TS-corrected absorption.
    :param c_mts: TS-corrected attenuation.
    :param epsilon: A default value of 0.14 is used for biological particle dominant waters.
        A value of 0.18 is recommended for waters dominated by sediments.
    :return: Fixed scattering corrected absorption coefficient.
    """

    scatcorr = a_mts - epsilon * (c_mts - a_mts)

    if isinstance(scatcorr, xr.DataArray):  # Assign attributes if output is an xarray.DataArray.
        scatcorr.attrs['ancillary_variables'] = [a_mts.name, c_mts.name]
        scatcorr.attrs['epsilon'] = epsilon
        scatcorr.attrs['scattering_correction_method'] = 'Fixed, Pegau et al. (2003)'
        scatcorr.attrs['scattering_correction_method_doi'] = ('https://ntrs.nasa.gov/api/citations/'
                                                              '20030093642/downloads/20030093642.pdf')
    return scatcorr


def proportional_scattering_correction(a_mts: NDArray[float] | xr.DataArray,
                                       c_mts: NDArray[float] | xr.DataArray,
                                       reference_a: NDArray[float] | xr.DataArray,
                                       reference_c: NDArray[float] | xr.DataArray) -> NDArray[float] | xr.DataArray:
    """
    Perform proportional scattering correction on the absorption coefficient.
    This is the proportional method from Zaneveld et al. (1994).

    :param a_mts: TS-corrected absorption.
    :param c_mts: TS-corrected attenuation.
    :param reference_a: The TS-corrected absorption coefficient at a reference wavelength. Historically 715nm.
    :param reference_c: The TS-corrected attenuation coefficient at a reference wavelength.  Historically 715nm.
    :return: Proportional scattering corrected absorption coefficient.
    """

    scatcorr = a_mts - ((reference_a / (reference_c - reference_a)) * (c_mts - a_mts))

    if isinstance(scatcorr, xr.DataArray):  # Assign attributes if output is an xarray.DataArray.
        scatcorr.attrs['ancillary_variables'] = [a_mts.name, c_mts.name]
        scatcorr.attrs['scattering_correction_method'] = 'Proportional, Zaneveld et al. (1994)'
        scatcorr.attrs['scattering_correction_method_doi'] = 'https://doi.org/10.1117/12.190095'
    return scatcorr


def baseline_plus_scattering_correction(a_mts: xr.DataArray,
                                        a_mts_715: xr.DataArray) -> xr.DataArray:
    """
    Perform baseline plus scattering correction for the absorption coefficient.
    This method originates from Rottgers et al. (2013). https://doi.org/10.1016/j.mio.2013.11.001
    True absorption at 715nm is empirically determined in Equation 5.

    :param a_mts: The TS-corrected absorption coefficient.
    :param a_mts_715: The absorption at the reference wavelength. Historically, 715nm is used.
    :return: Baseline plus scattering corrected absorption coefficient.
    """

    true_a_715 = 0.212 * np.pow(a_mts_715, 1.135)  # Equation 5 from Rottgers et al. (2013)
    scatcorr = a_mts - (a_mts_715 - true_a_715)  # Equation 3 from Rottgers et al. (2013)

    if isinstance(scatcorr, xr.DataArray):  # Assign attributes if output is an xarray.DataArray.
        scatcorr.attrs['ancillary_variables'] = a_mts.name
        scatcorr.attrs['scattering_correction_method'] = 'Baseline+, Rottgers et al. (2013)'
        scatcorr.attrs['scattering_correction_method_doi'] = 'https://doi.org/10.1016/j.mio.2013.11.001'
    return scatcorr


def proportional_plus_scattering_correction(a_mts: xr.DataArray,
                                            c_mts: xr.DataArray,
                                            a_mts_715: xr.DataArray,
                                            c_mts_715: xr.DataArray,
                                            e_c: float = 0.56) -> xr.DataArray:
    """
    Perform proportional plus scattering correction for the absorption coefficient.
    This method originates from Rottgers et al. (2013). https://doi.org/10.1016/j.mio.2013.11.001
    True absorption at 715nm is empirically determined in Equation 5.  The value e_c represents attenuation correction.

    This function only works with xarray DataArrays that share a common 'wavelength' dimension, thus the dataset
    must be linearly interpolated to a common wavelength range and bin size.

    :param a_mts: The TS-corrected absorption coefficient.
    :param c_mts: The TS-corrected attenuation coefficient.
    :param a_mts_715: The absorption at the reference wavelength. Historically, 715nm is used.
    :param c_mts_715: The attenuation at the reference wavelength. Historically, 715nm is used.
    :param e_c: The attenuation correction factor. Normally it is 1 (no attenuation correction),
        but Rottgers et al. (2013) suggest a value of 0.56 when running wavelength independent correction (which this
        function does).
    :return: Proportional plus scattering corrected absorption coefficient.
    """

    true_a_715 = 0.212 * np.pow(a_mts_715, 1.135)  # EQ5 from Rottgers et al. (2013)
    scatcorr = a_mts - (a_mts_715 - true_a_715) * (
            ((1 / e_c) * c_mts - a_mts) / ((1 / e_c) * c_mts_715 - true_a_715))  # EQ4

    if isinstance(scatcorr, xr.DataArray):  # Assign attributes if output is an xarray.DataArray.
        scatcorr.attrs['ancillary_variables'] = a_mts.name, c_mts.name
        scatcorr.attrs['reference_wavelength'] = 715
        scatcorr.attrs['scattering_correction_method'] = 'Proportional+, Rottgers et al. (2013)'
        scatcorr.attrs['scattering_correction_method_doi'] = 'https://doi.org/10.1016/j.mio.2013.11.001'
    return scatcorr


# ----------- Experimental and Untested Functions ----------- #

def _estimate_reference_wavelength_index(a_spectra: np.array) -> int | float:
    """
    Estimate the index of the reference wavelength of an absorption spectra. This function finds the first
    wavelength in the last 1/5 wavelength bins that is greater than and nearest to zero.
    If the entire spectra is NaN, then a wvl_idx of NaN is returned.

    :param a_spectra: A 1D array of absorption spectra.
    :return: The index of the reference wavelength.
    """
    num_wvls = len(a_spectra)
    if np.isnan(num_wvls):
        return -999
    elif np.all(np.isnan(a_spectra)):
        return -999
    len_non_reds = len(a_spectra[:int(num_wvls * 4 / 5)])
    reds = a_spectra[int(num_wvls * 4 / 5):]
    reds_pos = np.where(reds <= 0, np.nan, reds)
    if np.all(np.isnan(reds_pos)):
        return -999
    else:
        reds_wvl_idx = np.nanargmin(reds_pos)
        wvl_idx = int(len_non_reds + reds_wvl_idx)
        return wvl_idx


def estimate_reference_wavelength(a_mts: xr.DataArray, wavelength_dim: str) -> float:
    """
    Estimate a reference wavelength for an ND absorption dataset.
    The index closest to zero in the last 1/5 of each spectrum sample is determined. The index with the most repeats
    across the dataset estimated to be the reference wavelength.

    :param a_mts: TS-corrected absorption coefficient.
    :param wavelength_dim: The wavelength dimension name for a_mts.
    :return: The reference wavelength in nm.
    """
    wvl_idxs = xr.apply_ufunc(_estimate_reference_wavelength_index,
                              a_mts,
                              input_core_dims=[[wavelength_dim]],
                              vectorize=True)
    wvl_idxs = wvl_idxs.where(wvl_idxs != -999, drop=True)  # Drop bad estimations.
    idxs, counts = np.unique(wvl_idxs, return_counts=True)
    wvl_idx = idxs[np.nanargmax(counts)]
    if np.isnan(wvl_idx):
        raise ValueError('No reference wavelength found. Please check the input spectra.')
    wvl = float(a_mts[wavelength_dim].values[int(wvl_idx)])
    return wvl
