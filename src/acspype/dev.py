"""
This module contains functions and classes for parsing and accessing data from ACS .dev files.
Currently, only structure version 3 is supported.
"""

from datetime import datetime
from dateutil import parser
import numpy as np
import re
import xarray as xr
# from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

from .core import NUM_PAT


class ACSDev:
    """
    A class for parsing ACS .dev files and putting them into a format that is easier to work with for larger or
    multiple file datasets.

    Generally, users will not call individual functions, but rather use the class to obtain attributes, which are
    created at class instantiation or convert the data to an xarray dataset using the to_xarray function.
    """

    def __init__(self, filepath: str) -> None:
        """
        Run the following functions at instantiation to parse the .dev file and store the info as class attributes.

        :param filepath: The filepath to the .dev file.
        :return: None
        """

        self._filepath = filepath
        self.__read_dev()
        self.__parse_metadata()
        self.__parse_tbins()
        self.__parse_offsets()
        # self.__build_interp_funcs()
        # self.__check_parse()

    def __read_dev(self) -> None:
        """
        Import the .dev file as a text file.
        The file contents are stored as a list of strings in the class attribute self._content.

        :return: None
        """

        with open(self._filepath, "r") as _file:
            self._content = _file.readlines()

    def __parse_metadata(self) -> None:
        """
        Parse the .dev file for individual sensor metadata.
        Sensor specific metadata are stored as class attributes.

        :return: None
        """

        metadata_lines = [
            line for line in self._content if "C and A offset" not in line
        ]
        for line in metadata_lines:
            if "ACS Meter" in line:
                self.sensor_type = re.findall("(.*?)\n", line)[0].replace("\t", "")
            elif "Serial" in line:
                self.sn_hexdec = re.findall("(.*?)\t", line)[0]
                self.serial_number = "ACS-" + str(int(self.sn_hexdec[-6:], 16)).zfill(5)
            elif "structure version" in line:
                self.structure_version = int(re.findall(f"({NUM_PAT})\t", line)[0])
            elif "tcal" in line or "Tcal" in line:
                try:
                    self.tcal, self.ical = [
                        float(v) for v in re.findall(f"[:=]\\s?({NUM_PAT})\\s?", line)
                    ]
                except:
                    self.tcal, self.ical = [
                        float(v) for v in re.findall(f"\\s?({NUM_PAT})", line)
                    ][:2]

                if "file" in line and "on" in line and "files" not in line:
                    cal_date_str = re.findall("\\s+file\\s+on\\s?(.*?)[.\\s\\n]", line)[
                        0
                    ].replace(" ", "")
                if "files" in line:
                    cal_date_str = re.findall(
                        "\\s+file[s]?\\s+on\\s?(.*?)[.\\s\\n]", line
                    )[0].replace(" ", "")
                elif "file" not in line and "on" in line:
                    cal_date_str = re.findall("\\s+on\\s?(.*?)[.\\s\\n]", line)[
                        0
                    ].replace(" ", "")
                elif "file" in line and "on" not in line:
                    cal_date_str = re.findall("\\s+file\\s?(.*?)[.\\s\\n]", line)[
                        0
                    ].replace(" ", "")
                cal_date_str = cal_date_str.replace("\t", "")
                cal_date_str = cal_date_str.replace("'", "")
                cal_date_str = cal_date_str.replace('"', "")
                self.cal_date = parser.parse(cal_date_str)
                # try:  # Sometimes the file date is entered as yyyy or yy. This should handle both cases.
                #     self.cal_date = datetime.strptime(cal_date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
                # except:
                #     self.cal_date = datetime.strptime(cal_date_str, '%m/%d/%y').strftime('%Y-%m-%d')
            elif "Depth calibration" in line:
                (self.depth_offset, self.depth_scale_factor) = [
                    float(v) for v in re.findall(f"({NUM_PAT})", line)
                ]
            elif "Baud" in line:
                self.baudrate = int(re.findall(f"({NUM_PAT})\t", line)[0])
            elif "Path" in line:
                self.path_length = float(re.findall(f"({NUM_PAT})\t", line)[0])
            elif "wavelengths" in line:
                self.num_wavelength = int(re.findall(f"({NUM_PAT})\t", line)[0])
            elif "number of temperature bins" in line:
                self.num_tbin = int(re.findall(f"({NUM_PAT})\t", line)[0])
            elif "maxANoise" in line:
                (
                    self.max_a_noise,
                    self.max_c_noise,
                    self.max_a_nonconform,
                    self.max_c_nonconform,
                    self.max_a_difference,
                    self.max_c_difference,
                    self.min_a_counts,
                    self.min_c_counts,
                    self.min_r_counts,
                    self.max_temp_sd,
                    self.max_depth_sd,
                ) = [float(v) for v in re.findall(f"({NUM_PAT})\t", line)]

    def __parse_tbins(self) -> None:
        """
        Parse the .dev file for temperature bin information.

        :return: None
        """
        try:
            tbin_line = [
                line for line in self._content if "; temperature bins" in line
            ][0]
        except:
            tbin_line = [
                line for line in self._content if ";\ttemperature\tbins" in line
            ][0]

        tbins = tbin_line.split("\t")
        tbins = [v for v in tbins if v]  # Toss empty strings.
        tbins = [v for v in tbins if v != "\n"]  # Toss newline characters.
        tbins = [v for v in tbins if v != ";"]  # Toss newline characters.
        tbins = [v for v in tbins if v != "temperature"]  # Toss newline characters.
        tbins = [v for v in tbins if v != "bins"]  # Toss newline characters.
        self.tbin = np.array(
            [float(v) for v in tbins if "temperature bins" not in v]
        )  # Convert to float.

    def __parse_offsets(self) -> None:
        """
        Parse the .dev file for a and c offsets. Data are saved as class attributes for access at a later time.

        :return: None
        """

        offset_lines = [line for line in self._content if "C and A offset" in line]
        if len(offset_lines) == 0:  # That means someone used tabs instead of spaces.
            offset_lines = [
                line for line in self._content if "offset" in line and "cal" not in line
            ]

        # Create holder arrays to loop over and append data to.
        c_wvls = []
        a_wvls = []
        c_offs = []
        a_offs = []
        c_deltas = []
        a_deltas = []
        wavelength_color_schemes = []

        for line in offset_lines:
            try:
                offsets, c_delta, a_delta = line.split("\t\t")[:-1]
                c_wvl, a_wvl, wvl_color, c_off, a_off = offsets.split("\t")
            except:
                if "\t\t" in line:
                    try:
                        offsets, deltas = line.split("\t\t")[:-1]
                    except:
                        offsets, deltas = line.split("\t\t")
                    c_wvl, a_wvl, wvl_color, c_off, a_off = offsets.split("\t")
                    deltas_list = deltas.split("\t")
                    c_delta = deltas_list[: int(len(deltas_list) / 2)]
                    a_delta = deltas_list[int(len(deltas_list) / 2) :]
                else:
                    sep = line.split("\t")
                    c_wvl = sep[0]
                    a_wvl = sep[1]
                    wvl_color = sep[2]
                    c_off = sep[3]
                    a_off = sep[4]
                    c_delta = sep[5 : len(self.tbin)]
                    a_delta = sep[5 + len(self.tbin) : 5 + 2 * len(self.tbin)]

                    # Convert strings to proper pythonic datatypes.
            c_wvl = float(c_wvl.replace("C", ""))
            a_wvl = float(a_wvl.replace("A", ""))
            c_off = float(c_off)
            a_off = float(a_off)
            if isinstance(a_delta, str) and isinstance(c_delta, str):
                c_delta = [float(v) for v in c_delta.split("\t")]
                a_delta = [float(v) for v in a_delta.split("\t")]
            elif isinstance(a_delta, list) and isinstance(c_delta, list):
                c_delta = [
                    float(v)
                    for v in c_delta
                    if v
                    not in [
                        ";",
                        "C",
                        "and",
                        "A",
                        '"offset,"',
                        "temperature",
                        "correction",
                        "info\n",
                    ]
                ]
                a_delta = [
                    float(v)
                    for v in a_delta
                    if v
                    not in [
                        ";",
                        "C",
                        "and",
                        "A",
                        '"offset,"',
                        "temperature",
                        "correction",
                        "info\n",
                    ]
                ]

            # Append files to holder arrays.
            c_wvls.append(c_wvl)
            a_wvls.append(a_wvl)
            c_offs.append(c_off)
            a_offs.append(a_off)
            c_deltas.append(c_delta)
            a_deltas.append(a_delta)
            wavelength_color_schemes.append(wvl_color)

        # Convert holder arrays to numpy arrays.
        self.c_wavelength = np.array(c_wvls)
        self.a_wavelength = np.array(a_wvls)
        self.c_offset = np.array(c_offs)
        self.a_offset = np.array(a_offs)
        self.c_delta_t = np.array(c_deltas)
        self.a_delta_t = np.array(a_deltas)
        self.wavelength_color_schemes = wavelength_color_schemes

    def __build_interp_funcs(self) -> None:
        """
        Build interpolation functions for the a and c delta_t values and store as class attributes.

        Note: In SciPy v 1.15.0, scipy.interpolate.interp1d has become legacy and it is recommended that other code
        be used. 1st degree make_interp_spline has been implemented as a replacement. Technically, legacy code is not
        considered deprecated, so users could overwrite the function class attributes with interp1d if preferred.

        :return: None
        """
        self.func_a_delta_t = interp1d(self.tbin, self.a_delta_t, axis=1)
        self.func_c_delta_t = interp1d(self.tbin, self.c_delta_t, axis=1)
        self.delta_t_interp_method = "scipy.interpolate.interp1d"

        # self.func_a_delta_t = make_interp_spline(self.tbin, self.a_delta_t, k=1, axis=1)
        # self.func_c_delta_t = make_interp_spline(self.tbin, self.c_delta_t, k=1, axis=1)
        # self.delta_t_interp_method = 'scipy.interpolate.make_interp_spline'

    def __check_parse(self) -> None:
        """
        Verify that the shape of the data is as expected.

        :return: None
        """

        if len(self.a_wavelength) != self.num_wavelength:
            raise ValueError(
                "Mismatch between number of wavelengths extracted for A and expected from file."
                "Please verify the .dev file integrity."
            )
        if len(self.c_wavelength) != self.num_wavelength:
            raise ValueError(
                "Mismatch between number of wavelengths extracted for C and expected from file."
                "Please verify the .dev file integrity."
            )
        if len(self.c_wavelength) != len(self.a_wavelength):
            raise ValueError(
                "Mismatch between number of wavelengths extracted for A and C."
                "Please verify the .dev file integrity."
            )
        if np.array(self.a_delta_t).shape != (len(self.a_wavelength), self.num_tbin):
            raise ValueError(
                "Mismatch between length of A wavelengths and number of temperature bins."
                "Please verify the .dev file integrity."
            )
        if np.array(self.c_delta_t).shape != (len(self.a_wavelength), self.num_tbin):
            raise ValueError(
                "Mismatch between length of C wavelengths and number of temperature bins."
                "Please verify the .dev file integrity."
            )

    def to_xarray(self) -> xr.Dataset:
        """
        Convert the parsed .dev file files to an xarray dataset

        :return: An appropriately dimensioned xarray dataset containing device file files.
        """
        ds = xr.Dataset()
        ds = ds.assign_coords(
            {
                "a_wavelength": self.a_wavelength,
                "c_wavelength": self.c_wavelength,
                "temperature_bin": self.tbin,
            }
        )

        ds["a_offset"] = (["a_wavelength"], self.a_offset)
        ds["a_delta_t"] = (["a_wavelength", "temperature_bin"], self.a_delta_t)

        ds["c_offset"] = (["c_wavelength"], np.array(self.c_offset))
        ds["c_delta_t"] = (["c_wavelength", "temperature_bin"], self.c_delta_t)

        ds.attrs["device_filepath"] = self._filepath
        ds.attrs["sensor_type"] = self.sensor_type
        ds.attrs["serial_number_hexdec"] = self.sn_hexdec
        ds.attrs["serial_number"] = self.serial_number
        ds.attrs["device_file_structure_version"] = self.structure_version
        ds.attrs["tcal"] = self.tcal
        ds.attrs["ical"] = self.ical
        ds.attrs["calibration_date"] = self.cal_date
        ds.attrs["depth_offset"] = self.depth_offset
        ds.attrs["depth_scale_factor"] = self.depth_scale_factor
        ds.attrs["baudrate"] = self.baudrate
        ds.attrs["path_length"] = self.path_length
        ds.attrs["number_of_wavelength_bins"] = self.num_wavelength
        ds.attrs["number_of_temperature_bins"] = self.num_tbin
        ds.attrs["max_a_noise"] = self.max_a_noise
        ds.attrs["max_c_noise"] = self.max_c_noise
        ds.attrs["max_a_nonconform"] = self.max_a_nonconform
        ds.attrs["max_c_nonconform"] = self.max_c_nonconform
        ds.attrs["max_a_difference"] = self.max_a_difference
        ds.attrs["max_c_difference"] = self.max_c_difference
        ds.attrs["min_a_counts"] = self.min_a_counts
        ds.attrs["min_c_counts"] = self.min_c_counts
        ds.attrs["min_r_counts"] = self.min_r_counts
        ds.attrs["max_temp_sd"] = self.max_temp_sd
        ds.attrs["max_depth_sd"] = self.max_depth_sd
        return ds
