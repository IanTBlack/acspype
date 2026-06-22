import numpy as np
from scipy.interpolate._bsplines import BSpline

from acspype import ACSDev

TEST_FILEPATH = '../dev_tools/test_files/ACS-00011_2022-10-20.dev'


def test():
    dev = ACSDev(TEST_FILEPATH)
    assert len(dev.a_wavelength) == len(dev.c_wavelength)
    assert isinstance(dev.a_offset, np.ndarray)
