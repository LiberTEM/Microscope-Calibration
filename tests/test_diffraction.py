from numpy.testing import assert_equal, assert_almost_equal, assert_allclose


import os
from diffpy.structure import load_structure
import numpy as np

from microscope_calibration.util.diffraction import get_twothetas


def test_twothetas(request):
    # From https://www.globalsino.com/EM/page4171.html
    # d-spacing 111 is 0.2350 nm there
    # 200 kV

    # to pm
    reference_d = 0.2350 * 1000

    reference_au = [  # mrad
        10.68,  # 111, 0.2350 nm
        12.33,  # 200, 0.2035 nm
        17.44,  # 220, 0.1439 nm
        20.45,  # 113, 0.1227 nm
        21.36,  # 222, 0.1175 nm
        24.67,  # 400, 0.1018 nm
        26.88,  # 133, 0.0934 nm
    ]

    cif_path = os.path.join(os.path.dirname(request.path), 'AuEntryWithCollCode163723.cif')

    structure = load_structure(cif_path)

    # Sanity check of structure
    # Reference values for gold
    la = structure.lattice
    # angstrom it seems
    assert_almost_equal(la.a, 4.0709)
    assert_equal(la.a, la.b)
    assert_equal(la.a, la.c)
    assert_equal(la.alpha, 90)
    assert_equal(la.beta, 90)
    assert_equal(la.gamma, 90)

    # spacing for 111
    # https://en.wikipedia.org/wiki/Miller_index#Cubic_structures
    # convert to pm
    d_spacing = la.a * 100 / np.sqrt(3)

    # https://www.jeol.com/words/emterms/20121023.071258.php#gsc.tab=0
    wavelength = 2.5079  # pm
    # both in pm
    # sin(x) approx. x
    expected_111_twotheta = wavelength / d_spacing

    # Account for slight mismatch of d in reference and CIF
    corrected_au = np.array(reference_au) * reference_d / d_spacing

    twothetas = get_twothetas(cif_path, 200000)

    without_zero = twothetas[1:]

    print(without_zero, expected_111_twotheta, reference_au, corrected_au)

    # 111 is the first reflection with intensity
    # get_twothetas rounds to 5 decimals
    assert_almost_equal(without_zero[0], np.round(expected_111_twotheta, decimals=5), decimal=4)

    if len(without_zero) > len(corrected_au):
        without_zero = without_zero[:len(corrected_au)]
    elif len(without_zero) < len(corrected_au):
        corrected_au = corrected_au[:len(without_zero)]

    # convert reference to radian
    assert_allclose(corrected_au / 1000, without_zero, rtol=5e-3)
