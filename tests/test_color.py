from pathlib import Path

import numpy as np
import pytest

from camera_iq.color import (WHITE_D65, ciede2000, linear_to_srgb, linear_to_xyz,
                             srgb_to_lab, srgb_to_linear, xyz_to_lab)

SHARMA = np.loadtxt(Path(__file__).parent / "data" / "sharma2005.txt")


def test_sharma_has_34_pairs():
    assert SHARMA.shape == (34, 7)


@pytest.mark.parametrize("row", SHARMA, ids=lambda r: f"{r[0]:.0f},{r[1]:.2f},{r[2]:.2f}|{r[4]:.2f},{r[5]:.2f}")
def test_ciede2000_sharma(row):
    lab1, lab2, expected = row[0:3], row[3:6], row[6]
    assert ciede2000(lab1, lab2) == pytest.approx(expected, abs=5e-5)
    assert ciede2000(lab2, lab1) == pytest.approx(expected, abs=5e-5)  # symmetric


def test_ciede2000_vectorised_matches_scalar():
    d = ciede2000(SHARMA[:, 0:3], SHARMA[:, 3:6])
    assert d.shape == (34,)
    np.testing.assert_allclose(d, SHARMA[:, 6], atol=5e-5)


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 257)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_srgb_linear_known_values():
    assert srgb_to_linear(0.0) == 0.0
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    assert srgb_to_linear(0.04045) == pytest.approx(0.04045 / 12.92)
    assert srgb_to_linear(0.5) == pytest.approx(0.2140411, abs=1e-6)


def test_lab_white_black_grey():
    np.testing.assert_allclose(srgb_to_lab([1, 1, 1]), [100, 0, 0], atol=1e-6)
    np.testing.assert_allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=1e-9)
    grey = srgb_to_lab([128 / 255] * 3)  # 8-bit mid-grey
    assert grey[0] == pytest.approx(53.585, abs=1e-2)
    np.testing.assert_allclose(grey[1:], 0, atol=1e-6)


def test_lab_known_primaries():
    # Widely published sRGB/D65 values (e.g. Bruce Lindbloom, brucelindbloom.com).
    np.testing.assert_allclose(srgb_to_lab([1, 0, 0]), [53.24, 80.09, 67.20], atol=0.05)
    np.testing.assert_allclose(srgb_to_lab([0, 1, 0]), [87.73, -86.18, 83.18], atol=0.05)
    np.testing.assert_allclose(srgb_to_lab([0, 0, 1]), [32.30, 79.19, -107.86], atol=0.05)


def test_xyz_white_point():
    np.testing.assert_allclose(linear_to_xyz([1, 1, 1]), WHITE_D65)
    assert WHITE_D65 == pytest.approx([0.95047, 1.0, 1.08883], abs=1e-4)
    np.testing.assert_allclose(xyz_to_lab(WHITE_D65), [100, 0, 0], atol=1e-9)


def test_lab_linear_segment_continuity():
    # Both branches of f(t) must agree at the threshold.
    t = (6 / 29) ** 3
    lo = xyz_to_lab(WHITE_D65 * (t * (1 - 1e-9)))
    hi = xyz_to_lab(WHITE_D65 * (t * (1 + 1e-9)))
    np.testing.assert_allclose(lo, hi, atol=1e-6)
