import warnings

import numpy as np
import pytest
from synth import blurred_edge, expected_mtf50

from camera_iq.mtf import edge_mtf

SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5]
CLEAN_TOL = 0.02     # measured worst case is ~1.2% (angle-dependent bin phase, Hamming window)
NOISE = 0.02         # std, on an edge of contrast 0.8
# Seed-to-seed sd of the MTF50 error at NOISE (30 seeds): 2.3% / 0.9% / 0.5% for sigma 0.5 / 1.5 / 2.5.
NOISY_TOL = {0.5: 0.05, 1.5: 0.03, 2.5: 0.03}


def err(sigma, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a healthy edge must not warn
        r = edge_mtf(blurred_edge(sigma, **kw))
    return r.mtf50 / expected_mtf50(sigma) - 1


def test_expected_formula():
    assert expected_mtf50(1.0) == pytest.approx(0.187391, abs=1e-6)
    # and it really is where exp(-2 pi^2 sigma^2 f^2) crosses 0.5
    assert np.exp(-2 * np.pi ** 2 * 1.7 ** 2 * expected_mtf50(1.7) ** 2) == pytest.approx(0.5)


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("angle", [5, -7, 10])
def test_clean_edge(sigma, angle):
    assert abs(err(sigma, angle_deg=angle)) < CLEAN_TOL


@pytest.mark.parametrize("sigma", [0.5, 1.5, 2.5])
def test_horizontal_edge_matches_vertical(sigma):
    v = edge_mtf(blurred_edge(sigma, 5))
    h = edge_mtf(blurred_edge(sigma, 5, horizontal=True))
    assert h.mtf50 == pytest.approx(v.mtf50, rel=1e-9)
    assert abs(err(sigma, angle_deg=5, horizontal=True)) < CLEAN_TOL


@pytest.mark.parametrize("sigma", [0.5, 1.5, 2.5])
def test_flipped_polarity_matches(sigma):
    a = edge_mtf(blurred_edge(sigma, 5))
    b = edge_mtf(blurred_edge(sigma, 5, flip=True))
    assert b.mtf50 == pytest.approx(a.mtf50, rel=1e-9)
    assert abs(err(sigma, angle_deg=5, flip=True)) < CLEAN_TOL


@pytest.mark.parametrize("sigma", [0.5, 1.5, 2.5])
def test_noisy_edge_fixed_seed(sigma):
    assert abs(err(sigma, angle_deg=5, noise=NOISE, seed=0)) < NOISY_TOL[sigma]


@pytest.mark.parametrize("sigma", [0.5, 1.5, 2.5])
def test_noise_adds_no_systematic_bias(sigma):
    errs = [err(sigma, angle_deg=5, noise=NOISE, seed=k) for k in range(30)]
    assert abs(np.mean(errs)) < 0.015


def test_reported_angle():
    for a in (5, -7, 10):
        assert edge_mtf(blurred_edge(1.0, a)).angle_deg == pytest.approx(a, abs=0.01)


def test_mtf_curve_shape():
    r = edge_mtf(blurred_edge(1.0, 5))
    assert r.mtf[0] == pytest.approx(1.0)
    truth = np.exp(-2 * np.pi ** 2 * r.freq ** 2)
    sel = r.freq <= 0.5     # up to Nyquist
    assert np.abs(r.mtf[sel] - truth[sel]).max() < 0.02


@pytest.mark.parametrize("angle", [0.0, 0.5, 0.9])
def test_warns_on_shallow_angle(angle):
    with pytest.warns(UserWarning, match="edge angle"):
        edge_mtf(blurred_edge(1.5, angle))


def test_no_warning_at_1_5_degrees():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        edge_mtf(blurred_edge(1.5, 1.5))
