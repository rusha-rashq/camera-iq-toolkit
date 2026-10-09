"""Synthetic Gaussian-blurred slanted edge with a known MTF50."""
import numpy as np
from scipy.special import erf


def expected_mtf50(sigma):
    """Gaussian PSF: MTF = exp(-2 pi^2 sigma^2 f^2) = 0.5  =>  f = sqrt(ln2 / 2pi^2) / sigma."""
    return np.sqrt(np.log(2) / (2 * np.pi ** 2)) / sigma


def blurred_edge(sigma, angle_deg, shape=(128, 64), lo=0.1, hi=0.9, noise=0.0, seed=0,
                 horizontal=False, flip=False):
    """Edge blurred by a Gaussian of std `sigma` px, sampled at pixel centres (no pixel aperture).
    angle_deg: slant from vertical (x grows with y when positive)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = np.tan(np.radians(angle_deg))
    d = (xx - (w / 2 + t * (yy - h / 2))) * np.cos(np.radians(angle_deg))
    img = lo + (hi - lo) * 0.5 * (1 + erf(d / (sigma * np.sqrt(2))))
    if flip:
        img = lo + hi - img
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img.T if horizontal else img
