"""Colour accuracy, white balance and noise measured on the 24-patch chart (linear light in)."""
from dataclasses import dataclass

import numpy as np

from .charts import MID_GREY, PATCH_SRGB, WB_PATCHES, colour_chart, patch_polygons, polygon_mask
from .color import ciede2000, linear_to_xyz, srgb_to_lab, srgb_to_linear, xyz_to_lab

LUMA = np.array([0.2126729, 0.7151522, 0.0721750])
INNER = 0.6      # central fraction (per side) of each patch that is measured
CHART = colour_chart()


def _masks(linear, H, inner):
    return [polygon_mask(linear.shape, p) for p in patch_polygons(CHART, H, inner)]


@dataclass
class ColourResult:
    gain: float             # single exposure gain applied to the measured patches
    patch_rgb: np.ndarray   # (24, 3) mean linear RGB, before gain
    delta_e: np.ndarray     # (24,) CIEDE2000 after gain
    mean_delta_e: float
    wb_error: float         # mean chroma sqrt(a*^2 + b*^2) of the four middle neutrals, after gain
    neutral_chroma: np.ndarray


def colour_accuracy(linear, H, inner=INNER):
    """`linear`: (h, w, 3) linear-light capture; `H`: chart units -> image pixels."""
    rgb = np.array([linear[m].mean(axis=0) for m in _masks(linear, H, inner)])
    ref = srgb_to_linear(PATCH_SRGB)
    mid = list(MID_GREY)
    gain = float((ref[mid] * rgb[mid]).sum() / (rgb[mid] ** 2).sum())    # least squares, one scalar
    lab = xyz_to_lab(linear_to_xyz(rgb * gain))
    de = np.asarray(ciede2000(lab, srgb_to_lab(PATCH_SRGB)))
    chroma = np.hypot(lab[list(WB_PATCHES), 1], lab[list(WB_PATCHES), 2])
    return ColourResult(gain, rgb, de, float(de.mean()), float(chroma.mean()), chroma)


@dataclass
class NoiseResult:
    mean: np.ndarray        # (24, 3) patch means, linear
    sigma: np.ndarray       # (24, 3) residual std after plane removal
    snr_db: np.ndarray      # (24, 3) 20 log10(mean / sigma)
    luma_snr_db: np.ndarray  # (24,) same on Y = luma-weighted linear RGB


def _plane_residual(vals, ys, xs):
    """Residuals of `vals` (n, c) after a least-squares plane a + b*x + c*y per channel."""
    A = np.c_[np.ones(len(xs)), xs - xs.mean(), ys - ys.mean()]
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return vals - A @ coef, A.shape[1]


def noise(linear, H, inner=INNER):
    mean = np.zeros((24, 3))
    sigma = np.zeros((24, 3))
    luma_mean = np.zeros(24)
    luma_sigma = np.zeros(24)
    for i, m in enumerate(_masks(linear, H, inner)):
        ys, xs = np.nonzero(m)
        px = linear[ys, xs].astype(float)
        v = np.c_[px, px @ LUMA]                     # R, G, B, then luma, all plane-detrended alike
        res, k = _plane_residual(v, ys, xs)
        s = np.sqrt((res ** 2).sum(axis=0) / (len(xs) - k))
        mu = v.mean(axis=0)
        mean[i], sigma[i], luma_mean[i], luma_sigma[i] = mu[:3], s[:3], mu[3], s[3]
    with np.errstate(divide="ignore", invalid="ignore"):    # clipped-to-black channels give 0/0 = nan
        return NoiseResult(mean, sigma, 20 * np.log10(mean / sigma), 20 * np.log10(luma_mean / luma_sigma))
