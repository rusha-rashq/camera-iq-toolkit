"""Slanted-edge MTF (ISO 12233 e-SFR style) from a linear-light ROI containing one edge."""
import warnings
from dataclasses import dataclass

import numpy as np

BIN = 0.25          # ESF bin width in pixels (4x oversampling)
MAX_FREQ = 1.0      # cycles/pixel; Nyquist is 0.5, results above it are not physical
MIN_ANGLE_DEG = 1.0


@dataclass
class MTFResult:
    freq: np.ndarray    # cycles/pixel
    mtf: np.ndarray     # normalised, mtf[0] == 1
    mtf50: float        # cycles/pixel, nan if the curve never crosses 0.5
    angle_deg: float    # edge slant from the nearest image axis (signed)
    esf_x: np.ndarray   # pixels, perpendicular distance from the edge
    esf: np.ndarray


def _fit_edge(roi, centre_hw=None):
    """Per-row edge centroids -> line x = a + b*y. Returns (a, b)."""
    h, w = roi.shape
    d = np.diff(roi, axis=1)              # rising edge => positive
    pos = np.arange(w - 1) + 0.5
    y = np.arange(h)
    hw = w / 4 if centre_hw is None else centre_hw

    def centroids(window_centre):
        wgt = d if window_centre is None else d * (np.abs(pos[None, :] - window_centre[:, None]) <= hw)
        s = wgt.sum(axis=1)
        return np.where(s > 0, (wgt * pos).sum(axis=1) / np.where(s > 0, s, 1), np.nan), s > 0

    xc, ok = centroids(None)
    a, b = None, None
    for _ in range(3):  # first pass: whole row; then re-centre a window on the fitted line
        keep = ok & np.isfinite(xc)
        a_b = np.polyfit(y[keep], xc[keep], 1)
        res = xc[keep] - np.polyval(a_b, y[keep])
        mad = np.median(np.abs(res)) + 1e-12
        good = np.abs(res) < 4 * 1.4826 * mad
        a_b = np.polyfit(y[keep][good], xc[keep][good], 1)
        b, a = a_b
        xc, ok = centroids(a + b * y)
    return a, b


def edge_mtf(roi):
    """MTF of the edge in `roi` (2D, linear light, one edge, 2-10 deg off an axis ideally)."""
    roi = np.asarray(roi, dtype=float)
    if np.abs(np.diff(roi, axis=0)).sum() > np.abs(np.diff(roi, axis=1)).sum():
        roi = roi.T                       # horizontal edge -> vertical
    if np.diff(roi.mean(axis=0)).sum() < 0:
        roi = -roi                        # falling edge -> rising
    h, w = roi.shape

    a, b = _fit_edge(roi)
    angle = float(np.degrees(np.arctan(b)))
    if abs(angle) < MIN_ANGLE_DEG:
        warnings.warn(f"edge angle {angle:.2f} deg is under {MIN_ANGLE_DEG} deg: "
                      "too little sub-pixel phase diversity, MTF will be unreliable")

    # Signed perpendicular distance of every pixel centre to the fitted line, binned at 0.25 px.
    yy, xx = np.mgrid[0:h, 0:w]
    dist = (xx - (a + b * yy)) / np.sqrt(1 + b * b)
    k = np.rint(dist / BIN).astype(int).ravel()
    k -= k.min()
    cnt = np.bincount(k)
    esf = np.bincount(k, weights=roi.ravel()) / np.maximum(cnt, 1)
    filled = np.flatnonzero(cnt)
    esf, cnt = esf[filled[0]:filled[-1] + 1], cnt[filled[0]:filled[-1] + 1]
    xs = np.arange(len(esf))
    esf = np.interp(xs, xs[cnt > 0], esf[cnt > 0])    # fill empty bins
    esf_x = (xs + filled[0] + np.rint(dist.min() / BIN)) * BIN

    lsf = (esf[2:] - esf[:-2]) / 2                    # centred difference
    p = int(np.argmax(lsf))
    half = min(p, len(lsf) - 1 - p)
    lsf = lsf[p - half:p + half + 1] * np.hamming(2 * half + 1)

    n_fft = max(4096, 1 << (len(lsf) - 1).bit_length())
    spec = np.abs(np.fft.rfft(lsf, n_fft))
    freq = np.fft.rfftfreq(n_fft, d=BIN)
    mtf = spec / spec[0]
    mtf = mtf / np.sinc(2 * freq * BIN)               # undo the centred-difference response
    mtf = mtf / np.sinc(freq * BIN)                   # undo the 0.25 px bin-averaging (box) aperture
    sel = freq <= MAX_FREQ
    freq, mtf = freq[sel], mtf[sel]

    below = np.flatnonzero(mtf < 0.5)
    if len(below) and below[0] > 0:
        i = below[0]
        mtf50 = float(freq[i - 1] + (0.5 - mtf[i - 1]) * (freq[i] - freq[i - 1]) / (mtf[i] - mtf[i - 1]))
    else:
        mtf50 = float("nan")
    return MTFResult(freq, mtf, mtf50, angle, esf_x, esf)
