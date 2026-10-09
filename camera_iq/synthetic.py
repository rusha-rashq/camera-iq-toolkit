"""Simulated captures with known blur, noise, exposure, colour cast and perspective, for end-to-end tests.

Pipeline (all in linear light): render the chart -> perspective-warp onto a 4x-supersampled frame over a
flat background -> Gaussian blur (the lens/processing PSF, sigma in output pixels) -> box-average to the
output grid (the pixel aperture) -> exposure gain and per-channel cast -> additive Gaussian noise ->
clip -> 8-bit sRGB. So the true MTF is exp(-2 pi^2 sigma^2 f^2) * sinc(f), see `expected_mtf`."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .charts import render_linear
from .color import linear_to_srgb, srgb_to_linear

SS = 4


@dataclass
class Scene:
    srgb8: np.ndarray                              # (H, W, 3) uint8
    homographies: dict = field(default_factory=dict)   # chart name -> units -> output pixel coords


def expected_mtf(freq, sigma):
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * freq ** 2) * np.sinc(freq)


def expected_mtf50(sigma):
    from scipy.optimize import brentq
    return brentq(lambda f: expected_mtf(f, sigma) - 0.5, 1e-4, 0.5)


def simulate(placements, shape=(600, 800), blur_sigma=0.8, noise_sigma=0.0, gain=1.0, cast=(1, 1, 1),
             background=0.78, seed=0):
    """placements: [(chart, four outer-frame corners (TL, TR, BR, BL order in chart space) in output px)].
    background: linear-light level of the surroundings (0 gives a black background that merges with the frame)."""
    h, w = shape
    canvas = np.full((h * SS, w * SS, 3), background, np.float32)
    homographies = {}
    T = np.array([[SS, 0, SS / 2 - 0.5], [0, SS, SS / 2 - 0.5], [0, 0, 1.0]])   # output px -> supersampled px
    for chart, corners in placements:
        cw, ch = chart.size
        quad = np.float32(corners)
        side = max(np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4))
        ppu = 1.5 * SS * side / max(cw, ch)               # render finer than the supersampled frame
        ren = render_linear(chart, ppu, supersample=2)
        rh, rw = ren.shape[:2]
        src = np.float32([(-0.5, -0.5), (rw - 0.5, -0.5), (rw - 0.5, rh - 0.5), (-0.5, rh - 0.5)])   # pixel-centre convention
        M = cv2.getPerspectiveTransform(src, (quad * SS + (SS / 2 - 0.5)).astype(np.float32))
        size = (w * SS, h * SS)
        warped = cv2.warpPerspective(ren, M, size, flags=cv2.INTER_LINEAR)
        cover = cv2.warpPerspective(np.ones(ren.shape[:2], np.float32), M, size, flags=cv2.INTER_LINEAR)[..., None]
        canvas = canvas * (1 - cover) + warped * cover
        U = np.array([[rw / cw, 0, -0.5], [0, rh / ch, -0.5], [0, 0, 1.0]])
        homographies[chart.name] = np.linalg.inv(T) @ M @ U
    if blur_sigma > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur_sigma * SS)
    img = canvas.reshape(h, SS, w, SS, 3).mean(axis=(1, 3)) * gain * np.asarray(cast, np.float32)
    if noise_sigma:
        img = img + np.random.default_rng(seed).normal(0, noise_sigma, img.shape)
    srgb8 = np.round(np.clip(linear_to_srgb(np.clip(img, 0, 1)), 0, 1) * 255).astype(np.uint8)
    return Scene(srgb8, homographies)
