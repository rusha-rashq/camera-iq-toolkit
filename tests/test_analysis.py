import cv2
import numpy as np
import pytest

from camera_iq import analysis, charts
from camera_iq.loading import UnsupportedImage, load_capture_bytes

PPU = 6.0
CHART = charts.colour_chart()
H = np.diag([PPU, PPU, 1.0])


def capture(gain=1.0, cast=(1, 1, 1), sigma=0.0, tilt=0.0, seed=0):
    """Linear-light capture of the colour chart with known exposure, colour cast, noise, shading."""
    img = charts.render_linear(CHART, PPU).astype(float) * gain * np.array(cast)
    h, w, _ = img.shape
    if tilt:
        img = img * (1 + tilt * (np.arange(w)[None, :, None] / w - 0.5))
    if sigma:
        img = img + np.random.default_rng(seed).normal(0, sigma, img.shape)
    return img


def test_perfect_capture_any_exposure():
    for g in (1.0, 0.4, 1.3):
        r = analysis.colour_accuracy(capture(gain=g), H)
        assert r.gain == pytest.approx(1 / g, rel=1e-3)
        assert r.mean_delta_e < 0.05 and r.wb_error < 0.05


def test_colour_cast_shows_in_wb_and_delta_e():
    clean = analysis.colour_accuracy(capture(), H)
    warm = analysis.colour_accuracy(capture(cast=(1.15, 1.0, 0.85)), H)
    assert warm.wb_error > 5 and warm.mean_delta_e > clean.mean_delta_e + 3
    # a pure exposure change must not look like a cast
    assert analysis.colour_accuracy(capture(gain=0.5), H).wb_error < 0.05


def test_noise_snr_and_plane_removal():
    sigma, gain = 0.01, 0.5
    for tilt in (0.0, 0.3):          # 30% shading across the frame must not count as noise
        r = analysis.noise(capture(gain=gain, sigma=sigma, tilt=tilt), H)
        expect = 20 * np.log10(r.mean[:, 1] / sigma)
        assert np.abs(r.snr_db[:, 1] - expect).max() < 0.4
    assert r.sigma[:, 1].mean() == pytest.approx(sigma, rel=0.05)
    assert r.luma_snr_db[19] > r.snr_db[19, 1]      # luma averages three independent channels


def _jpeg_with_orientation(img, orient):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    exif = (b"Exif\0\0MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01" + bytes([0, orient]) + b"\0\0\0\0\0\0")
    seg = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    j = jpg.tobytes()
    return j[:2] + seg + j[2:]


def test_loading_applies_exif_orientation_and_linearises():
    bgr = np.zeros((40, 80, 3), np.uint8)
    bgr[:, :40] = (0, 0, 200)        # red left half
    cap = load_capture_bytes(_jpeg_with_orientation(bgr, 6))     # 6: rotate 90 deg clockwise
    assert cap.srgb8.shape == (80, 40, 3)
    assert cap.srgb8[5, 20, 0] > 150 and cap.srgb8[75, 20, 0] < 50      # red half is now on top
    assert cap.linear[5, 20, 0] == pytest.approx(((cap.srgb8[5, 20, 0] / 255 + 0.055) / 1.055) ** 2.4, rel=1e-4)


def test_loading_rejects_16_bit():
    ok, png = cv2.imencode(".png", np.full((8, 8, 3), 1000, np.uint16))
    with pytest.raises(UnsupportedImage):
        load_capture_bytes(png)
