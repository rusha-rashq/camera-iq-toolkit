import io

import cv2
import numpy as np
import pytest

from camera_iq import analysis, charts
from PIL import Image

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


def test_wb_uses_middle_four_neutrals():
    img = capture()
    # corrupt white and black only: WB error must not move
    for i in (18, 23):
        x0, y0, x1, y1 = CHART.patches[i]
        img[int(y0 * PPU):int(y1 * PPU), int(x0 * PPU):int(x1 * PPU)] *= (1.3, 1.0, 0.7)
    assert analysis.colour_accuracy(img, H).wb_error < 0.05


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


def test_p3_tagged_image_converted_to_srgb():
    from icc import P3_XY, make_profile, rgb_to_xyz_d65
    from camera_iq.color import linear_to_srgb, srgb_to_linear
    p3_codes = np.array([[0.85, 0.30, 0.25], [0.30, 0.65, 0.35], [0.5, 0.5, 0.5], [0.95, 0.80, 0.20]])
    img = np.tile((p3_codes * 255).round().astype(np.uint8)[:, None, :], (1, 8, 1))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", icc_profile=make_profile("Display P3 test"))
    cap = load_capture_bytes(buf.getvalue())
    assert cap.meta.profile == "Display P3 test"

    to_xyz_p3 = rgb_to_xyz_d65(P3_XY)
    to_xyz_srgb = rgb_to_xyz_d65(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)))
    lin_p3 = srgb_to_linear(img[:, 0] / 255.0)
    expect = linear_to_srgb(np.clip(np.linalg.solve(to_xyz_srgb, to_xyz_p3 @ lin_p3.T).T, 0, 1)) * 255
    assert np.abs(cap.srgb8[:, 0].astype(float) - expect).max() < 2.5
    assert np.abs(cap.srgb8[:, 0].astype(int) - img[:, 0]).max() > 10      # i.e. it really changed


def test_srgb_tagged_and_untagged_pass_through():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    from PIL import ImageCms
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    assert (load_capture_bytes(buf.getvalue()).srgb8 == img).all()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    assert (load_capture_bytes(buf.getvalue()).srgb8 == img).all()


def test_exif_metadata_read():
    im = Image.new("RGB", (16, 16), (90, 90, 90))
    exif = Image.Exif()
    exif[271], exif[272] = "Apple", "Apple iPhone 15 Pro"
    exif.get_ifd(0x8769).update({33434: 1 / 120, 34855: 64, 37386: 6.86, 33437: 1.78})
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    m = load_capture_bytes(buf.getvalue()).meta
    assert m.camera == "Apple iPhone 15 Pro" and m.iso == 64
    assert m.exposure_time == pytest.approx(1 / 120, rel=1e-3)
    assert m.focal_length == pytest.approx(6.86, rel=1e-3) and m.f_number == pytest.approx(1.78, rel=1e-3)


def test_heic_keeps_icc_profile_and_exif():
    from icc import make_profile
    img = np.tile(np.array([[0.85, 0.30, 0.25]]) * 255, (32, 48, 1)).round().astype(np.uint8)
    exif = Image.Exif()
    exif[271], exif[272] = "Apple", "iPhone 15 Pro"
    exif.get_ifd(0x8769).update({33434: 1 / 60, 34855: 125, 37386: 6.86})
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "HEIF", quality=95, exif=exif, icc_profile=make_profile("Display P3 test"))
    cap = load_capture_bytes(buf.getvalue())
    assert cap.meta.profile == "Display P3 test"
    assert (cap.meta.camera, cap.meta.iso) == ("Apple iPhone 15 Pro", 125)
    assert cap.meta.exposure_time == pytest.approx(1 / 60, rel=1e-3)
    # P3 (0.85, 0.30, 0.25) is a more saturated red than sRGB can encode as-is: the red channel rises, green/blue fall
    r, g, b = cap.srgb8[16, 24].astype(int)
    assert r > 0.85 * 255 and g < 0.30 * 255 - 10 and b < 0.25 * 255
