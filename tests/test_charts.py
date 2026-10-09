import cv2
import numpy as np
import pytest

from camera_iq import charts
from camera_iq.charts import apply_h


def photo(chart, corners, bg, shape=(480, 640), ppu=6.0, rot=0):
    """Warp the rendered chart so its outer corners land on `corners` over a flat background."""
    ren = (charts.render_srgb(chart, ppu) * 255).round().astype(np.uint8)
    w, h = chart.size
    src = np.float32([(0, 0), (w, 0), (w, h), (0, h)]) * ppu
    dst = np.roll(np.float32(corners), -rot, axis=0)
    M = cv2.getPerspectiveTransform(src, dst)
    img = np.full(shape + (3,), bg, np.uint8)
    warped = cv2.warpPerspective(ren, M, shape[::-1], flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(ren.shape[:2], 255, np.uint8), M, shape[::-1])
    img[mask > 127] = warped[mask > 127]
    return img, M @ np.diag([ppu, ppu, 1])


CORNERS = [(120, 90), (520, 120), (500, 400), (100, 370)]


@pytest.mark.parametrize("make", [charts.slanted_chart, charts.colour_chart])
@pytest.mark.parametrize("bg", [200, 0])      # 0: background merges with the black frame
@pytest.mark.parametrize("rot", [0, 1, 2, 3])
def test_detect(make, bg, rot):
    chart = make()
    img, Htrue = photo(chart, CORNERS, bg, rot=rot)
    det = charts.detect_chart(img, chart)
    w, h = chart.size
    pts = [(0, 0), (w, 0), (w, h), (0, h), (w / 2, h / 2), (FR := charts.FRAME, FR)]
    err = np.linalg.norm(apply_h(det.H, pts) - apply_h(Htrue, pts), axis=1)
    assert err.max() < 1.5, err
    assert det.score > 0.9


def test_no_chart():
    with pytest.raises(charts.ChartNotFound):
        charts.detect_chart(np.full((300, 300, 3), 128, np.uint8), charts.colour_chart())


def test_rois_and_patches():
    chart = charts.colour_chart()
    img, H = photo(chart, CORNERS, 200)
    polys = charts.patch_polygons(chart, H)
    assert polys.shape == (24, 4, 2)
    lin = img.astype(float) / 255
    for i in (0, 10, 18, 23):
        m = charts.polygon_mask(img.shape, polys[i])
        assert np.abs(lin[m].mean(axis=0) - charts.PATCH_SRGB[i]).max() < 0.02

    sl = charts.slanted_chart()
    img, H = photo(sl, [(100, 50), (500, 50), (500, 450), (100, 450)], 200)
    rois = charts.edge_rois(sl, H)
    assert len(rois) == 4
    for x0, y0, x1, y1 in rois:
        roi = img[y0:y1, x0:x1, 0]
        assert roi.min() < 140 and roi.max() > 215        # dark inside, paper outside, nothing else


def test_slanted_contrast_is_4_to_1():
    ren = charts.render_linear(charts.slanted_chart(), 4.0)
    centre, paper = ren[200, 200, 0], ren[200, 40, 0]
    assert paper / centre == pytest.approx(4.0, rel=0.01)
