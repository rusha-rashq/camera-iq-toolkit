"""Test charts: rendering from design values, and detection of the frame in a photo.

Chart coordinates are in abstract "units" (origin top-left, x right, y down). The outer edge of the
black frame is the rectangle (0, 0)-(w, h). A small white notch is cut into the frame band at the
top-left corner, which fixes the chart's orientation.

Homographies `H` always map chart units -> image pixels.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .color import linear_to_srgb, srgb_to_linear

FRAME = 8.0
NOTCH = 3.0                     # side of the white notch, centred in the frame band at the top-left
# The slanted square follows ISO 12233's ~4:1 contrast (in linear reflectance, not black on white),
# which avoids clipping and heavy sharpening: linear 0.8 paper vs 0.2 square. The frame is near-black.
PAPER_LIN, SQUARE_LIN = 0.8, 0.2
PAPER_SRGB = tuple(float(v) * 255 for v in linear_to_srgb([PAPER_LIN] * 3))
SQUARE_SRGB = tuple(float(v) * 255 for v in linear_to_srgb([SQUARE_LIN] * 3))
BLACK_SRGB = (10, 10, 10)

# Design sRGB values, row-major 6x4. Neutrals are the bottom row.
PATCH_SRGB = np.array([
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
    (243, 243, 243), (200, 200, 200), (160, 160, 160), (122, 122, 122), (85, 85, 85), (52, 52, 52),
], dtype=float) / 255
NEUTRAL = tuple(range(18, 24))
MID_GREY = (20, 21)             # used for the single exposure gain


@dataclass
class Chart:
    name: str
    size: tuple                                   # (w, h) in units
    squares: list = field(default_factory=list)   # (centre_xy, side, angle_deg, srgb) dark squares
    patches: list = field(default_factory=list)   # (x0, y0, x1, y1) in units; colour = PATCH_SRGB[i]

    @property
    def inner(self):
        w, h = self.size
        return (FRAME, FRAME, w - FRAME, h - FRAME)


def slanted_chart():
    """100x100: one dark square rotated 5 degrees on light paper. Its four edges are the MTF targets."""
    return Chart("slanted", (100.0, 100.0), squares=[((50.0, 50.0), 50.0, 5.0, SQUARE_SRGB)])


def colour_chart(cols=6, rows=4, patch=20.0, gap=4.0, margin=6.0):
    pitch = patch + gap
    w = 2 * (FRAME + margin) + cols * patch + (cols - 1) * gap
    h = 2 * (FRAME + margin) + rows * patch + (rows - 1) * gap
    x0 = y0 = FRAME + margin
    patches = [(x0 + c * pitch, y0 + r * pitch, x0 + c * pitch + patch, y0 + r * pitch + patch)
               for r in range(rows) for c in range(cols)]
    return Chart("colour", (w, h), patches=patches)


def _square_corners(centre, side, angle_deg):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    half = side / 2
    return np.array(centre) + np.array([(-half, -half), (half, -half), (half, half), (-half, half)]) @ R.T


def _fill(canvas, poly, rgb_lin, s):
    pts = np.round(np.asarray(poly) * s * 16).astype(np.int32)   # 4 fractional bits
    cv2.fillPoly(canvas, [pts], tuple(float(v) for v in rgb_lin), lineType=cv2.LINE_8, shift=4)


def shapes(chart):
    """Painter's-order list of (polygon in units, linear RGB) making up the chart."""
    lin = lambda srgb: srgb_to_linear(np.asarray(srgb, float) / 255)
    rect = lambda x0, y0, x1, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    w, h = chart.size
    out = [(rect(0, 0, w, h), lin(BLACK_SRGB)), (rect(*chart.inner), lin(PAPER_SRGB))]
    for centre, side, ang, srgb in chart.squares:
        out.append((_square_corners(centre, side, ang), lin(srgb)))
    for i, box in enumerate(chart.patches):
        out.append((rect(*box), srgb_to_linear(PATCH_SRGB[i])))
    c = FRAME / 2
    out.append((rect(c - NOTCH / 2, c - NOTCH / 2, c + NOTCH / 2, c + NOTCH / 2), lin(PAPER_SRGB)))
    return out


def render_linear(chart, px_per_unit=4.0, supersample=4):
    """Chart as linear-light RGB float32, shape (H, W, 3). Antialiased by box-averaging in linear light."""
    s = px_per_unit * supersample
    w, h = chart.size
    W, H = int(round(w * px_per_unit)), int(round(h * px_per_unit))
    canvas = np.zeros((H * supersample, W * supersample, 3), np.float32)
    for poly, rgb in shapes(chart):
        _fill(canvas, poly, rgb, s)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)


def render_srgb(chart, px_per_unit=4.0, supersample=4):
    """Chart as sRGB-encoded float32 in [0, 1]."""
    return linear_to_srgb(render_linear(chart, px_per_unit, supersample)).astype(np.float32)


# ---------------------------------------------------------------- geometry

def apply_h(H, pts):
    pts = np.asarray(pts, float)
    p = np.c_[pts.reshape(-1, 2), np.ones(pts.size // 2)] @ H.T
    return (p[:, :2] / p[:, 2:3]).reshape(pts.shape)


def edge_rois(chart, H, half_len=10.0, half_wid=6.0):
    """Axis-aligned image boxes (x0, y0, x1, y1) around the midpoint of each dark-square edge.

    The box is the bounding box of the mapped (2*half_len x 2*half_wid) strip, so it contains only
    that one edge as long as the strip stays clear of the square's corners (true for the defaults).
    The pixels are taken from the original, un-warped image."""
    (centre, side, ang, _), = chart.squares
    corners = _square_corners(centre, side, ang)
    out = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        mid, d = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        strip = np.array([mid + sx * half_len * d + sy * half_wid * n
                          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        q = apply_h(H, strip)
        x0, y0 = np.floor(q.min(axis=0)).astype(int)
        x1, y1 = np.ceil(q.max(axis=0)).astype(int)
        out.append((x0, y0, x1, y1))
    return out


def patch_polygons(chart, H, inner=0.6):
    """Image-space quads (n, 4, 2) of the central `inner` fraction (linear, per side) of each patch."""
    polys = []
    for x0, y0, x1, y1 in chart.patches:
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * inner / 2, (y1 - y0) * inner / 2
        polys.append(apply_h(H, [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]))
    return np.array(polys)


def polygon_mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


# ---------------------------------------------------------------- detection

class ChartNotFound(RuntimeError):
    pass


@dataclass
class Detection:
    H: np.ndarray       # chart units -> image pixels
    score: float        # normalised correlation of the rectified image against the rendered chart
    corners: np.ndarray  # outer frame corners in the image, from H


def _quads(mask, min_area):
    """Convex 4-gons among every contour (outer boundaries and holes) of the binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                q = ap.reshape(4, 2).astype(float)
                ang = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
                out.append(q[np.argsort(ang)])        # clockwise on screen (y down)
                break
    return out


def detect_chart(image, chart, verify_px_per_unit=3.0, min_score=0.75):
    """Locate `chart` in `image` (gray or RGB, uint8 or float in [0, 1], display-encoded).

    Candidates are quadrilaterals of the dark mask: the frame's outer boundary and its inner
    boundary (a dark background merges with the outer one, but the inner one stays a clean quad).
    Each candidate x 4 corner rotations is rectified and scored against the rendered chart, which
    both rejects non-chart quads and resolves orientation through the notch."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (0, 0), max(1.0, min(gray.shape) / 800))
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # The low-contrast slanted square can land on either side of the Otsu split, so also try
    # stricter thresholds that keep only the frame.
    darks = [(blur <= t).astype(np.uint8) * 255 for t in (otsu, otsu * 0.6, otsu * 0.35)]

    w, h = chart.size
    outer = [(0, 0), (w, 0), (w, h), (0, h)]
    x0, y0, x1, y1 = chart.inner
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ref = cv2.cvtColor(np.round(render_srgb(chart, verify_px_per_unit) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    refz = (ref - ref.mean()) / ref.std()
    S = np.diag([verify_px_per_unit, verify_px_per_unit, 1.0])

    hyps = []
    min_area = 0.001 * gray.size
    for q in (q for dark in darks for q in _quads(dark, min_area)):
        for target in (outer, inner):
            for k in range(4):
                dst = np.roll(np.array(target, np.float32), k, axis=0)   # wrap-around of quad corners
                H = cv2.getPerspectiveTransform(dst, q.astype(np.float32))
                warped = cv2.warpPerspective(gray, H @ np.linalg.inv(S), ref.shape[::-1],
                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                             borderMode=cv2.BORDER_REPLICATE)
                if warped.std() < 1e-6:
                    continue
                score = float(((warped - warped.mean()) / warped.std() * refz).mean())
                hyps.append((score, H))
    if not hyps or max(h[0] for h in hyps) < min_score:
        raise ChartNotFound(f"no candidate matched the {chart.name} chart (best score "
                            f"{max((h[0] for h in hyps), default=float('nan')):.2f}, need {min_score})")
    # The notch is a tiny cue, so near-ties (e.g. a 180 degree flip of a symmetric chart) are
    # settled by reading the notch directly rather than by the global correlation.
    top = max(h[0] for h in hyps)
    near = [h for h in hyps if h[0] >= top - 0.05]
    score, H = max(near, key=lambda h: _notch_contrast(gray, chart, h[1]))
    H = _refine(gray, ref, H, S)
    return Detection(H, score, apply_h(H, outer))


def _notch_contrast(gray, chart, H):
    """Brightness at the top-left notch minus the mean at the other three frame corners."""
    w, h = chart.size
    c = FRAME / 2
    pts = apply_h(H, [(c, c), (w - c, c), (w - c, h - c), (c, h - c)])
    v = ndimage.map_coordinates(gray.astype(float), [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    return v[0] - v[1:].mean()


def _refine(gray, ref, H, S):
    """Sub-pixel polish of H by maximising ECC against the rendered chart (contour corners are
    only good to about a pixel). Keeps the original H if ECC fails to converge."""
    M = (H @ np.linalg.inv(S)).astype(np.float32)      # template raster px -> image px
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, M = cv2.findTransformECC(ref.astype(np.float32), gray.astype(np.float32), M,
                                    cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    except cv2.error:
        return H
    Hr = M.astype(float) @ S
    return Hr / Hr[2, 2]
