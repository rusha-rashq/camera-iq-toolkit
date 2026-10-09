"""End to end: capture -> chart detection -> MTF / colour / noise -> regression flags vs a baseline."""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analysis
from .charts import ChartNotFound, MID_GREY, colour_chart, detect_chart, edge_rois, slanted_chart
from .config import DEFAULT, Thresholds
from .loading import CaptureMeta, load_capture
from .mtf import edge_mtf

FREQ_GRID = np.linspace(0, 0.5, 101)    # cycles/pixel, up to Nyquist


@dataclass
class MTFSummary:
    edge_mtf50: list            # per usable edge, cycles/pixel
    mtf50: float                # mean over edges
    curve: np.ndarray           # mean MTF on FREQ_GRID


@dataclass
class CaptureResult:
    name: str
    meta: CaptureMeta
    mtf: MTFSummary | None = None
    colour: analysis.ColourResult | None = None
    noise: analysis.NoiseResult | None = None
    notes: list = field(default_factory=list)

    @property
    def grey_snr_db(self):
        """Luma SNR averaged over the mid-grey patches."""
        return None if self.noise is None else float(self.noise.luma_snr_db[list(MID_GREY)].mean())


def _measure_mtf(capture, chart, det, notes):
    luma = capture.linear @ analysis.LUMA
    curves, m50s = [], []
    for k, (x0, y0, x1, y1) in enumerate(edge_rois(chart, det.H)):
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, luma.shape[1]), min(y1, luma.shape[0])
        if x1 - x0 < 16 or y1 - y0 < 16:
            notes.append(f"slanted edge {k + 1} is outside the frame or too small")
            continue
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = edge_mtf(luma[y0:y1, x0:x1])
            notes += [f"slanted edge {k + 1}: {x.message}" for x in w]
        except Exception as e:
            notes.append(f"slanted edge {k + 1}: MTF failed ({e})")
            continue
        if np.isfinite(r.mtf50):
            m50s.append(r.mtf50)
            curves.append(np.interp(FREQ_GRID, r.freq, r.mtf))
        else:
            notes.append(f"slanted edge {k + 1}: MTF never crosses 0.5")
    if not m50s:
        return None
    return MTFSummary(m50s, float(np.mean(m50s)), np.mean(curves, axis=0))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}


def analyze_captures(items, name):
    """One capture made of several images, e.g. the edge chart and the colour chart shot separately.
    `items`: [(filename, Capture)]. Each chart is looked for in the images in order; the first image
    where it is found supplies that measurement. Metadata comes from the first image."""
    res = CaptureResult(name, items[0][1].meta)
    src = {}
    for chart_name, label in (("slanted", "MTF"), ("colour", "colour/noise")):
        chart = slanted_chart() if chart_name == "slanted" else colour_chart()
        for fname, cap in items:
            try:
                det = detect_chart(cap.srgb8, chart)
            except ChartNotFound:
                continue
            if chart_name == "slanted":
                res.mtf = _measure_mtf(cap, chart, det, res.notes)
            else:
                res.colour = analysis.colour_accuracy(cap.linear, det.H)
                res.noise = analysis.noise(cap.linear, det.H)
            src[label] = fname
            break
        else:
            res.notes.append(f"{chart_name} chart not found" + (" in any image" if len(items) > 1 else ""))
    if len(items) > 1:
        res.notes.append("from " + ", ".join(f"{k}: {v}" for k, v in src.items()))
        m0 = items[0][1].meta
        for fname, cap in items[1:]:
            diff = [f for f in ("camera", "iso", "focal_length") if getattr(cap.meta, f) != getattr(m0, f)]
            if diff:
                res.notes.append(f"{fname} differs from {items[0][0]} in {', '.join(diff)}")
    return res


def analyze_capture(capture, name):
    return analyze_captures([(name, capture)], name)


def image_files(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def analyze_path(path):
    """A file is one capture; a directory is one capture made of all the images inside it."""
    path = Path(path)
    if path.is_dir():
        files = image_files(path)
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        return analyze_captures([(f.name, load_capture(f)) for f in files], path.name)
    return analyze_capture(load_capture(path), path.name)


analyze_file = analyze_path


@dataclass
class Flag:
    metric: str
    baseline: float
    current: float
    message: str


def compare(baseline, current, thresholds: Thresholds = DEFAULT):
    """Regression flags of `current` against `baseline` (the first capture)."""
    flags = []
    if baseline.mtf and current.mtf:
        b, c = baseline.mtf.mtf50, current.mtf.mtf50
        if c < b * (1 - thresholds.mtf50_drop):
            flags.append(Flag("mtf50", b, c, f"MTF50 down {100 * (1 - c / b):.1f}% "
                                              f"(limit {100 * thresholds.mtf50_drop:.0f}%)"))
    if baseline.colour and current.colour:
        b, c = baseline.colour.mean_delta_e, current.colour.mean_delta_e
        if c - b > thresholds.delta_e_rise:
            flags.append(Flag("delta_e", b, c, f"mean ΔE00 up {c - b:.2f} (limit {thresholds.delta_e_rise:g})"))
    return flags
