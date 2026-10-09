"""End to end: capture -> chart detection -> MTF / colour / noise -> regression flags vs a baseline."""
import warnings
from dataclasses import dataclass, field

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


def analyze_capture(capture, name):
    """Measure whichever charts are visible in `capture`; missing charts are noted, not fatal."""
    res = CaptureResult(name, capture.meta)
    try:
        chart = slanted_chart()
        res.mtf = _measure_mtf(capture, chart, detect_chart(capture.srgb8, chart), res.notes)
    except ChartNotFound:
        res.notes.append("slanted chart not found")
    try:
        det = detect_chart(capture.srgb8, colour_chart())
        res.colour = analysis.colour_accuracy(capture.linear, det.H)
        res.noise = analysis.noise(capture.linear, det.H)
    except ChartNotFound:
        res.notes.append("colour chart not found")
    return res


def analyze_file(path):
    from pathlib import Path
    return analyze_capture(load_capture(path), Path(path).name)


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
