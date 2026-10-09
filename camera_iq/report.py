"""Self-contained HTML report: summary table, MTF overlay and grey-patch tone curve as inline SVG. No JS.
Hovering a point or curve shows its value through native SVG <title> tooltips."""
from html import escape

import numpy as np

from .analysis import LUMA
from .charts import NEUTRAL, PATCH_SRGB
from .color import linear_to_srgb
from .config import DEFAULT
from .pipeline import compare

# Categorical slots 1-8 of the reference palette; series k wears --s{k}, never recoloured.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MAX_SERIES = len(LIGHT)

CSS = """
.viz{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e3e2de;%s
 font:14px/1.45 -apple-system,system-ui,sans-serif;color:var(--ink);background:var(--surface);max-width:960px;margin:0 auto;padding:24px}
@media (prefers-color-scheme:dark){.viz{color-scheme:dark;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#34332f;%s}}
.viz h1{font-size:20px;margin:0 0 4px}.viz h2{font-size:15px;margin:28px 0 6px}.viz p{color:var(--ink2);margin:4px 0}
.viz table{border-collapse:collapse;width:100%%}.viz th,.viz td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
.viz th{color:var(--ink2);font-weight:600}.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0;padding:0;list-style:none}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.viz svg{width:100%%;max-width:640px;height:auto;display:block}.viz td:first-child{white-space:nowrap}.viz svg text{fill:var(--ink2);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--ink2);stroke-width:1}.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4;fill:none}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.flag{font-weight:600}
""" % ("".join(f"--s{i + 1}:{c};" for i, c in enumerate(LIGHT)), "".join(f"--s{i + 1}:{c};" for i, c in enumerate(DARK)))

W, H, L, R, T, B = 560, 340, 52, 16, 14, 44


def _fmt_exposure(t):
    if t is None:
        return "–"
    return f"1/{round(1 / t)} s" if t < 0.95 else f"{t:g} s"


def _plot(xlim, ylim, xticks, yticks, xlabel, ylabel, body, refs=""):
    sx = lambda x: L + (x - xlim[0]) / (xlim[1] - xlim[0]) * (W - L - R)
    sy = lambda y: H - B - (y - ylim[0]) / (ylim[1] - ylim[0]) * (H - T - B)
    g = "".join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
                f'<text x="{L - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:g}</text>' for y in yticks)
    g += "".join(f'<text x="{sx(x):.1f}" y="{H - B + 16}" text-anchor="middle">{x:g}</text>' for x in xticks)
    g += (f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>'
          f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{xlabel}</text>'
          f'<text transform="translate(13 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    return sx, sy, g, refs, body


def _svg(label, pieces):
    sx, sy, g, refs, body = pieces
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}">{g}{refs}{body}</svg>'


def _ticks(top, step):
    return [round(i * step, 4) for i in range(int(round(top / step)) + 1)]


def _mtf_svg(results):
    shown = [r for r in results[:MAX_SERIES] if r.mtf is not None]
    ytop = max([1.0] + [float(r.mtf.curve.max()) for r in shown])
    ytop = np.ceil(ytop * 4) / 4                                    # room for sharpening overshoot
    xtop = max([r.mtf.freq_mm[-1] for r in shown] or [1.0])         # Nyquist of the finest capture
    xstep = next(st for st in (0.25, 0.5, 1, 2, 5, 10) if xtop / st <= 8)
    xtop = np.ceil(xtop / xstep) * xstep
    sx, sy, g, _, _ = _plot((0, xtop), (0, ytop), _ticks(xtop, xstep), _ticks(ytop, 0.25 if ytop <= 1.5 else 0.5),
                            "spatial frequency on the chart (cycles/mm)", "MTF", "")
    refs = (f'<line class="ref" x1="{L}" x2="{W - R}" y1="{sy(.5):.1f}" y2="{sy(.5):.1f}"/>'
            + (f'<line class="ref" style="stroke-dasharray:1 4" x1="{L}" x2="{W - R}" y1="{sy(1):.1f}" y2="{sy(1):.1f}"/>' if ytop > 1 else ""))
    body = ""
    for k, r in enumerate(results[:MAX_SERIES]):
        if r.mtf is None:
            continue
        m = r.mtf
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(m.freq_mm, m.curve))
        body += (f'<g><title>{escape(r.name)}: MTF50 {m.mtf50_mm:.2f} cycles/mm ({m.mtf50:.3f} cycles/pixel), peak MTF {m.peak:.2f}</title>'
                 f'<polyline class="line" style="stroke:var(--s{k + 1})" points="{pts}"/>'
                 f'<circle cx="{sx(m.mtf50_mm):.1f}" cy="{sy(.5):.1f}" r="4" style="fill:var(--s{k + 1});stroke:var(--surface);stroke-width:2"/></g>')
    return _svg("MTF overlay", (sx, sy, g, refs, body))


def _tone_points(r):
    ref = PATCH_SRGB[list(NEUTRAL), 1]
    luma = (r.colour.patch_rgb[list(NEUTRAL)] * r.colour.gain) @ LUMA
    return ref, np.asarray(linear_to_srgb(np.clip(luma, 0, None)))


def _tone_svg(results):
    sx, sy, g, _, _ = _plot((0, 1), (0, 1), [0, .25, .5, .75, 1], [0, .25, .5, .75, 1],
                            "chart value (sRGB)", "measured, after gain (sRGB)", "")
    refs = f'<line class="ref" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>'
    body = ""
    for k, r in enumerate(results[:MAX_SERIES]):
        if r.colour is None:
            continue
        ref, y = _tone_points(r)
        pts = " ".join(f"{sx(a):.1f},{sy(min(b, 1.05)):.1f}" for a, b in zip(ref, y))
        dots = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(min(b, 1.05)):.1f}" r="4" style="fill:var(--s{k + 1});stroke:var(--surface);stroke-width:2">'
                       f'<title>{escape(r.name)}: chart {a:.2f} → measured {b:.3f}</title></circle>' for a, b in zip(ref, y))
        body += f'<g><polyline class="line" style="stroke:var(--s{k + 1})" points="{pts}"/>{dots}</g>'
    return _svg("Tone curve of the grey patches", (sx, sy, g, refs, body))


def _cell(v, fmt):
    return "–" if v is None else fmt.format(v)


def _snr(v):
    return "–" if v is None else ("&gt;100 dB" if v > 100 else f"{v:.1f} dB")


def render_report(results, thresholds=DEFAULT, title="Camera image quality report"):
    """HTML string. results[0] is the baseline; every later capture is compared against it."""
    rows, base = [], results[0]
    for k, r in enumerate(results):
        if k == 0:
            status = "baseline"
        else:
            flags = compare(base, r, thresholds)
            status = ('<span class="flag">⚠ regression: ' + escape("; ".join(f.message for f in flags)) + "</span>"
                      if flags else "✓ OK")
        m = r.meta
        cam = " · ".join(x for x in (m.camera, f"ISO {m.iso}" if m.iso else None, _fmt_exposure(m.exposure_time) if m.exposure_time else None,
                                     f"{m.focal_length:g} mm" if m.focal_length else None, m.profile) if x) or "–"
        notes = f'<br><span style="color:var(--ink2)">{escape("; ".join(r.notes))}</span>' if r.notes else ""
        swatch = f'<span class="sw" style="background:var(--s{k + 1})"></span>' if k < MAX_SERIES else ""
        rows.append(f"<tr><td>{swatch}{escape(r.name)}</td><td>{escape(cam)}</td>"
                    f"<td>{_cell(r.mtf and r.mtf.mtf50_mm, '{:.2f}')}</td><td>{_cell(r.mtf and r.mtf.mtf50, '{:.3f}')}</td><td>{_cell(r.mtf and r.mtf.peak, '{:.2f}')}</td><td>{_cell(r.colour and r.colour.mean_delta_e, '{:.2f}')}</td>"
                    f"<td>{_cell(r.colour and r.colour.wb_error, '{:.2f}')}</td><td>{_snr(r.grey_snr_db)}</td>"
                    f"<td>{status}{notes}</td></tr>")
    legend = "".join(f'<li><span class="sw" style="background:var(--s{k + 1})"></span>{escape(r.name)}{" (baseline)" if k == 0 else ""}</li>'
                     for k, r in enumerate(results[:MAX_SERIES]))
    extra = f"<p>Charts show the first {MAX_SERIES} captures; the table lists all {len(results)}.</p>" if len(results) > MAX_SERIES else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body><div class="viz">
<h1>{escape(title)}</h1>
<p>Baseline: {escape(base.name)}. Flags: MTF50 (cycles/mm) down more than {100 * thresholds.mtf50_drop:g}%, or mean ΔE00 up more than {thresholds.delta_e_rise:g}.</p>
<h2>Summary</h2>
<table><thead><tr><th>Capture</th><th>Camera</th><th>MTF50 (cy/mm)</th><th>MTF50 (cy/px)</th><th>Peak MTF</th><th>Mean ΔE00</th><th>WB error (C*)</th><th>Grey SNR</th><th>Status</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>{extra}
<ul class="legend">{legend}</ul>
<h2>MTF overlay</h2><p>Mean of the four slanted edges, in cycles per mm on the chart so captures of different resolution compare directly. Dashed line marks MTF = 0.5, dot marks MTF50; values above 1 are sharpening overshoot.</p>{_mtf_svg(results)}
<h2>Tone curve of the grey patches</h2><p>Six neutral patches, luma after the single exposure gain; dashed line is the ideal response.</p>{_tone_svg(results)}
</div></body></html>
"""
