"""Print-ready chart files: US Letter landscape, 1 chart unit = 1 mm, 100% scale, centred.

Print at "Actual size" / 100% (not "fit to page"). The PDF is vector; the PNG is raster with its dpi
recorded. Neither embeds an ICC profile, so for the colour chart use your printer's sRGB/no-colour-
management path if you intend to compare printed colours against the design values."""
import struct
import zlib
from pathlib import Path

import numpy as np

from .charts import render_srgb, shapes
from .color import linear_to_srgb

MM_PER_IN = 25.4
PAGE_MM = (11.0 * MM_PER_IN, 8.5 * MM_PER_IN)   # Letter landscape


def _origin(chart):
    return (PAGE_MM[0] - chart.size[0]) / 2, (PAGE_MM[1] - chart.size[1]) / 2


def chart_pdf(chart):
    """Vector PDF bytes: one page, 792 x 612 pt."""
    k = 72 / MM_PER_IN
    ox, oy = _origin(chart)
    ops = []
    for poly, rgb in shapes(chart):
        c = np.clip(linear_to_srgb(rgb), 0, 1)
        pts = [((ox + x) * k, (PAGE_MM[1] - oy - y) * k) for x, y in poly]    # PDF y points up
        ops.append("%.5f %.5f %.5f rg" % tuple(c))
        ops.append(" ".join(("%.3f %.3f m" if i == 0 else "%.3f %.3f l") % p for i, p in enumerate(pts)) + " h f")
    stream = "\n".join(ops).encode()
    w, h = (round(v * k, 2) for v in PAGE_MM)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %g %g] /Contents 4 0 R >>" % (w, h),
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def _png(rgb8, dpi):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    h, w, _ = rgb8.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb8.reshape(h, w * 3)], axis=1).tobytes()
    ppm = round(dpi / 0.0254)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"sRGB", b"\x00")
            + chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def chart_png(chart, dpi=300):
    """PNG bytes of the full Letter-landscape page at `dpi`, chart centred at 100% scale."""
    ppu = dpi / MM_PER_IN
    W, H = round(PAGE_MM[0] * ppu), round(PAGE_MM[1] * ppu)
    page = np.ones((H, W, 3), np.float32)
    ren = render_srgb(chart, ppu, supersample=3)
    h, w, _ = ren.shape
    ox, oy = (round(v * ppu) for v in _origin(chart))
    page[oy:oy + h, ox:ox + w] = ren
    return _png(np.round(np.clip(page, 0, 1) * 255).astype(np.uint8), dpi)


def export_charts(outdir, dpi=300):
    """Write <name>.pdf and <name>.png for both charts; returns the paths."""
    from .charts import colour_chart, slanted_chart
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for chart in (slanted_chart(), colour_chart()):
        for ext, data in (("pdf", chart_pdf(chart)), ("png", chart_png(chart, dpi))):
            p = outdir / f"{chart.name}_chart_letter.{ext}"
            p.write_bytes(data)
            paths.append(p)
    return paths
