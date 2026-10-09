import re
import struct

import cv2
import numpy as np

from camera_iq import charts, export
from camera_iq.cli import main


def test_pdf_is_letter_landscape_and_scaled():
    pdf = export.chart_pdf(charts.slanted_chart())
    assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/MediaBox [0 0 792 612]" in pdf
    # outer frame is the first path: 100 mm = 283.465 pt wide, centred on the page
    xs = [float(m) for m in re.findall(rb"([\d.]+) [\d.]+ [ml]", pdf.split(b"stream\n")[1])[:4]]
    assert abs(max(xs) - min(xs) - 100 * 72 / 25.4) < 0.01
    assert abs((min(xs) + max(xs)) / 2 - 396) < 0.01


def test_png_size_dpi_and_physical_scale(tmp_path):
    dpi = 150
    data = export.chart_png(charts.slanted_chart(), dpi)
    i = data.index(b"pHYs")
    assert struct.unpack(">II", data[i + 4:i + 12]) == (5906, 5906)
    p = tmp_path / "a.png"
    p.write_bytes(data)
    img = cv2.imread(str(p))
    assert img.shape == (1275, 1650, 3)                      # 11 x 8.5 in at 150 dpi
    frame = np.flatnonzero((img[img.shape[0] // 2, :, 0] < 30))
    assert abs((frame.max() - frame.min() + 1) - 100 / 25.4 * dpi) <= 2     # 100 mm frame


def test_cli(tmp_path):
    main(["export-charts", str(tmp_path), "--dpi", "50"])
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "colour_chart_letter.pdf", "colour_chart_letter.png",
        "slanted_chart_letter.pdf", "slanted_chart_letter.png"]
