import io

import numpy as np
import pytest
from PIL import Image

from camera_iq import charts, pipeline, report, synthetic
from camera_iq.cli import main
from camera_iq.config import Thresholds
from camera_iq.loading import load_capture_bytes

SL = charts.slanted_chart()
COL = charts.colour_chart()
SL_QUAD = [(60, 80), (360, 95), (350, 395), (50, 380)]
COL_QUAD = [(500, 110), (1110, 95), (1130, 480), (520, 510)]


def scene(**kw):
    kw.setdefault("shape", (620, 1200))
    return synthetic.simulate([(SL, SL_QUAD), (COL, COL_QUAD)], **kw)


def result(name="c", **kw):
    s = scene(**kw)
    buf = io.BytesIO()
    Image.fromarray(s.srgb8).save(buf, "PNG")
    return pipeline.analyze_capture(load_capture_bytes(buf.getvalue()), name)


def test_measured_mtf50_matches_ground_truth():
    for sigma in (0.6, 1.0, 1.5):
        r = result(blur_sigma=sigma)
        assert r.mtf.mtf50 == pytest.approx(synthetic.expected_mtf50(sigma), rel=0.04)
        assert len(r.mtf.edge_mtf50) == 4


def test_colour_and_noise_ground_truth():
    clean = result(blur_sigma=0.8)
    assert clean.colour.mean_delta_e < 0.5          # 8-bit quantisation only
    assert clean.colour.wb_error < 0.5
    noisy = result(blur_sigma=0.8, noise_sigma=0.01, gain=0.6)
    assert noisy.colour.gain == pytest.approx(1 / 0.6, rel=0.02)
    mid = noisy.noise.mean[20, 1]
    assert noisy.noise.snr_db[20, 1] == pytest.approx(20 * np.log10(mid / 0.01), abs=0.6)
    cast = result(blur_sigma=0.8, cast=(1.15, 1.0, 0.85))
    assert cast.colour.wb_error > 4 and cast.colour.mean_delta_e > clean.colour.mean_delta_e + 2


def test_no_flags_for_equal_or_sharper_or_re_exposed():
    base = result("base", blur_sigma=0.8)
    for kw in (dict(blur_sigma=0.8, noise_sigma=0.004), dict(blur_sigma=0.6), dict(blur_sigma=0.8, gain=0.5)):
        assert pipeline.compare(base, result(**kw)) == []


def test_flags_for_blur_and_colour_shift():
    base = result("base", blur_sigma=0.8)
    flags = pipeline.compare(base, result(blur_sigma=1.3))
    assert [f.metric for f in flags] == ["mtf50"]
    flags = pipeline.compare(base, result(blur_sigma=0.8, cast=(1.2, 1.0, 0.8)))
    assert [f.metric for f in flags] == ["delta_e"]
    # thresholds are configurable in one place
    assert pipeline.compare(base, result(blur_sigma=1.3), Thresholds(mtf50_drop=0.5)) == []


@pytest.mark.parametrize("background", [0.78, 0.0])
def test_detection_with_perspective_and_dark_background(background):
    r = result(blur_sigma=0.8, background=background)
    assert r.mtf is not None and r.colour is not None and not r.notes


def test_missing_chart_is_noted_not_fatal():
    s = synthetic.simulate([(COL, COL_QUAD)], shape=(620, 1200))
    buf = io.BytesIO()
    Image.fromarray(s.srgb8).save(buf, "PNG")
    r = pipeline.analyze_capture(load_capture_bytes(buf.getvalue()), "x")
    assert r.mtf is None and r.colour is not None and "slanted chart not found" in r.notes


def test_report_is_static_html_with_svgs():
    rs = [result("a.png", blur_sigma=0.8), result("b.png", blur_sigma=1.4)]
    html = report.render_report(rs)
    assert html.count("<svg") == 2 and "<script" not in html.lower()
    assert "regression" in html and "baseline" in html


def test_cli_analyze_exit_code_and_report(tmp_path, capsys):
    paths = []
    for name, sigma in (("base.png", 0.8), ("blurry.png", 1.5)):
        p = tmp_path / name
        Image.fromarray(scene(blur_sigma=sigma).srgb8).save(p)
        paths.append(str(p))
    out = tmp_path / "r.html"
    assert main(["analyze", *paths, "-o", str(out)]) == 1
    assert "REGRESSION blurry.png: MTF50 down" in capsys.readouterr().out
    assert main(["analyze", paths[0], paths[0], "-o", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")


def _save(path, placements, **kw):
    Image.fromarray(synthetic.simulate(placements, shape=(620, 1200), **kw).srgb8).save(path)


def test_separate_shots_in_a_folder_form_one_capture(tmp_path):
    for name, sigma in (("base", 0.8), ("soft", 1.5)):
        d = tmp_path / name
        d.mkdir()
        _save(d / "1_edge.png", [(SL, SL_QUAD)], blur_sigma=sigma)
        _save(d / "2_colour.png", [(COL, COL_QUAD)], blur_sigma=sigma)
    one = pipeline.analyze_path(tmp_path / "base")
    assert one.name == "base" and one.mtf is not None and one.colour is not None
    assert one.mtf.mtf50 == pytest.approx(synthetic.expected_mtf50(0.8), rel=0.04)
    assert "from MTF: 1_edge.png, colour/noise: 2_colour.png" in one.notes
    assert not any("not found" in n for n in one.notes)

    out = tmp_path / "r.html"
    assert main(["analyze", str(tmp_path / "base"), str(tmp_path / "soft"), "-o", str(out)]) == 1
    assert "baseline" in out.read_text()


def test_folder_with_only_one_chart_notes_the_other(tmp_path):
    _save(tmp_path / "edge.png", [(SL, SL_QUAD)])
    r = pipeline.analyze_path(tmp_path)
    assert r.mtf is not None and r.colour is None and "colour chart not found" in r.notes
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        pipeline.analyze_path(tmp_path / "empty")


def test_saturated_edge_is_flagged():
    r = result(blur_sigma=0.8, gain=1.5)
    assert any("saturated" in n for n in r.notes)
    assert not any("saturated" in n for n in result(blur_sigma=0.8).notes)


def test_wrong_chart_is_not_matched():
    # a colour chart must not be accepted as the slanted chart (the frame and paper alone correlate well)
    s = synthetic.simulate([(COL, COL_QUAD)], shape=(620, 1200), background=0.0)
    with pytest.raises(charts.ChartNotFound):
        charts.detect_chart(s.srgb8, SL)


# ---- units: cycles per pixel vs cycles per mm on the chart

def test_edge_scales_similarity_and_perspective():
    assert charts.edge_scales(SL, np.array([[4.0, 0, 30], [0, 4.0, 50], [0, 0, 1]])) == pytest.approx([4.0] * 4)
    s = scene(blur_sigma=0.8)
    Htrue = s.homographies["slanted"]
    side = np.mean([np.linalg.norm(np.subtract(SL_QUAD[(i + 1) % 4], SL_QUAD[i])) for i in range(4)]) / 100
    assert np.mean(charts.edge_scales(SL, Htrue)) == pytest.approx(side, rel=0.05)
    # perspective: the edge nearer the camera (larger in the image) has the larger scale
    quad = [(100, 100), (500, 100), (560, 480), (40, 480)]           # bottom wider than top
    sc = charts.edge_scales(SL, synthetic.simulate([(SL, quad)], shape=(600, 700)).homographies["slanted"])
    assert sc[2] > sc[0] * 1.15


def test_mtf50_mm_is_px_times_scale():
    r = result(blur_sigma=0.8)
    m = r.mtf
    assert len(m.edge_mtf50_mm) == 4
    assert m.mtf50_mm == pytest.approx(m.mtf50 * m.px_per_mm, rel=0.02)
    assert m.px_per_mm == pytest.approx(3.0, rel=0.1)                 # ~300 px frame side / 100 mm
    assert m.freq_mm[-1] == pytest.approx(0.5 * m.px_per_mm)
    # cycles/mm = cycles/px * px/mm, per edge
    s = scene(blur_sigma=0.8)
    cap = load_capture_bytes(_png_bytes(s.srgb8))
    det = charts.detect_chart(cap.srgb8, SL)
    for px, mm, sc in zip(pipeline_edge_mtf50s(cap, det), result(blur_sigma=0.8).mtf.edge_mtf50_mm, charts.edge_scales(SL, det.H)):
        assert mm == pytest.approx(px * sc, rel=1e-6)


def _png_bytes(srgb8):
    buf = io.BytesIO()
    Image.fromarray(srgb8).save(buf, "PNG")
    return buf.getvalue()


def pipeline_edge_mtf50s(cap, det):
    return pipeline._measure_mtf(cap, SL, det, []).edge_mtf50


def test_same_optics_at_two_resolutions_agree_in_mm_not_in_px():
    """Same physical blur and framing, sensor with 2x the pixels: cycles/pixel halves, cycles/mm does not."""
    lo = result("lo", blur_sigma=1.0)
    s2 = synthetic.simulate([(SL, [(2 * x, 2 * y) for x, y in SL_QUAD]), (COL, [(2 * x, 2 * y) for x, y in COL_QUAD])],
                            shape=(1240, 2400), blur_sigma=2.0)
    hi = pipeline.analyze_capture(load_capture_bytes(_png_bytes(s2.srgb8)), "hi")
    assert hi.mtf.mtf50 / lo.mtf.mtf50 == pytest.approx(0.5, rel=0.08)
    assert hi.mtf.mtf50_mm / lo.mtf.mtf50_mm == pytest.approx(1.0, rel=0.06)
    assert pipeline.compare(lo, hi) == []          # a px-based flag would have fired: -50%
    soft = result("soft", blur_sigma=1.6)
    assert [f.metric for f in pipeline.compare(lo, soft)] == ["mtf50"]


def test_peak_mtf_shows_sharpening_overshoot():
    import cv2
    plain = result(blur_sigma=1.2).mtf
    s = scene(blur_sigma=1.2).srgb8
    sharp8 = np.clip(s.astype(float) + 1.5 * (s - cv2.GaussianBlur(s, (0, 0), 1.5).astype(float)), 0, 255).astype(np.uint8)
    sharp = pipeline.analyze_capture(load_capture_bytes(_png_bytes(sharp8)), "sharp").mtf
    assert plain.peak < 1.02 and sharp.peak > 1.08
    html = report.render_report([result("a", blur_sigma=1.2), pipeline.analyze_capture(load_capture_bytes(_png_bytes(sharp8)), "b")])
    assert ">1.25<" in html          # y-axis extends above 1
    assert "Peak MTF" in html and "MTF50 (cy/mm)" in html


def test_report_note_is_shown_and_escaped(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(scene(blur_sigma=0.8).srgb8).save(p)
    out = tmp_path / "r.html"
    assert main(["analyze", str(p), "-o", str(out), "--note", "mm are <design units>"]) == 0
    assert "<strong>Note:</strong> mm are &lt;design units&gt;" in out.read_text()
