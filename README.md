# camera-iq-toolkit

Measure a phone or camera against two printed charts (a slanted-edge sharpness chart and a 24-patch colour
chart), and flag regressions between captures: change a setting, update the OS, swap a lens, and see
whether sharpness or colour got worse.

Per capture it reports MTF50 (sharpness), mean CIEDE2000 colour error, white-balance error and grey-patch
SNR, plus the camera, ISO, exposure time and focal length from EXIF. Output is a single static HTML
report (inline SVG, no JavaScript). The first capture is the baseline; every later one is compared to it.

## Install

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest

Needs Python 3.10+. Dependencies: numpy, scipy, opencv-python-headless, pillow, pillow-heif.

## Use

1. Print the charts (US Letter landscape, 1 chart unit = 1 mm):

       camera-iq export-charts charts/          # PDF (vector) + PNG (300 dpi) of both

   Print at **100% / "Actual size"**, not "fit to page". The slanted square should measure 50 mm and
   the frame 100 mm. Use matte paper. The charts are tested on synthetic images only so far (see
   Limitations), so treat the shooting advice below as starting points, not validated requirements.

2. Shoot. Keep the chart flat, evenly lit, in focus, with the whole black frame visible and no glare. The
   charts can be in one frame or shot separately. For repeatable comparisons keep distance, framing and
   light the same, and change one thing at a time.

3. Analyse:

       camera-iq analyze baseline/ new_firmware/ other.heic -o report.html

   Each argument is one capture: a **file**, or a **folder** whose images together make one capture
   (for example `edge.heic` and `colour.heic` shot separately). Each chart is looked for in every
   image of the folder, in filename order, and the first image where it is found supplies that
   measurement. EXIF is taken from the first image, and a note is added if the other images differ in
   camera, ISO or focal length. Prints `REGRESSION ...` lines and exits 1 if any capture is flagged.

   Thresholds default to MTF50 (cycles/mm) down >10% and mean ΔE00 up >1.5; override with `--mtf-drop` / `--de-rise`
   or edit `camera_iq/config.py`.

Supported inputs: 8-bit JPEG, PNG, TIFF and HEIC/HEIF (iPhone default). Embedded ICC profiles
(e.g. Display P3) are converted to sRGB; untagged images are assumed sRGB. EXIF orientation is applied.
16-bit and 10-bit files are rejected rather than silently reduced.

## Method

**Charts** (`charts.py`). Both have a thick black frame with a small white notch in the top-left corner
for orientation. The slanted chart is a dark square rotated 5° on light paper at 4:1 linear contrast
(0.2 vs 0.8), as ISO 12233 recommends, to avoid clipping and heavy sharpening. The colour chart is
6×4 patches; the reference colours are the chart's own design sRGB values (ColorChecker-like), not
measurements of the printed chart.

**Detection.** Threshold and take the quadrilaterals of the dark regions, both the frame's outer boundary
and its inner boundary (a dark background merges with the outer one but not the inner one). Each
candidate, in four corner rotations, is rectified and scored by normalised correlation against the
rendered chart, which rejects non-chart quads. Orientation near-ties are settled by reading the notch,
and the winner is polished to sub-pixel accuracy with ECC. The rectified image is only used to locate
regions of interest: MTF is computed on the original, un-warped pixels.

**Loading** (`loading.py`). Decode, apply EXIF orientation, convert any embedded ICC profile to sRGB
(relative colourimetric), read EXIF, then linearise with the sRGB curve. All measurements are in linear light.

**MTF** (`mtf.py`). Slanted-edge e-SFR on each of the four edges (luma, linear light): per-row edge
centroids → robust line fit → project pixels on the edge normal and bin at 0.25 px (4× oversampling) →
edge spread function → differentiate → Hamming window → FFT. MTF50 is the first crossing of 0.5, averaged
over the four edges. It is reported both in cycles per pixel and in **cycles per mm on the chart**:
the detected chart geometry gives the local image scale across each edge (pixels per chart unit,
perspective included, with 1 unit = 1 mm when the chart is printed at 100%), and cycles/mm =
cycles/pixel × pixels/mm. The regression flag uses cycles/mm so captures of different resolution
(e.g. the 24 MP main camera vs the 12 MP 2× tele) compare directly. The report also shows peak MTF, the highest value of
the curve above 0.02 cycles/pixel; values above 1 are sharpening overshoot.

**Colour** (`analysis.py`). One exposure gain, fitted by least squares on the two mid-grey patches, is applied
to all patches, then ΔE00 (Sharma et al. 2005) against the design values in D65 Lab. White-balance error
is the mean chroma √(a*²+b*²) of the four middle neutrals (white may clip, black is noisy).

**Noise.** Within the inner 60% (per side) of each patch, a plane is fitted and removed per channel, and
SNR = mean / residual std in linear light (reported in dB, also on luma).

**Regression and report** (`pipeline.py`, `report.py`). Flags compare each capture to the first; the report
has the summary table, an MTF overlay and a tone curve of the grey patches.

## Verification

Run `pytest` (126 tests). What is checked, and against what:

**CIEDE2000.** All 34 test pairs from Sharma, Wu & Dalal (2005), as published (`tests/data/sharma2005.txt`),
agree to 5·10⁻⁵ (the published precision), in both argument orders and in the vectorised path. sRGB↔Lab is
checked against known values (white, grey, primaries) and the D65 white point.

**MTF50 on synthetic edges.** Gaussian-blurred edges with a known analytic MTF (`tests/synth.py`);
expected MTF50 = √(ln2 / 2π²) / σ. Regenerate with `python tests/mtf_table.py`:

```
case                                    expected  measured    error
sigma=0.5 angle=+5                        0.3748    0.3788   +1.06%
sigma=0.5 angle=-7                        0.3748    0.3750   +0.07%
sigma=0.5 angle=+10                       0.3748    0.3755   +0.19%
sigma=1.0 angle=+5                        0.1874    0.1882   +0.41%
sigma=1.0 angle=-7                        0.1874    0.1880   +0.32%
sigma=1.0 angle=+10                       0.1874    0.1877   +0.15%
sigma=1.5 angle=+5                        0.1249    0.1255   +0.43%
sigma=1.5 angle=-7                        0.1249    0.1256   +0.57%
sigma=1.5 angle=+10                       0.1249    0.1253   +0.33%
sigma=2.0 angle=+5                        0.0937    0.0943   +0.68%
sigma=2.0 angle=-7                        0.0937    0.0945   +0.82%
sigma=2.0 angle=+10                       0.0937    0.0942   +0.56%
sigma=2.5 angle=+5                        0.0750    0.0758   +1.07%
sigma=2.5 angle=-7                        0.0750    0.0758   +1.14%
sigma=2.5 angle=+10                       0.0750    0.0756   +0.85%
sigma=0.5 horizontal 5deg                 0.3748    0.3788   +1.06%
sigma=0.5 flipped 5deg                    0.3748    0.3788   +1.06%
sigma=0.5 noisy(0.02, seed 0) 5deg        0.3748    0.3814   +1.76%
sigma=1.5 horizontal 5deg                 0.1249    0.1255   +0.43%
sigma=1.5 flipped 5deg                    0.1249    0.1255   +0.43%
sigma=1.5 noisy(0.02, seed 0) 5deg        0.1249    0.1252   +0.23%
sigma=2.5 horizontal 5deg                 0.0750    0.0758   +1.07%
sigma=2.5 flipped 5deg                    0.0750    0.0758   +1.07%
sigma=2.5 noisy(0.02, seed 0) 5deg        0.0750    0.0758   +1.13%
```

Worst case is +1.76% (σ = 0.5 with noise); the clean cases are within +1.2%. Errors are positive and
small, consistent with a finite Hamming window and bin-phase effects. Horizontal and polarity-flipped
edges give identical results to the vertical case. Over 30 seeds the noise adds no systematic bias; the
seed-to-seed s.d. of the MTF50 error at this noise level is 2.3% / 0.9% / 0.5% for σ = 0.5 / 1.5 / 2.5
(figures recorded in `tests/test_mtf.py`).

**End to end on simulated captures** (`synthetic.py`, `tests/test_pipeline.py`). Scenes with known blur,
noise, exposure, colour cast and perspective, with a light or black background. Measured MTF50 of the
tilted chart is within 4% of the analytic value including the pixel aperture (in practice within about 3%);
the single gain recovers the exposure change to 2%; pure exposure changes produce no white-balance error
or ΔE00 flag; blur raises only the MTF50 flag and a colour cast raises only the ΔE00 flag.

**Colour management.** A Display P3 profile built in the tests converts to the same sRGB values as an
independent numpy calculation (within 2.5 levels); HEIC round-trips ICC and EXIF; 16-bit input is rejected.

## Deviation from ISO 12233

The e-SFR here follows the ISO 12233 slanted-edge procedure (4× binned ESF, derivative, window, FFT,
correction for the finite-difference derivative) with **one addition: the MTF is also divided by the
0.25 px bin-averaging aperture, sinc(f·0.25)**. Averaging samples into 0.25 px bins is a box filter
that belongs to the measurement, not the camera, so it is removed. As I understand the standard it
corrects only the derivative filter; I have not had the standard text to hand to confirm this, so check
before citing numbers as ISO-conformant. The correction is small: it raises the MTF by 0.4% at 0.19
cycles/px, 0.8% at 0.28 and 1.5% at 0.375, which moves MTF50 by about 0.3% when MTF50 is 0.19
cycles/px and about 1% when it is 0.375. Results are in cycles/pixel and cycles/mm on the chart, with no
cycles/picture-height normalisation.

## Limitations

- **Little real-world testing.** Beyond the synthetic tests, it has only been run on one iPhone 15 Plus
  session (4 captures, JPEG, Display P3; see Results). Other cameras, lighting, glare and print
  quality are untested.
- **Colour is relative to the chart's design values**, not to a measured print. Printer, ink, paper and
  the light on the chart all contribute to ΔE00. It is meaningful for comparing captures made under the
  same light of the same print, not as an absolute colour accuracy. Lab is computed against a D65
  white, so a non-D65 light shows up as white-balance error.
- **Phone processing is in the measurement.** Sharpening, local tone mapping and noise reduction change
  MTF, SNR and ΔE00; sharpening shows up as MTF above 1 (the plot shows it). Results compare pipelines, not sensors.
- **Cycles/mm needs a known chart size and distance.** It assumes the chart was printed at 100%. The
  results in this README were shot off a screen, so they are in design units (see Results). It is also a
  measure at the chart, so it falls as the camera moves away: compare captures taken from the same distance. At a different distance or crop, only
  cycles/pixel at equal resolution is comparable. Edges that end up under 1° from an image axis (camera rolled ~4–6° against the chart) produce a
  warning and unreliable MTF.
- **The display limits what can be measured.** The chart in these results was shown on a MacBook Retina
  screen (~9 pixels per mm). If a design mm is about one screen mm, the screen's own pixel aperture alone
  would lower MTF by about 11% at 2.4 cycles/mm (sinc(2.4/9) = 0.89; my estimate, not measured), and its
  pixel grid sets a hard cap near 4.5 cycles/mm. The 1× vs 2× comparison is therefore partly limited by
  the screen, not only by the cameras, and the tele curve (which extends to about 3.8 cycles per design mm)
  is the most affected. The display's brightness, white point and gamut likewise enter the colour numbers,
  which measure phone plus screen together.
- **Repeatability comes from a single pair of shots** (plus one failed close-range pair). It gives an
  idea of the size of the scatter, not a distribution; the thresholds are not calibrated by it.
- **Grey SNR is noisy.** It varied by 3.2 dB between two identical shots, so any SNR regression threshold
  would have to be well above 3.2 dB. The tool currently has no SNR flag; SNR is reported only.
- **Noise SNR** is from a single frame with plane removal; fine-scale texture or compression artefacts
  count as noise, and a noiseless synthetic frame reports >100 dB.
- **8-bit only.** No RAW, 10-bit or HDR gain maps (the SDR base image is used). Out-of-gamut colours are clipped when converting
  to sRGB. ICC conversion assumes a matrix/TRC or LUT RGB profile that Pillow's LittleCMS can read.
- **Detection** needs the whole black frame visible and mostly unoccluded. Orientation relies on a small
  notch, which needs enough resolution to see.
- **Flag thresholds (10% / 1.5) are defaults, not calibrated.** Shoot the same scene several times to
  find your own repeatability before trusting them.
- The report plots the first 8 captures (fixed categorical palette); the table lists all.

## Results

Apple iPhone 15 Plus, main camera (26 mm equivalent) and 2× (52 mm), JPEG, Display P3 converted to sRGB. Edge
and colour charts were shot separately and analysed as one capture per folder (`photos/<n>/edge.jpeg` +
`colour.jpeg`). Baseline = `1-baseline`.

**These charts were not printed.** They were photographed off a MacBook screen at the same display size
for every capture. So "mm" below means the chart's **design units on screen**, not physical millimetres,
and "cy/mm" is **cycles per design mm**: it compares these captures with each other, but it is not a
physical resolution and depends on the display size and the camera-to-screen distance. Print the charts at 100%
(see Use) if you want physical cycles/mm.

| Capture | Settings (edge shot) | px per design mm | MTF50 (cy per design mm) | MTF50 (cy/px) | Peak MTF | Mean ΔE00 | WB error | Grey SNR | Flags |
|---|---|---|---|---|---|---|---|---|---|
| 1-baseline | 26 mm, ISO 80, 1/121 s | 5.63 | 2.35 | 0.417 | 1.43 | 3.22 | 0.85 | 22.9 dB | baseline |
| 2-repeat | same | 5.67 | 2.31 (−1.8%) | 0.407 | 1.36 | 3.32 | 0.97 | 26.1 dB | none |
| 3-tele | 52 mm, ISO 32, 1/121 s | 7.54 | 2.53 (+7.6%) | 0.335 | 1.58 | 3.25 | 2.34 | 22.8 dB | none |
| 4-dim | 26 mm, ISO 400, 1/60 s | 6.08 | 2.44 (+4.0%) | 0.402 | 1.25 | 2.99 | 2.01 | 20.5 dB | none |

In cycles/pixel the tele capture is 19.7% below the baseline, only because it has fewer, larger pixels across the
chart; in cycles per design mm it is 7.6% above. The peak MTF of 1.25–1.58 means the iPhone's JPEG pipeline sharpens
strongly, which also raises MTF50: compare captures with each other, not with lens specifications. The
tele and dim captures show a larger white-balance error (2.0–2.3 against 0.9–1.0), which could be a
real colour shift or a change in the light between shots; not investigated. (ISO and exposure for the
colour shots differ from the edge shots, e.g. tele ISO 50 vs 32; the table gives the edge shots.)

### Repeatability (baseline vs repeat, same settings)

| Shot distance | MTF50 baseline vs repeat | Notes |
|---|---|---|
| Far (chart frame ≈ 545 px wide) | −2.4% in cycles/px, −1.8% in cycles per design mm; per edge +0.9%, −8.1%, −2.4%, +0.4% | all 4 edges measured on both shots; edge-to-edge spread 4–6% |
| Close (chart frame ≈ 1200 px wide) | −34% | **not a real difference**: 2 of 4 edges failed on the baseline and one edge on the repeat read 0.011 cy/px, so the means are over different, partly broken edges |

For the close shots the horizontal edges gave absurd fitted angles (41°, −53°); the ROI crops showed a
fine pixel pattern in the white, consistent with photographing a screen, but I did not confirm the
cause. Treat the close-shot figure as a failure case, not repeatability. For the far shots, ΔE00 differed by
0.10 and WB error by 0.12, the exposure gain by 1.8%, and grey SNR by 3.2 dB between two identical shots,
so SNR is the noisiest of the metrics. This is one pair of shots, so it supports "the 10% MTF50 / 1.5 ΔE00 defaults are not
obviously too tight" but does not calibrate them; repeat the baseline several times (and the colour chart
under the same light) before relying on the thresholds.

## Layout

`camera_iq/`: `charts.py` (render, detect), `export.py` (print files), `loading.py`, `mtf.py`, `color.py`,
`analysis.py` (colour, noise), `pipeline.py`, `report.py`, `config.py`, `synthetic.py`, `cli.py`.
`tests/`: unit and end-to-end tests, `mtf_table.py`, `data/`. `PLAN.md`: design decisions.
`DEVLOG.md`: development log, including where the agent's output or the plan needed correcting.
