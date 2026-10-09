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

   Thresholds default to MTF50 down >10% and mean ΔE00 up >1.5; override with `--mtf-drop` / `--de-rise`
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
over the four edges. Units are cycles per pixel.

**Colour** (`analysis.py`). One exposure gain, fitted by least squares on the two mid-grey patches, is applied
to all patches, then ΔE00 (Sharma et al. 2005) against the design values in D65 Lab. White-balance error
is the mean chroma √(a*²+b*²) of the four middle neutrals (white may clip, black is noisy).

**Noise.** Within the inner 60% (per side) of each patch, a plane is fitted and removed per channel, and
SNR = mean / residual std in linear light (reported in dB, also on luma).

**Regression and report** (`pipeline.py`, `report.py`). Flags compare each capture to the first; the report
has the summary table, an MTF overlay and a tone curve of the grey patches.

## Verification

Run `pytest` (119 tests). What is checked, and against what:

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
cycles/px and about 1% when it is 0.375. Results are in cycles/pixel, with no cycles/picture-height
normalisation.

## Limitations

- **Only synthetic images have been tested.** No real photograph has been through detection, MTF or the
  thresholds yet; real lenses, glare, paper texture and phone processing may break assumptions.
- **Colour is relative to the chart's design values**, not to a measured print. Printer, ink, paper and
  the light on the chart all contribute to ΔE00. It is meaningful for comparing captures made under the
  same light of the same print, not as an absolute colour accuracy. Lab is computed against a D65
  white, so a non-D65 light shows up as white-balance error.
- **Phone processing is in the measurement.** Sharpening, local tone mapping and noise reduction change
  MTF, SNR and ΔE00; MTF values above 1 are clipped in the plot. Results compare pipelines, not sensors.
- **MTF is in cycles/pixel**, so captures at different resolutions or crops are not directly comparable.
  Edges that end up under 1° from an image axis (camera rolled ~4–6° against the chart) produce a
  warning and unreliable MTF.
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

_To be filled in with real photos._

| Capture | Device / settings | MTF50 (cy/px) | Mean ΔE00 | WB error | Grey SNR | Flags |
|---|---|---|---|---|---|---|
| | | | | | | |

Repeatability (same scene shot N times): _MTF50 spread, ΔE00 spread_ → suggested thresholds: _tbd_

## Layout

`camera_iq/`: `charts.py` (render, detect), `export.py` (print files), `loading.py`, `mtf.py`, `color.py`,
`analysis.py` (colour, noise), `pipeline.py`, `report.py`, `config.py`, `synthetic.py`, `cli.py`.
`tests/`: unit and end-to-end tests, `mtf_table.py`, `data/`. `PLAN.md`: design decisions.
