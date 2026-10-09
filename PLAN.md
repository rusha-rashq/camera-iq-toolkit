# camera-iq-toolkit plan

## Steps
1. Scaffold — done
2. Colour science (sRGB/Lab, CIEDE2000) — done
3. Slanted-edge MTF (e-SFR) — done
4. `charts.py`: chart rendering and detection
5. Loading (`io`), colour analysis, noise analysis
6. Regression comparison and report
7. `synthetic.py` end-to-end tests, CLI

## Decisions

**Charts**
- Two charts: a 5° slanted square and a 24-patch colour chart.
- Each has a thick black frame and a small corner notch for orientation.
- Reference colours are the chart's own design sRGB values.

**Detection**
- Threshold, then find the frame quadrilateral.
- Must be robust to a dark background merging with the frame: try the inner frame boundary, or verify candidates against the rendered chart.
- Rectify only to locate ROIs. MTF is computed on the original un-warped pixels.

**Loading**
- 8-bit only; apply EXIF orientation; convert any embedded ICC profile (e.g. iPhone Display P3) to sRGB, assume sRGB if untagged; read EXIF camera/ISO/exposure/focal length for the report; linearise with the sRGB curve.

**Colour**
- Normalise exposure with a single gain matched on the mid-grey patches before computing ΔE00.
- WB error = chroma of a*/b* on the four middle neutrals (white may clip, black is noisy).

**Noise**
- Fit and remove a plane within each patch before computing SNR.
- Linear light, inner 60% of each patch.

**Regression flags** (vs the first capture)
- MTF50 down more than 10%.
- Mean ΔE00 up more than 1.5.
- Thresholds configurable in one place.

**Report**
- Inline SVG MTF overlay plus a tone curve for the grey patches. No JS.

**Testing**
- `synthetic.py` simulates captures with known blur, noise, colour cast and perspective for end-to-end tests.
