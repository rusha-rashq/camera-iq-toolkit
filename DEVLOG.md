# Development log

Built in one evening with Claude Code as the coding agent. I set the scope,
reviewed each step, and verified results before moving on. This log records
where the agent's output or my plan needed correction.

## Planning
- Asked for a plan and an explanation of slanted-edge MTF before any code.
- Changes I made to the agent's plan: compute MTF on original (un-warped)
  pixels, since warping interpolates and biases MTF; add a synthetic capture
  simulator for end-to-end tests; normalise exposure before ΔE; remove a
  lighting plane before measuring noise.

## Colour science
- The agent fetched the Sharma (2005) CIEDE2000 data with curl rather than a
  summarising web tool, to avoid corrupted numbers, and checked its shape (34 rows).
- One of its tests failed: the expected value was for 128/255, not 0.5. The code
  was right; the test input was fixed without loosening the tolerance.

## Slanted-edge MTF
- A noisy test at σ = 0.5 px showed a −2.2% bias. The agent's first diagnosis
  (line-fit error) was disproved by measurement. The real cause was the 0.25 px
  binning acting as a small blur; corrected with a sinc divisor (a deviation
  from ISO 12233, documented in the README).
- Final accuracy against the analytic Gaussian MTF: within +1.8% across all cases.

## Charts, loading, colour
- Switched the slanted edge from ~270:1 to 4:1 contrast (ISO 12233 practice);
  this broke Otsu thresholding, which the agent fixed.
- The agent found two design neutrals were off by one level and corrected them.
- My catch: iPhone photos are Display P3, not sRGB. Added ICC conversion so
  saturated patches don't show false colour error.
- White-balance error moved to the four middle neutrals (white can clip, black is noisy).

## Environment
- iCloud Desktop sync kept hiding the editable install's .pth file, breaking
  imports. Moved the project out of ~/Desktop.

## Real photos
- First edge shots (close to the screen): baseline vs repeat differed 34%, and
  two edges per photo failed. The repeatability check caught an invalid
  measurement: screen-pixel aliasing.
- Reshot from farther away: baseline vs repeat agree within 2.4%.
- My catch: captures were different resolutions (24 MP vs 12 MP), so cycles/pixel
  wasn't comparable. Switched to cycles per chart mm. In cycles/pixel the 2x
  capture looked 19.7% worse; in cycles/mm it was 7.6% better.
