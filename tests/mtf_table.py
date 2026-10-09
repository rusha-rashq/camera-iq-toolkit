"""Print expected vs measured MTF50 for every synthetic case: python tests/mtf_table.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without the editable install

from synth import blurred_edge, expected_mtf50

from camera_iq.mtf import edge_mtf

NOISE, SEED = 0.02, 0


def cases():
    for s in (0.5, 1.0, 1.5, 2.0, 2.5):
        for a in (5, -7, 10):
            yield f"sigma={s} angle={a:+d}", s, dict(angle_deg=a)
    for s in (0.5, 1.5, 2.5):
        yield f"sigma={s} horizontal 5deg", s, dict(angle_deg=5, horizontal=True)
        yield f"sigma={s} flipped 5deg", s, dict(angle_deg=5, flip=True)
        yield f"sigma={s} noisy({NOISE}, seed {SEED}) 5deg", s, dict(angle_deg=5, noise=NOISE, seed=SEED)


if __name__ == "__main__":
    print(f"{'case':38s} {'expected':>9s} {'measured':>9s} {'error':>8s}")
    for name, s, kw in cases():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = edge_mtf(blurred_edge(s, **kw))
        e = expected_mtf50(s)
        print(f"{name:38s} {e:9.4f} {r.mtf50:9.4f} {100 * (r.mtf50 / e - 1):+7.2f}%")
