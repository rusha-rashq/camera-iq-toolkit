"""Build a minimal ICC v2 matrix/TRC RGB profile (sRGB transfer curve) for any set of primaries."""
import struct

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
D65_XY = (0.3127, 0.3290)
P3_XY = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))


def _xyz(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz_d65(prim_xy, white_xy=D65_XY):
    P = np.array([_xyz(*p) for p in prim_xy]).T
    S = np.linalg.solve(P, _xyz(*white_xy))
    return P * S


def adapt_to_d50(M, white_xy=D65_XY):
    src, dst = BRADFORD @ _xyz(*white_xy), BRADFORD @ D50
    return np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD @ M


def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(v):
    return b"XYZ \0\0\0\0" + b"".join(_s15(c) for c in v)


def _srgb_curve(n=1024):
    t = np.linspace(0, 1, n)
    y = np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
    return b"curv\0\0\0\0" + struct.pack(">I", n) + b"".join(struct.pack(">H", int(round(v * 65535))) for v in y)


def _desc(text):
    a = text.encode() + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(a)) + a + b"\0" * 8 + b"\0\0\0" + b"\0" * 67


def make_profile(description, prim_xy=P3_XY):
    M = adapt_to_d50(rgb_to_xyz_d65(prim_xy))
    curve = _srgb_curve()
    tags = {b"desc": _desc(description), b"cprt": b"text\0\0\0\0test\0", b"wtpt": _xyz_tag(D50),
            b"rXYZ": _xyz_tag(M[:, 0]), b"gXYZ": _xyz_tag(M[:, 1]), b"bXYZ": _xyz_tag(M[:, 2]),
            b"rTRC": curve, b"gTRC": curve, b"bTRC": curve}
    off = 128 + 4 + 12 * len(tags)
    table, body, placed = b"", b"", {}
    for sig, data in tags.items():
        if data is curve and "curve" in placed:
            o = placed["curve"]
        else:
            o = off + len(body)
            body += data + b"\0" * (-len(data) % 4)
            if data is curve:
                placed["curve"] = o
        table += sig + struct.pack(">II", o, len(data))
    size = off + len(body)
    hdr = (struct.pack(">I", size) + b"\0\0\0\0" + b"\x02\x10\0\0" + b"mntr" + b"RGB " + b"XYZ " + b"\0" * 12
           + b"acsp" + b"\0" * 24 + b"\0\0\0\0" + b"".join(_s15(c) for c in D50) + b"\0" * 48)
    assert len(hdr) == 128
    return hdr + struct.pack(">I", len(tags)) + table + body
