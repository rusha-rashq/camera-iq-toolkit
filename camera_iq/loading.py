"""Image loading: 8-bit sRGB only, EXIF orientation applied, returned as sRGB and linear light."""
from dataclasses import dataclass

import cv2
import numpy as np

from .color import srgb_to_linear


class UnsupportedImage(ValueError):
    pass


@dataclass
class Capture:
    srgb8: np.ndarray       # (H, W, 3) uint8, RGB, display-encoded; used for chart detection
    linear: np.ndarray      # (H, W, 3) float32, sRGB curve removed; used for all measurements


def _from_array(img):
    if img is None:
        raise UnsupportedImage("could not decode image")
    if img.dtype != np.uint8:
        raise UnsupportedImage(f"only 8-bit images are supported, got {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.ndim == 3 else cv2.COLOR_GRAY2RGB)
    lut = srgb_to_linear(np.arange(256) / 255).astype(np.float32)
    return Capture(rgb, lut[rgb])


# IMREAD_COLOR applies EXIF orientation; ANYDEPTH keeps 16-bit data so we can reject it rather than
# have it silently squashed to 8 bits. (IMREAD_UNCHANGED would skip the orientation.)
_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH


def load_capture(path):
    """The file is assumed to be sRGB: embedded ICC profiles are not read."""
    data = np.fromfile(path, np.uint8)       # imread chokes on non-ASCII paths on some platforms
    return load_capture_bytes(data)


def load_capture_bytes(data):
    return _from_array(cv2.imdecode(np.frombuffer(data, np.uint8), _FLAGS))
