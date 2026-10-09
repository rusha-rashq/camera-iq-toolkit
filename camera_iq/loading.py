"""Image loading: 8-bit, EXIF orientation applied, embedded ICC profile converted to sRGB,
EXIF capture metadata read, returned as sRGB and linear light. Untagged images are assumed sRGB."""
import io
import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .color import srgb_to_linear

_SRGB = ImageCms.createProfile("sRGB")
_EXIF_IFD = 0x8769


class UnsupportedImage(ValueError):
    pass


@dataclass
class CaptureMeta:
    camera: str | None = None           # "Make Model"
    iso: int | None = None
    exposure_time: float | None = None  # seconds
    focal_length: float | None = None   # mm
    f_number: float | None = None
    profile: str | None = None          # description of the embedded ICC profile, if any


@dataclass
class Capture:
    srgb8: np.ndarray       # (H, W, 3) uint8, RGB, display-encoded; used for chart detection
    linear: np.ndarray      # (H, W, 3) float32, sRGB curve removed; used for all measurements
    meta: CaptureMeta


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_meta(im):
    exif = im.getexif()
    ifd = exif.get_ifd(_EXIF_IFD) if exif else {}
    camera = " ".join(str(exif.get(t)).strip() for t in (271, 272) if exif.get(t))
    if camera:       # models often repeat the make ("Apple iPhone 15" vs make "Apple")
        make, model = str(exif.get(271, "")).strip(), str(exif.get(272, "")).strip()
        camera = model if model.lower().startswith(make.lower()) else camera
    iso = ifd.get(34855)
    iso = iso[0] if isinstance(iso, (tuple, list)) and iso else iso
    return CaptureMeta(camera or None, int(iso) if iso else None, _num(ifd.get(33434)),
                       _num(ifd.get(37386)), _num(ifd.get(33437)))


def _to_srgb(im, meta):
    """Convert to 8-bit sRGB using the embedded profile (if any); drops alpha."""
    icc = im.info.get("icc_profile")
    if im.mode in ("RGBA", "LA", "P", "L", "1"):
        im = im.convert("RGB")
    if im.mode != "RGB":
        raise UnsupportedImage(f"unsupported image mode {im.mode}")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        meta.profile = ImageCms.getProfileDescription(src).strip() or "unnamed profile"
        if src.profile.xcolor_space.strip() != "RGB":
            raise ImageCms.PyCMSError("not an RGB profile")
        if "srgb" in meta.profile.lower():
            return im
        return ImageCms.profileToProfile(im, src, _SRGB, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                         outputMode="RGB")
    except ImageCms.PyCMSError as e:
        warnings.warn(f"could not use embedded ICC profile ({e}); assuming sRGB")
        return im


def _reject_deep(data, im):
    if im.mode.startswith(("I", "F")) or ";16" in im.mode:
        raise UnsupportedImage(f"only 8-bit images are supported, got mode {im.mode}")
    # PIL silently reduces 16-bit RGB PNGs to 8 bits; read the bit depth from the header instead.
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[24] != 8 and im.mode != "P":
        raise UnsupportedImage(f"only 8-bit images are supported, PNG has {data[24]} bits per channel")


def load_capture_bytes(data):
    data = bytes(data)
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise UnsupportedImage(f"could not decode image: {e}") from e
    _reject_deep(data, im)
    meta = _read_meta(im)
    im = ImageOps.exif_transpose(im)        # keeps the ICC profile in .info
    rgb = np.asarray(_to_srgb(im, meta))
    lut = srgb_to_linear(np.arange(256) / 255).astype(np.float32)
    return Capture(rgb, lut[rgb], meta)


def load_capture(path):
    with open(path, "rb") as f:
        return load_capture_bytes(f.read())
