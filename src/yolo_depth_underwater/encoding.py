"""Encoding helpers for the scaled uint16 depth PNG convention.

Ultralytics stores depth as uint16 PNG where ``value = depth * scale`` and
``0`` marks invalid pixels; 
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics.data.utils import load_depth

DEPTH_SCALE = 1000.0  # Ultralytics' default: uint16 PNG value = depth * scale
MAX_ENCODABLE = 65535 / DEPTH_SCALE  # 65.535 m, the uint16 ceiling at scale 1000

LOGGER = logging.getLogger(__name__)


def to_uint16(depth: np.ndarray, scale: float = DEPTH_SCALE) -> np.ndarray:
    """Encode float depth to the scaled uint16 PNG convention (0 stays invalid)."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    if valid.any():
        encoded[valid] = np.clip(np.rint(depth[valid] * scale), 1, 65535).astype(
            np.uint16
        )
    return encoded


def save_depth_png(
    path: str | Path, depth: np.ndarray, scale: float = DEPTH_SCALE
) -> Path:
    """Save float depth meters as a scaled uint16 PNG."""
    depth = np.asarray(depth, dtype=np.float32)
    ceiling = 65535 / scale
    too_far = np.count_nonzero(np.isfinite(depth) & (depth > ceiling))
    bad = np.count_nonzero(~np.isfinite(depth) | (depth <= 0))
    if too_far or bad:
        LOGGER.warning(
            "%s: %d px above %.2f m clipped, %d px invalid set to 0 "
            "(range %.3f..%.3f m)",
            Path(path).name,
            too_far,
            ceiling,
            bad,
            float(np.nanmin(depth)),
            float(np.nanmax(depth)),
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint16(depth, scale)).save(path, format="PNG")
    return path


def load_depth_img(path: str | Path, scale: float = DEPTH_SCALE) -> np.ndarray:
    """Load a scaled uint16 depth PNG as float32 **meters** (0 = invalid)"""
    return load_depth(Path(path), scale)
