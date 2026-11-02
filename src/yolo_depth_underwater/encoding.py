"""Encoding helpers for the scaled uint16 depth PNG convention.

Ultralytics stores depth as uint16 PNG where ``value = depth * scale`` and
``0`` marks invalid pixels; ``.npy`` files hold float depth directly.
``load_depth`` is re-exported so callers get read and write from one place.
"""

from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_SCALE = 1000.0  # Ultralytics' default: uint16 PNG value = depth * scale


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
    """Save float depth as a scaled uint16 PNG, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint16(depth, scale)).save(path, format="PNG")
    return path
