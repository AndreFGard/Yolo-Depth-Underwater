"""Encoding helpers for depth PNGs and tabular artifacts.

Ultralytics stores depth as uint16 PNG where ``value = depth * scale`` and
``0`` marks invalid pixels.
"""

import csv
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pandas as pd
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
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        raise ValueError(
            f"Depth img '{path}' is a tiff, but this uint16-PNG loader does not support FLSea images"
        )
    return load_depth(path, scale)


def _format_value(v: Any, float_places: int | None) -> Any:
    _FLOAT_TYPES: tuple[type, ...] = (float, np.floating)
    _INT_TYPES: tuple[type, ...] = (int, np.integer)
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, _FLOAT_TYPES):
        fv = float(v)
        if float_places is None:
            return repr(fv)
        return f"{fv:.{float_places}f}"
    if isinstance(v, _INT_TYPES):
        return str(int(v))
    if isinstance(v, Path):
        return str(v)
    return v


def write_csv(
    rows: Iterable[dict[str, Any]] | pd.DataFrame,
    path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    float_places: int | None = 6,
    delimiter: str = ",",
) -> Path:
    """Write rows in incoming order with fixed rounding so they can be versioned."""

    path = Path(path)
    if isinstance(rows, pd.DataFrame):
        if columns is None:
            columns = list(rows.columns)
        records: list[dict] = rows.to_dict(orient="records")
    else:
        records = list(rows)
        if columns is None:
            if not records:
                raise ValueError("write_csv: empty rows need explicit columns=")
            first = records[0]
            if not isinstance(first, Mapping):
                raise TypeError(
                    f"write_csv: expected Mapping rows, got {type(first)!r}"
                )
            columns = list(first.keys())

    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            delimiter=delimiter,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="raise",
        )
        writer.writeheader()
        for r in records:
            writer.writerow({k: _format_value(r.get(k), float_places) for k in columns})
    os.replace(tmp, path)
    return path
