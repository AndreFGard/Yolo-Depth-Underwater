"""One native-resolution image/depth pair."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yolo_depth_underwater.split import Split


@dataclass(frozen=True)
class DepthSample:
    split: Split
    index: int
    image: np.ndarray
    """float32 depth"""
    depth: np.ndarray
    image_path: Path
    depth_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    @property
    def valid_depth_mask(self) -> np.ndarray:
        return self.depth > 0

    @property
    def stats(self) -> dict[str, float | int]:
        """Return summary statistics over the valid depth pixels."""
        valid = self.depth[self.valid_depth_mask]
        if valid.size == 0:
            return {
                "count": 0,
                "min": 0.0,
                "median": 0.0,
                "max": 0.0,
                "valid_frac": 0.0,
            }
        return {
            "count": int(valid.size),
            "min": float(valid.min()),
            "median": float(np.median(valid)),
            "max": float(valid.max()),
            "valid_frac": float(valid.size / self.depth.size),
        }
