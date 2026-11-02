"""One native-resolution image/depth pair."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yolo_depth_underwater.encoding import DEPTH_SCALE, save_depth_png
from yolo_depth_underwater.split import Split




@dataclass(frozen=True)
class DepthSample:
    """One image/depth pair at native resolution.

    Attributes:
        split: Which split the sample came from.
        index: Index within that split (a value yielded by ``DepthData.indices``).
        image: ``(H, W, 3)`` uint8 BGR image.
        depth: ``(H, W)`` float32 depth; ``0`` marks invalid/unlabeled pixels.
        image_path: Source image path.
        depth_path: Paired depth path.
    """

    split: Split
    index: int
    image: np.ndarray
    depth: np.ndarray
    image_path: Path
    depth_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        """Return the original ``(height, width)`` of the pair."""
        return self.image.shape[:2]

    @property
    def valid_mask(self) -> np.ndarray:
        """Return the boolean mask of pixels carrying a valid depth."""
        return self.depth > 0

    @property
    def stats(self) -> dict[str, float | int]:
        """Return summary statistics over the valid depth pixels."""
        valid = self.depth[self.valid_mask]
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

    def save_depth(self, path: str | Path, scale: float = DEPTH_SCALE) -> Path:
        """Write this sample's depth to a scaled uint16 PNG."""
        return save_depth_png(path, self.depth, scale)
