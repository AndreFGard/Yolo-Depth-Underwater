from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import StrEnum

import numpy as np

from yolo_depth_underwater import sample, split


class OutputKind(StrEnum):
    DEPTH = "depth"
    DISPARITY = "disparity"


class InferenceModel(ABC):
    """A depth model that can be scored by ``evaluation``."""

    @property
    @abstractmethod
    def output_kind(self) -> OutputKind: ...
    @abstractmethod
    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Yield float32 depth maps"""

    @abstractmethod
    def predict_split(self, split: split.Split) -> Iterator[sample.DepthSample]:
        """Yield ``DepthSample`` objects with predicted depth for a whole split."""
