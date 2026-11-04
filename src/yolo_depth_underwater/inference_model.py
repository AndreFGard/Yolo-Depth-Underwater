from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from yolo_depth_underwater import sample, split


class InferenceModel(ABC):
    @abstractmethod
    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Yield float32 depth maps"""

    @abstractmethod
    def predict_split(self, split: split.Split) -> Iterator[sample.DepthSample]:
        """Yield ``DepthSample`` objects with predicted depth for a whole split."""
