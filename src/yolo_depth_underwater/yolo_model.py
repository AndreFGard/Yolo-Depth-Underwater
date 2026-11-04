"""YOLO depth estimator over a ``DepthDataset``."""

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import torch
import ultralytics

from yolo_depth_underwater import inference_model
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split

_CHECKPOINT = "yolo26n-depth.pt"


class YoloModel(inference_model.InferenceModel):
    """YOLO depth estimator."""

    def __init__(
        self,
        dataset: DepthDataset,
        device: str | None = None,
    ) -> None:

        self._dataset = dataset
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = ultralytics.YOLO(_CHECKPOINT).to(self._device)

    def _predict_image(self, image: np.ndarray) -> np.ndarray:
        results = self._model.predict(image, device=self._device, verbose=False)
        depth = results[0].depth
        if depth is None:
            raise RuntimeError(f"{_CHECKPOINT} did not return a depth map")
        return depth.data.float().cpu().numpy()

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (self._predict_image(image) for image in images)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))
