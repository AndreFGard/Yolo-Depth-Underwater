"""YOLO depth estimator over a ``DepthDataset``."""

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum

import numpy as np
import torch
import ultralytics

from yolo_depth_underwater import inference_model
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split

class YoloCheckpoints(StrEnum):
    NANO = "yolo26n-depth.pt"
    SMALL = "yolo26s-depth.pt"
    EXTRA = "yolo26x-depth.pt"

class YoloModel(inference_model.InferenceModel):
    """YOLO depth estimator."""

    @property
    def output_kind(self) -> inference_model.OutputKind:
        return inference_model.OutputKind.DEPTH

    def __init__(
        self,
        dataset: DepthDataset | None = None,
        checkpoint: str =YoloCheckpoints.NANO,
        device: str | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._dataset = dataset
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Starting yolo: {checkpoint}")

        self._model = ultralytics.YOLO(checkpoint).to(
            self._device
        )

    def _predict_image(self, image: np.ndarray) -> np.ndarray:
        results = self._model.predict(image, device=self._device, verbose=False)
        depth = results[0].depth
        if depth is None:
            raise RuntimeError(f"{self._checkpoint} did not return a depth map")
        return depth.data.float().cpu().numpy()

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (self._predict_image(image) for image in images)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        if self._dataset is None:
            raise ValueError("predict_split needs a DepthDataset")
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))
