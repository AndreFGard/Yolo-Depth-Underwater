"""UniDepthV2 metric depth estimator over a ``DepthDataset``."""

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import torch
from unidepth.models import UniDepthV2

from yolo_depth_underwater import inference_model
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split

MAX_FRACTION = 0.01  # Hard cap: never process more than 1% of a split.


class UniDepthV2Model(inference_model.InferenceModel):
    """UniDepthV2 metric depth estimator."""

    def __init__(
        self,
        dataset: DepthDataset,
        device: str | None = None,
    ) -> None:
        self._dataset = dataset
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = "lpiccinelli/unidepth-v2-vitl14"
        self._model = UniDepthV2.from_pretrained(checkpoint).to(self._device).eval()

    def _predict_image(self, image: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        with torch.inference_mode():
            prediction = self._model.infer(tensor, None)
        return prediction["depth"].float().squeeze().cpu().numpy()

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (self._predict_image(image) for image in images)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))
