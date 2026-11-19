"""Depth Anything V2 relative-depth estimator (Hugging Face ``transformers``).

Depth Anything V2 predicts **relative disparity** (affine-invariant inverse depth:
higher = closer, unknown scale and shift), *not* meters. That is why this class sets
``output_kind = "disparity"``: ``yolo_depth_underwater.evaluation`` aligns it to the
reference (scale + shift in disparity space) before computing metrics, exactly as
for the other models.

Needs the ``depth-anything`` extra (``transformers``). Images are BGR ``uint8``
arrays, like in the other ``InferenceModel`` implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from yolo_depth_underwater import inference_model
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split


class DepthAnythingVariant(StrEnum):
    # Check the license of each checkpoint before using it beyond coursework.
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


_CHECKPOINT_FSTR = "depth-anything/Depth-Anything-V2-{}-hf"


class DepthAnythingV2Model(inference_model.InferenceModel):
    """Depth Anything V2 (zero-shot, relative disparity)."""

    output_kind = "disparity"

    def __init__(
        self,
        dataset: DepthDataset | None = None,
        variant: str | DepthAnythingVariant = DepthAnythingVariant.SMALL,
        device: str | None = None,
    ) -> None:
        variant = DepthAnythingVariant(variant)
        self._dataset = dataset
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = _CHECKPOINT_FSTR.format(variant.value.capitalize())
        print(f"Starting Depth Anything V2: {checkpoint}")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(checkpoint)
            .to(self._device)
            .eval()
        )

    def _predict_image(self, image: np.ndarray) -> np.ndarray:
        """BGR ``uint8`` image -> float32 disparity map at the image resolution."""
        rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        disparity = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=rgb.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        return disparity.squeeze().float().cpu().numpy()

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (self._predict_image(image) for image in images)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        """Like the other models, but note ``DepthSample.depth`` holds **disparity**
        here (see ``output_kind``), not meters."""
        if self._dataset is None:
            raise ValueError("predict_split needs a DepthDataset")
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))
