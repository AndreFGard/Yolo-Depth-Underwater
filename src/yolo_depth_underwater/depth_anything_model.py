"""Depth Anything V2 relative-depth estimator (Hugging Face ``transformers``).

Depth Anything V2 predicts **relative disparity** (affine-invariant inverse depth:
higher = closer, unknown scale and shift), *not* meters. That is why this class returns
``OutputKind.DISPARITY``: ``yolo_depth_underwater.evaluation`` aligns it to the
reference (scale + shift in disparity space) before computing metrics, exactly as
for the other models.

Needs the ``depth-anything`` extra (``transformers``). Images are BGR ``uint8``
arrays, like in the other ``InferenceModel`` implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
import itertools

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from yolo_depth_underwater import inference_model
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.encoding import DEPTH_SCALE
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split

DEFAULT_EPS = 0.25
# Relative depth is stored in label units: ``save_depth_png`` scales by
# DEPTH_SCALE, so one unit of relative depth becomes 65535 uint16, matching the
# USOD10k labels written by ``scripts/convert_usd10k_to_coco.py``.
_RELATIVE_DEPTH_UNITS = 65535.0 / DEPTH_SCALE


def _disparity_to_relative_depth(
    disparity: np.ndarray, eps: float = DEFAULT_EPS
) -> np.ndarray:
    """Map a DA V2 disparity map to a relative depth"""
    disp = np.asarray(disparity, dtype=np.float64)
    lo, hi = float(disp.min()), float(disp.max())
    d = (disp - lo) / max(hi - lo, 1e-9)
    rel = eps / (d + eps) * _RELATIVE_DEPTH_UNITS
    return np.minimum(rel, np.nextafter(_RELATIVE_DEPTH_UNITS, 0.0)).astype(np.float32)


class DepthAnythingVariant(StrEnum):
    # Check the license of each checkpoint before using it beyond coursework.
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


_CHECKPOINT_FSTR = "depth-anything/Depth-Anything-V2-{}-hf"


class DepthAnythingV2Model(inference_model.InferenceModel):
    """Depth Anything V2 (zero-shot, relative disparity)."""

    @property
    def output_kind(self) -> inference_model.OutputKind:
        return inference_model.OutputKind.DISPARITY

    def __init__(
        self,
        dataset: DepthDataset | None = None,
        variant: str | DepthAnythingVariant = DepthAnythingVariant.SMALL,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        variant = DepthAnythingVariant(variant)
        self._dataset = dataset
        self._batch_size = max(1, batch_size)
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
        return self._predict_batch([image])[0]

    def _predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Several BGR ``uint8`` images -> float32 disparity maps"""
        rgbs = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1])) for im in images]
        inputs = self._processor(images=rgbs, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            predicted = self._model(**inputs).predicted_depth
        return [
            torch.nn.functional.interpolate(
                predicted[i].unsqueeze(0).unsqueeze(0),
                size=rgb.size[::-1],  # (height, width)
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .float()
            .cpu()
            .numpy()
            for i, rgb in enumerate(rgbs)
        ]

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        it = iter(images)
        while batch := list(itertools.islice(it, self._batch_size)):
            yield from self._predict_batch(batch)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        """Like the other models, but note ``DepthSample.depth`` holds **disparity**
        here (see ``output_kind``), not meters."""
        if self._dataset is None:
            raise ValueError("predict_split needs a DepthDataset")
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))


class DepthAnythingV2RelativeDepthModel(inference_model.InferenceModel):
    """Depth Anything V2 remapped to output relative depth, for convenience."""

    @property
    def output_kind(self) -> inference_model.OutputKind:
        return inference_model.OutputKind.DEPTH

    def __init__(
        self,
        dataset: DepthDataset | None = None,
        variant: str | DepthAnythingVariant = DepthAnythingVariant.LARGE,
        eps: float = DEFAULT_EPS,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self._model = DepthAnythingV2Model(dataset, variant, device, batch_size)
        self._eps = eps

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (
            _disparity_to_relative_depth(disparity, self._eps)
            for disparity in self._model.predict(images)
        )

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        return (
            replace(sample, depth=_disparity_to_relative_depth(sample.depth, self._eps))
            for sample in self._model.predict_split(split)
        )
