"""YOLO depth estimator over a ``DepthDataset``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import Any, Protocol, cast

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


class DepthHead(Protocol):
    cal_a: torch.Tensor
    cal_b: torch.Tensor


def depth_head(model: object) -> DepthHead | None:
    """The depth head carrying the ``cal_a``/``cal_b`` buffers, if there is one."""
    seq = getattr(model, "model", model)
    if not isinstance(seq, torch.nn.Sequential):
        return None
    head = seq[-1]
    if not isinstance(head, torch.nn.Module):
        return None
    if not isinstance(getattr(head, "cal_a", None), torch.Tensor):
        return None
    if not isinstance(getattr(head, "cal_b", None), torch.Tensor):
        return None
    return cast(DepthHead, head)


def _scalar(buffer: torch.Tensor) -> float:
    return float(buffer.reshape(-1)[0].item())


class YoloModel(inference_model.InferenceModel):
    """YOLO depth estimator."""

    def __init__(
        self,
        dataset: DepthDataset | None = None,
        checkpoint: str | YoloCheckpoints = YoloCheckpoints.NANO,
        device: str | torch.device | None = None,
    ) -> None:
        self._checkpoint: str = str(checkpoint)
        self._dataset: DepthDataset | None = dataset
        self._device: torch.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Starting yolo: {self._checkpoint}")

        self._model: Any = ultralytics.YOLO(self._checkpoint).to(self._device)

    @property
    def output_kind(self) -> inference_model.OutputKind:
        return inference_model.OutputKind.DEPTH

    def _predict_image(self, image: np.ndarray) -> np.ndarray:
        raw: Any = self._model.predict(image, device=self._device, verbose=False)
        results: list[Any] = list(raw) if not isinstance(raw, list) else raw
        first: Any = results[0]
        if isinstance(first, torch.Tensor):
            raise RuntimeError(f"{self._checkpoint} did not return a depth map")
        depth_map: Any = first.depth
        if depth_map is None:
            raise RuntimeError(f"{self._checkpoint} did not return a depth map")
        data: Any = depth_map.data
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)
        if not isinstance(data, torch.Tensor):
            raise RuntimeError(f"{self._checkpoint} returned depth of type {type(data)}")
        return data.float().cpu().numpy()

    def calibration(self) -> tuple[float, float]:
        """The ``(cal_a, cal_b)`` this checkpoint applies in eval mode."""
        head = depth_head(self._model.model)
        if head is None:
            raise RuntimeError(f"{self._checkpoint} has no depth-head calibration")
        return _scalar(head.cal_a), _scalar(head.cal_b)

    def predict(self, images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        return (self._predict_image(image) for image in images)

    def predict_split(self, split: Split) -> Iterator[DepthSample]:
        if self._dataset is None:
            raise ValueError("predict_split needs a DepthDataset")
        for sample in self._dataset.depth_samples(split):
            yield replace(sample, depth=self._predict_image(sample.image))
