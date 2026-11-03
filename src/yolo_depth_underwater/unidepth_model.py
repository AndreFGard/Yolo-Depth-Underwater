"""UniDepthV2 metric depth estimator over a ``DepthDataset``."""

from collections.abc import Iterator
from dataclasses import replace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from unidepth.models import UniDepthV2

from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.encoding import DEPTH_SCALE, save_depth_png
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split

MAX_FRACTION = 0.01  # Hard cap: never process more than 1% of a split.


class UniDepthV2Model:
    """UniDepthV2 metric depth estimator."""

    def __init__(
        self,
        dataset: DepthDataset,
        device: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = "lpiccinelli/unidepth-v2-vitl14"
        self.model = UniDepthV2.from_pretrained(checkpoint).to(self.device).eval()

    def predict(self, split: Split) -> Iterator[DepthSample]:
        """Returns metric depth in float32."""
        for sample in self.dataset.depth_samples(split):
            rgb = np.ascontiguousarray(sample.image[:, :, ::-1])
            prediction = self.model.infer(torch.from_numpy(rgb).permute(2, 0, 1), None)
            depth = prediction["depth"].float().squeeze().cpu().numpy()
            yield replace(sample, depth=depth)
