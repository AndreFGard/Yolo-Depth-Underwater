import math
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ultralytics.utils.plotting
from matplotlib.figure import Figure

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def render_depth_img(depth: np.ndarray | pathlib.Path) -> np.ndarray:
    if isinstance(depth, pathlib.Path):
        depth = cv2.imread(str(depth))  # pyright: ignore[reportAssignmentType]
        assert depth is not None
    MODE = "disparity"
    depth = np.asarray(depth, dtype=np.float32)
    color = ultralytics.utils.plotting.colorize_depth(depth, mode=MODE)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def render_images_in_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    cols: int = 2,
    width: float = 4.0,
    h_per_img: float = 2.0,
) -> Figure:
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width * cols, h_per_img * rows),
        squeeze=False,
        constrained_layout=True,
    )
    if titles is None:
        titles = [f"{i}" for i in range(n)]
    for ax, img, title in zip(axes.flat, images, titles):
        if img.ndim == 2:
            img = render_depth_img(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    return fig
