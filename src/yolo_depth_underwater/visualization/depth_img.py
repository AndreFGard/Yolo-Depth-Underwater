import pathlib

import cv2
import numpy as np
import ultralytics.utils.plotting 

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def render_depth_img(depth: np.ndarray | pathlib.Path) -> np.ndarray:
    if isinstance(depth, pathlib.Path):
        depth = cv2.imread(str(depth))
        assert depth is not None
    MODE= "disparity"
    depth = np.asarray(depth, dtype=np.float32)
    color = ultralytics.utils.plotting .colorize_depth(depth, mode=MODE)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
