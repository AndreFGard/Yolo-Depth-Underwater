"""Run depth inference on each dataset split and save predictions to disk."""

import cv2
import pathlib
from tqdm import tqdm

from yolo_depth_underwater import inference_model, split
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.encoding import save_depth_png
from yolo_depth_underwater.unidepth_model import UniDepthV2Model


def _read_image(p:pathlib.Path):
    img = cv2.imread(str(p))
    assert img is not None
    return img



def store_predictions(
    model: inference_model.InferenceModel,
    ds: DepthDataset,
    dest: str | pathlib.Path,
    splits: list[split.Split] | None = None,
) -> None:
    if splits is None:
        splits = [split.Split.TRAIN, split.Split.VAL, split.Split.TEST]

    dest = pathlib.Path(dest)

    for spl in splits:
        paths = ds.image_paths(spl)
        images = (_read_image(p) for p in paths)
        predictions = model.predict(images)

        base = dest / spl.value
        base.mkdir(exist_ok=True, parents=True)

        print(f"Saving {spl.value} split ({len(paths)} images) into '{base}'")
        for path, pred in tqdm(
            zip(paths, predictions),
            total=len(paths),
            desc=f"{spl.value} predictions",
        ):
            assert pred is not None
            save_depth_png(base / path.name, pred)


