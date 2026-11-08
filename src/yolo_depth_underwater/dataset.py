"""Read-only facade over the Ultralytics ``DepthDataset`` of each split."""

from collections.abc import Iterator
from pathlib import Path
import ultralytics.data.dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.patches import imread

from yolo_depth_underwater.encoding import DEPTH_SCALE, load_depth_img
from yolo_depth_underwater.sample import DepthSample
from yolo_depth_underwater.split import Split


class DepthDataset:
    """A COCO Depth dataset.

        Args:
            data_yaml: Path to an Ultralytics data YAML (as built by
                ``scripts/convert_usd10k_to_coco.py``).
    .
            cache: ultralytics cache option.
            fraction: Fraction or count of the *train* split to use.
    """

    def __init__(
        self,
        data_yaml: str | Path,
        imgsz: int = 640,
        fraction: float = 1.0,
    ):
        self._data_yaml = Path(data_yaml)
        self.data = check_det_dataset(str(self._data_yaml))
        self._cfg = get_cfg(overrides={"task": "depth", "imgsz": imgsz, "cache": False})
        self._fraction = fraction
        self._datasets: dict[Split, ultralytics.data.dataset.DepthDataset] = {}

    def _dataset(self, split: Split) -> ultralytics.data.dataset.DepthDataset:
        """Return the raw Ultralytics ``DepthDataset``."""
        if split not in self._datasets:
            img_path = self.data.get(split.value)
            if not img_path:
                raise KeyError(
                    f"Split {split.value!r} is not defined in {self._data_yaml}"
                )
            self._datasets[split] = ultralytics.data.dataset.DepthDataset(
                img_path=img_path,
                data=self.data,
                task="depth",
                imgsz=self._cfg.imgsz,
                augment=False,
                hyp=self._cfg,
                cache=False,
                fraction=self._fraction,
                prefix=f"{split.value}: ",
            )
        return self._datasets[split]

    def indices(self, split: Split) -> range:
        """Iterate the image indices of a split."""
        return range(len(self._dataset(split)))

    def image_paths(self, split: Split) -> list[Path]:
        """Return the image paths of a split, in index order."""
        return [Path(p) for p in self._dataset(split).im_files]

    def depth_paths(self, split: Split) -> list[Path]:
        """Return the depth paths of a split, in index order."""
        ds = self._dataset(split)
        return [Path(ds.depth_files_by_image[p]) for p in ds.im_files]

    def get(self, split: Split, index: int) -> DepthSample:
        """Load the image/depth pair at ``index``."""
        ds = self._dataset(split)
        im_file = ds.im_files[index]
        depth_file = ds.depth_files_by_image[im_file]
        image = imread(im_file)
        if image is None:
            raise FileNotFoundError(f"Could not read image {im_file}")
        depth = load_depth_img(depth_file, ds.data.get("depth_scale", DEPTH_SCALE))
        return DepthSample(split, index, image, depth, Path(im_file), Path(depth_file))

    def depth_samples(self, split: Split) -> Iterator[DepthSample]:
        """Stream ``DepthSample`` objects for a whole split, in index order."""

        return (self.get(split, i) for i in self.indices(split))
