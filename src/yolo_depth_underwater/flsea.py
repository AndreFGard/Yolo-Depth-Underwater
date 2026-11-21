"""FLSea-VI evaluation frames (Kaggle dataset ``viseaonlab/flsea-vi``).

The full dataset is ~92 GB, so frames are fetched one file at a time. A *manifest*
(``configs/flsea_eval_frames.csv``: ``sequence,frame_id``) pins the exact frames
used for evaluation, so every model is scored on the same images.

Remote layout (``<site>/<sequence>/...``):

* images: ``imgs/<frame_id>.tiff`` (RGB, uint8)
* reference depth: ``depth/<frame_id>_SeaErra_abs_depth.tif`` (float32, meters,
  from SfM photogrammetry; invalid pixels are NaN or <= 0)

Images follow the repository convention (BGR ``uint8``, like ``cv2.imread``).
``kagglehub``, ``kaggle`` and ``tifffile`` are imported lazily, so importing this
module needs none of them (install them with the ``flsea`` extra).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from yolo_depth_underwater.encoding import write_csv

LOGGER = logging.getLogger(__name__)

HANDLE = "viseaonlab/flsea-vi"
DATASET_VERSION = 5  # version used for the reported numbers (informational)

SEQUENCES: tuple[str, ...] = (
    "canyons/flatiron",
    "canyons/horse_canyon",
    "canyons/tiny_canyon",
    "canyons/u_canyon",
    "red_sea/big_dice_loop",
    "red_sea/coral_table_loop",
    "red_sea/cross_pyramid_loop",
    "red_sea/dice_path",
    "red_sea/landward_path",
    "red_sea/northeast_path",
    "red_sea/pier_path",
    "red_sea/sub_pier",
)
"""The 12 FLSea-VI sequences with reference depth (``calibration`` is excluded)."""

SUBSETS: dict[str, tuple[str, ...]] = {
    "all": SEQUENCES,
    # Six FLSea sequences used for zero-shot evaluation in arXiv:2507.02148
    # ("Real-World Benchmarking and Synthetic Fine-Tuning of Monocular Metric
    # Depth Estimation in Underwater Environments", Sec. 3.5). Only the choice of
    # sequences is shared: that paper does not describe per-image alignment.
    "arxiv-2507.02148": (
        "canyons/u_canyon",
        "canyons/flatiron",
        "red_sea/big_dice_loop",
        "red_sea/cross_pyramid_loop",
        "red_sea/coral_table_loop",
        "red_sea/sub_pier",
    ),
}

_ID_RE = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class FLSeaFrame:
    """One image/reference pair. Local paths are set by :func:`download_frames`
    or :func:`attach_local_root`."""

    sequence: str
    frame_id: str
    image_path: Path | None = None
    depth_path: Path | None = None

    @property
    def image_remote(self) -> str:
        return f"{self.sequence}/imgs/{self.frame_id}.tiff"

    @property
    def depth_remote(self) -> str:
        return f"{self.sequence}/depth/{self.frame_id}_SeaErra_abs_depth.tif"


# --------------------------------------------------------------------------- manifest


def read_manifest(
    path: str | Path, sequences: tuple[str, ...] | list[str] | None = None
) -> list[FLSeaFrame]:
    """Read a ``sequence,frame_id`` CSV, optionally keeping only some sequences."""
    with open(path, newline="") as f:
        frames = [FLSeaFrame(r["sequence"], r["frame_id"]) for r in csv.DictReader(f)]
    if sequences is not None:
        keep = set(sequences)
        frames = [fr for fr in frames if fr.sequence in keep]
    return frames


def write_manifest(frames: list[FLSeaFrame], path: str | Path) -> None:
    write_csv(
        [{"sequence": fr.sequence, "frame_id": fr.frame_id} for fr in frames],
        path,
        columns=["sequence", "frame_id"],
    )


# --------------------------------------------------------------------------- selection


def _is_rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429 or "429" in str(exc)


def list_remote_files(
    cache: str | Path | None = "flsea_files.json",
    page_size: int = 200,
    pause: float = 0.7,
) -> list[str]:
    """List every file of the dataset through the Kaggle API (a few minutes).

    Needs Kaggle credentials (``KAGGLE_API_TOKEN`` or ``KAGGLE_USERNAME`` +
    ``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``). The API rate-limits (HTTP 429),
    so pages are paced and retried with backoff. The result is cached as JSON.
    You only need this to build a *new* manifest, not to reuse the committed one.
    """
    if cache is not None and Path(cache).exists():
        return json.loads(Path(cache).read_text())

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    def fetch(token: str | None):
        for attempt in range(8):
            try:
                return api.dataset_list_files(HANDLE, page_token=token, page_size=page_size)
            except Exception as exc:  # noqa: BLE001 - kaggle raises several types
                if not _is_rate_limited(exc):
                    raise
                wait = min(60, 5 * 2**attempt)
                LOGGER.warning("rate limited (429); waiting %ds", wait)
                time.sleep(wait)
        raise RuntimeError("Kaggle API rate limit persisted")

    names: list[str] = []
    token: str | None = None
    while True:
        res = fetch(token)
        names += [f.name for f in res.files]
        token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not token:
            break
        time.sleep(pause)
    if cache is not None:
        Path(cache).write_text(json.dumps(names))
    return names


def _frame_id(path: str) -> str | None:
    match = _ID_RE.match(path.rsplit("/", 1)[-1])
    return match.group(1) if match else None


def pick_frames(names: list[str], sequence: str, n: int) -> list[FLSeaFrame]:
    """Pick ``n`` frames evenly spaced in time among those that have both an image
    and a reference depth (frame ids are capture timestamps, so sorting = time)."""
    imgs = {_frame_id(x) for x in names if x.startswith(f"{sequence}/imgs/")}
    depth = {_frame_id(x) for x in names if x.startswith(f"{sequence}/depth/")}
    common = sorted((imgs & depth) - {None})
    if not common:
        return []
    idx = sorted(set(np.linspace(0, len(common) - 1, n).round().astype(int)))
    return [FLSeaFrame(sequence, common[i]) for i in idx]


# --------------------------------------------------------------------------- files


def attach_local_root(frames: list[FLSeaFrame], root: str | Path) -> list[FLSeaFrame]:
    """Point frames at a local copy of the dataset (e.g. ``/kaggle/input/flsea-vi``)."""
    root = Path(root)
    out = [
        replace(fr, image_path=root / fr.image_remote, depth_path=root / fr.depth_remote)
        for fr in frames
    ]
    for fr in out:
        for p in (fr.image_path, fr.depth_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} (is {root} the dataset root?)")
    return out


def _cached_path(remote: str) -> Path | None:
    """Path of ``remote`` in the kagglehub cache, if it is already there.

    kagglehub asks the Kaggle API even for files it already has, which hits HTTP 429
    when the same frames are evaluated again (second model, second run). Looking at
    the cache first avoids those requests.
    """
    cache = Path(os.environ.get("KAGGLEHUB_CACHE") or Path.home() / ".cache" / "kagglehub")
    path = cache / "datasets" / HANDLE / "versions" / str(DATASET_VERSION) / remote
    return path if path.is_file() and path.stat().st_size > 0 else None


def download_frames(
    frames: list[FLSeaFrame], workers: int = 2, tries: int = 8
) -> list[FLSeaFrame]:
    """Download the image and reference of each frame with ``kagglehub``.

    Files already in the kagglehub cache are used as they are (no API request). Frames that fail
    after ``tries`` attempts are logged and left out of the result. Keep ``workers``
    low: every file costs one API request and Kaggle answers 429 when it is too fast.
    """
    import kagglehub

    def fetch_one(remote: str) -> Path | None:
        cached = _cached_path(remote)
        if cached is not None:
            return cached
        for attempt in range(tries):
            try:
                return Path(kagglehub.dataset_download(HANDLE, path=remote))
            except Exception as exc:  # noqa: BLE001
                wait = min(60, 3 * 2**attempt)
                LOGGER.warning("%s: %s; retrying in %ds", remote, str(exc)[:80], wait)
                time.sleep(wait)
        return None

    def fetch_frame(fr: FLSeaFrame) -> FLSeaFrame | None:
        img = fetch_one(fr.image_remote)
        dep = fetch_one(fr.depth_remote) if img else None
        return replace(fr, image_path=img, depth_path=dep) if (img and dep) else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch_frame, frames))
    ok = [r for r in results if r is not None]
    if len(ok) < len(frames):
        LOGGER.warning("%d of %d frames could not be downloaded", len(frames) - len(ok), len(frames))
    return ok


def load_frame(frame: FLSeaFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image, depth)``: BGR ``uint8`` image and ``float32`` depth in meters
    (invalid pixels NaN or <= 0; see ``metrics.valid_mask``).

    Do not use ``cv2.imread`` for the depth file: use this loader.
    """
    if frame.image_path is None or frame.depth_path is None:
        raise ValueError("frame has no local paths: use download_frames or attach_local_root")
    import tifffile

    img = tifffile.imread(frame.image_path)[..., :3]
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / np.iinfo(img.dtype).max * 255).astype(np.uint8)
    depth = tifffile.imread(frame.depth_path).astype(np.float32)
    return np.ascontiguousarray(img[..., ::-1]), depth