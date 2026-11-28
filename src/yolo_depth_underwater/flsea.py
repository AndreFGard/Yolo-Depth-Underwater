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
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

from yolo_depth_underwater.encoding import write_csv

LOGGER = logging.getLogger(__name__)

CACHE_EVERY = 25
"""Write a partial Kaggle listing every this many pages so it can be resumed."""

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


class Subset(StrEnum):
    ALL = "all"

    # Cai & Metzler, "Real-World Benchmarking and Synthetic Fine-Tuning of
    # Monocular Metric Depth Estimation in Underwater Environments"
    # (arXiv:2507.02148, Sec. 3.5). Only the sequence choice is shared, because the
    # paper does not describe per-image alignment and uses the entire scenes.
    CAI_METZLER = "arxiv-2507.02148"


SUBSETS: dict[Subset, tuple[str, ...]] = {
    Subset.ALL: SEQUENCES,
    Subset.CAI_METZLER: (
        "canyons/u_canyon",
        "canyons/flatiron",
        "red_sea/big_dice_loop",
        "red_sea/cross_pyramid_loop",
        "red_sea/coral_table_loop",
        "red_sea/sub_pier",
    ),
}

TRAINING_SCENES: tuple[str, ...] = tuple(
    s for s in SEQUENCES if s not in SUBSETS[Subset.CAI_METZLER]
)
"""The six sequences that no evaluation in this repo scores, so they are the only ones
usable for training without touching the paper's zero-shot table."""

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
    cache_path = Path(cache) if cache is not None else None
    names: list[str] = []
    token: str | None = None
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if isinstance(cached, list):
            return cached
        names = list(cached["files"])
        token = cached.get("next_page_token")
        if not token:
            return names
        LOGGER.info("resuming the listing after %d cached files", len(names))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    def fetch(page_token: str | None):
        for attempt in range(8):
            try:
                return api.dataset_list_files(
                    HANDLE, page_token=page_token, page_size=page_size
                )
            except Exception as exc:
                if not _is_rate_limited(exc):
                    raise
                wait = min(60, 5 * 2**attempt)
                LOGGER.warning("rate limited (429); waiting %ds", wait)
                time.sleep(wait)
        raise RuntimeError("Kaggle API rate limit persisted")

    pages = 0
    while True:
        res = fetch(token)
        names += [f.name for f in res.files]
        token = getattr(res, "next_page_token", None) or getattr(
            res, "nextPageToken", None
        )
        pages += 1
        if cache_path is not None and (pages % CACHE_EVERY == 0 or not token):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"files": names, "next_page_token": token})
            )
        if not token:
            break
        if pages % CACHE_EVERY == 0:
            LOGGER.info("%d files listed", len(names))
        time.sleep(pause)
    return names


def _frame_id(path: str) -> str | None:
    match = _ID_RE.match(path.rsplit("/", 1)[-1])
    return match.group(1) if match else None


def scene_frame_ids(names: list[str], sequence: str) -> list[str]:
    """Frame ids of ``sequence`` that have both an image and a reference depth,
    in time order (frame ids are 100 ns timestamps)."""
    imgs = {_frame_id(x) for x in names if x.startswith(f"{sequence}/imgs/")}
    depth = {_frame_id(x) for x in names if x.startswith(f"{sequence}/depth/")}
    return sorted((imgs & depth) - {None}, key=int)


def training_split(
    names: list[str],
    scenes: tuple[str, ...] = TRAINING_SCENES,
    test_per_scene: int = 50,
    val_per_scene: int = 167,
) -> dict[str, list[FLSeaFrame]]:
    """first frames test, then val, the rest train.
    """
    overlap = set(scenes) & set(SUBSETS[Subset.CAI_METZLER])
    if overlap:
        raise ValueError(f"scenes are in the evaluation subset: {sorted(overlap)}")
    out: dict[str, list[FLSeaFrame]] = {"test": [], "val": [], "train": []}
    for sequence in scenes:
        ids = scene_frame_ids(names, sequence)
        if len(ids) <= test_per_scene + val_per_scene:
            raise ValueError(
                f"{sequence} has {len(ids)} image/depth pairs, fewer than the "
                f"{test_per_scene + val_per_scene} requested for test+val"
            )
        test = ids[:test_per_scene]
        val = ids[test_per_scene : test_per_scene + val_per_scene]
        train = ids[test_per_scene + val_per_scene :]
        out["test"] += [FLSeaFrame(sequence, i) for i in test]
        out["val"] += [FLSeaFrame(sequence, i) for i in val]
        out["train"] += [FLSeaFrame(sequence, i) for i in train]
    return out


def sequence_frames(
    names: list[str], sequence: str, count: int, start: int = 0, stride: int = 1
) -> list[FLSeaFrame]:
    """Pick ``count`` frames of ``sequence`` that are consecutive in time."""
    common = scene_frame_ids(names, sequence)
    picked = common[start : start + count * stride : stride][:count]
    if len(picked) < count:
        LOGGER.warning(
            "%s: only %d of the %d requested frames are available (start=%d, stride=%d)",
            sequence,
            len(picked),
            count,
            start,
            stride,
        )
    return [FLSeaFrame(sequence, fid) for fid in picked]


def pick_frames(names: list[str], sequence: str, n: int) -> list[FLSeaFrame]:
    """Pick ``n`` frames evenly spaced in time among those that have both an image
    and a reference depth (frame ids are capture timestamps, so sorting = time)."""
    common = scene_frame_ids(names, sequence)
    if not common:
        return []
    idx = sorted(set(np.linspace(0, len(common) - 1, n).round().astype(int)))
    return [FLSeaFrame(sequence, common[i]) for i in idx]


def attach_local_root(frames: list[FLSeaFrame], root: str | Path) -> list[FLSeaFrame]:
    """Point frames at a local copy of the dataset (e.g. ``/kaggle/input/flsea-vi``)."""
    root = Path(root)
    out = [
        replace(
            fr, image_path=root / fr.image_remote, depth_path=root / fr.depth_remote
        )
        for fr in frames
    ]
    for fr in out:
        for p in (fr.image_path, fr.depth_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} (is {root} the dataset root?)")
    return out


def _cached_path(remote: str) -> Path | None:
    """Path of ``remote`` in the kagglehub cache, if it is already there."""
    cache = Path(
        os.environ.get("KAGGLEHUB_CACHE") or Path.home() / ".cache" / "kagglehub"
    )
    path = cache / "datasets" / HANDLE / "versions" / str(DATASET_VERSION) / remote
    return path if path.is_file() and path.stat().st_size > 0 else None


def download_frames(
    frames: list[FLSeaFrame], workers: int = 2, tries: int = 8
) -> list[FLSeaFrame]:
    """Download the image and reference of each frame with ``kagglehub``."""
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
        LOGGER.warning(
            "%d of %d frames could not be downloaded",
            len(frames) - len(ok),
            len(frames),
        )
    return ok


_KIND_MARKERS: tuple[str, ...] = ("/imgs/", "/depth/")

_SCENE_TO_SEQUENCE: dict[str, str] = {s.rsplit("/", 1)[-1]: s for s in SEQUENCES}
"""Scene directory name -> canonical ``<site>/<scene>`` sequence. Scene names are unique
across sites, so the same indexer serves a nested archive (``red_sea/dice_path/...``), a
prefix above the site (``flsea-vi/red_sea/dice_path/...``) and a flattened one where the
site directory is absent (``dice_path/...``, e.g. the per-site ``red_sea.zip``)."""


def canonical_remote(member: str) -> str | None:
    """Map an archive member to ``<site>/<scene>/<kind>/<file>``, or ``None`` when it is
    neither an image nor a reference depth map."""
    positions = [pos for pos in (member.find(k) for k in _KIND_MARKERS) if pos >= 0]
    if not positions:
        return None
    position = min(positions)
    sequence = _SCENE_TO_SEQUENCE.get(member[:position].rsplit("/", 1)[-1])
    return None if sequence is None else f"{sequence}{member[position:]}"


def decode_frame(
    image: np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn decoded TIFF arrays into the repo convention: BGR ``uint8`` + metres."""
    image = np.asarray(image)[..., :3]
    if image.dtype != np.uint8:
        image = (image.astype(np.float32) / np.iinfo(image.dtype).max * 255).astype(
            np.uint8
        )
    depth = np.asarray(depth).astype(np.float32)
    return np.ascontiguousarray(image[..., ::-1]), depth


def load_frame(frame: FLSeaFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image, depth)``: BGR ``uint8`` image and ``float32`` depth in meters
    (invalid pixels NaN or <= 0; see ``metrics.valid_mask``). Use this instead of any other loader for FLSea frames.
    """
    if frame.image_path is None or frame.depth_path is None:
        raise ValueError(
            "frame has no local paths: use download_frames, attach_local_root or "
            "ArchiveFrames"
        )
    import tifffile

    return decode_frame(tifffile.imread(frame.image_path), tifffile.imread(frame.depth_path))


class ArchiveFrames:
    """Read FLSea frames straight out of a dataset ``.zip``, without extracting it.

    The archive may keep any prefix above the scene directories (``flsea-vi/``,
    ``versions/5/``, ...) or drop the site directory entirely (the per-site archives such as
    ``red_sea.zip`` hold ``big_dice_loop/imgs/...``): members are indexed by their *canonical*
    name (``<site>/<scene>/imgs/<frame_id>.tiff``), which is what the rest of this module
    uses.

    Example:
        with ArchiveFrames.open("flsea-vi.zip") as archive:
            split = training_split(archive.names())
            image, depth = archive.load_frame(split["train"][0])
    """

    def __init__(self, path: Path, archive: zipfile.ZipFile, index: dict[str, str]):
        self.path = path
        self._archive = archive
        self._index = index

    @classmethod
    def open(cls, path: str | Path) -> "ArchiveFrames":
        """Open the archive and index every image/reference member by remote name."""
        path = Path(path)
        archive = zipfile.ZipFile(path)
        index: dict[str, str] = {}
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            remote = canonical_remote(member)
            if remote is not None:
                index[remote] = member
        if not index:
            archive.close()
            raise ValueError(
                f"{path} has no '<sequence>/imgs|depth/' members; "
                f"first members: {archive.namelist()[:3]}"
            )
        return cls(path, archive, index)

    def names(self) -> list[str]:
        """Remote names present in the archive, comparable to a Kaggle listing."""
        return sorted(self._index)

    def sequences(self) -> dict[str, tuple[int, int]]:
        """``sequence -> (image members, depth members)``, to check the layout at a glance."""
        counts: dict[str, list[int]] = {}
        for remote in self._index:
            sequence = remote.split("/imgs/")[0] if "/imgs/" in remote else remote.split("/depth/")[0]
            counts.setdefault(sequence, [0, 0])[0 if "/imgs/" in remote else 1] += 1
        return {name: (imgs, depth) for name, (imgs, depth) in sorted(counts.items())}

    def read(self, remote: str) -> bytes:
        """Raw bytes of one member."""
        return self._archive.read(self._index[remote])

    def load_frame(self, frame: FLSeaFrame) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as :func:`load_frame`, reading from the archive."""
        import io

        import tifffile

        image = tifffile.imread(io.BytesIO(self.read(frame.image_remote)))
        depth = tifffile.imread(io.BytesIO(self.read(frame.depth_remote)))
        return decode_frame(image, depth)

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "ArchiveFrames":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
