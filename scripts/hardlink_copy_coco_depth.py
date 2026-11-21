#!/usr/bin/env python3
"""Hardlink images and depth maps into a new COCO-depth dataset.

SOURCE provides ``images/`` and ``data.yaml``; DEPTH_SOURCE provides the
``<split>/*.png`` (or ``.npy``) depth tree. Both are hardlinked into DEST, so no
image or depth bytes are copied. The manifest and data.yaml are copied (with
paths repointed at DEST), and DEST/README.md records where everything came from.

Pass ``SOURCE/depth`` as DEPTH_SOURCE to duplicate a dataset's own depth maps.
All trees must live on the same filesystem. Existing DEST files are reused only
when they are hardlinks of the source file; anything else is an error. Every
image must have exactly one depth map with the same relative path, and
ultralytics ``*.cache`` files are never linked.

Usage:
    python scripts/hardlink_copy_coco_depth.py SOURCE DEPTH_SOURCE DEST
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
DEPTH_SUFFIXES = frozenset({".png", ".npy"})
CACHE_SUFFIX = ".cache"


class CopyError(Exception):
    ...


def link_file(src: Path, dst: Path) -> bool:
    """Hardlink src to dst; return True if a new link was made."""
    if dst.exists() or dst.is_symlink():
        if not dst.exists() or not os.path.samefile(src, dst):
            raise CopyError(f"{dst} exists and is not a hardlink of {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as exc:
        raise CopyError(f"cannot hardlink {src} -> {dst}: {exc}") from exc
    return True


def link_tree(
    src_root: Path, dst_root: Path, suffixes: frozenset[str]
) -> tuple[int, int]:
    """Hardlink matching files from src_root into dst_root.

    Returns:
        (new links, reused links).
    """
    if not src_root.is_dir():
        raise CopyError(f"missing directory: {src_root}")
    linked = reused = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file() or src.name.endswith(CACHE_SUFFIX):
            continue
        if src.suffix.lower() not in suffixes:
            continue
        if link_file(src, dst_root / src.relative_to(src_root)):
            linked += 1
        else:
            reused += 1
    return linked, reused


def stems(root: Path, suffixes: frozenset[str]) -> set[str]:
    """Return relative paths without extension for matching files under root."""
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not path.name.endswith(CACHE_SUFFIX)
    }


def count_pairs(images_root: Path, depth_root: Path) -> dict[str, int]:
    """Verify one depth map per image and return split -> pair count."""
    splits = sorted(path.name for path in images_root.iterdir() if path.is_dir())
    counts: dict[str, int] = {}
    for split in splits:
        images = stems(images_root / split, IMAGE_SUFFIXES)
        depths = stems(depth_root / split, DEPTH_SUFFIXES)
        if images != depths:
            problems = []
            missing = sorted(images - depths)
            extra = sorted(depths - images)
            if missing:
                problems.append(f"{len(missing)} without depth, e.g. {missing[0]}")
            if extra:
                problems.append(f"{len(extra)} without image, e.g. {extra[0]}")
            raise CopyError(f"{split}: " + "; ".join(problems))
        counts[split] = len(images)
    if not counts:
        raise CopyError(f"no split directories under {images_root}")
    return counts


def read_depth_scale(data_yaml: Path) -> float:
    """Read depth_scale from a data.yaml file."""
    match = re.search(r"(?m)^depth_scale:\s*([0-9.]+)", data_yaml.read_text())
    if match is None:
        raise CopyError(f"no depth_scale in {data_yaml}; pass --depth-scale")
    return float(match.group(1))


def copy_metadata(
    source: Path, dest: Path, depth_source: Path, depth_scale: float, note: str
) -> Path:
    """Copy data.yaml and the manifest into dest, describing the linked depth."""
    yaml_source = source / "data.yaml"
    text, count = re.subn(
        r"(?m)^path:.*$", f"path: {dest}", yaml_source.read_text(), count=1
    )
    if count == 0:
        raise CopyError(f"no path: line in {yaml_source}")
    (dest / "data.yaml").write_text(text)

    manifest_source = source / "annotations" / "annotations.json"
    if not manifest_source.is_file():
        raise CopyError(f"missing manifest: {manifest_source}")
    manifest_path = dest / "annotations" / "annotations.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_source, manifest_path)

    manifest = json.loads(manifest_path.read_text())
    info = manifest.setdefault("info", {})
    info["description"] = (
        f"Depth dataset: images from {source}, depth maps from {depth_source}"
    )
    info["depth"] = {
        "source": str(depth_source),
        "depth_scale": depth_scale,
        "semantics": note,
        "transform": "none: hardlinked unchanged from the depth source",
        "provenance_file": "README.md",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def write_readme(
    dest: Path,
    source: Path,
    depth_source: Path,
    depth_scale: float,
    note: str,
    pairs: dict[str, int],
) -> None:
    """Write the dataset's README.md."""
    rows = "\n".join(f"| {split} | {count} |" for split, count in pairs.items())
    images_source = source / "images"
    text = f"""# COCO-depth dataset: {dest.name}

Built by `scripts/hardlink_copy_coco_depth.py`.

| Item | Value |
|---|---|
| Images | hardlinked from `{images_source}` |
| Depth maps | hardlinked from `{depth_source}` |
| `depth_scale` | {depth_scale:g} (the loader divides stored values by this) |
| Depth semantics | {note or "not documented in the depth source"} |
| Pairs | {sum(pairs.values())} |

| Split | Image/depth pairs |
|---|---|
{rows}

The depth maps are linked unchanged: no conversion or rescaling was done here.
Do not rescale them to match another dataset's value range; the training loss is
scale invariant, but direction (farther = larger) and per-pixel non-linearity
are not.

## Hardlinks

- Editing a linked file in place edits the source and every other tree linked
  to it. Delete and re-create instead of overwriting.
- Deleting a link here does not free disk space while another link survives.
- `os.link` cannot cross filesystems; all trees must be on the same mount.
- Ultralytics `*.cache` files are never linked; regenerate them if needed.
"""
    (dest / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source", type=Path, help="COCO-depth dataset with images/ and data.yaml"
    )
    parser.add_argument(
        "depth_source", type=Path, help="directory of <split>/* depth maps"
    )
    parser.add_argument("dest", type=Path, help="destination dataset directory")
    parser.add_argument(
        "--depth-scale",
        type=float,
        help="override the depth_scale read from SOURCE/data.yaml",
    )
    parser.add_argument(
        "--note",
        default="",
        help="one-line description of the depth maps for README.md",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    depth_source = args.depth_source.resolve()
    dest = args.dest.resolve()

    try:
        for root, label in ((source, "SOURCE"), (depth_source, "DEPTH_SOURCE")):
            if not root.is_dir():
                raise CopyError(f"{label} is not a directory: {root}")
        if (
            source == dest
            or depth_source == dest
            or dest.is_relative_to(source)
            or dest.is_relative_to(depth_source)
            or source.is_relative_to(dest)
            or depth_source.is_relative_to(dest)
        ):
            raise CopyError(
                "SOURCE, DEPTH_SOURCE and DEST must be separate, non-nested directories"
            )

        data_yaml = source / "data.yaml"
        if not data_yaml.is_file():
            raise CopyError(f"missing data.yaml: {data_yaml}")
        depth_scale = args.depth_scale
        if depth_scale is None:
            depth_scale = read_depth_scale(data_yaml)
        if depth_scale <= 0:
            raise CopyError("--depth-scale must be positive")

        images_linked, images_reused = link_tree(
            source / "images", dest / "images", IMAGE_SUFFIXES
        )
        depth_linked, depth_reused = link_tree(
            depth_source, dest / "depth", DEPTH_SUFFIXES
        )
        pairs = count_pairs(dest / "images", dest / "depth")
        manifest = copy_metadata(source, dest, depth_source, depth_scale, args.note)
        write_readme(dest, source, depth_source, depth_scale, args.note, pairs)
    except CopyError as exc:
        parser.error(str(exc))

    print(f"images: {images_linked} linked, {images_reused} reused")
    print(f"depth: {depth_linked} linked, {depth_reused} reused")
    print("pairs: " + ", ".join(f"{split}={count}" for split, count in pairs.items()))
    print(f"wrote {dest / 'data.yaml'}, {manifest}, {dest / 'README.md'}")


if __name__ == "__main__":
    main()
