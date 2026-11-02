#!/usr/bin/env python3
"""
Run the refactored converter to produce a fresh COCO-depth dataset,
then compare it pixel-for-pixel against the existing one by hashing
every PNG in both trees.

Usage:
    python3 scripts/verify_coco.py

Compares:
    datasets/USDO10k-COCO/          (existing, from old script)
    datasets/usod10k_new_script/    (fresh, from refactored script)
"""

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("verify_coco")

EXISTING = Path("datasets/USDO10k-COCO")
FRESH = Path("datasets/usod10k_new_script")


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes(root: Path) -> dict[str, str]:
    """Walk root recursively, hash every PNG, return {relpath: sha256}."""
    results = {}
    for f in sorted(root.rglob("*.png")):
        rel = f.relative_to(root)
        results[str(rel)] = hash_file(f)
    return results


def run_conversion() -> None:
    """Run the refactored converter to produce the fresh dataset."""
    cmd = [
        sys.executable,
        "scripts/convert_usd10k_to_coco.py",
        "--usod10k-root",
        "datasets/USOD10k",
        "--output-root",
        str(FRESH),
    ]
    LOGGER.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"conversion failed with exit code {result.returncode}")


def compare(left_root: Path, right_root: Path) -> int:
    left = collect_hashes(left_root)
    right = collect_hashes(right_root)

    left_keys = set(left.keys())
    right_keys = set(right.keys())

    only_left = left_keys - right_keys
    only_right = right_keys - left_keys
    shared = left_keys & right_keys

    print(f"Left  ({left_root}):  {len(left)} PNGs")
    print(f"Right ({right_root}): {len(right)} PNGs")
    print(f"Shared: {len(shared)}")
    print()

    if only_left:
        print(f"--- Only in left ({len(only_left)}):")
        for k in sorted(only_left)[:20]:
            print(f"  {k}")
        if len(only_left) > 20:
            print(f"  ... and {len(only_left) - 20} more")
        print()

    if only_right:
        print(f"--- Only in right ({len(only_right)}):")
        for k in sorted(only_right)[:20]:
            print(f"  {k}")
        if len(only_right) > 20:
            print(f"  ... and {len(only_right) - 20} more")
        print()

    mismatches = 0
    for k in sorted(shared):
        if left[k] != right[k]:
            mismatches += 1
            print(f"  MISMATCH: {k}")
            print(f"    left  sha256: {left[k]}")
            print(f"    right sha256: {right[k]}")

    if mismatches == 0 and not only_left and not only_right:
        print("All PNGs match (identical hashes).")
    else:
        print()
        print(
            f"Summary: {mismatches} mismatches, "
            f"{len(only_left)} only-left, {len(only_right)} only-right"
        )

    return 1 if (mismatches or only_left or only_right) else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path("datasets/USOD10k").is_dir():
        LOGGER.error("USOD10k source not found at datasets/USOD10k")
        return 2

    # Produce fresh dataset
    if FRESH.is_dir():
        LOGGER.warning("%s already exists, removing it first", FRESH)
        import shutil

        shutil.rmtree(FRESH)
    run_conversion()

    # Compare the two COCO-depth outputs
    if not EXISTING.is_dir():
        LOGGER.error("existing dataset not found: %s", EXISTING)
        return 2

    print(f"Comparing {EXISTING}  vs  {FRESH}")
    print("=" * 60)
    return compare(EXISTING, FRESH)


if __name__ == "__main__":
    sys.exit(main())
