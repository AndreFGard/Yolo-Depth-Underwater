#!/usr/bin/env python3
"""Compare YOLO FLSea finetunes"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TABLES = ROOT / "report" / "tables"
OUTPUT = TABLES / "flsea_teacher_arms.csv"
OUTPUT_TESTS = TABLES / "flsea_teacher_arms_tests.csv"

BASELINE = "yolo_baseline_evaluation_flsea.csv"
VARIANTS = (
    ("YOLO-F-GT", "flsea_ladder_YOLO-GT-FLSea.csv"),
    ("YOLO-F-UniDepth", "flsea_ladder_YOLO-UniDepth-FLSea.csv"),
    ("YOLO-F-DA", "flsea_ladder_YOLO-DA-FLSea.csv"),
)
CONTRASTS = (
    ("UniDepth − GT", "YOLO-F-UniDepth", "YOLO-F-GT"),
    ("DA − GT", "YOLO-F-DA", "YOLO-F-GT"),
    ("UniDepth − DA", "YOLO-F-UniDepth", "YOLO-F-DA"),
)
SELECTED = "best"
ALIGNMENTS = ("ssi", "median")
METRICS = (
    ("ssi", "AbsRel", "SSI AbsRel↓"),
    ("ssi", "d1", "SSI δ₁↑"),
    ("median", "AbsRel", "Mediana AbsRel↓"),
    ("median", "d1", "Mediana δ₁↑"),
)
DECIMALS = 3
TEST_DECIMALS = 4
RESAMPLES = 4000
SEED = 0
Row = dict[str, str]
Frame = tuple[str, str]


def _load(name: str) -> list[Row]:
    path = ARTIFACTS / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def _select(
    rows: list[Row], alignment: str, checkpoint: str | None = None
) -> list[Row]:
    return [
        row
        for row in rows
        if row["alignment"] == alignment
        and (checkpoint is None or row["checkpoint"] == checkpoint)
    ]


def _frame_keys(rows: list[Row]) -> set[Frame]:
    return {(row["sequence"], row["frame_id"]) for row in rows}


def _validate_frames(reference: list[Row], candidate: list[Row], name: str) -> None:
    if _frame_keys(reference) != _frame_keys(candidate):
        raise ValueError(f"{name}: evaluated frames differ from the YOLO baseline")


def _mean(rows: list[Row], metric: str) -> str:
    return f"{sum(float(row[metric]) for row in rows) / len(rows):.{DECIMALS}f}"


def _by_frame(rows: list[Row]) -> dict[Frame, tuple[float, float]]:
    return {
        (row["sequence"], row["frame_id"]): (float(row["AbsRel"]), float(row["d1"]))
        for row in rows
    }


def _bootstrap(
    reference: dict[Frame, tuple[float, float]],
    candidate: dict[Frame, tuple[float, float]],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float, bool]]:
    keys = sorted(set(reference) & set(candidate))
    sequences = sorted({key[0] for key in keys})
    groups = [[key for key in keys if key[0] == sequence] for sequence in sequences]
    position = {key: index for index, key in enumerate(keys)}
    ref = {m: np.array([reference[key][m] for key in keys]) for m in (0, 1)}
    cand = {m: np.array([candidate[key][m] for key in keys]) for m in (0, 1)}
    rng = np.random.default_rng(seed)

    draws = {m: np.empty(resamples) for m in (0, 1)}
    for draw in range(resamples):
        index = np.concatenate(
            [
                np.array([position[key] for key in group])[
                    rng.integers(0, len(group), len(group))
                ]
                for group in groups
            ]
        )
        for m in (0, 1):
            draws[m][draw] = (cand[m][index] - ref[m][index]).mean()

    out: dict[str, tuple[float, float, float, bool]] = {}
    for name, m in (("AbsRel", 0), ("d1", 1)):
        delta = float((cand[m] - ref[m]).mean())
        low, high = (float(value) for value in np.percentile(draws[m], [2.5, 97.5]))
        out[name] = (delta, low, high, not low <= 0 <= high)
    return out


def build_table(baseline: dict[str, list[Row]], variants: dict[str, list[Row]]) -> None:
    output: list[tuple[str, ...]] = [
        ("Métrica", "Baseline", *(name for name, _ in VARIANTS))
    ]
    for alignment, metric, label in METRICS:
        values = [_mean(baseline[alignment], metric)]
        for name, _ in VARIANTS:
            values.append(_mean(_select(variants[name], alignment, SELECTED), metric))
        output.append((label, *values))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(output)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def build_contrasts(variants: dict[str, list[Row]]) -> None:
    """Write the paired bootstrap of every label-source contrast."""
    header = (
        "contraste",
        "checkpoints",
        "AbsRel Δ",
        "AbsRel IC2.5%",
        "AbsRel IC97.5%",
        "AbsRel significativo",
        "δ₁ Δ",
        "δ₁ IC2.5%",
        "δ₁ IC97.5%",
        "δ₁ significativo",
    )
    output = [header]
    frames = {
        name: _by_frame(_select(rows, "ssi", SELECTED))
        for name, rows in variants.items()
    }
    for label, candidate, reference in CONTRASTS:
        stats = _bootstrap(frames[reference], frames[candidate])
        cells = [label, SELECTED]
        for metric in ("AbsRel", "d1"):
            delta, low, high, significant = stats[metric]
            cells += [
                f"{delta:.{TEST_DECIMALS}f}",
                f"{low:.{TEST_DECIMALS}f}",
                f"{high:.{TEST_DECIMALS}f}",
                "sim" if significant else "não",
            ]
        output.append(tuple(cells))

    OUTPUT_TESTS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TESTS.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(output)
    print(f"Wrote {OUTPUT_TESTS.relative_to(ROOT)}")


def build() -> None:
    baseline_rows = _load(BASELINE)
    baseline = {
        alignment: _select(baseline_rows, alignment) for alignment in ALIGNMENTS
    }
    for alignment, rows in baseline.items():
        if not rows:
            raise ValueError(f"{BASELINE}: missing {alignment!r} rows")

    variants = {name: _load(path) for name, path in VARIANTS}
    for name, rows in variants.items():
        for alignment in ALIGNMENTS:
            selected = _select(rows, alignment, SELECTED)
            if not selected:
                raise ValueError(f"{name}: missing {alignment!r}/{SELECTED!r}")
            _validate_frames(baseline[alignment], selected, name)

    build_table(baseline, variants)
    build_contrasts(variants)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
