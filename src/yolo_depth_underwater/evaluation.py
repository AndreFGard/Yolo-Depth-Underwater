"""Evaluate any ``InferenceModel`` on FLSea frames with the shared protocol.

A model only needs ``predict(images)`` (BGR ``uint8`` arrays in, one float map per
image out) plus the ``output_kind`` property: ``DEPTH`` (farther = larger) or
``DISPARITY`` (closer = larger, e.g. Depth Anything V2). See ``metrics`` for the
alignment and the metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_depth_underwater import metrics
from yolo_depth_underwater.encoding import save_depth_png, write_csv
from yolo_depth_underwater.flsea import FLSeaFrame, load_frame
from yolo_depth_underwater.inference_model import InferenceModel, OutputKind

LOGGER = logging.getLogger(__name__)



ROW_FIELDS = ("sequence", "frame_id", "valid_frac", "alignment", *metrics.METRIC_NAMES)

DEFAULT_MIN_VALID_FRAC = 0.02
"""Frames whose reference covers less than this fraction of pixels are skipped."""


def _resize(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float map to ``shape`` (height, width) if needed."""
    if pred.shape == shape:
        return pred
    img = Image.fromarray(np.ascontiguousarray(pred, dtype=np.float32))
    return np.asarray(img.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)


def _predict_one(model: InferenceModel, image: np.ndarray) -> np.ndarray:
    pred = next(iter(model.predict(iter([image]))))
    return np.asarray(pred, dtype=np.float32).squeeze()


def evaluate_model(
    model: InferenceModel,
    frames: list[FLSeaFrame],
    *,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    include_raw: bool = True,
    rescaled: bool = True,
    global_ssi: bool = True,
    predictions_dir: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], list[tuple[FLSeaFrame, str]]]:
    """Score ``model`` on ``frames`` (which must have local paths).

    Per frame it records the alignments ``ssi`` (scale + shift) and ``median``
    (scale only), both per-image oracle fits, plus the ``constant`` control (a flat
    map: a model should beat it). With ``include_raw=True`` (default) and a
    depth-output model it also records ``raw`` (no alignment; only meaningful for
    models that predict metric depth).
    With ``rescaled=True`` (default) it also records ``rescaled``: one scale for
    the whole run, a least squares fit using a cross fit similar to global ssi.
    With ``global_ssi=True`` (default) it also records ``global-ssi``: one scale +
    shift for the whole run, cross-fitted the same way. The ``rescaled`` to
    ``global-ssi`` gap isolates the global shift. While either dataset-level row
    is on, each scored frame is predicted twice.
    If ``predictions_dir`` is set, the raw meter prediction for depth models, or the oracle ``ssi``-aligned depth in
    meters for disparity models (their raw output is not in meters).

    Returns ``(rows, skipped)``: one dict per (frame, alignment) with the keys in
    ``ROW_FIELDS``, and the frames that were skipped with the reason.
    """
    kind = model.output_kind
    if kind not in (OutputKind.DEPTH, OutputKind.DISPARITY):
        raise ValueError(f"output_kind must be 'depth' or 'disparity', got {kind!r}")

    rows: list[dict] = []
    skipped: list[tuple[FLSeaFrame, str]] = []
    kept: list[int] = []
    fits = (metrics.GlobalFit(), metrics.GlobalFit())
    scale_fits = (metrics.RescaledFit(), metrics.RescaledFit())
    for i, frame in enumerate(tqdm(frames, desc="frames", disable=not show_progress)):
        image, gt = load_frame(frame)
        if gt.shape != image.shape[:2]:
            skipped.append((frame, "image/depth shape mismatch"))
            continue
        valid = metrics.valid_mask(gt)
        valid_frac = float(valid.mean())
        if valid_frac < min_valid_frac:
            skipped.append((frame, f"valid fraction {valid_frac:.1%}"))
            continue
        kept.append(i)

        pred = _resize(_predict_one(model, image), gt.shape)
        disp = pred if kind is OutputKind.DISPARITY else metrics.depth_to_disparity(pred)
        base = {"sequence": frame.sequence, "frame_id": frame.frame_id, "valid_frac": valid_frac}

        if predictions_dir is not None:
            depth_to_save = (
                pred
                if kind is OutputKind.DEPTH
                else metrics.align_to_depth(disp, gt, valid, mode="ssi")
            )
            save_depth_png(Path(predictions_dir) / f"{frame.frame_id}.png", depth_to_save)

        for mode in metrics.ALIGN_MODES:
            depth = metrics.align_to_depth(disp, gt, valid, mode)
            rows.append({**base, "alignment": mode, **metrics.depth_metrics(depth, gt, valid)})
        const = metrics.constant_baseline(gt, valid)
        rows.append({**base, "alignment": "constant", **metrics.depth_metrics(const, gt, valid)})
        if include_raw and kind is OutputKind.DEPTH:
            #TODO: if we ever wanna take metric depth metrics, we will have to calibrate yolo on a validation set or similar.
            raw = np.clip(pred, metrics.MIN_DEPTH_M, metrics.DEPTH_CAP_M)
            rows.append({**base, "alignment": "raw", **metrics.depth_metrics(raw, gt, valid)})
        fits[i % 2].add(disp, gt, valid)
        scale_fits[i % 2].add(disp, gt, valid)

    if (rescaled or global_ssi) and kept:
        ssi_params = (fits[0].solve(), fits[1].solve()) if global_ssi else None
        scale_params = (
            (scale_fits[0].solve(), scale_fits[1].solve()) if rescaled else None
        )
        for i in tqdm(kept, desc="global", disable=not show_progress):
            frame = frames[i]
            image, gt = load_frame(frame)
            valid = metrics.valid_mask(gt)
            pred = _resize(_predict_one(model, image), gt.shape)
            disp = pred if kind is OutputKind.DISPARITY else metrics.depth_to_disparity(pred)
            base = {
                "sequence": frame.sequence,
                "frame_id": frame.frame_id,
                "valid_frac": float(valid.mean()),
            }
            if rescaled:
                depth = metrics.apply_rescaled_alignment(disp, scale_params[(i + 1) % 2])
                rows.append(
                    {**base, "alignment": metrics.RESCALED_ALIGNMENT, **metrics.depth_metrics(depth, gt, valid)}
                )
            if global_ssi:
                depth = metrics.apply_global_alignment(disp, *ssi_params[(i + 1) % 2])
                rows.append(
                    {**base, "alignment": metrics.GLOBAL_ALIGNMENT, **metrics.depth_metrics(depth, gt, valid)}
                )
    return rows, skipped


def summarize(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean and sample std (over frames) of each metric, per alignment."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for alignment in dict.fromkeys(r["alignment"] for r in rows):
        sel = [r for r in rows if r["alignment"] == alignment]
        out[alignment] = {
            name: (
                float(np.mean([r[name] for r in sel])),
                float(np.std([r[name] for r in sel], ddof=1)) if len(sel) > 1 else float("nan"),
            )
            for name in metrics.METRIC_NAMES
        }
    return out


def format_summary(summary: dict[str, dict[str, tuple[float, float]]], n_frames: int) -> str:
    """Plain-text table: ``mean ± std`` of each metric per alignment."""
    header = f"{'alignment':<10}" + "".join(f"{m:^16}" for m in metrics.METRIC_NAMES)
    lines = [f"{n_frames} frames (mean ± std over frames)", header.rstrip()]
    for alignment, vals in summary.items():
        cells = "".join(f"{vals[m][0]:>9.4f}±{vals[m][1]:<6.3f}" for m in metrics.METRIC_NAMES)
        lines.append(f"{alignment:<10}{cells}".rstrip())
    return "\n".join(lines)


def write_rows(rows: list[dict], path: str | Path) -> None:
    write_csv(rows, path, columns=list(ROW_FIELDS), float_places=6)
