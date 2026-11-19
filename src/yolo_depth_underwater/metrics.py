"""Depth metrics and scale alignment for relative and metric depth predictions.

Protocol (identical for every model, so the numbers are comparable):

* A model that outputs *depth* is converted to *disparity* (``1 / depth``) first;
  a model that outputs *disparity* (e.g. Depth Anything V2) is used as is.
* ``ssi``: per-image least-squares **scale + shift** in disparity space (MiDaS
  style), fitted on the valid reference pixels.
* ``median``: per-image **scale only**, from the ratio of the medians.
* After alignment the disparity is clipped to ``>= 1 / DEPTH_CAP_M`` and inverted
  back to meters, so the metrics are computed in depth space.
* Metrics are computed on valid reference pixels only (finite and ``> 0``).

The alignment uses the reference itself (an *oracle* alignment): it measures how
well a model recovers the structure of the scene, not what it would score in the
field without any reference. It is the same for all models.
"""

from __future__ import annotations

import numpy as np

DEPTH_CAP_M = 20.0
"""Upper bound (m) for aligned predictions; avoids ``disparity <= 0`` blow-ups."""

MIN_DEPTH_M = 0.01
"""Floor (m) applied to a predicted depth map before inverting it to disparity."""

METRIC_NAMES = ("AbsRel", "RMSE", "d1", "d2", "d3", "SILog")

ALIGN_MODES = ("ssi", "median")


def valid_mask(gt: np.ndarray) -> np.ndarray:
    """Pixels of the reference that carry a depth value (finite and > 0)."""
    return np.isfinite(gt) & (gt > 0)


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    """Invert a predicted depth map (floored at ``MIN_DEPTH_M``)."""
    return 1.0 / np.clip(np.asarray(depth, dtype=np.float64), MIN_DEPTH_M, None)


def align_to_depth(
    pred_disp: np.ndarray, gt: np.ndarray, valid: np.ndarray, mode: str = "ssi"
) -> np.ndarray:
    """Align a predicted disparity map to the reference and return depth in meters.

    Args:
        pred_disp: predicted disparity (higher = closer), same shape as ``gt``.
        gt: reference depth in meters (invalid pixels NaN or <= 0).
        valid: boolean mask of valid reference pixels (see :func:`valid_mask`).
        mode: ``"ssi"`` (scale + shift, least squares) or ``"median"`` (scale only).
    """
    if int(valid.sum()) < 2:
        raise ValueError("need at least 2 valid reference pixels to align")
    p = np.asarray(pred_disp, dtype=np.float64)
    pv = p[valid]
    g_disp = 1.0 / gt[valid].astype(np.float64)
    if mode == "ssi":
        design = np.stack([pv, np.ones_like(pv)], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, g_disp, rcond=None)
    elif mode == "median":
        scale, shift = np.median(g_disp) / np.median(pv), 0.0
    else:
        raise ValueError(f"unknown alignment mode {mode!r}; use one of {ALIGN_MODES}")
    disp = np.clip(scale * p + shift, 1.0 / DEPTH_CAP_M, None)
    return 1.0 / disp


def depth_metrics(
    pred_depth: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """AbsRel, RMSE (m), delta_1..3 and SILog over the valid reference pixels."""
    p = np.asarray(pred_depth, dtype=np.float64)[valid]
    g = gt[valid].astype(np.float64)
    ratio = np.maximum(g / p, p / g)
    log_err = np.log(p) - np.log(g)
    silog = np.mean(log_err**2) - 0.85 * np.mean(log_err) ** 2
    return {
        "AbsRel": float(np.mean(np.abs(g - p) / g)),
        "RMSE": float(np.sqrt(np.mean((g - p) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25**2)),
        "d3": float(np.mean(ratio < 1.25**3)),
        "SILog": float(100.0 * np.sqrt(max(silog, 0.0))),
    }


def constant_baseline(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Best constant depth map (a control that has no scene structure).

    It uses the reference, like the alignment does. A model that does not beat it
    adds no structure over a flat map.
    """
    return np.full(gt.shape, 1.0 / np.mean(1.0 / gt[valid]))
