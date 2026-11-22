# FLSea evaluation

Scores any `InferenceModel` on the same FLSea-VI frames with the same protocol, so models can
go in one table. The default subset is the six sequences of arXiv:2507.02148 (120 frames, 20
per sequence, evenly spaced in time); pass `--subset all` for all 12 sequences (240 frames).

## Setup
```toml
# pyproject.toml, [project.optional-dependencies]
flsea = ["kagglehub", "kaggle>=1.7", "tifffile", "imagecodecs"]
depth-anything = ["transformers>=4.45"]
```
Kaggle credentials (`KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`, or `~/.kaggle/kaggle.json`).
The dataset is 92 GB: only the files for the selected frames are downloaded (240 files for the
default subset, 480 for `--subset all`; kagglehub caches them).

## Run
```bash
uv run --extra cu --extra flsea --extra depth-anything python scripts/evaluate_flsea.py --model depth-anything
uv run --extra cu --extra flsea python scripts/evaluate_flsea.py --model yolo --variant n
uv run --extra cu --extra flsea python scripts/evaluate_flsea.py --model unidepth
# all 12 sequences instead of the default paper subset
... --subset all
# dataset already on disk
... --local-root /kaggle/input/flsea-vi
# also save the generated depth maps, one <frame_id>.png per scored frame
... --save-predictions results/flsea_yolo_n_depth
```
Output: per-frame CSV in `results/` and a summary table (mean ± std over frames).
Keep `--workers` at 2: Kaggle answers HTTP 429 when asked too fast.
The `raw` row is on by default; `--no-include-raw` disables it.

With `--save-predictions DIR`, disparity-output models store the oracle `ssi`-aligned depth in metres, since their raw output is not in metres. Skipped frames are not saved.

## Protocol
* Reference: SfM depth (metric, float32), valid = finite and > 0; frames with < 2% valid pixels are skipped.
* Predictions are resized to the reference resolution.
* Models that output **disparity** (higher = closer; Depth Anything V2) set `output_kind = "disparity"`;
  otherwise the output is treated as depth.
* Per-image oracle alignment in disparity space: `ssi` (scale + shift, least squares) and `median` (scale only);
  aligned disparity is clipped to >= 1/20 m and inverted. The same is applied to every model, metric ones included.
  The per-image scale fit means these scores measure **structure only**: they cannot detect wrong absolute scale
  or frame-to-frame scale drift. `raw` (default on) scores depth-output models without alignment.
* `control` = best flat map (a flat-map baseline, not a calibration); a model must beat it.
  CSVs written before 2026-11-22 label this row `constant` — same row, renamed.
* Metrics: AbsRel, RMSE, delta1/2/3 (1.25^i), SILog with `metrics.SILOG_LAMBDA = 1.0`
  (the variance form, `100 * std(log pred - log gt)`), on valid pixels.

### SILog: why lambda must be 1.0 (changed 2026-11-22)

With `g = log pred - log gt`, a prediction scaled by `s` shifts `g` by `c = log s`:

```
SILog^2(lambda) = mean(g^2) - lambda * mean(g)^2 = std(g)^2 + (1 - lambda) * mean(g)^2
```

So any `lambda != 1` retains a `(1 - lambda)` share of the mean log error, which is
exactly the global scale error. At 0.85 a pure scale error of +-50% moves SILog by ~1.3
points on a model whose spread is std = 0.15, and a perfectly calibrated model is
unaffected — which is why the mistake is invisible. 1.0 is the evaluation convention in
`ultralytics.utils.metrics` ("lambda=1 variance form (ZoeDepth/KITTI)") and in
`3rd_party/unidepth` (`100 * std(log pred - log gt)`); 0.85 is AdaBins' `variance_focus`,
a loss hyperparameter. SILog columns in CSVs written before 2026-11-22 use 0.85 and must
not be compared with new ones.

Every alignment row is a published protocol: `ssi` is the MiDaS scale-and-shift-invariant
protocol for relative depth, `median` is per-image median scaling, `raw` is how metric models
are scored. Nothing dataset-level is fitted.

### Why there is no dataset-level fit (removed 2026-11-22)

Earlier runs also recorded `rescaled` (one scale for the whole run) and `global-ssi` (one
scale + shift), both cross-fitted on even/odd halves of the manifest. Both were dropped:

* **The estimator is published only per capture.** A global `s·z + t` in inverse depth is
  Wofk et al. (2020), used by SPADE (eqs. 1-4) and by Depth Anything 3 (eq. 8) — always
  fitted for a *single* capture or pseudo-label batch, against measurements available at
  inference. Pooling one fit over a 6-site benchmark's references is an unpublished estimand.
  On our frames it is also mis-specified across sites: for UDepth it scores AbsRel 0.350, worse
  than the flat-map control (0.323), and for Depth Anything V2 Large 0.289, worse than
  per-image median scaling (0.144) by a factor of two.
* **It confounds paired comparisons.** `ssi`/`median` are fitted against the scored frame's
  own reference, so in a per-frame difference between two models the alignment cancels and
  only the prediction changes. A dataset-level fit is re-derived per model, so a delta mixes
  "the model changed" with "the fit changed".
* **It cannot carry a claim on 119 frames.** Measured on `yolo_headneck_e16` vs the zero-shot
  baseline, the paired delta on `ssi` has a bootstrap CI of width 0.017; on `global-ssi` the
  width is 0.155 (~9x) and only 44% of frames move in the same direction, i.e. a coin flip,
  because a single affine pooled over 119 frames is fitted on only ~60, and the two halves
  disagree by ~15% of the metric.

The consequence: the epoch-by-epoch claim (fine-tuning degrades FLSea transfer) rests on
`ssi`/`median`, where every metric agrees, plus `raw` at the endpoints as the no-oracle
check. Anyone reviving these rows should first re-read the two caveats above; the code was
removed rather than disabled so it cannot be re-enabled by accident.

## How the scores are aggregated (important when comparing papers)

We score each frame **separately**, using only the pixels where the SfM reference is valid. The summary then
takes an ordinary mean of those per-frame numbers. A frame with 11% valid pixels therefore counts exactly as
much as one with 100% valid pixels. The `±` value is the standard deviation across frames — it describes how
much scores vary from frame to frame, not statistical uncertainty in the mean.

Many papers instead pool all valid pixels from the whole subset and compute one metric over that combined set
(e.g. arXiv:2507.02148). All our frames have the same size, so that approach effectively gives each frame a
weight proportional to its `valid_frac`. On these frames, the two choices produce noticeably different numbers:
AbsRel differs by 0–6% relative, δ1 by about 0.01, and **RMSE by about 15%**. RMSE changes most because it
squares errors: pooling gives more influence to the frames with lots of valid reference, and in this dataset
those include distant frames where the errors are larger. Consequently, do not compare our RMSE (or the other
scores) with a paper's table without first checking which aggregation it used.

The per-frame CSV in `results/` lets you reproduce the pooled numbers for the same frames:

* AbsRel, δ1, δ2 and δ3: take the weighted average of the per-frame values, with `valid_frac` as the weight.
* RMSE: compute `sqrt(sum(valid_frac * RMSE_frame²) / sum(valid_frac))`. Averaging the per-frame RMSE values
  directly is **not** the same thing.
* SILog: it cannot be recovered from the CSV. Pooling requires `mean(log(error))` and `mean(log(error)²)`
  across pixels, but the CSV stores only each frame's already-combined SILog value.

## Oracle alignment

Per-image alignment uses the reference of the scored frame, so the `ssi`/`median` numbers are an upper bound
for relative models and are **not comparable** with the arXiv:2507.02148 tables (those use no alignment; only the
sequence choice is shared). For claims about physical meters, use `raw` (with the checkpoint's calibration
provenance stated), not the aligned rows; for structure claims and paired comparisons, use `ssi`/`median`.

## Reference numbers (Depth Anything V2 Small, zero-shot, Colab T4)

The `all` rows require `--subset all`; the default subset produces the `arxiv subset` row.

| subset | alignment | AbsRel | RMSE | d1 | d2 | d3 | SILog |
|---|---|---|---|---|---|---|---|
| all (238) | ssi | 0.0674 | 0.5654 | 0.9601 | 0.9912 | 0.9961 | 10.19 |
| all (238) | median | 0.1495 | 1.1858 | 0.8481 | 0.9564 | 0.9820 | 18.33 |
| all (238) | `constant` (old label of `control`) | 0.3127 | 1.5550 | 0.4151 | 0.7311 | 0.8866 | 42.44 |
| arxiv subset (119) | ssi | 0.0656 | 0.6305 | 0.9647 | 0.9896 | 0.9948 | 10.34 |

2 of 240 frames are discarded (too little reference). Reproduction should match up to GPU numerics.

## Adding a model

UDepth-RMI is available as `--model udepth` with the `udepth` extra.
Its inverse-relative output uses the shared disparity interface. In addition to
the auxiliary SSI-aligned PNGs, it exports unaligned float32 relative-depth NPYs.
See [UDepth setup, artifacts and results](udepth_flsea.md).

Implement `InferenceModel.predict(images)` (BGR uint8 in, one float map per image out), set
`output_kind = "disparity"` if needed, and add it to `build_model` in `scripts/evaluate_flsea.py`.
Note: `YoloModel` loads a hardcoded `yolo26{variant}-depth.pt`; to score fine-tuned weights it needs a weights-path option.

`configs/flsea_eval_frames.csv` was made by `scripts/make_flsea_manifest.py` (not needed to reproduce).
