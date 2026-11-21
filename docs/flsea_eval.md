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
`raw` and `global-ssi` rows are on by default; `--no-include-raw` / `--no-global-ssi`
disable them (the latter also halves inference cost).

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
* `constant` = best flat map; a model must beat it.
* `global-ssi` (default on): one scale + shift fitted in disparity space on the even-indexed frames of the run and
  applied to the odd-indexed ones, and vice versa — no frame contributes to its own fit. This mirrors the
  dataset-level rescaling of arXiv:2507.02148 ("rescaled to match the dataset-specific depth units"), so it is the
  row to place next to their Depth Anything V2 numbers, and the only metric-style row for disparity-output models.
  Each scored frame is predicted twice while it is on.
* Metrics: AbsRel, RMSE, delta1/2/3 (1.25^i), SILog (lambda = 0.85), on valid pixels.

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

Alignment uses the reference, so the aligned numbers are an upper bound for relative models and are **not
comparable** with the arXiv:2507.02148 tables (those use no alignment; only the sequence choice is shared).
For claims about physical meters, use `--include-raw` (possibly after calibrating on a separate split), not
the aligned rows.

## Reference numbers (Depth Anything V2 Small, zero-shot, Colab T4)

The `all` rows require `--subset all`; the default subset produces the `arxiv subset` row.

| subset | alignment | AbsRel | RMSE | d1 | d2 | d3 | SILog |
|---|---|---|---|---|---|---|---|
| all (238) | ssi | 0.0674 | 0.5654 | 0.9601 | 0.9912 | 0.9961 | 10.19 |
| all (238) | median | 0.1495 | 1.1858 | 0.8481 | 0.9564 | 0.9820 | 18.33 |
| all (238) | constant | 0.3127 | 1.5550 | 0.4151 | 0.7311 | 0.8866 | 42.44 |
| arxiv subset (119) | ssi | 0.0656 | 0.6305 | 0.9647 | 0.9896 | 0.9948 | 10.34 |

2 of 240 frames are discarded (too little reference). Reproduction should match up to GPU numerics.

## Adding a model
Implement `InferenceModel.predict(images)` (BGR uint8 in, one float map per image out), set
`output_kind = "disparity"` if needed, and add it to `build_model` in `scripts/evaluate_flsea.py`.
Note: `YoloModel` loads a hardcoded `yolo26{variant}-depth.pt`; to score fine-tuned weights it needs a weights-path option.

`configs/flsea_eval_frames.csv` was made by `scripts/make_flsea_manifest.py` (not needed to reproduce).
