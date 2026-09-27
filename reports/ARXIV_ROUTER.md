# arXiv metadata topic router for Laya

Status: **completed**. Model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, device `cpu`,
seed 42, context budget 640 tokens with a 256-token question head.

## What was run

A Laya multiple-choice question with eight options, one per arXiv subject area, over
the title and the first 100 abstract words. The base checkpoint and the fine-tuned
checkpoint are scored on the **same 800 held-out papers**, with one scalar
temperature per model fitted on the same separate calibration split.

**Training did not run to a completed schedule.** The run was bounded by a
7200-second wall-clock budget and stopped inside epoch 2:
2 epochs were started, epoch 1 ran all
500 steps, and epoch 2 ran
338 of them before the budget fired. The checkpoint
is a partial-schedule artefact, and no claim is made that more training would not
continue to help. The stop reason is recorded as
`time budget 7200.0s reached in epoch 2` in the report file and in the
training log.

| metric | base Laya | fine-tuned | delta |
| --- | --- | --- | --- |
| accuracy | 0.4650 | 0.6288 | +0.1638 |
| macro F1 | 0.3846 | 0.6215 | +0.2369 |
| macro precision | 0.5792 | 0.6490 | +0.0698 |
| macro recall | 0.4650 | 0.6288 | +0.1638 |
| ECE (10 bins) | 0.0584 | 0.0436 | -0.0148 |
| Brier | 0.6846 | 0.4749 | -0.2097 |
| temperature | 0.71 | 0.97 | |

- accuracy 95% bootstrap CI: base [0.4313, 0.5000], fine-tuned [0.5962, 0.6637]
- macro F1 95% bootstrap CI: base [0.3618, 0.4075], fine-tuned [0.5894, 0.6541]
- paired McNemar exact p: 7.691e-22 with 199 discordant items
  (34 only the base model got right, 165 only the fine-tune)

Chance is 0.125 accuracy for eight balanced classes.

## Calibration method, and which numbers the temperature can move

Each model gets one scalar temperature, fitted on the separate 400-item
calibration split by minimising negative log likelihood of the correct class. The fit
runs on **raw logits**, not on a probability vector: temperature scaling is defined as
`softmax(z / T)`, and feeding an already-normalised distribution back in fits the
log-probability family instead, where the transform is the identity, because `p / T`
renormalises to `p`.

| model | temperature | calibration NLL | ECE (10 equal-width bins) | Brier |
| --- | --- | --- | --- | --- |
| base | 0.71 | 1.488 | 0.0584 | 0.6846 |
| fine-tuned | 0.97 | 0.881 | 0.0436 | 0.4749 |

**Everything ranked is invariant to the temperature.** A positive scalar cannot
change an argmax, so accuracy, macro F1, macro precision, macro recall, every
per-class precision, recall and F1, both confusion matrices, the bootstrap intervals
and the McNemar test are identical under any positive temperature. They are reported
at the fitted value for consistency and are unaffected by the calibration step.
**ECE and Brier are not invariant**: they read the probability vector, so the
temperature fit changes them. An earlier version of this analysis fitted the
temperature on probabilities, reported base ECE 0.5139 and Brier
1.0414, and returned a temperature pinned at the edge of the search grid. Those
confidence figures were artefacts of the wrong objective and are superseded by the
table above; the accuracy and macro F1 figures were never affected.

## Per-class scores, held-out split

| subject area | n | base P | base R | base F1 | tuned P | tuned R | tuned F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cond-mat.mes-hall` | 100 | 0.000 | 0.000 | 0.000 | 0.458 | 0.380 | 0.415 |
| `cond-mat.mtrl-sci` | 100 | 1.000 | 0.030 | 0.058 | 0.648 | 0.590 | 0.618 |
| `gr-qc` | 100 | 0.497 | 0.830 | 0.622 | 0.587 | 0.740 | 0.655 |
| `hep-ph` | 100 | 1.000 | 0.150 | 0.261 | 0.886 | 0.390 | 0.542 |
| `hep-th` | 100 | 0.167 | 0.010 | 0.019 | 0.416 | 0.420 | 0.418 |
| `math.AG` | 100 | 0.830 | 0.880 | 0.854 | 0.784 | 0.910 | 0.843 |
| `math.PR` | 100 | 0.906 | 0.870 | 0.888 | 0.871 | 0.810 | 0.839 |
| `quant-ph` | 100 | 0.233 | 0.950 | 0.375 | 0.541 | 0.790 | 0.642 |

## Confusion matrices

Base:

| true \ predicted | `cond-mat.mes-hall` | `cond-mat.mtrl-sci` | `gr-qc` | `hep-ph` | `hep-th` | `math.AG` | `math.PR` | `quant-ph` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cond-mat.mes-hall` | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 97 |
| `cond-mat.mtrl-sci` | 0 | 3 | 0 | 0 | 1 | 3 | 1 | 92 |
| `gr-qc` | 0 | 0 | 83 | 0 | 0 | 0 | 0 | 17 |
| `hep-ph` | 0 | 0 | 15 | 15 | 3 | 0 | 1 | 66 |
| `hep-th` | 0 | 0 | 61 | 0 | 1 | 5 | 1 | 32 |
| `math.AG` | 0 | 0 | 5 | 0 | 1 | 88 | 1 | 5 |
| `math.PR` | 0 | 0 | 1 | 0 | 0 | 9 | 87 | 3 |
| `quant-ph` | 0 | 0 | 2 | 0 | 0 | 1 | 2 | 95 |

Fine-tuned:

| true \ predicted | `cond-mat.mes-hall` | `cond-mat.mtrl-sci` | `gr-qc` | `hep-ph` | `hep-th` | `math.AG` | `math.PR` | `quant-ph` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cond-mat.mes-hall` | 38 | 27 | 0 | 0 | 1 | 0 | 4 | 30 |
| `cond-mat.mtrl-sci` | 33 | 59 | 0 | 0 | 1 | 1 | 0 | 6 |
| `gr-qc` | 0 | 0 | 74 | 2 | 13 | 3 | 0 | 8 |
| `hep-ph` | 5 | 0 | 7 | 39 | 37 | 0 | 1 | 11 |
| `hep-th` | 2 | 2 | 40 | 2 | 42 | 6 | 1 | 5 |
| `math.AG` | 0 | 0 | 1 | 0 | 3 | 91 | 1 | 4 |
| `math.PR` | 0 | 0 | 1 | 0 | 1 | 14 | 81 | 3 |
| `quant-ph` | 5 | 3 | 3 | 1 | 3 | 1 | 5 | 79 |

## Title-only ablation

The same held-out papers with the abstract removed, leaving the title.

| model | accuracy | macro F1 | ECE | mean tokens |
| --- | --- | --- | --- | --- |
| base | 0.4738 | 0.4427 | 0.0396 | 212.5 |
| fine-tuned | 0.5450 | 0.5346 | 0.1450 | 212.5 |

Read this next to the headline table. Most of the fine-tune's gain survives the
abstract being removed, so for these eight subject areas the title alone carries much
of the routable signal, and a large part of the improvement is a shift in how the
model weighs title keywords rather than new subject knowledge. The base model is
essentially unchanged by the removal, which is the expected shape if it was reading
title cues all along.

## Latency

Measured on the run machine, macOS-26.4.1-arm64-arm-64bit, 10 torch threads,
CPU only (no GPU).

| model | batch 8 ms/example mean | batch 8 p95 | batch 1 ms mean | batch 1 p95 |
| --- | --- | --- | --- | --- |
| base | 557 | 717 | 675 | 788 |
| fine-tuned | 668 | 1007 | 668 | 1108 |

Both passes ran on an idle machine, separately from training, and the first call of
each is a cold start excluded from the mean. The batch-8 durations are per call and are
divided by the batch size. The two checkpoints share an architecture, and the ordering
reverses between the two columns, so the difference between them is run-to-run
variation and not a property of the fine-tune. Batching buys roughly 20% per example
over batch 1 on CPU. No GPU, serving or concurrency numbers were measured, and none
should be inferred from this table.

## Training

| item | value |
| --- | --- |
| trainable parameters | 50762753 |
| parameter groups | head, encoder_top |
| epochs requested / started | 2 / 2 |
| steps per epoch | 500 |
| steps completed | 838 |
| steps in the final partial epoch | 338 |
| wall-clock budget | 7200 s |
| optimizer steps | 838 |
| batch size | 8 |
| top encoder layers trained | 2 |
| wall seconds | 7209 |
| stopped early | time budget 7200.0s reached in epoch 2 |

The decision head and the top 2 encoder layers are trained; the
rest of ModernBERT-large and the SDK action head stay frozen. Training was bounded by
an explicit wall-clock budget and stopped mid-epoch, so this is a short partial
fine-tune, not a converged run. An independent replication from the frozen artefacts
reproduced every accuracy-side figure exactly.

## Dataset, revision, selection rules and licences

- source: `secemp9/arxiv-complete` config `metadata`, revision `cee894837962fede5612cccf2a4c7cacf49b4c3a`
- columns read: `paper_id`, `title`, `abstract`, `primary_category`, `license`, `first_version_date`
- never read: paper text, LaTeX, source archives, PDFs, PostScript
- access: HTTP range reads of projected Parquet columns over the Hub; the 16.08 TB
  repository is never cloned and no full-text config is opened
- split sizes: {'calibration': 400, 'test': 800, 'train': 4000}
- submission date ranges: calibration 2008-02-19 to 2008-03-17, test 2008-12-19 to 2009-02-08, train 2007-03-31 to 2007-10-22
- labels: `cond-mat.mes-hall`, `cond-mat.mtrl-sci`, `gr-qc`, `hep-ph`, `hep-th`, `math.AG`, `math.PR`, `quant-ph`
- licence: Dataset compilation: CC0 organisational layer. Paper rights stay with the authors under each recorded per-paper licence (metadata.license), or none recorded. No paper full text is used.

Selection rules, as recorded in `data/router/provenance.json` and covered by tests:

1. three disjoint Parquet row groups in file order, which is paper_id ascending, so
   train, calibration and test are ordered in submission time;
2. an eight-label set curated a priori for multi-archive coverage and per-split
   availability, never by model output;
3. legacy identifiers containing `/` are dropped, and legacy arXiv identifiers cited
   inside titles and abstracts are replaced by `REF`, because strings such as
   `hep-th/0607218` spell out the archive;
4. any remaining paper whose text mentions one of the label codes is dropped;
5. a normalized title may appear in at most one split, which removes cross-split
   duplicates and version pairs of the same work;
6. class balance by deterministic round robin, so no split can be dominated by one
   archive.

Licence counts across the kept rows: {}

The dataset compilation carries a CC0 organisational layer, but the papers stay
under their own per-author licences, and 69% of the kept rows record no licence at
all. Titles and abstracts are used here as model input, not redistributed: the
built splits are git-ignored and rebuilt by the script.

## Limitations

- Labels are arXiv author-declared primary categories, not expert annotations.
- The window is 2007-2009 arXiv metadata; the router is not a general 2026 topic router.
- CPU-only training reached a bounded number of steps; this is not a converged full fine-tune.
- Eight subject areas only, selected a priori for per-split availability.
- Latency is Mac CPU batch-1 and batch-8; GPU and serving numbers are not measured.
- No paper full text was used, and no model artefact is a safety or authorisation claim.

## Reproduce

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt pyarrow fsspec
.venv/bin/python src/arxiv_router_data.py --output-dir data/router
ROUTER_THREADS=10 .venv/bin/python src/router_run.py all --epochs 2 \
    --encoder-layers 2 --max-train-seconds 7200 \
    --report reports/router_results.json
.venv/bin/python src/router_recompute.py reports/router_results.json
```

`src/router_recompute.py` rebuilds every table above from the per-item logits stored
in the report, and `--verify-only` re-derives the reported accuracies from those same
logits. Metrics recomputed in place: {"bootstrap_samples": 2000, "note": null, "source": "reports/router_results.json", "temperature_refit_from_calibration_logits": true}.

## Verification

`./verify` compiles the sources and runs the unit tests, including the selection
rules, the leak guards and the metric implementations used for the tables above.
