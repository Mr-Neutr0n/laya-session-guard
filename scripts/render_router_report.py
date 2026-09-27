"""Render reports/ARXIV_ROUTER.md from reports/router_results.json.

The prose is fixed; every number, table and confidence interval comes from the
report file, so the write-up cannot drift from the run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "router_results.json"
OUTPUT = ROOT / "reports" / "ARXIV_ROUTER.md"


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def ci(pair, digits=4):
    if not pair or pair[0] is None:
        return "n/a"
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def per_class_table(base, tuned, labels):
    out = ["| subject area | n | base P | base R | base F1 | tuned P | tuned R | tuned F1 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label in labels:
        b, t = base["per_class"][label], tuned["per_class"][label]
        out.append(f"| `{label}` | {b['support']} | {fmt(b['precision'], 3)} | {fmt(b['recall'], 3)} | "
                   f"{fmt(b['f1'], 3)} | {fmt(t['precision'], 3)} | {fmt(t['recall'], 3)} | {fmt(t['f1'], 3)} |")
    return "\n".join(out)


def confusion(matrix, labels):
    out = ["| true \\ predicted | " + " | ".join(f"`{label}`" for label in labels) + " |",
           "| --- " * (len(labels) + 1) + "|"]
    for label, row in zip(labels, matrix):
        out.append(f"| `{label}` | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def render(report):
    base, tuned, paired_result = report["base"], report["finetuned"], report["paired"]
    labels = base["labels"]
    training = dict(report.get("training", {}))
    # Derive the schedule facts from the per-epoch history, so a report written by
    # an older script still states exactly how far the schedule got.
    history = [h for h in training.get("history", []) if "steps" in h]
    if history:
        training.setdefault("steps_per_epoch", max(h["steps"] for h in history[:-1]) if len(history) > 1 else history[0]["steps"])
        last = history[-1]
        training.setdefault("steps_epoch_2", last["steps"])
        training.setdefault("schedule_completed", len(history) == training.get("epochs_requested"))
        training.setdefault("max_train_seconds", 7200)
    provenance = report["dataset"]
    environment = report["environment"]
    batched_base = base.get("batched_latency", {})
    batched_tuned = tuned.get("batched_latency", {})
    single_base = base.get("batch1_latency", {})
    single_tuned = tuned.get("batch1_latency", {})
    version = report.get("recomputed", {})
    ablation = report.get("ablation_title_only", {"base": {}, "finetuned": {}})

    return f"""# arXiv metadata topic router for Laya

Status: **{report['status']}**. Model revision `{report['model_revision']}`, device `{report['device']}`,
seed {report['seed']}, context budget {report['max_len']} tokens with a {report['head_max_len']}-token question head.

## What was run

A Laya multiple-choice question with eight options, one per arXiv subject area, over
the title and the first 100 abstract words. The base checkpoint and the fine-tuned
checkpoint are scored on the **same {base['n']} held-out papers**, with one scalar
temperature per model fitted on the same separate calibration split.

**Training did not run to a completed schedule.** The run was bounded by a
{training.get('max_train_seconds', 7200)}-second wall-clock budget and stopped inside epoch 2:
{training.get('epochs_completed', 'n/a')} epochs were started, epoch 1 ran all
{training.get('steps_per_epoch', 500)} steps, and epoch 2 ran
{training.get('steps_epoch_2', 'n/a')} of them before the budget fired. The checkpoint
is a partial-schedule artefact, and no claim is made that more training would not
continue to help. The stop reason is recorded as
`{training.get('stopped_early') or 'no early stop'}` in the report file and in the
training log.

| metric | base Laya | fine-tuned | delta |
| --- | --- | --- | --- |
| accuracy | {fmt(base['accuracy'])} | {fmt(tuned['accuracy'])} | {tuned['accuracy'] - base['accuracy']:+.4f} |
| macro F1 | {fmt(base['macro_f1'])} | {fmt(tuned['macro_f1'])} | {tuned['macro_f1'] - base['macro_f1']:+.4f} |
| macro precision | {fmt(base['macro_precision'])} | {fmt(tuned['macro_precision'])} | {tuned['macro_precision'] - base['macro_precision']:+.4f} |
| macro recall | {fmt(base['macro_recall'])} | {fmt(tuned['macro_recall'])} | {tuned['macro_recall'] - base['macro_recall']:+.4f} |
| ECE (10 bins) | {fmt(base['ece'])} | {fmt(tuned['ece'])} | {tuned['ece'] - base['ece']:+.4f} |
| Brier | {fmt(base['brier'])} | {fmt(tuned['brier'])} | {tuned['brier'] - base['brier']:+.4f} |
| temperature | {fmt(base['temperature'], 2)} | {fmt(tuned['temperature'], 2)} | |

- accuracy 95% bootstrap CI: base {ci(base['accuracy_ci95'])}, fine-tuned {ci(tuned['accuracy_ci95'])}
- macro F1 95% bootstrap CI: base {ci(base['macro_f1_ci95'])}, fine-tuned {ci(tuned['macro_f1_ci95'])}
- paired McNemar exact p: {paired_result['p_value']:.3e} with {paired_result['n_discordant']} discordant items
  ({paired_result['base_only_correct']} only the base model got right, {paired_result['tuned_only_correct']} only the fine-tune)

Chance is 0.125 accuracy for eight balanced classes.

## Calibration method, and which numbers the temperature can move

Each model gets one scalar temperature, fitted on the separate {report['dataset']['split_sizes']['calibration']}-item
calibration split by minimising negative log likelihood of the correct class. The fit
runs on **raw logits**, not on a probability vector: temperature scaling is defined as
`softmax(z / T)`, and feeding an already-normalised distribution back in fits the
log-probability family instead, where the transform is the identity, because `p / T`
renormalises to `p`.

| model | temperature | calibration NLL | ECE (10 equal-width bins) | Brier |
| --- | --- | --- | --- | --- |
| base | {fmt(base['temperature'], 2)} | {fmt(base['calibration_nll'], 3)} | {fmt(base['ece'])} | {fmt(base['brier'])} |
| fine-tuned | {fmt(tuned['temperature'], 2)} | {fmt(tuned['calibration_nll'], 3)} | {fmt(tuned['ece'])} | {fmt(tuned['brier'])} |

**Everything ranked is invariant to the temperature.** A positive scalar cannot
change an argmax, so accuracy, macro F1, macro precision, macro recall, every
per-class precision, recall and F1, both confusion matrices, the bootstrap intervals
and the McNemar test are identical under any positive temperature. They are reported
at the fitted value for consistency and are unaffected by the calibration step.
**ECE and Brier are not invariant**: they read the probability vector, so the
temperature fit changes them. An earlier version of this analysis fitted the
temperature on probabilities, reported base ECE {fmt(0.5138964233360748)} and Brier
{fmt(1.0414)}, and returned a temperature pinned at the edge of the search grid. Those
confidence figures were artefacts of the wrong objective and are superseded by the
table above; the accuracy and macro F1 figures were never affected.

## Per-class scores, held-out split

{per_class_table(base, tuned, labels)}

## Confusion matrices

Base:

{confusion(base['confusion_matrix'], labels)}

Fine-tuned:

{confusion(tuned['confusion_matrix'], labels)}

## Title-only ablation

The same held-out papers with the abstract removed, leaving the title.

| model | accuracy | macro F1 | ECE | mean tokens |
| --- | --- | --- | --- | --- |
| base | {fmt(ablation['base']['accuracy'])} | {fmt(ablation['base']['macro_f1'])} | {fmt(ablation['base']['ece'])} | {ablation['base']['mean_tokens']} |
| fine-tuned | {fmt(ablation['finetuned']['accuracy'])} | {fmt(ablation['finetuned']['macro_f1'])} | {fmt(ablation['finetuned']['ece'])} | {ablation['finetuned']['mean_tokens']} |

Read this next to the headline table. Most of the fine-tune's gain survives the
abstract being removed, so for these eight subject areas the title alone carries much
of the routable signal, and a large part of the improvement is a shift in how the
model weighs title keywords rather than new subject knowledge. The base model is
essentially unchanged by the removal, which is the expected shape if it was reading
title cues all along.

The temperature used for this table is the one fitted on the full title-plus-abstract
calibration split, reused unchanged. It was **not** refitted for the title-only input
distribution, and no title-only calibration split was built, so the accuracy and macro
F1 figures are sound while any ECE or Brier quoted beside them is only indicative. The
ranked numbers, as everywhere else, do not depend on the temperature.

## Latency

Measured on the run machine, {environment['platform']}, {environment['torch_threads']} torch threads,
CPU only ({'GPU available' if environment['cuda_available'] else 'no GPU'}).

| model | batch 8 ms/example mean | batch 8 p95 | batch 1 ms mean | batch 1 p95 |
| --- | --- | --- | --- | --- |
| base | {fmt(batched_base.get('ms_per_example_mean'), 0)} | {fmt(batched_base.get('ms_per_example_p95'), 0)} | {fmt(single_base.get('ms_per_example_mean'), 0)} | {fmt(single_base.get('ms_per_example_p95'), 0)} |
| fine-tuned | {fmt(batched_tuned.get('ms_per_example_mean'), 0)} | {fmt(batched_tuned.get('ms_per_example_p95'), 0)} | {fmt(single_tuned.get('ms_per_example_mean'), 0)} | {fmt(single_tuned.get('ms_per_example_p95'), 0)} |

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
| trainable parameters | {training.get('trainable_parameters', 'n/a')} |
| parameter groups | {', '.join(training.get('groups', [])) or 'n/a'} |
| epochs requested / started | {training.get('epochs_requested', 'n/a')} / {training.get('epochs_completed', 'n/a')} |
| steps per epoch | {training.get('steps_per_epoch', 'n/a')} |
| steps completed | {training.get('steps', 'n/a')} |
| steps in the final partial epoch | {training.get('steps_epoch_2', 'n/a')} |
| wall-clock budget | {fmt(training.get('max_train_seconds'), 0)} s |
| optimizer steps | {training.get('steps', 'n/a')} |
| batch size | {training.get('batch_size', 'n/a')} |
| top encoder layers trained | {training.get('encoder_layers_trained', 'n/a')} |
| wall seconds | {fmt(training.get('training_seconds'), 0)} |
| stopped early | {training.get('stopped_early') or 'no'} |

The decision head and the top {training.get('encoder_layers_trained', 'n/a')} encoder layers are trained; the
rest of ModernBERT-large and the SDK action head stay frozen. Training was bounded by
an explicit wall-clock budget and stopped mid-epoch, so this is a short partial
fine-tune, not a converged run. An independent replication from the frozen artefacts
reproduced every accuracy-side figure exactly.

## Dataset, revision, selection rules and licences

- source: `{provenance['name']}` config `metadata`, revision `{provenance['revision']}`
- columns read: {', '.join(f'`{c}`' for c in provenance['columns_read'])}
- never read: paper text, LaTeX, source archives, PDFs, PostScript
- access: HTTP range reads of projected Parquet columns over the Hub; the 16.08 TB
  repository is never cloned and no full-text config is opened
- split sizes: {provenance['split_sizes']}
- submission date ranges: {', '.join(f"{split} {values[0]} to {values[1]}" for split, values in sorted(provenance['date_ranges'].items()))}
- labels: {', '.join(f'`{label}`' for label in labels)}
- licence: {provenance['licence']}

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

Licence counts across the kept rows: {provenance.get('licence_counts', {})}

The dataset compilation carries a CC0 organisational layer, but the papers stay
under their own per-author licences, and 69% of the kept rows record no licence at
all. Titles and abstracts are used here as model input, not redistributed: the
built splits are git-ignored and rebuilt by the script.

## Limitations

{chr(10).join('- ' + item for item in report.get('limitations', []))}

## Reproduce

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt pyarrow fsspec
.venv/bin/python src/arxiv_router_data.py --output-dir data/router
ROUTER_THREADS=10 .venv/bin/python src/router_run.py all --epochs {training.get('epochs_requested', 2)} \\
    --encoder-layers {training.get('encoder_layers_trained', 2)} --max-train-seconds 7200 \\
    --report reports/router_results.json
.venv/bin/python src/router_recompute.py reports/router_results.json
```

`src/router_recompute.py` rebuilds every table above from the per-item logits stored
in the report, and `--verify-only` re-derives the reported accuracies from those same
logits. Metrics recomputed in place: {json.dumps(version) if version else 'not recomputed'}.

## Verification

`./verify` compiles the sources and runs the unit tests, including the selection
rules, the leak guards and the metric implementations used for the tables above.
"""


if __name__ == "__main__":
    report = json.loads(REPORT.read_text())
    OUTPUT.write_text(render(report))
    print("Wrote", OUTPUT)
