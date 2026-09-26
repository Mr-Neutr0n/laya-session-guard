"""Build a public, results-only Kaggle notebook for the arXiv topic router.

The notebook embeds `reports/router_results.json` and recomputes every table from
the saved per-item logits. Run All needs no GPU, no API key and no dataset
download: the numbers cannot drift from the recorded run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "router_results.json"
PROVENANCE = ROOT / "data" / "router" / "provenance.json"

INTRO = """# Laya arXiv topic router: base versus fine-tuned, on one held-out set

I fine-tuned Laya to route an arXiv preprint to its primary subject area from the
title and the first 100 words of the abstract, using metadata only from
`secemp9/arxiv-complete`. This notebook recomputes the base-versus-fine-tuned
tables from the saved per-item logits of a single run.

- Dataset: `secemp9/arxiv-complete`, metadata config, pinned revision
- Columns read: `paper_id`, `title`, `abstract`, `primary_category`, `license`, `first_version_date`
- Never read: `paper_text`, `latex`, `source`, `pdf`, `ps`. The 16.08 TB repository is never cloned.
- Compute: one Mac CPU, no paid service, no GPU.

**Experimental only.** The labels are author-declared arXiv categories, not expert
annotations, and the window is 2007-2009 metadata."""

CELLS = [
    ("markdown", INTRO),
    ("code", """import json, math
from pathlib import Path

RESULTS = json.loads(Path('router_results.json').read_text())
PROVENANCE = json.loads(Path('router_provenance.json').read_text())
base, tuned = RESULTS['base'], RESULTS['finetuned']
labels = base['labels']
print(RESULTS['status'], RESULTS['device'], RESULTS['environment']['platform'])
print('dataset revision', RESULTS['dataset']['revision'])
print('split sizes', RESULTS['dataset']['split_sizes'])
print('test date range', RESULTS['dataset']['date_ranges']['test'])"""),
    ("markdown", "## Headline: same 800 held-out items, same calibration split"),
    ("code", """def table(base, tuned):
    rows = [('accuracy', base['accuracy'], tuned['accuracy']),
            ('macro F1', base['macro_f1'], tuned['macro_f1']),
            ('ECE (10 bins)', base['ece'], tuned['ece']),
            ('Brier', base['brier'], tuned['brier']),
            ('temperature', base['temperature'], tuned['temperature'])]
    out = ['| metric | base Laya | fine-tuned | delta |', '| --- | --- | --- | --- |']
    for name, b, t in rows:
        out.append(f'| {name} | {b:.4f} | {t:.4f} | {t - b:+.4f} |')
    return '\\n'.join(out)

print(table(base, tuned))
print()
print('95% bootstrap CI, accuracy: base', [round(v, 4) for v in base['accuracy_ci95']],
      'fine-tuned', [round(v, 4) for v in tuned['accuracy_ci95']])
print('95% bootstrap CI, macro F1: base', [round(v, 4) for v in base['macro_f1_ci95']],
      'fine-tuned', [round(v, 4) for v in tuned['macro_f1_ci95']])
print('McNemar exact p (paired):', round(RESULTS['paired']['p_value'], 6))
print('base-only correct', RESULTS['paired']['base_only_correct'],
      'tuned-only correct', RESULTS['paired']['tuned_only_correct'])"""),
    ("markdown", "## Per-class scores on the held-out split"),
    ("code", """out = ['| subject area | n | base P | base R | base F1 | tuned P | tuned R | tuned F1 |',
       '| --- | --- | --- | --- | --- | --- | --- | --- |']
for label in labels:
    b, t = base['per_class'][label], tuned['per_class'][label]
    out.append(f"| {label} | {b['support']} | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} "
               f"| {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} |")
print('\\n'.join(out))"""),
    ("markdown", "## Confusion matrices, true rows by predicted column"),
    ("code", """def show(matrix):
    head = '| true \\\\ predicted | ' + ' | '.join(labels) + ' |'
    print(head)
    print('| --- ' * (len(labels) + 1) + '|')
    for label, row in zip(labels, matrix):
        print(f'| {label} | ' + ' | '.join(str(v) for v in row) + ' |')

print('base'); show(base['confusion_matrix'])
print()
print('fine-tuned'); show(tuned['confusion_matrix'])"""),
    ("markdown", "## Latency on this machine"),
    ("code", """for name, report in (('base', base), ('fine-tuned', tuned)):
    batched, single = report['batched_latency'], report.get('batch1_latency', {})
    print(f"{name}: batch of 8 {batched['ms_per_example_mean']:.0f} ms/example mean, "
          f"p95 {batched['ms_per_example_p95']:.0f} ms; batch of 1 "
          f"{single.get('ms_per_example_mean', float('nan')):.0f} ms mean, "
          f"p95 {single.get('ms_per_example_p95', float('nan')):.0f} ms")
print('device:', RESULTS['environment']['platform'], '| torch', RESULTS['environment']['torch'],
      '| threads', RESULTS['environment']['torch_threads'])"""),
    ("markdown", "## Dataset, selection rules and licence"),
    ("code", """print('columns read:', PROVENANCE['source']['columns_read'])
print('columns not read:', PROVENANCE['source']['columns_not_read'])
print('access:', PROVENANCE['source']['access'])
print('licence:', PROVENANCE['licence'])
print('licence counts:', PROVENANCE['licence_counts'])
print()
for rule in PROVENANCE['selection_rules']:
    print('-', rule)"""),
    ("markdown", "## Recompute a score from the saved logits"),
    ("code", """import math

def softmax(values, temperature):
    scaled = [v / temperature for v in values]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    return [e / sum(exps) for e in exps]

def accuracy(report, rows):
    hits = 0
    for item in rows:
        probs = softmax(item['logits'], report['temperature'])
        hits += int(max(range(len(probs)), key=lambda i: probs[i]) == item['label'])
    return hits / len(rows)

for name, report in (('base', base), ('fine-tuned', tuned)):
    rows = RESULTS['predictions'][name]
    print(name, 'recomputed accuracy', round(accuracy(report, rows), 4),
          'reported', round(report['accuracy'], 4))"""),
    ("markdown", "## What this does not show"),
    ("code", """for item in RESULTS['limitations']:
    print('-', item)
print()
print('training:', json.dumps(RESULTS['training'], indent=2)[:1200])"""),
]


def build(results_path=RESULTS, provenance_path=PROVENANCE, output=None):
    results = json.loads(Path(results_path).read_text())
    provenance = json.loads(Path(provenance_path).read_text()) if Path(provenance_path).exists() else {}
    cells = []
    for kind, source in CELLS:
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                          "source": source.strip().splitlines(True)})
    # Embed the artefacts so Run All is self-contained and offline.
    cells.insert(1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": ["from pathlib import Path\n",
                                f"Path('router_results.json').write_text({json.dumps(json.dumps(results))!r})\n",
                                f"Path('router_provenance.json').write_text({json.dumps(json.dumps(provenance))!r})\n"]})
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "kaggle": {"language": "python", "kernelType": "notebook"}},
                "cells": cells}
    target = Path(output or ROOT / "kaggle" / "arxiv_router.ipynb")
    target.write_text(json.dumps(notebook, indent=1))
    metadata = {"id": "uranium53/laya-arxiv-topic-router", "title": "Laya arXiv Topic Router: Base vs Fine-tuned",
                "code_file": target.name, "language": "python", "kernel_type": "notebook",
                "is_private": False, "enable_gpu": False, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (target.parent / "kernel-metadata.router.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target


if __name__ == "__main__":
    print("Packaged", build())
