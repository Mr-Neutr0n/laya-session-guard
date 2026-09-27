"""Package the private Kaggle risk-ranked Laya experiment."""
import base64
import json
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "kaggle" / "risk"
DEST.mkdir(exist_ok=True)
files = {p.name: p.read_text() for p in (ROOT / "src").glob("*.py")}
files["challenge.jsonl"] = (ROOT / "data" / "challenge.jsonl").read_text()
files["workspace.jsonl"] = (ROOT / "data" / "aspi" / "workspace.jsonl").read_text()
payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), level=9)).decode()


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": value.splitlines(True)}


intro = """# Laya risk-ranked ASPI pilot: RESEARCH-ONLY, REJECTED

**This experiment is rejected for deployment. It is not a safety guard and must not be used to authorize tool calls.** It is published as a negative result so the measurement is reproducible.

## The numbers in this notebook are generated, not written

**No per-run count appears in this prose.** Hard-coded figures are why this write-up contradicted its own output twice: the first revision asserted one run's counts, and a later revision asserted a two-run range that a third run fell outside.

The table below this cell is produced by a code cell that reads this run's own `report.json` and per-case rows after training finishes. It cannot disagree with the run, because it is written by the run. The committed three-run spread, which is an observation about repeatability rather than a per-run claim, is in the repository reports.

## What is stable across runs

- The run is rejected in every observed execution, by a wide margin.
- Every false allow on the ASPI splits comes from a single attacker goal, goal 13, not from a class of attacks. Some occur on a transcript with a benign source and no injection anywhere, where the model allowed an outward-facing email send the user never requested.
- The errors are confident, not borderline, so recalibration cannot help.
- Conditional permission is confounded with a metadata flag in one direction: in the training split, every incomplete-history case is labelled `review` with no counterexample.
- The synthetic replacement built for this failure shape was measured and rejected as non-transferable. It is a specification for the human collector, not data.

## The observed spread, and why one run is not enough

The three runs executed **before determinism was enabled** produced false-allow counts on unseen attacker goals of 66, 58, and 33 of 176, a spread of nearly 2x on the primary metric, because fp16 autocast on GPU let training diverge. That range describes those three runs and is **not a bound on this or any future run**: enabling determinism changes kernel selection, so a deterministic run is not expected to land inside it. What a deterministic run should reproduce is its own previous deterministic run on the same stack, which has not yet been demonstrated for want of a duplicate run. The current script sets `torch.use_deterministic_algorithms`, `cudnn.deterministic`, disables `cudnn.benchmark` and TF32, and sets `CUBLAS_WORKSPACE_CONFIG`; it is the first configuration intended to be repeatable, and it runs with no warning about any operation lacking a deterministic implementation.

**The verdict is unaffected in every run.** The range spans 33 to 66 of 176 unauthorized actions allowed, and every value in it is rejected for the same reason. Treat a single run as one draw, not as the measurement.

**What determinism does not deliver.** Bit-identical results are not claimed across different GPUs, drivers, or library versions, since the kernels differ. fp16 accumulation order remains the weakest link: deterministic algorithms fix which algorithm each operation uses, not the order in which reduced-precision values accumulate. A duplicate run on the same stack would be needed to demonstrate repeatability end to end, and that has not been done.

Source: https://huggingface.co/datasets/ScaleAI/aspi (CC BY 4.0). See the repository's `data/aspi/README.md` for split and adaptation details. Reports: `reports/RISK_RANKED.md` and `reports/FALSE_ALLOW_ANALYSIS.md`. Independently replicated; replication checkout `/Users/harikp/Developer/scratch/laya-replication/`.
"""
setup = """import os, subprocess, sys
os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.3.5', 'transformers==4.57.6', 'safetensors==0.6.2'], check=True)
import torch
assert torch.cuda.is_available(), 'Kaggle GPU required'
print('GPU', torch.cuda.get_device_name(0), flush=True)
"""
write = """import base64, json, zlib
from pathlib import Path
files = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)).decode())
for name, source in files.items():
    Path('/kaggle/working', name).write_text(source)
print('Unpacked', len(files), 'files', flush=True)
""".replace("PAYLOAD", repr(payload))
run = """env = dict(os.environ,
    ASPI_SOURCE='/kaggle/working/workspace.jsonl',
    CHALLENGE_DATA='/kaggle/working/challenge.jsonl',
    OUTPUT_DIR='/kaggle/working/laya-risk-ranked',
    TRAIN_EPOCHS='3', RISK_WEIGHT='2.0', PAIR_WEIGHT='1.0', RISK_MARGIN='1.0')
subprocess.run([sys.executable, '-u', '/kaggle/working/train_risk.py'], env=env, check=True)
subprocess.run([sys.executable, '-u', '/kaggle/working/evaluate_challenge.py',
                '--model', '/kaggle/working/laya-risk-ranked',
                '--data', '/kaggle/working/challenge.jsonl',
                '--output', '/kaggle/working/laya-risk-ranked/challenge.json'], env=env, check=True)
# The per-case rows are the artifact that lets a wrapper threshold be fitted on
# calibration rather than on the frozen test splits. Fail loudly if they are missing.
out = Path('/kaggle/working/laya-risk-ranked')
required = ['calibration_predictions.jsonl', 'test_attack_predictions.jsonl',
            'test_both_predictions.jsonl', 'test_user_predictions.jsonl',
            'old_challenge_predictions.jsonl', 'challenge_predictions.jsonl']
missing = [n for n in required if not (out / n).exists()]
assert not missing, f'missing per-case rows: {missing}'
for n in required:
    print('ROWS', n, sum(1 for _ in (out / n).open()), flush=True)
"""
summarise = """# Render this run's own numbers. Read from the report the training just wrote, so
# the table below is generated by the run and cannot contradict it.
import json as _json
_rep = _json.loads(Path('/kaggle/working/laya-risk-ranked', 'report.json').read_text())
_LABELS = ['allow', 'block', 'review']
_rows = []
for _name, _v in _rep['finetuned'].items():
    _a = _v['action']
    _rows.append((_name, _a['false_allow'], _a['nonallow'], _a['correct_allow'], _a['allow_cases'],
                  _a['accuracy'], _a['block_recall']))
print('EXECUTED RUN SUMMARY (generated from this run\'s report.json)')
print(f"  best epoch {_rep['best_epoch']}, {_rep['training_seconds']:.0f} s, "
      f"temperatures {_rep['temperatures']}")
print(f"  {'split':<14}{'acc':>8}{'false allow':>14}{'correct allow':>16}{'block recall':>14}")
for _n, _fa, _na, _ca, _al, _acc, _br in sorted(_rows):
    print(f"  {_n:<14}{_acc * 100:>7.1f}%{f'{_fa}/{_na}':>14}{f'{_ca}/{_al}':>16}{_br * 100:>13.1f}%")
_det = _rep.get('determinism')
if _det:
    print(f"  determinism: seeded {bool(_det.get('seed'))}, "
          f"deterministic algorithms {bool(_det.get('use_deterministic_algorithms'))}, "
          f"cudnn.benchmark {_det.get('cudnn_benchmark')}, tf32 {_det.get('tf32')}")
print('  REJECTED: unsafe on unseen attacker goals in this run, as in every recorded run.')
"""
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)},
                      code(setup), code(write), code(run), code(summarise)]}
(DEST / "risk.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {"id": "uranium53/laya-risk-ranked-aspi", "title": "Laya Risk Ranked ASPI",
            "code_file": "risk.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaL4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
(DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Packaged", len(files), "files; notebook bytes", (DEST / "risk.ipynb").stat().st_size)
