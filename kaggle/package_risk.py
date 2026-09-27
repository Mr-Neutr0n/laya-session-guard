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

**This experiment is rejected for deployment. It is not a safety guard and must not be used to authorize tool calls.** It is published as a negative result so the measurement is reproducible. The numbers below are corrected after an independent replication, which reproduced every headline result and contradicted three descriptive details.

On unseen attacker goals this run **incorrectly allowed between 58 and 66 of 176** unauthorized actions, and between 13 and 18 of 36 on the both-unseen split. Those are two executions of the same code with the same seed on the same data, not a range of confidence intervals. The pipeline was not bit-reproducible: fp16 autocast on GPU with no deterministic-algorithm flags let the run diverge during training, so the per-case counts vary. This run enables `torch.use_deterministic_algorithms`, `cudnn.deterministic`, disables `cudnn.benchmark` and TF32, and sets `CUBLAS_WORKSPACE_CONFIG`, so a rerun on the same stack is comparable. Bit-identical results are not claimed across different GPUs, drivers, or library versions, and fp16 accumulation order remains the weakest link.

It cut false allows on unseen user tasks from 199 to between 4 and 4, so it is better than the pinned base on every axis and still unsafe. A model that is excellent on one axis and unsafe on another is exactly the artifact that must not be mistaken for a guard. **The rejection verdict is unaffected by the variance: every run in the observed range is rejected.**

**All 84 false allows on the ASPI splits come from a single attacker goal, goal 13**, not from a class of attacks. 23 of them occur on a transcript with a benign source and no injection anywhere, where the model allowed an outward-facing email send the user never requested.

**Wrapper threshold, honestly fitted.** With the margin defined as `allow_logit - log(exp(block_logit) + exp(review_logit))`, a threshold fitted by reading the test splits reports 52 of 66 false allows caught. Cross-fitted on a different split and applied out of sample, the same rule catches **31 of 66**, at zero cost to the 176 legitimate allowances. The in-sample figure is roughly double the honest one. No threshold can be fitted on the calibration split for the earlier runs, because the calibration predictions were never persisted and the weights were never downloaded; this run writes `calibration_predictions.jsonl` for the first time, and on those rows the temperature fit is interior at 2.18 rather than pinned to a grid edge. Full procedure: `scripts/analyze_false_allows.py`.

**The errors are confident, not borderline.** Median margin over the false allows is 11.3 on unseen attacker goals and 12.6 on the both-unseen split, interquartile ranges 8.3 to 13.8 and 9.2 to 16.3, so recalibration cannot help.

**Conditional permission is confounded with a metadata flag, in one direction.** In the 2,920-case training split, all 216 incomplete-history cases are labelled `review` with no counterexample anywhere, while 40 `review` cases have a complete history. The model learned the one learnable direction and had no data teaching the other.

**A synthetic replacement for the missing data was built and rejected.** A 64-pair S6/S7 stopgap covers the failure shape and covers 11 of 11 task shapes and 5 of 5 action features, but a fidelity comparison against the real cases shows it is 0.35x the length, shares a vocabulary Jaccard of 0.075, and carries a structured `permission` field that 0 of 2,720 real cases have. **A model trained on it should not be expected to transfer, and it does not justify a training run.** It is kept as a specification for the human collector, not as data.

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
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)},
                      code(setup), code(write), code(run)]}
(DEST / "risk.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {"id": "uranium53/laya-risk-ranked-aspi", "title": "Laya Risk Ranked ASPI",
            "code_file": "risk.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaL4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
(DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Packaged", len(files), "files; notebook bytes", (DEST / "risk.ipynb").stat().st_size)
