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


intro = """# Laya risk-ranked ASPI pilot

Private Kaggle GPU run. Laya trains with a cost for incorrect `allow` decisions and an action-pair ranking loss. The public ASPI workspace benchmark is adapted to unverified clarification events. Tests hold out user tasks, attacker goals, and both. The previous 24-case challenge is diagnostic only. ASPI labels are benchmark-generated, not human-reviewed authorization judgments. This experiment cannot authorize real tool calls.

Source: https://huggingface.co/datasets/ScaleAI/aspi (CC BY 4.0). See the repository's `data/aspi/README.md` for split and adaptation details.
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
