"""Package two controlled private Kaggle training runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "kaggle" / "followup"
DEST.mkdir(exist_ok=True)
files = {p.name: p.read_text() for p in (ROOT / "src").glob("*.py")}
files["challenge.jsonl"] = (ROOT / "data" / "challenge.jsonl").read_text()


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": value.splitlines(True)}


intro = """# Laya session guard follow-up

Two controlled runs on Kaggle GPU. Run A repeats the original 480-session training set for eight epochs. Run B trains for eight epochs on 576 new counterfactual sessions. Both start from the identical pinned base checkpoint. The original 24 challenge cases are used only after training. All data is synthetic. This does not establish production safety.
"""
setup = """import os, subprocess, sys, json
from pathlib import Path
os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.3.5', 'transformers==4.57.6', 'safetensors==0.6.2'], check=True)
import torch
assert torch.cuda.is_available(), 'Kaggle GPU required'
print('GPU', torch.cuda.get_device_name(0), flush=True)
"""
write = "files = json.loads(" + repr(json.dumps(files)) + ")\nfor name, source in files.items():\n    Path('/kaggle/working', name).write_text(source)\n"
run = """for variant in ('original', 'counterfactual'):
    output = f'/kaggle/working/followup-{variant}'
    env = dict(os.environ, OUTPUT_DIR=output, DATA_VARIANT=variant, TRAIN_EPOCHS='8')
    print('START', variant, flush=True)
    subprocess.run([sys.executable, '-u', '/kaggle/working/train.py'], env=env, check=True)
    subprocess.run([sys.executable, '-u', '/kaggle/working/evaluate_challenge.py',
                    '--model', output, '--data', '/kaggle/working/challenge.jsonl',
                    '--output', f'{output}/challenge.json'], check=True)
    report = json.loads(Path(output, 'report.json').read_text())
    challenge = json.loads(Path(output, 'challenge.json').read_text())
    print('RESULT', variant, json.dumps({'train_loss': report['history'],
          'template': report['finetuned'], 'challenge': challenge['metrics'],
          'wrapper': challenge['wrapper']}), flush=True)
"""
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)},
                      code(setup), code(write), code(run)]}
(DEST / "followup.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {"id": "uranium53/laya-session-guard-follow-up", "title": "Laya Session Guard Follow-up",
            "code_file": "followup.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
(DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Packaged", len(files), "files")
