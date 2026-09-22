"""Embed auditable source files in a private Kaggle notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = {p.name: p.read_text() for p in (ROOT / "src").glob("*.py")}
setup = """import os, subprocess, sys
os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.3.5', 'transformers==4.57.6', 'safetensors==0.6.2'], check=True)
import torch
assert torch.cuda.is_available(), 'GPU required'
print('GPU:', torch.cuda.get_device_name(0), 'visible:', torch.cuda.device_count(), flush=True)
"""
write = "from pathlib import Path\nimport json\nfiles = json.loads(" + repr(json.dumps(files)) + ")\nfor name, source in files.items():\n    Path('/kaggle/working', name).write_text(source)\n"
run = "import subprocess, sys\nsubprocess.run([sys.executable, '-u', '/kaggle/working/train.py'], check=True)\n"
def cell(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}
nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
    "cells":[{"cell_type":"markdown","metadata":{},"source":["# Laya session guard\nPrivate synthetic pilot. Full supervised fine-tune with base comparison and disjoint calibration. Single T4 training, bounded to one hour. No real user transcripts.\n"]},cell(setup),cell(write),cell(run)]}
(ROOT/'kaggle'/'session_guard.ipynb').write_text(json.dumps(nb,indent=1))
metadata={"id":"uranium53/laya-session-guard-pilot","title":"Laya Session Guard Pilot","code_file":"session_guard.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":[],"kernel_sources":[]}
(ROOT/'kaggle'/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Packaged',list(files))
