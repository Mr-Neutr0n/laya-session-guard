"""Package the reproducible Wikispeedia experiment as a Kaggle notebook."""
import base64
import io
import json
import tarfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle/wikirace"
DATASET = ROOT / "kaggle/wikirace-dataset"
FILES = ["wikirace/__init__.py", "wikirace/graph.py", "wikirace/train.py"]
DATA_FILES = ["graph.json", "train.jsonl", "calibration.jsonl", "test.jsonl", "eval_races.json", "provenance.json"]


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    DATASET.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        shutil.copy2(ROOT / "data/wikirace" / name, DATASET / name)
    (DATASET / "datasets-metadata.json").unlink(missing_ok=True)
    (DATASET / "dataset-metadata.json").write_text(json.dumps({"title": "Laya Wikirace Wikispeedia Graph and Splits",
        "id": "uranium53/laya-wikirace-data", "licenses": [{"name": "other"}],
        "subtitle": "Archived Wikispeedia graph and target-disjoint splits"}, indent=2))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name in FILES:
            archive.add(ROOT / name, arcname=name)
    payload = base64.b64encode(data.getvalue()).decode()
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [
                    cell("markdown", "# Laya Wikiracing: base versus fine-tuned\n\nA fixed Wikispeedia graph, target-disjoint splits, and identical 60-race evaluation. The notebook runs base Laya before training, then full supervised fine-tuning on one T4. It records decision accuracy, race success, and GPU model latency separately. Source: https://snap.stanford.edu/data/wikispeedia.html\n"),
                    cell("code", "!pip -q install laya==0.3.5 transformers==4.57.6 safetensors==0.6.2 numpy==2.0.2\n"),
                    cell("code", "import os, base64, io, tarfile, shutil\nfrom pathlib import Path\nos.environ['USE_TF']='0'\nos.environ['USE_FLAX']='0'\npayload = '" + payload + "'\nroot=Path('/kaggle/working/wikirace-project')\nwith tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as archive:\n    archive.extractall(root, filter='data')\n(root/'data/wikirace').mkdir(parents=True,exist_ok=True)\nfor source in Path('/kaggle/input/laya-wikirace-data').iterdir():\n    if source.is_file() and source.suffix in ('.json','.jsonl'):\n        shutil.copy2(source,root/'data/wikirace'/source.name)\nos.chdir(root)\nprint('Graph and splits unpacked')\n"),
                    cell("code", "!PYTHONPATH=. OUTPUT_DIR=/kaggle/working/laya-wikirace python -u -m wikirace.train\n"),
                    cell("code", "import json\nfrom pathlib import Path\nreport=json.loads(Path('/kaggle/working/laya-wikirace/report.json').read_text())\nprint(json.dumps(report, indent=2))\n")
                ]}
    (DEST / "wikirace.ipynb").write_text(json.dumps(notebook))
    metadata = {"id": "uranium53/laya-wikiracing-base-vs-fine-tuned", "title": "Laya Wikiracing Base vs Fine Tuned", "code_file": "wikirace.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": ["uranium53/laya-wikirace-data"], "competition_sources": [], "kernel_sources": []}
    (DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Packed {len(FILES)} files, {len(data.getvalue())} compressed bytes")


if __name__ == "__main__":
    main()
