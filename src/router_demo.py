"""Paper-triage demo: route a title and abstract to one of eight arXiv areas.

Loads the published checkpoint from the Hugging Face Hub at a pinned revision, so
the demo exercises exactly the weights that were released, not a local copy.

    .venv/bin/python src/router_demo.py --title "..." --abstract "..."
    .venv/bin/python src/router_demo.py --file cases.jsonl --out predictions.json
    cat cases.jsonl | .venv/bin/python src/router_demo.py --stdin --out predictions.json

CPU only, no paid service. Prints a table to stderr and writes JSON to stdout or a
file, so it pipes into the static view in `scripts/build_demo_view.py`.
"""
import argparse
import json
import os
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pinned to the released commit, not to `main`.
DEFAULT_MODEL = "hxrikp/laya-arxiv-topic-router"
DEFAULT_REVISION = "1c2bb313b6a94b674c2e2c57483760eb5ccab244"
MAX_WORDS = 100
# Same budget the released checkpoint was evaluated under.
MAX_LEN = 640
HEAD_MAX_LEN = 256


def load(model, revision, device):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import router_run
    path = snapshot_download(model, revision=revision)
    print(f"loaded {model}@{revision[:12]} from {path}", file=sys.stderr)
    network, _ = router_run.load_model(path, device)
    network.eval()
    tokenizer = AutoTokenizer.from_pretrained(Path(path, "tokenizer"))
    return network, tokenizer, path


def clean(text, max_words=MAX_WORDS):
    return " ".join(" ".join(text.split()).split(" ")[:max_words]).strip()


def run(network, tokenizer, cases, labels, device, batch_size=4, max_words=MAX_WORDS):
    import torch
    import router_run
    from router_task import build_items
    rows = [{"id": case.get("id", f"case-{i}"), "title": case.get("title", ""),
             "abstract": clean(case.get("abstract", ""), max_words), "label": None}
            for i, case in enumerate(cases)]
    items = build_items(tokenizer, rows, labels, MAX_LEN, HEAD_MAX_LEN, require_label=False)
    pad = tokenizer.pad_token_id
    results = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        began = time.perf_counter()
        with torch.no_grad():
            logits, _ = network(*router_run.collate(chunk, pad, device))
        elapsed = (time.perf_counter() - began) * 1000.0
        for item, values in zip(chunk, logits.float()):
            scores = values[:len(item["markers"])]
            best = max(range(len(scores)), key=lambda i: float(scores[i]))
            top = float(scores[best])
            exps = [float(v - top) for v in scores]
            total = sum(math.exp(v) for v in exps)
            distribution = {label: math.exp(exps[i]) / total for i, label in enumerate(labels)}
            results.append({"id": item["id"], "title": rows[start + chunk.index(item)]["title"],
                            "tokens": len(item["ids"]),
                            "latency_ms_per_example": elapsed / len(chunk),
                            "distribution": distribution,
                            "top_label": labels[best],
                            "top_probability": distribution[labels[best]],
                            "margin_over_runner_up": distribution[labels[best]] - max(
                                v for k, v in distribution.items() if k != labels[best])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--title", help="paper title; pair with --abstract")
    source.add_argument("--file", help="JSONL file of {\"id\", \"title\", \"abstract\"} objects")
    source.add_argument("--stdin", action="store_true", help="read that JSONL from stdin")
    parser.add_argument("--abstract", default="", help="paper abstract, truncated to 100 words")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS,
                        help="abstract words kept; the released evaluation used 100")
    parser.add_argument("--out", default=None, help="write the JSON result here instead of stdout")
    args = parser.parse_args()

    if args.title is not None:
        cases = [{"id": "cli", "title": args.title, "abstract": clean(args.abstract, args.max_words)}]
    else:
        handle = sys.stdin if args.stdin else open(args.file)
        cases = [json.loads(line) for line in handle if line.strip()]
        if not args.stdin:
            handle.close()

    from arxiv_router_data import LABELS
    labels = list(LABELS)

    import torch
    network, tokenizer, path = load(args.model, args.revision, torch.device(args.device))
    results = run(network, tokenizer, cases, labels, args.device, args.batch_size, args.max_words)
    payload = {"model": args.model, "revision": args.revision, "local_path": str(path),
               "device": args.device, "max_abstract_words": args.max_words, "labels": labels,
               "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "predictions": results}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    for item in results:
        ranked = sorted(item["distribution"].items(), key=lambda kv: -kv[1])
        summary = "  ".join(f"{label} {value:.2f}" for label, value in ranked[:4])
        print(f"{item['id']}: {item['top_label']} {item['top_probability']:.2f} | {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
