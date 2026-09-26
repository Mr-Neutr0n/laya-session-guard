"""Paired base and full fine-tune evaluation on the Wikispeedia decision task."""
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, build_sequence

from wikirace.graph import load_graph, shortlist, question, state

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/wikirace"
OUT = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/laya-wikirace"))
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def read_rows(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def encode_row(tokenizer, row, cfg):
    q = row["question"]
    schema = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    ids, positions = build_sequence(tokenizer, row["state"], schema, 1_000_000, cfg["head_max_len"])
    if len(ids) > cfg["max_len"] or len(positions) != len(row["candidates"]):
        raise ValueError(f"Truncated or malformed question: {row['id']}")
    return {"id": row["id"], "ids": ids, "markers": positions, "label": row["label"],
            "current": row["current"], "target": row["target"]}


def batch(rows, pad, device):
    n = len(rows)
    width = max(len(r["ids"]) for r in rows)
    choices = max(len(r["markers"]) for r in rows)
    ids = torch.full((n, width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    markers = torch.zeros((n, choices), dtype=torch.long)
    option_mask = torch.zeros((n, choices), dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"])
        attention[i, :len(row["ids"])] = 1
        markers[i, :len(row["markers"])] = torch.tensor(row["markers"])
        option_mask[i, :len(row["markers"])] = True
    return [x.to(device) for x in (ids, attention, markers, option_mask, torch.zeros(n, dtype=torch.long))]


def predictions(model, rows, tokenizer, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            chunk = rows[start:start + 8]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*batch(chunk, tokenizer.pad_token_id, device))
            for row, z in zip(chunk, logits.float().cpu()):
                result.append({"id": row["id"], "label": row["label"],
                               "logits": z[:len(row["markers"])].tolist()})
    return result


def fit_temperature(predictions):
    best = (float("inf"), 1.0)
    for t in np.linspace(0.5, 5.0, 91):
        loss = 0.0
        for row in predictions:
            z = np.asarray(row["logits"], dtype=np.float64) / t
            loss += float(np.logaddexp.reduce(z) - z[row["label"]])
        if loss < best[0]:
            best = (loss, float(t))
    return best[1]


def decision_metrics(predictions, temperature):
    correct = 0
    brier = 0.0
    confidences = []
    for row in predictions:
        z = np.asarray(row["logits"], dtype=np.float64) / temperature
        p = np.exp(z - np.logaddexp.reduce(z))
        choice = int(p.argmax())
        correct += choice == row["label"]
        confidences.append((float(p.max()), choice == row["label"]))
        truth = np.zeros(len(p)); truth[row["label"]] = 1
        brier += float(((p - truth) ** 2).sum())
    ece = sum(sum(1 for confidence, _ in confidences if a <= confidence < b) / len(confidences) *
              abs(np.mean([confidence for confidence, _ in confidences if a <= confidence < b]) -
                  np.mean([good for confidence, good in confidences if a <= confidence < b]))
              for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])
              if any(a <= confidence < b for confidence, _ in confidences))
    return {"n": len(predictions), "top1_accuracy": correct / len(predictions),
            "brier": brier / len(predictions), "ece_10_bin": float(ece), "temperature": temperature}


def race_metrics(model, tokenizer, cfg, graph, device, races, temperature, hop_limit=8):
    model.eval()
    result = []
    with torch.inference_mode():
        for race in races:
            current, target = race["start"], race["target"]
            visited = [current]
            timings = []
            for step in range(hop_limit):
                if current == target or not graph["links"][current]:
                    break
                candidates = shortlist(graph, current, target)
                row = {"id": "race", "state": state(graph, current, target),
                       "question": question(graph, candidates), "candidates": candidates, "label": 0,
                       "current": current, "target": target}
                encoded = encode_row(tokenizer, row, cfg)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(*batch([encoded], tokenizer.pad_token_id, device))
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
                z = logits[0, :len(candidates)].float().cpu().numpy() / temperature
                current = candidates[int(z.argmax())]
                visited.append(current)
                if current == target:
                    break
            result.append({"start": race["start"], "target": target, "path": visited,
                           "success": current == target, "hops": len(visited) - 1,
                           "shortest_hops": race["shortest_hops"], "latency_ms": timings})
    times = [t for r in result for t in r["latency_ms"]]
    wins = [r for r in result if r["success"]]
    return {"n": len(result), "wins": len(wins), "win_rate": len(wins) / len(result),
            "mean_hops_on_wins": float(np.mean([r["hops"] for r in wins])) if wins else None,
            "mean_excess_hops_on_wins": float(np.mean([r["hops"] - r["shortest_hops"] for r in wins])) if wins else None,
            "latency_ms_p50": float(np.percentile(times, 50)), "latency_ms_p95": float(np.percentile(times, 95)),
            "traces": result}


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA GPU required")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model_dir = snapshot_download("convaiinnovations/laya", revision=REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"])
    _fix_tokenizer_config(model_dir)
    cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
    cfg.update(max_len=512, head_max_len=192, amp_dtype="fp16", temperature=[1., 1., 1.], temperature_by_options={})
    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir, "tokenizer"))
    splits = {name: [encode_row(tokenizer, row, cfg) for row in read_rows(name)]
              for name in ("train", "calibration", "test")}
    print("PREPARED", {key: len(value) for key, value in splits.items()},
          "max_tokens", max(len(row["ids"]) for rows in splits.values() for row in rows), flush=True)
    graph = load_graph()
    races = json.loads((DATA / "eval_races.json").read_text())
    model = build_model(cfg, encoder_dir=str(Path(model_dir, "encoder")))
    model.encoder.config.reference_compile = False
    model.load_state_dict(load_file(str(Path(model_dir, "model.safetensors"))), strict=True)
    model.to(device)
    baseline_cal = predictions(model, splits["calibration"], tokenizer, device)
    baseline_temp = fit_temperature(baseline_cal)
    baseline_preds = predictions(model, splits["test"], tokenizer, device)
    baseline_decisions = decision_metrics(baseline_preds, baseline_temp)
    baseline_races = race_metrics(model, tokenizer, cfg, graph, device, races, baseline_temp)
    (OUT / "baseline.json").write_text(json.dumps({"decisions": baseline_decisions, "races": baseline_races}, indent=2))
    print("BASELINE", json.dumps({"decisions": baseline_decisions, "races": {k: v for k, v in baseline_races.items() if k != "traces"}}), flush=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 2e-5}, {"params": head, "lr": 5e-5}], weight_decay=.01)
    epochs, batch_size, accum = 3, 4, 4
    total_steps = epochs * math.ceil(math.ceil(len(splits["train"]) / batch_size) / accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, int(total_steps * .06)),
        max(0., (total_steps - step) / max(1, total_steps * .94))))
    scaler = torch.amp.GradScaler("cuda")
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        rows = splits["train"].copy()
        random.shuffle(rows)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, offset in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[offset:offset + batch_size]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*batch(chunk, tokenizer.pad_token_id, device))
                loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([r["label"] for r in chunk], device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss / accum).backward()
            losses.append(float(loss))
            if (step + 1) % accum == 0 or offset + batch_size >= len(rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 150 == 0:
                print("PROGRESS", epoch + 1, step + 1, len(rows) // batch_size, round(np.mean(losses[-150:]), 4), flush=True)
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
        save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()}, str(OUT / "model.safetensors"))
        print("EPOCH", history[-1], flush=True)
    training_seconds = time.perf_counter() - started
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    tuned_cal = predictions(model, splits["calibration"], tokenizer, device)
    tuned_temp = fit_temperature(tuned_cal)
    tuned_preds = predictions(model, splits["test"], tokenizer, device)
    tuned_decisions = decision_metrics(tuned_preds, tuned_temp)
    tuned_races = race_metrics(model, tokenizer, cfg, graph, device, races, tuned_temp)
    cfg["temperature_by_options"] = {f"choice:{bucket}": tuned_temp
                                          for bucket in ("2", "3-5", "6-10", "11+")}
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-wikirace-wikispeedia"
    (OUT / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer.save_pretrained(OUT / "tokenizer")
    model.encoder.config.save_pretrained(OUT / "encoder")
    report = {"status": "completed", "base_revision": REVISION, "seed": 42, "gpu": torch.cuda.get_device_name(0),
              "training_seconds": training_seconds, "history": history,
              "base": {"decisions": baseline_decisions, "races": {k: v for k, v in baseline_races.items() if k != "traces"}},
              "tuned": {"decisions": tuned_decisions, "races": {k: v for k, v in tuned_races.items() if k != "traces"}},
              "split_sizes": {k: len(v) for k, v in splits.items()}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "race_traces.json").write_text(json.dumps({"base": baseline_races["traces"], "tuned": tuned_races["traces"]}))
    (OUT / "decision_predictions.json").write_text(json.dumps({"base": baseline_preds, "tuned": tuned_preds}))
    print("COMPLETED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
