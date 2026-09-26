"""Metadata-only arXiv topic router: base versus fine-tuned Laya, one held-out set.

Free compute only. This script trains on CPU and never contacts a paid API.

  data     rebuild the bounded dataset (see src/arxiv_router_data.py)
  base     evaluate the pinned base checkpoint
  train    fine-tune the decision head and the top encoder layers
  eval     evaluate a checkpoint, base or fine-tuned, on the same held-out split
  extras   capture calibration logits for both checkpoints and run a title-only ablation

Base and fine-tuned scores come from identical test items, with temperatures
fitted on the same separate calibration split.
"""
import argparse
import json
import math
import os
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
import platform
import random
import time
from pathlib import Path

MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
MAX_LEN = 640
HEAD_MAX_LEN = 256
SEED = 42


def collate(items, pad, device):
    import torch
    n, length = len(items), max(len(x["ids"]) for x in items)
    k = max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    pos = torch.zeros((n, k), dtype=torch.long)
    mask = torch.zeros((n, k), dtype=torch.bool)
    for i, row in enumerate(items):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"])
        attention[i, :len(row["ids"])] = 1
        pos[i, :len(row["markers"])] = torch.tensor(row["markers"])
        mask[i, :len(row["markers"])] = True
    return [x.to(device) for x in (ids, attention, pos, mask, torch.zeros(n, dtype=torch.long))]


def base_model_dir():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    path = snapshot_download("convaiinnovations/laya", revision=MODEL_REVISION,
                             allow_patterns=["model.safetensors", "rl_agent_config.json",
                                             "encoder/config.json", "tokenizer/*"])
    _fix_tokenizer_config(path)
    return path


def load_model(model_dir, device):
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    source = Path(model_dir)
    config = json.loads((source / "rl_agent_config.json").read_text())
    config.update(max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, amp_dtype="fp32",
                  temperature=[1.0, 1.0, 1.0], temperature_by_options={})
    if (source / "encoder").exists():
        encoder_dir = str(source / "encoder")
    else:
        from huggingface_hub import snapshot_download
        encoder_dir = snapshot_download("convaiinnovations/laya", revision=MODEL_REVISION,
                                        allow_patterns=["encoder/config.json"])
    model = build_model(config, encoder_dir=encoder_dir)
    model.encoder.config.reference_compile = False
    model.load_state_dict(load_file(str(source / "model.safetensors")), strict=True)
    return model.to(device), config


def load_items(tok, rows, labels):
    from router_task import build_items
    return build_items(tok, rows, labels, MAX_LEN, HEAD_MAX_LEN)


def evaluate(model, items, device, pad, batch_size=8, tag=""):
    import torch
    from router_metrics import Latency
    model.eval()
    latency = Latency()
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            started = time.perf_counter()
            logits, _ = model(*collate(chunk, pad, device))
            latency.record((time.perf_counter() - started) * 1000.0)
            logits = logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite evaluation logits")
            for item, values in zip(chunk, logits):
                rows.append({"id": item["id"], "label": item["label"], "category": item["category"],
                             "logits": [round(float(v), 5) for v in values[:len(item["markers"])]]})
            if start and start % 100 == 0:
                print(f"  {tag} {start}/{len(items)} elapsed {latency.report()['total_seconds']:.0f}s", flush=True)
    return rows, latency


def single_example_latency(model, items, device, pad, sample):
    """Batch-1 latency: what a router pays per request."""
    import torch
    from router_metrics import Latency
    latency = Latency()
    model.eval()
    with torch.no_grad():
        for item in items[:sample]:
            started = time.perf_counter()
            model(*collate([item], pad, device))
            latency.record((time.perf_counter() - started) * 1000.0)
    return latency


def probabilities(rows, temperature=1.0):
    out = []
    for row in rows:
        scaled = [value / temperature for value in row["logits"]]
        top = max(scaled)
        exps = [math.exp(value - top) for value in scaled]
        total = sum(exps)
        probs = [value / total for value in exps]
        out.append({"id": row["id"], "label": row["label"], "category": row["category"],
                    "probs": probs, "prediction": max(range(len(probs)), key=lambda i: probs[i])})
    return out


def report_for(name, rows, labels, calibration_rows, bootstrap_samples, latency, batch_one_latency=None):
    from router_metrics import bootstrap_ci, fit_temperature, summary
    # The temperature is fitted on raw calibration logits, never on a probability
    # vector: scaling a normalised distribution and renormalising is the identity.
    temperature, cal_nll = fit_temperature([row["logits"] for row in calibration_rows],
                                          [row["label"] for row in calibration_rows])
    scored = probabilities(rows, temperature)
    y_true = [row["label"] for row in scored]
    y_pred = [row["prediction"] for row in scored]
    stats = summary(y_true, y_pred, labels, temperatures=temperature,
                    probabilities=[row["probs"] for row in scored])
    accuracy_ci = bootstrap_ci([(int(t == p),) for t, p in zip(y_true, y_pred)],
                               lambda draw: sum(v[0] for v in draw) / len(draw),
                               samples=bootstrap_samples, seed=SEED)

    def macro_f1(draw):
        matrix = [[0] * len(labels) for _ in labels]
        for truth, guess in draw:
            matrix[truth][guess] += 1
        total = 0.0
        for i in range(len(labels)):
            tp = matrix[i][i]
            predicted = sum(matrix[r][i] for r in range(len(labels)))
            actual = sum(matrix[i])
            precision = tp / predicted if predicted else 0.0
            recall = tp / actual if actual else 0.0
            total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return total / len(labels)

    macro_ci = bootstrap_ci(list(zip(y_true, y_pred)), macro_f1, samples=bootstrap_samples, seed=SEED)
    out = {"name": name, "n": stats["n"], "temperature": temperature, "calibration_nll": cal_nll,
           "accuracy": stats["accuracy"], "accuracy_ci95": list(accuracy_ci),
           "macro_f1": stats["macro_f1"], "macro_f1_ci95": list(macro_ci),
           "macro_precision": stats["macro_precision"], "macro_recall": stats["macro_recall"],
           "ece": stats["ece"], "brier": stats["brier"],
           "per_class": stats["per_class"], "confusion_matrix": stats["confusion_matrix"],
           "labels": labels, "batched_latency": latency.report(),
           "correct": [int(t == p) for t, p in zip(y_true, y_pred)]}
    if batch_one_latency is not None:
        out["batch1_latency"] = batch_one_latency.report()
    return out


def paired(base_report, tuned_report, labels):
    from router_metrics import mcnemar
    pairs = list(zip(base_report["correct"], tuned_report["correct"]))
    result = mcnemar([(bool(b), bool(t)) for b, t in pairs])
    result["base_only_ids"] = [i for i, (b, t) in enumerate(pairs) if b and not t][:20]
    result["tuned_only_ids"] = [i for i, (b, t) in enumerate(pairs) if t and not b][:20]
    result["accuracy_delta"] = tuned_report["accuracy"] - base_report["accuracy"]
    result["macro_f1_delta"] = tuned_report["macro_f1"] - base_report["macro_f1"]
    return result


def trainable_parameters(model, train_encoder_layers):
    import torch
    from contextlib import contextmanager

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # The SDK act_head is an agent action classifier, unused here, and stays frozen.
    head = [p for name, p in model.named_parameters()
            if not name.startswith("encoder.") and "act_head" not in name]
    for parameter in head:
        parameter.requires_grad_(True)
    groups = [{"params": head, "lr": 5e-5, "name": "head"}]
    if train_encoder_layers:
        wanted = [f"encoder.layers.{i}." for i in range(28 - train_encoder_layers, 28)]
        wanted += ["encoder.norm.", "encoder.layers.27."]
        encoder_params = [p for name, p in model.named_parameters()
                          if any(name.startswith(prefix) for prefix in wanted)]
        for parameter in encoder_params:
            parameter.requires_grad_(True)
        groups.append({"params": encoder_params, "lr": 1e-5, "name": "encoder_top"})
    # act_head is an agent action classifier, unused here, and stays frozen.
    return groups, sum(p.numel() for group in groups for p in group["params"])


def no_grad_encoder(model):
    """Wrap the encoder forward so frozen layers do not build a graph."""
    import torch
    original = model.encoder.forward

    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original(*args, **kwargs)

    model.encoder.forward = wrapped
    return original


def train(args, model, config, rows, labels, device, pad, output_dir):
    import torch
    from router_metrics import Latency
    random.seed(SEED)
    torch.manual_seed(SEED)
    items = load_items(args.tokenizer, rows["train"], labels)
    print(f"train items {len(items)}", flush=True)
    groups, trainable = trainable_parameters(model, args.encoder_layers)
    print(f"trainable parameters {trainable/1e6:.1f}M in {[g['name'] for g in groups]}", flush=True)
    optimiser = torch.optim.AdamW(groups, weight_decay=0.01)
    order = list(range(len(items)))
    history = []
    started = time.time()
    stopped_early = None
    original = None
    if args.encoder_layers == 0:
        original = no_grad_encoder(model)
    steps_per_epoch = math.ceil(len(order) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min((step + 1) / max(1, int(0.06 * total_steps)),
                                    max(0.0, (total_steps - step) / max(1, total_steps - int(0.06 * total_steps)))))
    try:
        for epoch in range(args.epochs):
            model.train()
            random.shuffle(order)
            running, seen, skipped = 0.0, 0, 0
            for step, start in enumerate(range(0, len(order), args.batch_size)):
                chunk = [items[i] for i in order[start:start + args.batch_size]]
                if len(chunk) < 2:
                    skipped += 1
                    continue
                logits, _ = model(*collate(chunk, pad, device))
                loss = torch.nn.functional.cross_entropy(logits.float(),
                                                        torch.tensor([r["label"] for r in chunk], device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                optimiser.step()
                scheduler.step()
                running += float(loss)
                seen += 1
                if (step + 1) % 25 == 0:
                    elapsed = time.time() - started
                    print(f"epoch {epoch+1}/{args.epochs} batch {step+1}/{steps_per_epoch} "
                          f"loss {running/seen:.4f} elapsed {elapsed:.0f}s", flush=True)
                if args.max_train_seconds and time.time() - started > args.max_train_seconds:
                    stopped_early = f"time budget {args.max_train_seconds}s reached in epoch {epoch+1}"
                    break
            history.append({"epoch": epoch + 1, "steps": seen, "mean_loss": running / max(1, seen),
                            "skipped_batches": skipped, "seconds": time.time() - started})
            print("EPOCH", json.dumps(history[-1]), flush=True)
            if stopped_early:
                break
    finally:
        if original is not None:
            model.encoder.forward = original
    training_seconds = time.time() - started
    if stopped_early:
        history.append({"note": stopped_early})
        print("STOPPED EARLY:", stopped_early, flush=True)
    save_checkpoint(model, config, args, output_dir)
    return {"history": history, "training_seconds": training_seconds, "steps": sum(h["steps"] for h in history if "steps" in h),
            "steps_per_epoch": steps_per_epoch,
            "steps_epoch_2": next((h["steps"] for h in history if h.get("epoch") == 2), None),
            "max_train_seconds": args.max_train_seconds,
            "schedule_completed": stopped_early is None,
            "trainable_parameters": trainable, "groups": [g["name"] for g in groups],
            "epochs_requested": args.epochs, "epochs_completed": len([h for h in history if "steps" in h]),
            "stopped_early": stopped_early, "batch_size": args.batch_size,
            "encoder_layers_trained": args.encoder_layers, "device": str(device)}


def save_checkpoint(model, config, args, output_dir):
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    args.tokenizer.save_pretrained(target / "tokenizer")
    model.encoder.config.save_pretrained(target / "encoder")
    config = dict(config)
    config["fine_tuned"] = True
    config["model_name"] = "laya-arxiv-topic-router"
    (target / "rl_agent_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print("SAVED", target, flush=True)


def extras(args, rows, labels, base_dir, base_model, device, pad, report, output):
    """Calibration logits for both checkpoints, plus a title-only ablation.

    The calibration logits let `router_recompute.py` refit the temperature on the
    logit scale, which is what the metric depends on. The ablation removes the
    abstract, so the write-up can say how much of the routing comes from the title.
    `--extras-part` chooses, because on a busy CPU the calibration pass is worth
    an hour and the ablation is optional.
    """
    if args.extras_part in ("calibration", "both"):
        base_calibration, _ = evaluate(base_model, load_items(args.tokenizer, rows["calibration"], labels),
                                       device, pad, batch_size=args.calibration_batch_size, tag="base-cal")
        report["calibration_predictions"] = {"base": base_calibration}
        tuned, _ = load_model(args.model or args.output_dir, device)
        tuned_calibration, _ = evaluate(tuned, load_items(args.tokenizer, rows["calibration"], labels),
                                        device, pad, batch_size=args.calibration_batch_size, tag="tuned-cal")
        report["calibration_predictions"]["finetuned"] = tuned_calibration
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("CALIBRATION CAPTURED", flush=True)
    if args.extras_part in ("ablation", "both"):
        base_model = base_model
        tuned, _ = load_model(args.model or args.output_dir, device)
        stripped = [dict(row, abstract="") for row in rows["test"]]
        ablation = {}
        for name, model in (("base", base_model), ("finetuned", tuned)):
            items = load_items(args.tokenizer, stripped, labels)
            logits, latency = evaluate(model, items, device, pad, tag=f"title-only-{name}")
            ablation[name] = {**report_for(name, logits, labels, report["calibration_predictions"][name],
                                           args.bootstrap_samples, latency),
                              "mean_tokens": round(sum(len(i["ids"]) for i in items) / len(items), 1),
                              "checkpoint": str(base_dir if name == "base" else (args.model or args.output_dir))}
        if ablation["base"]["confusion_matrix"] == ablation["finetuned"]["confusion_matrix"]:
            raise RuntimeError("the two checkpoints produced identical ablation predictions; "
                               "one of the loads did not use the intended checkpoint")
        report["ablation_title_only"] = {
            "description": "The same held-out papers with the abstract removed, title only.",
            "base": ablation["base"], "finetuned": ablation["finetuned"]}
    report["status"] = "completed"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EXTRAS DONE", flush=True)


def environment():
    import importlib.metadata
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "laya", "transformers", "safetensors", "pyarrow", "huggingface_hub")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["data", "base", "train", "eval", "all", "extras"])
    parser.add_argument("--data-dir", default="data/router")
    parser.add_argument("--model", default=None, help="checkpoint directory; default is the pinned base model")
    parser.add_argument("--output-dir", default="outputs/laya-arxiv-router")
    parser.add_argument("--report", default="reports/router_results.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=5400)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--latency-sample", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extras-part", choices=["calibration", "ablation", "both"], default="both")
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.stage == "data":
        import arxiv_router_data
        print(json.dumps(arxiv_router_data.build(args.data_dir), indent=2)[:2000])
        return

    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(int(os.environ.get("ROUTER_THREADS", "0")) or torch.get_num_threads())
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device(args.device)
    provenance = json.loads((Path(args.data_dir) / "provenance.json").read_text())
    labels = provenance["labels"]
    rows = {split: [json.loads(line) for line in
                                       (Path(args.data_dir) / f"{split}.jsonl").read_text().splitlines() if line.strip()]
                              for split in ("train", "calibration", "test")}
    for split, values in rows.items():
        for row in values:
            from router_task import assert_no_cue
            assert_no_cue(row, labels)
    if args.test_limit:
        rows["test"] = rows["test"][:args.test_limit]
    print(json.dumps({"labels": labels, "split_sizes": {k: len(v) for k, v in rows.items()}}), flush=True)

    model_dir = args.model or base_model_dir()
    if args.stage == "extras":
        # The extras stage compares the two checkpoints, so the first load must be
        # the base model even when --model points at a fine-tuned directory.
        model_dir = base_model_dir()
    model, config = load_model(model_dir, device)
    args.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir, "tokenizer"))
    pad = args.tokenizer.pad_token_id

    prepared = {split: load_items(args.tokenizer, values, labels) for split, values in rows.items()}
    print("mean tokens", {s: round(sum(len(i["ids"]) for i in v) / len(v), 1) for s, v in prepared.items()}, flush=True)

    report = {"status": "started", "model_revision": MODEL_REVISION, "model_dir": str(model_dir),
              "device": str(device), "seed": SEED, "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN,
              "dataset": {"name": provenance["source"]["dataset"], "revision": provenance["source"]["revision"],
                          "columns_read": provenance["source"]["columns_read"],
                          "licence": provenance["licence"], "split_sizes": provenance["split_sizes"],
                          "date_ranges": provenance["date_ranges"], "labels": labels},
              "environment": environment()}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "extras":
        # Reuse the report written by the training run instead of rebuilding it.
        if Path(args.report).exists():
            report.update(json.loads(Path(args.report).read_text()))
        extras(args, rows, labels, model_dir, model, device, pad, report, output)
        return

    base_rows, base_latency = evaluate(model, prepared["test"], device, pad, tag="base-test")
    base_cal, _ = evaluate(model, prepared["calibration"], device, pad, tag="base-cal")
    base_one = single_example_latency(model, prepared["test"], device, pad, args.latency_sample)
    report["base"] = report_for("base", base_rows, labels, base_cal, args.bootstrap_samples, base_latency, base_one)
    print("BASE", json.dumps({k: report["base"][k] for k in ("accuracy", "macro_f1", "temperature", "ece")}), flush=True)
    if args.stage in ("base",):
        report["status"] = "base_only"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return

    report["training"] = train(args, model, config, rows, labels, device, pad, args.output_dir)
    model, config = load_model(args.output_dir, device)
    tuned_rows, tuned_latency = evaluate(model, prepared["test"], device, pad, tag="tuned-test")
    tuned_cal, _ = evaluate(model, prepared["calibration"], device, pad, tag="tuned-cal")
    tuned_one = single_example_latency(model, prepared["test"], device, pad, args.latency_sample)
    report["finetuned"] = report_for("finetuned", tuned_rows, labels, tuned_cal, args.bootstrap_samples,
                                     tuned_latency, tuned_one)
    report["paired"] = paired(report["base"], report["finetuned"], labels)
    report["predictions"] = {"base": base_rows, "finetuned": tuned_rows}
    report["calibration_predictions"] = {"base": base_cal, "finetuned": tuned_cal}
    report["status"] = "completed"
    report["limitations"] = [
        "Labels are arXiv author-declared primary categories, not expert annotations.",
        "The window is 2007-2009 arXiv metadata; the router is not a general 2026 topic router.",
        "CPU-only training stopped on a wall-clock budget inside epoch 2, so the schedule was not completed; see the training table.",
        "Eight subject areas only, selected a priori for per-split availability.",
        "Latency is Mac CPU batch-1 and batch-8; GPU and serving numbers are not measured.",
        "No paper full text was used, and no model artefact is a safety or authorisation claim.",
    ]
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("COMPLETED", json.dumps({"base": report["base"]["accuracy"], "finetuned": report["finetuned"]["accuracy"],
                                   "base_macro_f1": report["base"]["macro_f1"],
                                   "finetuned_macro_f1": report["finetuned"]["macro_f1"],
                                   "mcnemar_p": report["paired"]["p_value"]}), flush=True)


if __name__ == "__main__":
    main()
