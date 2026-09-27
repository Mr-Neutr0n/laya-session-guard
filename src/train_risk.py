"""Risk-weighted and pair-ranked Laya fine-tune on a Kaggle GPU."""
import os
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZER_PARALLELISM="false",
                  TOKENIZERS_PARALLELISM="false")
# Must be set before CUDA initialises or cuBLAS picks its workspace config, otherwise
# the cuBLAS reductions this run depends on are not reproducible. See the determinism
# block in main() for what is and is not guaranteed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import importlib.metadata
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from aspi_data import generate as aspi_generate
from context import checked_sequence
from counterfactual_data import generate as counterfactual_generate
from eval_rows import rows_from_predictions, rows_path, write_rows
from session_data import QUESTIONS, generate as original_generate
from train import REVISION, collate, evaluate, fit_temperatures, metrics


OUT = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/laya-risk-ranked"))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "3"))
RISK_WEIGHT = float(os.environ.get("RISK_WEIGHT", "2.0"))
PAIR_WEIGHT = float(os.environ.get("PAIR_WEIGHT", "1.0"))
MARGIN = float(os.environ.get("RISK_MARGIN", "1.0"))


def prepare(tok, cfg, rows):
    output = []
    for row in rows:
        for qid, question in QUESTIONS.items():
            ids, markers = checked_sequence(tok, row["state"], question,
                                            cfg["max_len"], cfg["head_max_len"])
            output.append({"ids": ids, "markers": markers, "id": row["id"],
                           "qid": qid, "family": row["family"],
                           "label": list(question["criteria"]).index(row["labels"][qid]),
                           "pair_id": row.get("pair_id"), "action_label": row["labels"]["action"]})
    return output


def make_units(rows):
    pairs = defaultdict(list)
    singles = []
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
        else:
            singles.append(row)
    units = []
    for pair_id, members in pairs.items():
        assert len(members) == 2, pair_id
        assert {m["labels"]["action"] for m in members} == {"allow", "block"}, pair_id
        units.append(members)
    units.extend(singles[i:i+2] for i in range(0, len(singles), 2))
    return units


def risk_loss(logits, entries):
    labels = torch.tensor([r["label"] for r in entries], device=logits.device)
    per_row = F.cross_entropy(logits.float(), labels, reduction="none")
    weights = torch.tensor([0.5 if r["qid"] == "content" else 1.0 for r in entries],
                           device=logits.device)
    base = (per_row * weights).sum() / weights.sum()
    risk_terms = []
    pairs = defaultdict(dict)
    for i, row in enumerate(entries):
        if row["qid"] != "action":
            continue
        if row["label"] != 0:
            risk_terms.append(F.softplus(logits[i, 0].float() - logits[i, row["label"]].float() + MARGIN))
        if row["pair_id"]:
            pairs[row["pair_id"]][row["action_label"]] = i
    rank_terms = []
    for pair in pairs.values():
        if set(pair) != {"allow", "block"}:
            continue
        safe, unsafe = logits[pair["allow"]].float(), logits[pair["block"]].float()
        safe_gap = safe[0] - torch.logsumexp(safe[1:3], dim=0)
        unsafe_gap = unsafe[0] - torch.logsumexp(unsafe[1:3], dim=0)
        rank_terms.append(F.softplus(MARGIN - (safe_gap - unsafe_gap)))
    risk = torch.stack(risk_terms).mean() if risk_terms else base.new_zeros(())
    rank = torch.stack(rank_terms).mean() if rank_terms else base.new_zeros(())
    return base + RISK_WEIGHT * risk + PAIR_WEIGHT * rank, (base.item(), risk.item(), rank.item())


def action_summary(predictions):
    action = [p for p in predictions if p["qid"] == "action"]
    verdicts = []
    for row in action:
        z = torch.tensor(row["logits"][:3]).float()
        verdicts.append((row["label"], int(z.argmax())))
    n = len(verdicts)
    return {"n": n,
            "accuracy": sum(y == p for y, p in verdicts) / n,
            "false_allow": sum(y != 0 and p == 0 for y, p in verdicts),
            "nonallow": sum(y != 0 for y, _ in verdicts),
            "correct_allow": sum(y == p == 0 for y, p in verdicts),
            "allow_cases": sum(y == 0 for y, _ in verdicts),
            "block_recall": sum(y == p == 1 for y, p in verdicts) /
                            max(1, sum(y == 1 for y, _ in verdicts))}


def choose_key(summary):
    return (summary["false_allow"], -summary["correct_allow"], -summary["accuracy"])


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    # Reproducibility. Two runs of this script with the same seed and the same data
    # produced different per-case decisions, and different calibration accuracy at
    # epoch 1, because the seeds alone do not pin GPU kernel selection. The flags below
    # are what make a rerun comparable.
    #
    # What is now guaranteed: the same seed reproduces the same weights on the same
    # GPU, driver, and library stack, and a rerun is comparable run over run.
    #
    # What is NOT guaranteed, and must be read as a caveat rather than a guarantee:
    #   - bit-identical results across different GPU architectures, driver versions, or
    #     torch builds, since kernels differ;
    #   - that warn_only=True has no warnings, which is why it is logged below. Ops with
    #     no deterministic implementation fall back to a nondeterministic kernel and
    #     announce themselves;
    #   - fp16 autocast itself. Deterministic algorithms make each op's *algorithm*
    #     deterministic, but reduced-precision accumulation order can still depend on the
    #     kernel, so fp16 remains the weakest link in the chain.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark picks kernels by timing, so it is
                                             # nondeterministic by construction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Kaggle GPU required"
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {"original": original_generate(), "counterfactual": counterfactual_generate(),
               "aspi": aspi_generate()}
    splits = {
        "train": sources["original"]["train"] + sources["counterfactual"]["train"] + sources["aspi"]["train"],
        "calibration": sources["original"]["calibration"] + sources["counterfactual"]["calibration"] + sources["aspi"]["calibration"],
        "test_user": sources["aspi"]["test_user"],
        "test_attack": sources["aspi"]["test_attack"],
        "test_both": sources["aspi"]["test_both"],
    }
    challenge_path = Path(os.environ.get("CHALLENGE_DATA", Path(__file__).resolve().parent.parent /
                                         "data" / "challenge.jsonl"))
    splits["old_challenge"] = [json.loads(x) for x in challenge_path.read_text().splitlines()]
    for name, rows in splits.items():
        with (OUT / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    model_dir = snapshot_download("convaiinnovations/laya", revision=REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"])
    _fix_tokenizer_config(model_dir)
    cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
    cfg.update(max_len=1024, head_max_len=192, amp_dtype="fp16", temperature=[1., 1., 1.],
               temperature_by_options={})
    tok = AutoTokenizer.from_pretrained(Path(model_dir, "tokenizer"))
    prepared = {name: prepare(tok, cfg, rows) for name, rows in splits.items()}
    prepared_train_by_id = defaultdict(list)
    for entry in prepared["train"]:
        prepared_train_by_id[entry["id"]].append(entry)
    print("PREPARED", {k: len(v) for k, v in prepared.items()},
          "MAX_TOKENS", max(len(x["ids"]) for v in prepared.values() for x in v), flush=True)
    model = build_model(cfg, encoder_dir=str(Path(model_dir, "encoder")))
    model.encoder.config.reference_compile = False
    model.load_state_dict(load_file(str(Path(model_dir, "model.safetensors"))), strict=True)
    model.to(device)
    baseline = {name: action_summary(evaluate(model, prepared[name], tok, device))
                for name in ("test_user", "test_attack", "test_both", "old_challenge")}
    (OUT / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("BASELINE", baseline, flush=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": 1e-5}, {"params": head, "lr": 5e-5}], weight_decay=.02)
    units = make_units(splits["train"])
    steps = EPOCHS * math.ceil(len(units) / 8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: min((s+1)/max(1, int(steps*.06)), max(0., (steps-s)/max(1, steps-int(steps*.06)))))
    scaler = torch.amp.GradScaler("cuda")
    history = []; best_key = None; best_epoch = None
    started = time.time()
    for epoch in range(EPOCHS):
        model.train(); random.shuffle(units)
        optimizer.zero_grad(set_to_none=True)
        losses = np.zeros(3)
        for step, unit in enumerate(units):
            entries = [entry for row in unit for entry in prepared_train_by_id[row["id"]]]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*collate(entries, tok.pad_token_id, device))
                loss, components = risk_loss(logits, entries)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss / 8).backward()
            losses += components
            if (step+1) % 8 == 0 or step+1 == len(units):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1) % 150 == 0:
                print("PROGRESS", epoch+1, step+1, len(units), round(time.time()-started), flush=True)
        calibration = evaluate(model, prepared["calibration"], tok, device)
        summary = action_summary(calibration)
        key = choose_key(summary)
        history.append({"epoch": epoch+1, "loss_components": (losses/len(units)).tolist(),
                        "calibration_action": summary})
        print("EPOCH", history[-1], flush=True)
        if best_key is None or key < best_key:
            best_key, best_epoch = key, epoch+1
            save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
                      str(OUT / "model.safetensors"))
            tok.save_pretrained(OUT / "tokenizer")
            model.encoder.config.save_pretrained(OUT / "encoder")
            print("BEST", best_epoch, best_key, flush=True)
    model.load_state_dict(load_file(str(OUT / "model.safetensors")), strict=True)
    model.eval()
    calibration = evaluate(model, prepared["calibration"], tok, device)
    temperatures = fit_temperatures(calibration)
    cfg["temperature_by_options"] = {"choice:2": temperatures["content"],
                                     "choice:3-5": temperatures["action"]}
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-session-guard-risk-ranked-aspi-pilot"
    (OUT / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    final = {}
    # Persist per-case rows for every evaluated split, calibration included. Without the
    # calibration rows a wrapper threshold can only be fitted by reading the frozen test
    # splits, which is how the risk-ranked run ended up with a threshold whose apparent
    # gain halved out of sample.
    for name in ("calibration", "test_user", "test_attack", "test_both", "old_challenge"):
        predictions = evaluate(model, prepared[name], tok, device)
        path = write_rows(rows_path(OUT, name),
                          rows_from_predictions(predictions, name, model="risk-ranked"))
        print("ROWS", name, len(predictions) // len(QUESTIONS), path, flush=True)
        # Every evaluated split is summarised. An earlier version of this loop skipped
        # old_challenge here, which silently dropped the 24-case diagnostic from
        # report["finetuned"] and stopped writing predictions-old_challenge.json.
        final[name] = {"action": action_summary(predictions),
                       "metrics": metrics(predictions, temperatures)}
        (OUT / f"predictions-{name}.json").write_text(json.dumps(predictions))
    report = {"status": "completed", "gpu": torch.cuda.get_device_name(0),
              "training_seconds": time.time()-started, "base_revision": REVISION,
              "epochs_requested": EPOCHS, "best_epoch": best_epoch,
              "objective": {"risk_weight": RISK_WEIGHT, "pair_weight": PAIR_WEIGHT, "margin": MARGIN},
              "split_sizes": {k: len(v) for k, v in splits.items()},
              "history": history, "baseline": baseline, "finetuned": final,
              "temperatures": temperatures,
              "determinism": {
                  "seed": 42,
                  "use_deterministic_algorithms": True,
                  "deterministic_algorithms_warn_only": True,
                  "cudnn_deterministic": True,
                  "cudnn_benchmark": False,
                  "tf32": False,
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "amp_dtype": "float16",
                  "caveat": ("Bit-identical only on the same GPU, driver, and library stack. "
                             "warn_only means any op without a deterministic implementation "
                             "falls back silently apart from a warning, and fp16 accumulation "
                             "order can still depend on the kernel."),
              },
              "per_case_rows": {name: str(rows_path(OUT, name))
                                for name in ("calibration", "test_user", "test_attack",
                                             "test_both", "old_challenge")},
              "limitations": ["ASPI adaptation treats clarification as an unverified lower-trust channel",
                              "ASPI labels are benchmark-generated, not human-reviewed session judgments",
                              "Only workspace tasks; not a coding-session production benchmark",
                              "One seed; no automatic tool authorization"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "versions.json").write_text(json.dumps({k: importlib.metadata.version(k) for k in
        ["torch", "laya", "transformers", "safetensors", "huggingface_hub"]}, indent=2))
    print("COMPLETED", json.dumps({"best_epoch": best_epoch, "baseline": baseline,
                                  "finetuned_action": {k: v["action"] for k, v in final.items()}}), flush=True)


if __name__ == "__main__":
    main()
