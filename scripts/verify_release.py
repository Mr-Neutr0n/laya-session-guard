"""Verify the published release end to end, through `laya.load`.

Checks, in order:

1. the pinned Hub revision downloads and the config inside it is the one this
   repository would write now: a real training record, no upstream base
   checkpoint record, and the calibrated temperature;
2. `laya.load` actually applies that temperature, by comparing the probability
   the agent returns against softmax(logits / T) and against softmax(logits);
3. the model card's prompt reproduces the training prompt's behaviour, and the
   bare-label prompt that an earlier card used does not.

Writes a JSON verdict to stdout. Nothing is uploaded and nothing is trained.
"""
import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_REVISION = "1c2bb313b6a94b674c2e2c57483760eb5ccab244"
ROWS_PER_CLASS = 15  # 120 held-out rows in total


def held_out_rows(data_dir, labels, per_class=ROWS_PER_CLASS):
    """Deterministic first N rows per class from the held-out split, in file order."""
    rows = [json.loads(line) for line in (Path(data_dir) / "test.jsonl").read_text().splitlines() if line.strip()]
    taken, chosen = {label: 0 for label in labels}, []
    for row in rows:
        if taken[row["label"]] < per_class:
            taken[row["label"]] += 1
            chosen.append(row)
    return chosen


def truncate(text, words=100):
    return " ".join(" ".join(text.split()).split(" ")[:words])


CARD_SIZE_RE = re.compile(r"`model\.safetensors`, float16, ([\d,]+) bytes")
CARD_SHA_RE = re.compile(r"sha256\s+([0-9a-f]{64})")


def parse_card_checksum(card_text):
    """The size and hash the card claims for its own weights."""
    size = CARD_SIZE_RE.search(card_text)
    sha = CARD_SHA_RE.search(card_text)
    return {"size_bytes": int(size.group(1).replace(",", "")) if size else None,
            "sha256": sha.group(1) if sha else None}


def hub_lfs_metadata(model, revision):
    """Size and LFS oid for the weights, read back from the Hub, not from memory."""
    import json as _json
    import urllib.request
    url = f"https://huggingface.co/api/models/{model}/tree/{revision}?recursive=true"
    request = urllib.request.Request(url, headers={"User-Agent": "laya-router-release-check"})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = _json.loads(response.read())
    for entry in entries:
        if entry.get("path", "").endswith("model.safetensors"):
            lfs = entry.get("lfs") or {}
            return {"size_bytes": entry.get("size"), "lfs_size": lfs.get("size"),
                    "lfs_oid": lfs.get("oid"), "path": entry.get("path")}
    return {}


def check_card_checksum(model, revision, local_path):
    """The card must state the size and hash the Hub actually holds."""
    card_text = (Path(local_path) / "README.md").read_text()
    claimed = parse_card_checksum(card_text)
    hub = hub_lfs_metadata(model, revision)
    actual_size = hub.get("lfs_size") or hub.get("size_bytes")
    return {
        "claimed_size_bytes": claimed["size_bytes"],
        "hub_size_bytes": actual_size,
        "size_matches": claimed["size_bytes"] == actual_size,
        "claimed_sha256": claimed["sha256"],
        "hub_lfs_oid": hub.get("lfs_oid"),
        "hash_matches": claimed["sha256"] == hub.get("lfs_oid"),
    }


def run_card_snippet(local_path, config):
    """Execute the usage snippet exactly as the published card prints it.

    The card pins a revision inside its own text, so a pin that predates
    `router_prompt.json` makes the snippet fail with FileNotFoundError for anyone who
    copies it. Extracting and running the fenced block is the only way to catch that
    before publication rather than after.
    """
    import re
    import subprocess
    card = (Path(local_path) / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", card, re.S)
    if not blocks:
        return {"found": False, "ok": False, "detail": "no python block in the card"}
    snippet = blocks[0]
    pin = re.search(r'revision="([0-9a-f]{40})"', snippet)
    script = Path(tempfile.mkdtemp()) / "card_snippet.py"
    script.write_text(snippet)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            cwd=script.parent, timeout=1800)
    output = result.stdout.strip().splitlines()
    return {
        "found": True,
        "ok": result.returncode == 0,
        "pinned_revision": pin.group(1) if pin else None,
        "pins_a_revision": bool(pin),
        "returncode": result.returncode,
        "stdout": output[-1] if output else "",
        "stderr_tail": result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hxrikp/laya-arxiv-topic-router")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-dir", default="data/router")
    parser.add_argument("--report", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    import laya
    from router_task import router_question
    from arxiv_router_data import LABELS

    labels = list(LABELS)
    path = snapshot_download(args.model, revision=args.revision)
    config = json.loads((Path(path) / "rl_agent_config.json").read_text())
    prompt_file = Path(path) / "router_prompt.json"

    training = config.get("training", {})
    upstream_keys = {"epochs_completed", "hours", "updates", "world_size", "fine_tuned_from_checkpoint"}
    leaked = sorted(upstream_keys & set(training))
    by_options = config.get("temperature_by_options", {})
    temperature = by_options.get("choice:6-10", (config.get("temperature") or [1.0])[0])

    config_checks = {
        "revision": args.revision,
        "upstream_base_record_keys_present": leaked,
        "training_steps_completed": training.get("steps_completed"),
        "training_wall_clock_budget_seconds": training.get("wall_clock_budget_seconds"),
        "training_schedule_completed": training.get("schedule_completed"),
        "config_temperature_by_options": by_options,
        "config_temperature_choice_fallback": (config.get("temperature") or [None])[0],
        "temperature_that_laya_will_apply": temperature,
        "prompt_artifact_published": prompt_file.exists(),
    }

    agent = laya.load(str(path), device="cpu")
    applied = {"choice:6-10": agent.temperature_by_options.get("choice:6-10"),
               "temperature_choice": agent.temperature[0]}

    rows = held_out_rows(args.data_dir, labels)
    if args.limit:
        rows = rows[:args.limit]
    question = router_question(labels)
    external = {"q": {"type": "choice", "instructions": question["ins"],
                      "criteria": {label: label for label in labels}}}

    trained_question = {"q": {"type": "choice", "instructions": question["ins"],
                              "criteria": dict(question["crit"])}}

    def score(questions, check_temperature=False):
        hits, first = 0, None
        for row in rows:
            state = {"title": row["title"], "abstract": truncate(row["abstract"])}
            answers = agent.predict(state, questions)["answers"]
            probabilities = answers["q"]["probabilities"]
            choice = answers["q"]["choice"]
            if first is None:
                first = (choice, dict(probabilities), answers["q"].get("probabilities"))
            hits += int(choice == row["label"])
        return hits / len(rows), first

    trained_accuracy, sample = score(trained_question)
    naive_accuracy, _ = score(external)

    logits_check = {}
    if sample:
        _, probs, _ = sample
        # The agent divides by the temperature internally, so the returned vector
        # is softmax(z / T). Confirm it is not softmax(z) by re-running one row.
        state = {"title": rows[0]["title"], "abstract": truncate(rows[0]["abstract"])}
        from laya.common import build_sequence
        from transformers import AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(Path(path, "tokenizer"))
        ids, markers = build_sequence(tokenizer, state,
                                      {"t": "choice", "ins": question["ins"], "crit": question["crit"]},
                                      640, 256)
        pad = tokenizer.pad_token_id
        batch = (torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long),
                 torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                 torch.zeros(1, dtype=torch.long))
        with torch.no_grad():
            logits, _ = agent.model(*batch)
        z = [float(v) for v in logits[0][:len(markers)]]

        def softmax(values, t):
            scaled = [v / t for v in values]
            top = max(scaled)
            exps = [math.exp(v - top) for v in scaled]
            return [e / sum(exps) for e in exps]

        got = list(probs.values())
        for name, t in (("applied_temperature", temperature), ("unit_temperature", 1.0)):
            expected = softmax(z, t)
            logits_check[name] = {"max_abs_difference": max(abs(a - b) for a, b in zip(expected, got)),
                                  "expected_top": labels[max(range(len(expected)), key=lambda i: expected[i])]}
        # Recover the temperature the agent actually used, by fitting it to its own
        # output. A residual of a few times float noise means the recovered value is
        # the config value; a residual of 4e-3 would mean T=1.0 was applied instead.
        best, best_error = None, float("inf")
        for step in range(600, 1200):
            candidate = step / 1000.0
            error = sum((a - b) ** 2 for a, b in zip(softmax(z, candidate), got))
            if error < best_error:
                best, best_error = candidate, error
        logits_check["recovered_temperature"] = best
        logits_check["residual_at_recovered_temperature"] = best_error ** 0.5
        logits_check["residual_at_unit_temperature"] = sum(
            (a - b) ** 2 for a, b in zip(softmax(z, 1.0), got)) ** 0.5
        logits_check["config_temperature"] = temperature
        logits_check["agent_probabilities"] = {k: round(v, 6) for k, v in probs.items()}

    verdict = {
        "model": args.model,
        "config": config_checks,
        "applied_temperature": applied,
        "temperature_evidence": logits_check,
        "held_out_rows": len(rows),
        "rows_per_class": ROWS_PER_CLASS,
        "trained_prompt_accuracy": trained_accuracy,
        "bare_label_prompt_accuracy": naive_accuracy,
        "chance": 1.0 / len(labels),
    }
    verdict["card_snippet"] = run_card_snippet(path, config)
    verdict["card_checksum"] = check_card_checksum(args.model, args.revision, path)
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
