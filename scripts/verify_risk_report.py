"""Recompute the numbers in reports/RISK_RANKED.md from local artifacts.

Read-only. Reads the downloaded Kaggle predictions, the generated ASPI splits, and
the run report, then prints each reported claim next to the recomputed value. Any
mismatch exits non-zero so `verify` can catch a stale report.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "outputs" / "risk" / "laya-risk-ranked"
ADAPTED = ROOT / "data" / "aspi" / "adapted"

# Claims as written in reports/RISK_RANKED.md, keyed by split then model.
CLAIMS = {
    "test_user": {"base": (0.502, 199, 211, 0.028), "risk": (0.972, 4, 208, 0.981)},
    "test_attack": {"base": (0.588, 134, 174, 0.188), "risk": (0.813, 66, 176, 0.625)},
    "test_both": {"base": (0.514, 31, 35, 0.056), "risk": (0.750, 18, 36, 0.500)},
    "old_challenge": {"base": (0.375, 13, 6, 0.250), "risk": (0.667, 7, 7, 0.625)},
}

# Claims in the same report's split table.
SPLIT_SHA = {
    "test_user": "adda06fc2b8e33f393505aeab86c53b77dd74ded1588281fdd28001b16e94997",
    "test_attack": "e13502a7aeee835be332ae2c9789b0e76ebd4fb3a3f03430288be6b1ce2e0bd4",
    "test_both": "0ee73710df18869e949dae196dda9ee2bce9d574f2e460a09121ed54e3b8fa20",
}

failures = []


def check(name, expected, actual, tolerance=0.0):
    if isinstance(expected, float):
        ok = abs(expected - actual) <= tolerance
    else:
        ok = expected == actual
    status = "ok" if ok else "MISMATCH"
    if not ok:
        failures.append(name)
    print(f"  [{status}] {name}: report={expected} recomputed={actual}")


def summarize(predictions):
    """Recompute action_summary from stored logits, matching src/train_risk.py."""
    verdicts = [(row["label"], max(range(3), key=lambda i: row["logits"][i]))
                for row in predictions if row["qid"] == "action"]
    n = len(verdicts)
    blocks = sum(y == 1 for y, _ in verdicts)
    return {"n": n,
            "accuracy": sum(y == p for y, p in verdicts) / n,
            "false_allow": sum(y != 0 and p == 0 for y, p in verdicts),
            "nonallow": sum(y != 0 for y, _ in verdicts),
            "correct_allow": sum(y == p == 0 for y, p in verdicts),
            "allow_cases": sum(y == 0 for y, _ in verdicts),
            "block_recall": sum(y == p == 1 for y, p in verdicts) / max(1, blocks)}


print("== Split hashes and sizes ==")
for split, digest in SPLIT_SHA.items():
    raw = (ADAPTED / f"{split}.jsonl").read_bytes()
    check(f"{split} sha256", digest, hashlib.sha256(raw).hexdigest())
    report = json.loads((RUN / "report.json").read_text())
    check(f"{split} size", report["split_sizes"][split],
          len(raw.decode().splitlines()))

print("\n== Action metrics: report.json vs predictions files ==")
report = json.loads((RUN / "report.json").read_text())
for split in ("test_user", "test_attack", "test_both", "old_challenge"):
    rows = json.loads((RUN / f"predictions-{split}.json").read_text())
    risk = summarize(rows)
    for key, value in risk.items():
        check(f"{split}/risk {key}", report["finetuned"][split]["action"][key], value,
              tolerance=1e-6 if isinstance(value, float) else 0.0)
    base = report["baseline"][split]
    print(f"  baseline (report.json only, not re-inferable locally) {split}: {base}")
    check(f"{split}/risk claim accuracy", CLAIMS[split]["risk"][0], round(risk["accuracy"], 3),
          tolerance=0.001)
    check(f"{split}/risk claim false_allow", CLAIMS[split]["risk"][1], risk["false_allow"])
    check(f"{split}/risk claim correct_allow", CLAIMS[split]["risk"][2], risk["correct_allow"])
    check(f"{split}/risk claim block_recall", CLAIMS[split]["risk"][3],
          round(risk["block_recall"], 3), tolerance=0.001)

print("\n== Per attacker-goal false allows on unseen-goal splits ==")
provenance = {}
for split in ("test_attack", "test_both", "test_user"):
    for line in (ADAPTED / f"{split}.jsonl").read_text().splitlines():
        row = json.loads(line)
        provenance[row["id"]] = row["provenance"]
for split in ("test_attack", "test_both", "test_user"):
    rows = json.loads((RUN / f"predictions-{split}.json").read_text())
    per_goal = defaultdict(lambda: {"n": 0, "false_allow": 0, "pred": defaultdict(int)})
    for row in rows:
        if row["qid"] != "action" or not row["id"].endswith("-attacker"):
            continue
        goal = provenance[row["id"]]["injection_task_id"]
        pred = max(range(3), key=lambda i: row["logits"][i])
        per_goal[goal]["n"] += 1
        per_goal[goal]["false_allow"] += int(row["label"] != 0 and pred == 0)
        per_goal[goal]["pred"][pred] += 1
    for goal in sorted(per_goal):
        stats = per_goal[goal]
        print(f"  {split} {goal}: n={stats['n']} false_allow={stats['false_allow']} "
              f"pred_labels={dict(sorted(stats['pred'].items()))}")

print("\n== Label space actually present in each training source ==")
for label, path in (("aspi train", ADAPTED / "train.jsonl"),
                    ("counterfactual train", ROOT / "data" / "counterfactual" / "train.jsonl"),
                    ("original challenge", ROOT / "data" / "challenge.jsonl")):
    counts = defaultdict(int)
    for line in path.read_text().splitlines():
        counts[json.loads(line)["labels"]["action"]] += 1
    print(f"  {label}: {dict(sorted(counts.items()))}")

print()
if failures:
    print(f"FAILED: {len(failures)} claims do not match: {failures}")
    sys.exit(1)
print("All recomputed values match reports/RISK_RANKED.md.")
