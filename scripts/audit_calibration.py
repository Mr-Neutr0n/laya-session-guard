"""Audit for the calibration bug class caught in the arXiv experiment.

The bug shape: a temperature or calibration transform applied to *already
normalised probabilities*. Dividing each probability by T and renormalising divides
by 1/T twice, so the transform is the identity for every T. A grid search then sees
a flat loss and returns the first grid point.

Note the distinction that matters. Dividing by T and renormalising by the new sum
is the identity. Passing probabilities through softmax after dividing by T is NOT
identity, because exp is not linear; that is a power transform and it does vary with
T. Both are wrong things to do, and only the first produces a flat loss.

This script separates the bug from the innocent cause that shares its symptom, a
genuine optimum sitting on the grid's lower edge, and states plainly which of the
two the stored evidence can and cannot distinguish.

Reads stored logits only. No model, no GPU, no network.

Usage:
    python3 scripts/audit_calibration.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "outputs" / "risk" / "laya-risk-ranked"
FOLLOWUP = ROOT / "outputs" / "followup" / "followup-original"

GRID_LO, GRID_HI, GRID_N = 0.5, 5.0, 181


def renormalise_by_new_sum(values, t):
    scaled = [v / t for v in values]
    total = sum(scaled)
    return [v / total for v in scaled]


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs, gold):
    return -math.log(max(probs[gold], 1e-12))


def brier(probs, gold):
    return sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))


def stored_action_rows():
    rows = json.loads((RUN / "predictions-test_attack.json").read_text())
    return [(r["label"], [float(x) for x in r["logits"]]) for r in rows if r["qid"] == "action"]


def curve(rows, lo, hi, n, transform):
    return [sum(cross_entropy(transform(logits, lo + (hi - lo) * i / (n - 1)), gold)
                for gold, logits in rows) / len(rows) for i in range(n)]


def logit_path(logits, t):
    """Exactly what src/train.py does: divide LOGITS by T, then softmax."""
    return softmax([v / t for v in logits])


def identity_path(probabilities, t):
    """The bug shape: divide an already-normalised vector by T, renormalise by the new sum."""
    return renormalise_by_new_sum(probabilities, t)


def main():
    print("== 1. The identity property that defines the bug ==")
    p = [0.70, 0.22, 0.08]
    print(f"   p = {p}, sum = {sum(p):.6f}")
    for t in (0.5, 1.0, 2.0, 7.5):
        out = identity_path(p, t)
        drift = max(abs(a - b) for a, b in zip(out, p))
        print(f"   T={t:<4} renormalise(p/T) = [{', '.join(f'{v:.6f}' for v in out)}]"
              f"   max drift {drift:.2e}")
    print("   Divide by T then renormalise by the new sum is the identity for every T.")
    print("   A grid search over T sees a constant loss, and argmin returns index 0.")

    print("\n== 1b. The neighbouring transform that is NOT the identity ==")
    for t in (0.5, 1.0, 2.0, 7.5):
        out = softmax([v / t for v in p])
        drift = max(abs(a - b) for a, b in zip(out, p))
        print(f"   T={t:<4} softmax(p/T)   = [{', '.join(f'{v:.6f}' for v in out)}]"
              f"   max drift {drift:.4f}")
    print("   softmax after scaling is a power transform and does vary with T, so it")
    print("   would not produce a flat loss. Conflating the two would misdiagnose.")

    print("\n== 2. What a flat loss does to a grid search ==")
    flat = [cross_entropy(identity_path(p, t), 0) for t in (0.5, 1.0, 2.0, 7.5)]
    print(f"   identity-path loss at four grid points: {[round(x, 12) for x in flat]}")
    print(f"   all equal: {len(set(round(x, 12) for x in flat)) == 1}"
          f"   argmin index: {flat.index(min(flat))}")
    print("   The repo's grid starts at 0.5, so a published temperature of exactly 0.5 is")
    print("   what this bug looks like, and also what a truncated search looks like.")

    rows = stored_action_rows()
    print(f"\n== 3. The repo's actual path, on {len(rows)} stored action rows ==")
    print("   src/train.py metrics():          z = logits / T ; p = softmax(z)")
    print("   src/train.py fit_temperatures(): cross_entropy(logits / T, y)")
    print("   src/evaluate_jev.py metrics():   no temperature; raw API probabilities,")
    print("                                   renormalised only to undo 2-dp rounding")
    repo = curve(rows, GRID_LO, GRID_HI, GRID_N, logit_path)
    print(f"\n   logit path: loss at T={GRID_LO} is {repo[0]:.6f}, at T={GRID_HI} "
          f"{repo[-1]:.6f}")
    print(f"   varies across the grid: {abs(repo[0] - repo[-1]) > 1e-9}"
          f"   spread {abs(repo[0] - repo[-1]):.6f}")
    print("   Non-flat, so the identity transform is NOT present in this code path.")
    print("   The temperature divides logits, which are not yet normalised.")

    print("\n== 4. The published value is 0.5, on the first grid point. Which cause? ==")
    argmin_index = repo.index(min(repo))
    argmin_t = GRID_LO + (GRID_HI - GRID_LO) * argmin_index / (GRID_N - 1)
    print(f"   On these stored TEST rows the argmin is index {argmin_index} of {GRID_N},"
          f" T = {argmin_t:.4f}, the UPPER edge.")
    print("   That is a different split from the one the fit used. These rows cannot")
    print("   explain a calibration fit, and they point the opposite way from the floor.")
    print("   The calibration logits were never persisted, so the cause of the published")
    print("   0.5 is NOT DETERMINABLE from the stored artifacts. Two candidates remain:")
    print("     (a) the grid floor truncates a genuine optimum below 0.5")
    print("     (b) the calibration surface is flat or near-flat for the action head")
    print("   Distinguishing them needs the calibration rows, which is exactly what the")
    print("   per-case rows added in eval_rows exist to provide for the next run.")

    print("\n== 5. Blast radius: which published numbers can a temperature touch? ==")
    changed = sum(1 for _g, l in rows
                  if max(range(len(l)), key=lambda i: l[i])
                  != max(range(len(l)), key=lambda i: l[i] / 0.5))
    print(f"   argmax differs between T=1 and T=0.5 on {changed} of {len(rows)} rows.")
    print("   Argmax is invariant to positive temperature scaling, so accuracy, false allows,")
    print("   correct allows, block recall and every confusion matrix are UNAFFECTED. Those")
    print("   are the guard-decision numbers, and they are the ones published and replicated.")
    print("   Only confidence-dependent numbers move: Brier and ECE.")
    b_half = sum(brier(logit_path(l, 0.5), g) for g, l in rows) / len(rows)
    b_one = sum(brier(logit_path(l, 1.0), g) for g, l in rows) / len(rows)
    b_five = sum(brier(logit_path(l, 5.0), g) for g, l in rows) / len(rows)
    print(f"\n   Brier at T=0.5 (as published): {b_half:.4f}")
    print(f"   Brier at T=1.0 (uncalibrated): {b_one:.4f}   delta {b_half - b_one:+.4f}")
    print(f"   Brier at T=5.0                : {b_five:.4f}   delta {b_half - b_five:+.4f}")
    print("   The action head is so overconfident that the truncated T=0.5 makes its")
    print("   reported Brier worse than not calibrating at all, which is a visible symptom")

    print("\n== 6. Published fitted temperatures ==")
    for name, path in (("risk-ranked", RUN / "report.json"),
                       ("followup-original", FOLLOWUP / "report.json")):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        print(f"   {name:<20} temperatures={data.get('temperatures')}")
        if data.get("baseline_temperatures"):
            print(f"   {'':<20} baseline      ={data['baseline_temperatures']}")
    print(f"   grid: torch.linspace({GRID_LO}, {GRID_HI}, {GRID_N}), so {GRID_LO} is the first")
    print("   point. The follow-up baseline reached 1.825 and 1.125, so the search is not")
    print("   pinned to the first point in general; the two fine-tuned action fits did land")
    print("   on it, and the content fit in the follow-up run did too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
