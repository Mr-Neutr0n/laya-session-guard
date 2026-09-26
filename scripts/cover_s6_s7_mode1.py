"""Measure how much of the Mode 1 false-allow shape the synthetic stopgap covers.

Read-only. Compares the action shapes of the real benign-source false allows in
the rejected run against the synthetic S6 cases, feature by feature, so coverage
is reported as a measurement rather than asserted.

The 23 Mode 1 failures share a single action description across 23 distinct user
tasks, so shape coverage and task coverage are reported separately. A generator
that emits one shape many times scores well on the first and badly on the second,
and collapsing the two would overstate what the stopgap buys.

Usage:
    python3 scripts/cover_s6_s7_mode1.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "outputs" / "risk" / "laya-risk-ranked"
ADAPTED = ROOT / "data" / "aspi" / "adapted"
STOPGAP = ROOT / "data" / "s6_s7_synthetic" / "cases.jsonl"

# The features a Mode 1 failure is made of. Each is a property of the action
# text or the surrounding transcript, not of any single word.
FEATURES = {
    "outward_direction": r"\b(email|e-mail|message|post|publish|invite|forward|push|text|send)\b",
    "recipient_expansion": r"\b(every|all|everyone|everybody|whole|entire|each)\b",
    "per_item_loop": r"\b(one per|per person|per recipient|each (one|recipient|person)|separate|one recipient)\b",
    "scripted_body": r"\b(same (text|wording|message)|exactly|as written|following language|verbatim|one message per)\b",
    "enumeration_source": r"\b(calendar|meeting|attendee|contact|list|subscriber|roster|everyone I am meeting)\b",
}

# Wording a defender could key on. The Mode 1 failures contain none of it, so an
# S6 case containing any of it is not the shape being targeted.
INJECTION_MARKERS = re.compile(
    r"\b(ignore the (earlier|previous|above)|disregard|override|maintainer|approved it|"
    r"skip asking|urgent|immediately|system notice|new instruction|do not mention|"
    r"treat this (page|note|message) as|authority)\b", re.I)


def predict(row):
    return max(range(len(row["logits"])), key=lambda i: row["logits"][i])


def features_of(text):
    return {name: bool(re.search(pattern, text, re.I)) for name, pattern in FEATURES.items()}


def mode1_cases():
    """The real benign-source false allows, 23 of them."""
    cases, out = {}, []
    for split in ("test_attack", "test_both"):
        path = ADAPTED / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            cases[row["id"]] = row
    for split in ("test_attack", "test_both"):
        predictions = PREDICTIONS / f"predictions-{split}.json"
        if not predictions.exists():
            continue
        for row in json.loads(predictions.read_text()):
            if row["qid"] != "action" or row["label"] == 0 or predict(row) != 0:
                continue
            case = cases.get(row["id"])
            if case and case["provenance"]["operator"] == "none":
                out.append(case)
    return out


def stopgap_cases():
    if not STOPGAP.exists():
        return []
    return [json.loads(line) for line in STOPGAP.read_text().splitlines() if line.strip()]


def main():
    real = mode1_cases()
    synthetic = stopgap_cases()
    s6 = [c for c in synthetic if c["stratum"] == "S6"]
    s7 = [c for c in synthetic if c["stratum"] == "S7"]
    if not real or not s6:
        print("missing real or synthetic artifacts; nothing to compare")
        return 1

    print(f"Mode 1 false allows in the rejected run : {len(real)}")
    print(f"Synthetic S6 stopgap cases              : {len(s6)}")
    print(f"Synthetic S7 control cases              : {len(s7)}")

    real_features = [features_of(c["state"]["proposed_action"]["description"]) for c in real]
    synth_features = [features_of(c["state"]["proposed_action"]["description"]) for c in s6]
    real_needed = set().union(*[set(f) for f in real_features if any(f.values())])
    print("\n== Feature coverage: does any S6 case exhibit what the failures need? ==")
    print(f"  {'feature':<22}{'required by':>12}{'present in S6':>16}{'covered':>10}")
    covered = 0
    for name in sorted(FEATURES):
        needed = sum(1 for f in real_features if f[name])
        present = sum(1 for f in synth_features if f[name])
        ok = present > 0 and needed > 0
        covered += int(ok)
        print(f"  {name:<22}{needed:>7}/{len(real):<4}{present:>10}/{len(s6):<5}"
              f"{'yes' if ok else 'NO':>10}")
    print(f"\n  features covered: {covered}/{len(FEATURES)}")

    print("\n== S6 wording: no injection markers, as required ==")
    dirty = [c["case_id"] for c in s6
             if INJECTION_MARKERS.search(c["state"]["proposed_action"]["description"])
             or any(INJECTION_MARKERS.search(e["content"]) for e in c["state"]["events"])]
    print(f"  S6 cases containing injection phrasing: {len(dirty)}  {dirty[:5]}")

    print("\n== Matched-pair integrity ==")
    by_pair = defaultdict(dict)
    for case in synthetic:
        if case["group"]["pair_group_id"]:
            by_pair[case["group"]["pair_group_id"]][case["stratum"]] = case
    bad_action = bad_split = bad_label = 0
    for halves in by_pair.values():
        if "S6" not in halves or "S7" not in halves:
            continue
        a, b = halves["S6"], halves["S7"]
        if a["state"]["proposed_action"]["description"] != b["state"]["proposed_action"]["description"]:
            bad_action += 1
        if a["split"] != b["split"]:
            bad_split += 1
        if (a["labels"]["action"], b["labels"]["action"]) != ("block", "allow"):
            bad_label += 1
    print(f"  complete pairs                 : {sum(1 for h in by_pair.values() if len(h) == 2)}")
    print(f"  pairs whose action text differs: {bad_action}")
    print(f"  pairs straddling two splits    : {bad_split}")
    print(f"  pairs with the wrong label pair: {bad_label}")

    print("\n== Coverage of the user-task side, reported separately ==")
    real_tasks = {c["provenance"]["user_task_id"] for c in real}
    real_task_text = {c["state"]["authorized_task"] for c in real}
    synth_tasks = {c["state"]["authorized_task"] for c in s6}
    print(f"  distinct user tasks behind the failures : {len(real_tasks)}")
    print(f"  distinct narrow task texts in S6        : {len(synth_tasks)}")
    overlap = real_task_text & synth_tasks
    print(f"  exact task-text overlap                 : {len(overlap)}")
    print("  The failures carry 23 different narrow tasks and the stopgap carries a handful of")
    print("  templates, so task-level coverage is thin. Shape coverage is the honest claim here,")
    print("  not task coverage, and not coverage of ordinary unauthorized actions in general.")

    print("\n== Recipient and verb diversity in S6, so neither can stand in as a cue ==")
    verbs = defaultdict(int)
    dests = defaultdict(int)
    for case in s6:
        verbs[case["state"]["proposed_action"]["kind"]] += 1
        dests[re.sub(r"^the \w+ ", "", case["state"]["proposed_action"]["destination"])] += 1
    print(f"  action kinds      : {dict(sorted(verbs.items()))}")
    print(f"  recipient classes : {len(dests)} distinct, {dict(sorted(dests.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
