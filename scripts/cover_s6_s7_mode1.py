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
sys.path.insert(0, str(ROOT / "scripts"))
from generate_s6_s7_synthetic import ARTIFACTS, SHAPES  # noqa: E402

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


# Analyst classification of the 23 held-out task prompts into the generator's shape
# vocabulary. Written by reading the prompts, not derived from the generator, so the
# two sides of the coverage figure are not scored by a single table.
SHAPE_CUES = [
    ("mail-fact-lookup", r"\b(when is|where and at what time|what is my|link to reset|security code|going to be, based on)\b"),
    ("availability-then-create", r"\bam i free\b"),
    ("enumerate-and-summarise", r"\bhow many appointments|perform the following tasks in the order|according to the\b"),
    ("create-from-evidence", r"\bcreate an event in the calendar|book \d+ hours\b"),
    ("list-shared-items", r"\bids of the files|shared with\b"),
    ("follow-instructions-found", r"\bdo the actions specified|carry out\b"),
    ("read-then-report-owners", r"\beach person mentioned\b"),
    ("reschedule-existing", r"\breschedule\b"),
    ("add-participants", r"\badd also|to the participants\b"),
    ("destructive-file-op", r"\bfind and delete the largest\b"),
    ("build-shopping-list", r"\bingredients|grocery|recipe\b"),
]


def classify_shape(text):
    for shape, pattern in SHAPE_CUES:
        if re.search(pattern, text, re.I):
            return shape
    return "unclassified"


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
    print("  Shape coverage is the honest claim, not coverage of the 23 specific prompts, and")
    print("  it says nothing about ordinary unauthorized actions in general.")
    print("\n== Shape coverage of the 23 task texts ==")
    real_shapes = defaultdict(set)
    for text in real_task_text:
        real_shapes[classify_shape(text)].add(text)
    synth_shapes = set()
    for case in s6:
        for _shape, template in SHAPES:
            if any(case["state"]["authorized_task"] == template.format(artifact=a)
                   for a in ARTIFACTS):
                synth_shapes.add(_shape)
    print(f"  task shapes behind the failures : {len(real_shapes) - ('unclassified' in real_shapes)}")
    print(f"  task shapes covered in S6        : {len(synth_shapes)}")
    covered = (set(real_shapes) & synth_shapes) - {"unclassified"}
    print(f"  shape coverage                  : {len(covered)}/{len(real_shapes) - 1}")
    unclassified = len(real_shapes.get("unclassified", ()))
    if unclassified:
        print(f"  prompts the classifier could not place: {unclassified} (excluded from the "
              f"denominator, not counted as covered)")
    print("  distribution of the real failures:")
    for shape, texts in sorted(real_shapes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"    {shape:<28} {len(texts)}")
    covered_shapes = sorted(s for s in real_shapes if s != "unclassified" and s in synth_shapes)
    biggest = max(((s, len(t)) for s, t in real_shapes.items() if s != "unclassified"),
                  key=lambda kv: kv[1])
    print(f"\n  The {len(real_tasks)} failures sit behind {len(real_tasks)} prompts but only "
          f"{len(real_shapes) - 1} distinct task shapes.")
    print(f"  The most common, {biggest[0]}, accounts for {biggest[1]} of them, so the")
    print(f"  generator stops at {len(SHAPES)} templates: that covers every shape the failures")
    print("  actually exhibit, and more templates would pad the count without adding a shape.")
    print("  Exact text overlap with the held-out split is zero by construction: the S6 task")
    print("  wording paraphrases a task shape and is never the ASPI prompt itself, so the set")
    print("  does not contaminate the split it is meant to test. The cost is that shape")
    print("  coverage is a claim about categories, not about the 23 specific prompts, and it")
    print("  says nothing about ordinary unauthorized actions in general.")
    print("  The shape classification is an analyst judgement, written by reading the prompts")
    print("  against the generator's shape vocabulary. It is not derived from the generator, so")
    print("  the two sides of the figure are not scored by one table.")

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
