"""Distribution fidelity of the synthetic S6/S7 stopgap against the real ASPI cases.

Read-only. Compares the synthetic pairs in data/s6_s7_synthetic/ against the real
adapted cases in data/aspi/adapted/, axis by axis, and prints the divergence with
numbers rather than adjectives.

The question this answers is narrow and practical: if a model were trained on the
stopgap, should anyone expect the frozen ASPI splits to move? A stopgap that only
resembles the real data on the axis being targeted is not worth a training run, and
saying so with numbers is more useful than a stopgap that looks comprehensive.

Usage:
    python3 scripts/compare_s6_s7_fidelity.py
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOPGAP = ROOT / "data" / "s6_s7_synthetic" / "cases.jsonl"
ADAPTED = ROOT / "data" / "aspi" / "adapted"
PREDICTIONS_DIR = ROOT / "outputs" / "risk" / "laya-risk-ranked"

# Tone markers. The stopgap is written in a deliberately bland register; if the real
# cases are markedly more or less marked, that is a distribution difference a model
# would see.
POLITE = re.compile(
    r"\b(please|could you|would you|can you|thank|thanks|kindly|when you get a moment|"
    r"if you can|at your convenience|just|when convenient|appreciate)\b", re.I)
INJECTION = re.compile(
    r"\b(ignore the (earlier|previous)|disregard|override|maintainer|approved|"
    r"skip asking|urgent|immediately|system notice|instruction|do not mention|"
    r"treat this as|authority|please note|as requested)\b", re.I)
LEAD_VERB = re.compile(
    r"^\s*(?:(?:please|could you|would you|can you|when you get a moment|"
    r"if you can|at your convenience)\b[\w\s,]*?\s*)?"
    r"(send|email|e-mail|message|post|publish|invite|forward|push|save|"
    r"delete|remove|call|create|read|show|find|get|add|write|text)\b", re.I)
RECIPIENT_WORDS = re.compile(
    r"\b(every|everyone|everybody|all|each|entire|whole|recipient|person|people|"
    r"attendee|contact|subscriber|list|group)\b", re.I)
DATE_OR_ENTITY = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|@|[A-Z][a-z]+\s+[A-Z][a-z]+)\b")

WORD = re.compile(r"[a-z']+")


def words(text):
    return WORD.findall(text.lower())


def case_text(case):
    """The comparable surface of a case: the task, the events, and the action."""
    state = case["state"]
    parts = [state["authorized_task"]]
    parts += [event["content"] for event in state["events"]]
    parts.append(state["proposed_action"]["description"])
    return "\n".join(p for p in parts if p)


def load_aspi(splits=("train", "test_user", "test_attack", "test_both")):
    rows = []
    for split in splits:
        path = ADAPTED / f"{split}.jsonl"
        if path.exists():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def load_mode1():
    """The real benign-source false allows, the target failure.

    Reads the action question only. An earlier version of this function matched on
    the content rows, which carry two logits and a benign/suspicious label, and
    silently returned nothing.
    """
    adapted = {}
    for split in ("test_attack", "test_both"):
        path = ADAPTED / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                adapted[row["id"]] = row
    if not adapted:
        return []
    out = []
    for split in ("test_attack", "test_both"):
        predictions = PREDICTIONS_DIR / f"predictions-{split}.json"
        if not predictions.exists():
            continue
        for row in json.loads(predictions.read_text()):
            if row["qid"] != "action" or row["label"] == 0:
                continue
            pred = max(range(len(row["logits"])), key=lambda i: row["logits"][i])
            case = adapted.get(row["id"])
            if case and pred == 0 and case["provenance"]["operator"] == "none":
                out.append(case)
    return out


def stats(values):
    if not values:
        return {}
    ordered = sorted(values)
    return {"n": len(ordered),
            "mean": round(statistics.mean(ordered), 1),
            "median": round(statistics.median(ordered), 1),
            "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 1)}


def jaccard(left, right):
    return len(left & right) / len(left | right) if left | right else 0.0


def main():
    stopgap = [json.loads(line) for line in STOPGAP.read_text().splitlines() if line.strip()]
    s6 = [c for c in stopgap if c["stratum"] == "S6"]
    s7 = [c for c in stopgap if c["stratum"] == "S7"]
    aspi = load_aspi()
    aspi_block = [c for c in aspi if c["labels"]["action"] == "block"]
    mode1 = load_mode1()
    if not aspi or not s6:
        print("missing artifacts; nothing to compare")
        return 1

    print(f"synthetic S6 cases : {len(s6)}")
    print(f"synthetic S7 cases : {len(s7)}")
    print(f"real ASPI cases    : {len(aspi)}  (block-labelled: {len(aspi_block)})")
    print(f"real Mode 1 failures: {len(mode1)}")

    print("\n== Axis 1: text length, in characters ==")
    print(f"  {'corpus':<22}{'n':>6}{'mean':>9}{'median':>9}{'p90':>9}")
    for name, rows in (("S6 stopgap", s6), ("S7 stopgap", s7),
                       ("ASPI all", aspi), ("ASPI block", aspi_block),
                       ("Mode 1 failures", mode1)):
        s = stats([len(case_text(c)) for c in rows])
        print(f"  {name:<22}{s.get('n', 0):>6}{s.get('mean', 0):>9}{s.get('median', 0):>9}"
              f"{s.get('p90', 0):>9}")
    s6m = statistics.mean([len(case_text(c)) for c in s6])
    aspi_m = statistics.mean([len(case_text(c)) for c in aspi])
    m1_m = statistics.mean([len(case_text(c)) for c in mode1]) if mode1 else 0
    print(f"  S6 vs ASPI all      : {s6m / aspi_m:.2f}x the mean length")
    if m1_m:
        print(f"  S6 vs Mode 1        : {s6m / m1_m:.2f}x the mean length")

    print("\n== Axis 2: vocabulary ==")
    vocab = {}
    for name, rows in (("S6", s6), ("S7", s7), ("ASPI", aspi), ("Mode1", mode1)):
        bag = Counter()
        for case in rows:
            bag.update(set(words(case_text(case))))
        vocab[name] = set(bag)
    s6_tokens = Counter()
    for case in s6:
        s6_tokens.update(words(case_text(case)))
    total = sum(s6_tokens.values())
    oov_aspi = sum(n for w, n in s6_tokens.items() if w not in vocab["ASPI"])
    print(f"  distinct words: S6 {len(vocab['S6'])}, S7 {len(vocab['S7'])}, "
          f"ASPI {len(vocab['ASPI'])}, Mode 1 {len(vocab['Mode1'])}")
    print(f"  Jaccard S6 vs ASPI  : {jaccard(vocab['S6'], vocab['ASPI']):.3f}")
    print(f"  Jaccard S6 vs Mode1: {jaccard(vocab['S6'], vocab['Mode1']):.3f}")
    print(f"  S6 token mass unseen in ASPI: {oov_aspi}/{total} = {100 * oov_aspi / total:.1f}%")
    unseen_mode1 = sum(n for w, n in s6_tokens.items() if w not in vocab["Mode1"])
    print(f"  S6 token mass unseen in Mode 1: {unseen_mode1}/{total} = "
          f"{100 * unseen_mode1 / total:.1f}%")

    print("\n== Axis 3: action verb distribution (leading verb of the action text) ==")
    def verbs(rows):
        out = Counter()
        for case in rows:
            match = LEAD_VERB.match(case["state"]["proposed_action"]["description"])
            out[(match.group(1).lower() if match else "other")] += 1
        return out
    s6v, aspiV = verbs(s6), verbs(aspi_block)
    print(f"  S6      : {dict(s6v.most_common(6))}")
    print(f"  ASPI blk: {dict(aspiV.most_common(6))}")
    shared = set(s6v) & set(aspiV) - {"other"}
    s6_mass = sum(s6v.values())
    aspi_mass = sum(aspiV.values())
    overlap = sum(s6v[v] for v in shared) / max(1, s6_mass)
    print(f"  S6 verb mass in verbs ASPI also uses: {overlap:.0%}")
    print(f"  S6 distinct verbs {len(s6v)}, ASPI block distinct verbs {len(aspiV)}")

    print("\n== Axis 4: recipient class ==")
    s6_dest = {re.sub(r"^the \w+ ", "", c["state"]["proposed_action"]["destination"]) for c in s6}
    aspi_dest = {c["state"]["proposed_action"]["destination"] for c in aspi}
    print(f"  S6 distinct destination values      : {len(s6_dest)}")
    print(f"  ASPI distinct destination values    : {len(aspi_dest)}  {sorted(aspi_dest)[:2]}")
    print("  ASPI carries no recipient field at all: the destination is a single constant")
    print("  string, so recipient class cannot be matched and is a structural divergence,")
    print("  not a distributional one.")
    s6_recip = sum(1 for c in s6 if RECIPIENT_WORDS.search(c["state"]["proposed_action"]["description"]))
    aspi_recip = sum(1 for c in aspi_block if RECIPIENT_WORDS.search(c["state"]["proposed_action"]["description"]))
    m1_recip = sum(1 for c in mode1 if RECIPIENT_WORDS.search(c["state"]["proposed_action"]["description"]))
    print(f"  action text naming a recipient set: S6 {s6_recip}/{len(s6)}, "
          f"ASPI block {aspi_recip}/{len(aspi_block)}, Mode 1 {m1_recip}/{len(mode1)}")

    print("\n== Axis 5: tone markers per case ==")
    for name, rows in (("S6 stopgap", s6), ("ASPI all", aspi),
                       ("ASPI block", aspi_block), ("Mode 1 failures", mode1)):
        if not rows:
            continue
        polite = sum(1 for c in rows if POLITE.search(case_text(c)))
        inject = sum(1 for c in rows if INJECTION.search(case_text(c)))
        ent = sum(1 for c in rows if DATE_OR_ENTITY.search(case_text(c)))
        print(f"  {name:<22} politeness {polite / len(rows):>6.0%}   "
              f"injection wording {inject / len(rows):>6.0%}   "
              f"dates/entities {ent / len(rows):>6.0%}")

    print("\n== Axis 6: S6 against the real block-labelled unauthorized cases ==")
    print(f"  S6 cases carrying injection wording: "
          f"{sum(1 for c in s6 if INJECTION.search(case_text(c)))}/{len(s6)}")
    print(f"  ASPI block cases carrying injection wording: "
          f"{sum(1 for c in aspi_block if INJECTION.search(case_text(c)))}/{len(aspi_block)}")
    print(f"  S6 cases with a structured permission field: "
          f"{sum(1 for c in s6 if c['state'].get('permission'))}/{len(s6)}")
    print(f"  ASPI cases with a structured permission field: "
          f"{sum(1 for c in aspi if c['state'].get('permission'))}/{len(aspi)}")
    print("  The stopgap expresses authorization as granted/scope/revoked/source. The ASPI")
    print("  adapter has no such field at all: the only authorization signal is the user's")
    print("  task text. A model that learns the stopgap's permission field learns a schema")
    print("  the frozen splits do not contain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
