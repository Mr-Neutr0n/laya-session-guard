"""Validate a human-reviewed authorization dataset.

Two layers. First a small JSON Schema evaluator over the subset of keywords used by
data/review/case_schema_v1.json, so the schema file stays the single source of truth and
no new dependency is required. Second, the semantic checks a schema cannot express:
split leakage by session, task, pair group, and normalized content; the
history_complete shortcut, globally and per stratum; permission coherence; and the
consent and redaction preconditions that a consented capture must satisfy.

Identity strings are compared through normalize_key, which strips case, surrounding
whitespace, and internal separators. A reviewer typed as "Reviewer-A", "reviewer a", and
"reviewer_a" is one person, and a session id retyped with a trailing space is one
session. The comparison fails safe: two ids that genuinely differ only in punctuation
are merged and reported, which costs a false positive rather than a missed leak.

Exit status is 0 when there are no errors and 1 otherwise. Warnings never fail the run,
they report coverage shortfalls against the plan in reports/AUTHZ_GAP_AUDIT.md.

Usage:
    python3 scripts/validate_review_set.py data/review/examples/clean.jsonl
    python3 scripts/validate_review_set.py dev.jsonl calib.jsonl test.jsonl

Pass every split in one invocation. Leakage is a cross-split property, so validating
each split separately cannot detect it, and per-split runs are how a leak reaches a
training run unnoticed.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "data" / "review" / "case_schema_v1.json"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}

# Strata the current models demonstrably fail, per reports/AUTHZ_GAP_AUDIT.md.
PRIORITY_STRATA = ("S1", "S2", "S3", "S5", "S6")

# S7 is the authorized control for S6. A set with unauthorized cases and no control can
# satisfy the S6 minimum by blocking everything, so the pair is checked directly rather
# than by adding S7 to the 20-case minimum list, which it does not need.
CONTROL_STRATUM = "S7"
CONTROL_FOR = "S6"

# Residual template placeholders. A case carrying one is not a finished case, and a
# template that validated as a submittable batch would let an unedited consented
# capture, with consent_id "REPLACE-WITH-YOUR-CONSENT-ID", pass the gate.
PLACEHOLDER_MARKER = "REPLACE-"

# Below this many dev/calib cases a stratum is too small for the shortcut rate to mean
# anything, and the per-stratum check is skipped rather than guessed at.
MIN_SHORTCUT_STRATUM_CASES = 4

# Same reasoning for the global gate, and higher because the global number is the one
# that blocks a release. Validating a single split, or one slice of dev, leaves too few
# cases for the rate to carry information, and a one-case set scores 100% by construction.
MIN_SHORTCUT_FITTING_CASES = 6

# High-signal credential shapes. Presence in a case is an error because redaction is a
# precondition of storage, not a cleanup step. The scan covers the whole case, not just
# the session state, because a secret pasted into a rationale or a provenance note is
# still a secret sitting in the repository.
CREDENTIAL_PATTERNS = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "private_key_block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    "openai_style_key": r"\bsk-[A-Za-z0-9]{20,}",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
}
# Addresses are an error for a consented capture, because redaction is claimed and
# clearly did not happen. For synthetic origins they are a warning, since fictional
# benchmark domains are expected in the S6 and S7 strata.
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@([\w-]+\.[\w.]+)\b")
EMAIL_ALLOWLIST = {"example.com", "example.org", "example.net", "invalid", "test"}


def normalize_key(value):
    """Compare identity strings by content, ignoring case, padding, and separators."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def normalize_text(value):
    """Compare prose by content, ignoring case and punctuation."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[^a-z0-9 ]+", "", value.casefold()).strip()


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, case, code, message):
        self.errors.append((case, code, message))

    def warn(self, case, code, message):
        self.warnings.append((case, code, message))

    def failed_cases(self):
        return {case for case, _, _ in self.errors if case}

    def render(self, n_cases):
        for case, code, message in self.errors:
            print(f"ERROR   {case or '-'}: {code}: {message}")
        for case, code, message in self.warnings:
            print(f"WARNING {case or '-'}: {code}: {message}")
        print(f"\n{n_cases} cases, {len(self.errors)} errors, {len(self.warnings)} warnings")


# --------------------------------------------------------------------------- schema

TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int, "null": type(None),
}


def resolve(schema, root):
    while "$ref" in schema:
        target = schema["$ref"]
        if not target.startswith("#/"):
            raise ValueError(f"only local refs are supported, got {target}")
        node = root
        for part in target[2:].split("/"):
            node = node[part]
        schema = node
    return schema


def check_type(value, expected):
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        python_type = TYPES[name]
        if name != "boolean" and isinstance(value, bool) and python_type is not bool:
            continue  # bool is a subclass of int in Python but is not a number here
        if isinstance(value, python_type):
            return True
    return False


def validate_schema(instance, schema, root, path, report, case_id):
    schema = resolve(schema, root)
    if "const" in schema and instance != schema["const"]:
        report.error(case_id, "schema.const", f"{path} must equal {schema['const']!r}")
        return
    if "enum" in schema and instance not in schema["enum"]:
        report.error(case_id, "schema.enum",
                     f"{path} must be one of {schema['enum']}, got {instance!r}")
        return
    if "type" in schema and not check_type(instance, schema["type"]):
        report.error(case_id, "schema.type",
                     f"{path} must be {schema['type']}, got {type(instance).__name__}")
        return
    if isinstance(instance, str):
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            report.error(case_id, "schema.pattern",
                         f"{path} does not match {schema['pattern']!r}")
        if len(instance) < schema.get("minLength", 0):
            report.error(case_id, "schema.minLength", f"{path} is shorter than {schema['minLength']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            report.error(case_id, "schema.minimum", f"{path} is below {schema['minimum']}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            report.error(case_id, "schema.minItems", f"{path} has fewer than {schema['minItems']} items")
        if "items" in schema:
            for index, item in enumerate(instance):
                validate_schema(item, schema["items"], root, f"{path}[{index}]", report, case_id)
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                report.error(case_id, "schema.required", f"{path} is missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                validate_schema(value, properties[key], root, f"{path}.{key}", report, case_id)
            elif schema.get("additionalProperties") is False:
                report.error(case_id, "schema.additionalProperties",
                             f"{path} has unknown key {key!r}")


# ------------------------------------------------------------------ semantic checks

def check_consent(case, report):
    """A consented capture must carry a consent id and have been redacted."""
    case_id = case.get("case_id", "?")
    provenance = case.get("provenance", {})
    consented_origin = provenance.get("origin") == "consented_session"
    consent_id = provenance.get("consent_id")
    # data_origin says whether the material is synthetic or a real consented capture. It must
    # agree with origin, so a synthetic case generated to cover a known failure cannot be
    # recorded in a way that reads as an observation, and a real capture cannot be laundered
    # as template content.
    data_origin = provenance.get("data_origin")
    if data_origin is not None:
        expected = "consented" if consented_origin else "synthetic"
        if data_origin != expected:
            report.error(case_id, "consent.data_origin",
                         f"provenance.data_origin is {data_origin!r} but origin "
                         f"{provenance.get('origin')!r} implies {expected!r}")
    if consented_origin:
        if not normalize_key(consent_id):
            report.error(case_id, "consent.missing",
                         "provenance.origin is consented_session but consent_id is absent or "
                         "blank; consent is per use and is never implied by the session existing")
        if provenance.get("redaction") != "applied":
            report.error(case_id, "consent.redaction",
                         "provenance.redaction must be 'applied' for a consented session")
    # Scan the entire case, not just state: a secret in a rationale or a provenance note
    # is committed to the repository just the same.
    blob = json.dumps(case, sort_keys=True)
    for name, pattern in CREDENTIAL_PATTERNS.items():
        if re.search(pattern, blob):
            report.error(case_id, "redaction.credential",
                         f"case text matches the {name} pattern; redact before storing")
    domains = set(EMAIL_PATTERN.findall(blob))
    for domain in sorted(domains - EMAIL_ALLOWLIST):
        code = "redaction.email"
        if consented_origin:
            report.error(case_id, code,
                         f"case text contains addresses at {domain}, but the case claims a "
                         f"redacted consented capture")
        else:
            report.warn(case_id, code,
                        f"case text contains addresses at {domain}; confirm redaction")


def check_permission(case, report):
    """The grant recorded in the case must be able to support the decision recorded."""
    case_id = case.get("case_id", "?")
    state = case.get("state", {})
    permission = state.get("permission", {})
    events = state.get("events", [])
    granted = permission.get("granted")
    scope = permission.get("scope", [])
    source = permission.get("source")
    if source == "none" and granted:
        report.error(case_id, "permission.source",
                     "permission.source is 'none' but granted is true, so no user ever "
                     "authorized this action")
    if not granted and scope:
        report.error(case_id, "permission.scope",
                     f"permission.granted is false but scope lists {scope}; nothing was "
                     f"authorized, so a scope cannot be recorded")
    if granted and not scope:
        report.error(case_id, "permission.scope",
                     "permission.granted is true with an empty scope, which authorizes nothing")
    if permission.get("revoked") and case.get("labels", {}).get("action") == "allow":
        report.error(case_id, "permission.revoked",
                     "permission.revoked is true but the label is allow; a withdrawn grant "
                     "cannot support an allow")
    if source == "direct_user":
        user_turns = [event for event in events
                      if isinstance(event, dict) and event.get("source") == "direct_user"]
        if not user_turns:
            report.error(case_id, "permission.source",
                         "permission.source is 'direct_user' but no direct_user event appears in "
                         "the session; only a user turn can grant permission")
        elif not any(str(event.get("content", "")).strip() for event in user_turns):
            report.error(case_id, "permission.source",
                         "permission.source is 'direct_user' but every user turn is blank; an "
                         "empty turn cannot grant permission")
        if not str(state.get("authorized_task", "")).strip():
            report.error(case_id, "permission.source",
                         "permission.source is 'direct_user' but authorized_task is blank; a "
                         "grant with no stated task grants nothing to compare the action against")


def check_annotations(case, report):
    """Two distinct humans, and a decision consistent with what they said."""
    case_id = case.get("case_id", "?")
    annotations = case.get("annotations", [])
    adjudication = case.get("adjudication", {})
    reviewers = [normalize_key(a.get("reviewer_id")) for a in annotations]
    for annotation in annotations:
        if not normalize_key(annotation.get("reviewer_id")):
            report.error(case_id, "annotation.blank_reviewer",
                         "a reviewer_id is blank; an anonymous annotation is not independent "
                         "and cannot be audited")
    if len(set(reviewers)) != len(reviewers):
        report.error(case_id, "annotation.duplicate_reviewer",
                     "the same reviewer annotated the case more than once, allowing for case, "
                     "padding, and separator differences; a second pass by one person is not a "
                     "second opinion")
    actions = {a.get("action") for a in annotations}
    final = adjudication.get("final")
    if len(actions) == 1:
        if adjudication.get("status") != "unanimous":
            report.error(case_id, "annotation.status",
                         f"all reviewers chose {actions} but status is "
                         f"{adjudication.get('status')!r}, expected 'unanimous'")
        if adjudication.get("adjudicator_id") is not None:
            report.error(case_id, "annotation.adjudicator",
                         "a unanimous case must not name an adjudicator")
    else:
        if adjudication.get("status") != "adjudicated":
            report.error(case_id, "annotation.disagreement",
                         f"reviewers disagreed ({sorted(map(str, actions))}) but status is "
                         f"{adjudication.get('status')!r}; the case needs a third human")
        adjudicator = normalize_key(adjudication.get("adjudicator_id"))
        if not adjudicator:
            report.error(case_id, "annotation.adjudicator",
                         "an adjudicated case must name the deciding reviewer")
        elif adjudicator in reviewers:
            report.error(case_id, "annotation.adjudicator",
                         "the adjudicator must not be one of the two original annotators, "
                         "allowing for case, padding, and separator differences")
    if final is not None and case.get("labels", {}).get("action") != final:
        report.error(case_id, "annotation.label_mismatch",
                     f"labels.action is {case['labels'].get('action')!r} but the "
                     f"adjudicated decision is {final!r}")


def check_context_flag(case, report):
    """The two history fields must agree, so the flag is not a free label signal."""
    case_id = case.get("case_id", "?")
    context = case.get("state", {}).get("context", {})
    complete = context.get("history_complete")
    omitted = context.get("omitted_events")
    reason = context.get("truncation_reason")
    if complete is False:
        if not reason:
            report.error(case_id, "context.truncation_reason",
                         "history_complete is false but truncation_reason is null")
        if not isinstance(omitted, int) or omitted <= 0:
            report.error(case_id, "context.omitted_events",
                         f"history_complete is false but omitted_events is {omitted!r}; the "
                         f"count must show the history is genuinely short")
    elif complete is True and isinstance(omitted, int) and omitted > 0:
        report.error(case_id, "context.omitted_events",
                     f"history_complete is true but omitted_events is {omitted}; the flag "
                     f"hides a truncation cue the model can still read")


def fingerprint(case, *, include_prose=True):
    """Content identity for duplicate detection.

    The strict form includes the task text and the action wording, so a reworded case
    reads as a different case. The loose form drops both and keeps only the
    authorization-bearing state: the permission, the events, the action class, and the
    destination. Rewording cannot change those, which is exactly why a loose collision is
    a candidate for a human rather than proof.
    """
    state = case.get("state", {})
    action = state.get("proposed_action", {})
    events = state.get("events", [])
    if not isinstance(events, list):
        events = []
    parts = [
        json.dumps(state.get("permission", {}), sort_keys=True),
        " ".join(str(e.get("content", "")) for e in events if isinstance(e, dict)),
        " ".join(str(e.get("source", "")) for e in events if isinstance(e, dict)),
        str(action.get("kind", "")),
        str(action.get("destination", "")),
    ]
    if include_prose:
        parts = [str(state.get("authorized_task", "")),
                 str(action.get("description", ""))] + parts
    return normalize_text(" | ".join(parts))


def check_leakage(cases, report):
    """No session, task, pair group, or content fingerprint may straddle a split."""
    for key in ("session_id", "task_id", "pair_group_id"):
        splits = defaultdict(set)
        for case in cases:
            value = case.get("group", {}).get(key)
            if value is None:
                continue
            normalized = normalize_key(value)
            if not normalized:
                report.error(case.get("case_id", "?"), f"leakage.blank_{key}",
                             f"group.{key} is blank; a blank key cannot be checked for "
                             f"cross-split reuse and must be filled in")
                continue
            splits[normalized].add(case["split"])
        for value, seen in sorted(splits.items()):
            if len(seen) > 1:
                offenders = sorted(c["case_id"] for c in cases
                                   if normalize_key(c.get("group", {}).get(key) or "") == value)
                report.error(offenders[0], f"leakage.{key}",
                             f"{key}={value!r} appears in splits {sorted(seen)}, comparing ids "
                             f"ignoring case, padding, and separators; affected cases {offenders}")
    seen_ids = Counter(c.get("case_id") for c in cases)
    for case_id, count in sorted(seen_ids.items()):
        if count > 1:
            report.error(case_id, "leakage.case_id",
                         f"case_id appears {count} times; ids are stable and never reused")
    by_fingerprint = defaultdict(set)
    for case in cases:
        by_fingerprint[fingerprint(case)].add(case["split"])
    for value, seen in sorted(by_fingerprint.items()):
        if len(seen) > 1:
            offenders = sorted(c["case_id"] for c in cases if fingerprint(c) == value)
            report.error(offenders[0], "leakage.text",
                         f"the task, permission, events, and action are duplicated across "
                         f"splits {sorted(seen)}; affected cases {offenders}")
    # A reworded twin of a case keeps the permission, the events, and the destination but
    # not the phrasing, so it escapes the strict fingerprint. Surface those as candidates
    # for a human. This is a warning, not an error, because deciding whether two wordings
    # are the same case is a judgment the validator cannot make.
    by_loose = defaultdict(set)
    for case in cases:
        by_loose[fingerprint(case, include_prose=False)].add(case["split"])
    for value, seen in sorted(by_loose.items()):
        if len(seen) > 1 and value not in by_fingerprint:
            offenders = sorted(c["case_id"] for c in cases
                               if fingerprint(c, include_prose=False) == value)
            report.warn(offenders[0], "leakage.text_near",
                        f"the permission, events, and destination are shared across splits "
                        f"{sorted(seen)} while the task wording differs; affected cases "
                        f"{offenders}. Confirm by hand whether these are the same case "
                        f"reworded, which the validator cannot decide")


def one_bit_rate(rows):
    """Accuracy of predicting the action from history_complete alone, in percent."""
    if not rows:
        return 0.0
    by_flag = defaultdict(Counter)
    for case in rows:
        by_flag[case["state"]["context"]["history_complete"]][case["labels"]["action"]] += 1
    correct = sum(counts.most_common(1)[0][1] for counts in by_flag.values())
    return 100.0 * correct / len(rows)


def check_shortcut(cases, report, threshold, min_stratum_cases):
    """Guard against relabeling the history_complete shortcut under a new dataset.

    Earlier Laya training data made review a perfect one-bit function of
    history_complete, and the model learned the flag instead of the permission rule.
    Checked globally and per stratum, because a flag that predicts the label perfectly
    inside one stratum still teaches the shortcut and still hides in the global average.
    """
    fitting = [c for c in cases if c["split"] in ("dev", "calib")]
    if not fitting:
        report.warn(None, "shortcut.no_fitting_data",
                    "no dev or calib cases; the history_complete shortcut cannot be checked")
        return
    review_flags = Counter(c["state"]["context"]["history_complete"]
                           for c in fitting if c["labels"]["action"] == "review")
    if not review_flags:
        report.warn(None, "shortcut.no_review_cases",
                    "no review cases in dev or calib; the review class is untested")
    elif len(review_flags) == 1:
        only = next(iter(review_flags))
        report.error(None, "shortcut.review_confounded",
                     f"all {review_flags[only]} dev/calib review cases share "
                     f"history_complete={only}; in the earlier data review was a perfect "
                     f"one-bit function of this flag and the model learned the flag. Include "
                     f"review cases at both flag values. Incomplete-history cases that are "
                     f"decisively block or allow are fine and required; only the review "
                     f"class needs both.")
    accuracy = one_bit_rate(fitting)
    # The historical defect runs one way only. In the risk-ranked run's 2,920-case train
    # split, all 216 incomplete-history cases were labelled review with zero counterexamples,
    # while 40 review cases had complete history. So "truncated implies review" was learnable
    # and "review implies truncated" was not. A confounded review class does not catch that.
    incomplete = [c for c in fitting if not c["state"]["context"]["history_complete"]]
    if len(incomplete) >= MIN_SHORTCUT_FITTING_CASES and \
            all(c["labels"]["action"] == "review" for c in incomplete):
        report.error(incomplete[0]["case_id"], "shortcut.incomplete_is_review",
                     f"all {len(incomplete)} dev/calib cases with history_complete=false are "
                     f"labelled review, with no counterexample. The model can learn "
                     f"'truncated implies review' and never learn a rule for review on a "
                     f"complete history, which is exactly what happened. Include incomplete "
                     f"cases that are decisively allow or block")
    distinct = {c["labels"]["action"] for c in fitting}
    if len(fitting) >= MIN_SHORTCUT_FITTING_CASES and len(distinct) >= 2 \
            and accuracy >= threshold:
        report.error(None, "shortcut.single_bit_accuracy",
                     f"predicting the label from history_complete alone reaches "
                     f"{accuracy:.1f}% of dev/calib, at or above the {threshold:.0f}% gate")
    elif accuracy >= threshold:
        report.warn(None, "shortcut.single_bit_accuracy_small",
                    f"predicting the label from history_complete alone reaches "
                    f"{accuracy:.1f}% of dev/calib, but only {len(fitting)} cases and "
                    f"{len(distinct)} distinct label(s) are present, so the rate is not "
                    f"informative and the gate was not applied. Run the validator over the "
                    f"full dev and calib splits")
    by_stratum = defaultdict(list)
    for case in fitting:
        by_stratum[case["stratum"]].append(case)
    for stratum in sorted(by_stratum):
        rows = by_stratum[stratum]
        # Two guards, both needed. A stratum with too few cases gives a meaningless rate,
        # and a stratum with a single label scores 100% by construction whatever the flag
        # does, which would reject a legitimately one-directional stratum.
        if len(rows) < min_stratum_cases:
            continue
        if len({c["labels"]["action"] for c in rows}) < 2:
            continue
        local = one_bit_rate(rows)
        if local >= threshold:
            sample = rows[0]["case_id"]
            report.error(sample, "shortcut.stratum_single_bit",
                         f"within stratum {stratum}, history_complete alone predicts the label "
                         f"for {local:.1f}% of {len(rows)} dev/calib cases, at or above the "
                         f"{threshold:.0f}% gate. The global rate is {accuracy:.1f}%, so a "
                         f"stratum-local shortcut can hide in the average")
    table = {}
    for stratum in sorted(by_stratum):
        rows = by_stratum[stratum]
        labels = {c["labels"]["action"] for c in rows}
        if len(rows) >= min_stratum_cases and len(labels) >= 2:
            table[stratum] = {"n": len(rows), "one_bit": f"{one_bit_rate(rows):.1f}%"}
    print(f"INFO    history_complete single-bit accuracy on dev+calib: {accuracy:.1f}% "
          f"({len(fitting)} cases, gate {threshold:.0f}%)")
    if table:
        print(f"INFO    per-stratum one-bit accuracy, strata with >= {min_stratum_cases} cases "
              f"and >= 2 labels: {table}")
    single = sorted(s for s, rows in by_stratum.items()
                    if len(rows) >= min_stratum_cases and len({c["labels"]["action"] for c in rows}) < 2)
    if single:
        print(f"INFO    per-stratum rate not computed for single-label strata {single}: a stratum "
              f"with one label scores 100% by construction, whatever the flag does")
    test_cases = [c for c in cases if c["split"] == "test"]
    if test_cases:
        test_flags = Counter(c["state"]["context"]["history_complete"] for c in test_cases)
        test_labels = Counter(c["labels"]["action"] for c in test_cases)
        print(f"INFO    sealed test flags {dict(test_flags)} labels {dict(test_labels)}")


def check_placeholders(case, report):
    """Refuse a case that still carries template placeholders.

    An unedited template is a shape, not a case. Without this, the shipped
    review_kit template validated clean, including a consented_session row whose
    consent_id was the literal string REPLACE-WITH-YOUR-CONSENT-ID.
    """
    case_id = case.get("case_id", "?")
    found = []

    def walk(node, path):
        if isinstance(node, str):
            if PLACEHOLDER_MARKER in node:
                found.append(path)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(case, "$")
    for path in found:
        report.error(case_id, "template.unedited",
                     f"{path} still contains a {PLACEHOLDER_MARKER} placeholder; finish the "
                     f"case before submitting it")


def check_coverage(cases, report, minimum):
    """Plan coverage. Shortfalls are warnings: they block a training run, not a schema check.

    Counted over distinct authorized tasks, not cases. Twenty copies of one session are
    one observation, and counting cases let a single session satisfy a 20-case minimum.
    """
    for stratum in PRIORITY_STRATA:
        tasks = {c["group"].get("task_id") for c in cases
                 if c["split"] == "test" and c["stratum"] == stratum}
        if len(tasks) < minimum:
            report.warn(None, "coverage.test_stratum",
                        f"stratum {stratum} has {len(tasks)} distinct sealed test tasks, below "
                        f"the planned minimum of {minimum}")
    control = {c["group"].get("task_id") for c in cases
               if c["split"] == "test" and c["stratum"] == CONTROL_STRATUM}
    target = {c["group"].get("task_id") for c in cases
              if c["split"] == "test" and c["stratum"] == CONTROL_FOR}
    if target and not control:
        report.warn(None, "coverage.missing_control",
                    f"sealed test has {len(target)} distinct {CONTROL_FOR} tasks and no "
                    f"{CONTROL_STRATUM} control. Without the authorized control a model can "
                    f"satisfy {CONTROL_FOR} by blocking everything, which is the failure mode "
                    f"the matched pairs exist to prevent")


def load_cases(paths, report):
    cases = []
    for path in paths:
        if not path.exists():
            report.error(None, "input.missing", f"{path}: no such file")
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                report.error(f"{path.name} line {number}", "json.parse", str(error))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", type=Path,
                        help="one or more JSONL files of review cases. Pass every split in a "
                             "single invocation: leakage is a cross-split property.")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--min-test-per-stratum", type=int, default=20)
    parser.add_argument("--shortcut-accuracy-gate", type=float, default=95.0)
    parser.add_argument("--min-shortcut-stratum-cases", type=int,
                        default=MIN_SHORTCUT_STRATUM_CASES)
    parser.add_argument("--allow-placeholders", action="store_true",
                        help="accept cases that still carry template placeholders. Only for "
                             "checking that the shipped templates are shaped correctly; a "
                             "submittable batch must not use it.")
    args = parser.parse_args()

    report = Report()
    if not all(path.exists() for path in args.cases):
        print(f"ERROR   input.missing: {args.cases}: no such file")
        return 1
    root_schema = json.loads(args.schema.read_text())
    cases = load_cases(args.cases, report)
    if not cases:
        report.error(None, "batch.empty",
                     f"no cases found in {[str(p) for p in args.cases]}. An empty batch is not "
                     f"a valid batch; an accidentally truncated or wrong-path file must fail "
                     f"rather than pass silently")

    for case in cases:
        if not isinstance(case, dict):
            report.error(None, "schema.type", "a case is not a JSON object")
            continue
        case_id = case.get("case_id", "?")
        validate_schema(case, root_schema, root_schema, "$", report, case_id)
        # A case that already failed the schema may be missing or mistype anything the
        # semantic layer reads, so running it would crash on `"annotations": "..."` rather
        # than report. Schema errors are already fatal, so skip it and say so.
        if case_id in report.failed_cases():
            report.warn(case_id, "checks.skipped",
                        "semantic checks skipped because this case failed the schema; fix the "
                        "schema errors first, then re-run")
            continue
        if case.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            report.error(case_id, "schema.unsupported_version",
                         f"schema_version {case.get('schema_version')!r} is not implemented; "
                         f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
        check_consent(case, report)
        check_permission(case, report)
        check_annotations(case, report)
        check_context_flag(case, report)
        if not args.allow_placeholders:
            check_placeholders(case, report)

    check_leakage([c for c in cases if isinstance(c, dict)], report)
    check_shortcut([c for c in cases if isinstance(c, dict)], report,
                   args.shortcut_accuracy_gate, args.min_shortcut_stratum_cases)
    check_coverage([c for c in cases if isinstance(c, dict)], report,
                   args.min_test_per_stratum)
    report.render(len(cases))
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
