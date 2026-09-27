"""Turn local agent session logs into candidate authorization cases.

LOCAL ONLY. This script reads session logs from disk, decides what to keep, and writes
a draft batch next to this file. It sends nothing anywhere, opens no socket, and needs
no network. Nothing in the draft leaves the machine unless you copy it somewhere
deliberately.

What it produces is a **candidate**, not a labelled case. A candidate records that a
permission-relevant action was taken or proposed, and the user turn that precedes it as
the grant context. It deliberately carries **no label and no annotations**, because a
machine-collected case must not arrive pre-labelled. Two humans still have to judge it,
and the subject has to consent, before it means anything. Every candidate is written
with `origin` and `data_origin` of `pending_consent`, and its notes say so.

Redaction runs before anything is written. A candidate whose text still contains a
credential, a real address, a home directory path, or other personal content is
**dropped, not stored**, and the reason is counted on stderr. Only the minimum the
schema needs is kept: the preceding user turn, the action, and the target. Raw
transcripts are never copied.

Usage:
    python3 review_kit/collect_from_sessions.py --dry-run
    python3 review_kit/collect_from_sessions.py --out review_kit/draft_batch.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0.0"

# Redaction is a hard gate here, stricter than the validator's, because this script's
# whole job is to avoid writing something sensitive in the first place.
CREDENTIAL_PATTERNS = {
    # OpenAI, including the project- and org-scoped forms. The previous pattern used
    # [A-Za-z0-9]{20,} straight after "sk-", which cannot match "sk-proj-..." because the
    # hyphen is not in the class, so a live project key passed the gate in three fields of
    # one candidate. The separator set is now explicit.
    "openai_key": r"\bsk-(?:proj-|ant-|admin-|svcacct-)?[A-Za-z0-9_-]{16,}",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_-]{16,}",
    "stripe_key": r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}",
    # AWS. Access key ids are a fixed prefix plus uppercase alphanumerics; the secret is a
    # 40 character base64-ish run that appears after a label.
    "aws_access_key_id": r"\b(?:AKIA|ASIA|ABIA|ACCA|AIDA|AROA|ANPA|ANVA|AIPA|APKA)[A-Z0-9]{12,}",
    "aws_secret_access_key": r"(?i)aws[_a-z]{0,12}(?:secret|private)[_a-z ]{0,16}[\"'\s:=]{1,8}[A-Za-z0-9/+=]{40}",
    # GitHub classic and fine-grained, GitLab, Slack including app tokens and webhooks.
    "github_token": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{22,}",
    "gitlab_token": r"\bglpat-[A-Za-z0-9_-]{16,}",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}|\bxapp-[A-Za-z0-9-]{10,}",
    "slack_webhook": r"https://hooks\.slack\.com/services/[A-Za-z0-9/+_-]{10,}",
    # Generic authorization headers, whichever scheme.
    "bearer_token": r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}",
    "basic_auth": r"(?i)\bbasic\s+[A-Za-z0-9+/=]{16,}",
    # Structural material.
    "private_key_block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    "npm_token": r"\bnpm_[A-Za-z0-9]{30,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{30,}",
    # Long opaque runs. These over-match on purpose: a git SHA or a base64 asset is not
    # worth storing in a review batch either, and a false drop is cheap.
    "hex_blob": r"\b[0-9a-fA-F]{40,}\b",
    "base64_blob": r"\b[A-Za-z0-9+/]{40,}={0,2}",
}
EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+\.[\w.]+)\b")
EMAIL_ALLOWLIST = {"example.com", "example.org", "example.net", "invalid", "test", "localhost"}
HOME_PATH = re.compile(r"/(Users|home)/([A-Za-z0-9._-]+)")
LONG_DIGITS = re.compile(r"\b\d{9,}\b")
PHONEISH = re.compile(r"\b\+?\d[\d ()-]{9,}\d\b")
IPV4 = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")

# Tools that mutate, send, or reach outside. Reads are not authorization decisions.
PI_DECISION_TOOLS = {
    "write": ("write_file", "write a file"),
    "edit": ("write_file", "edit a file"),
    "multi_edit": ("write_file", "edit files"),
    "bash": ("run_command", "run a shell command"),
    "web_search": ("network_request", "make an external request"),
    "web_fetch": ("network_request", "fetch an external URL"),
    "send_email": ("send_message", "send a message"),
    # The scenario harness mock tool. An outward send to a recipient list is the
    # decision point the Mode 1 failures are made of.
    "send_batch": ("send_message", "send a message to each recipient in a list"),
}
CODEX_DECISION_NAMES = {
    "shell": ("run_command", "run a shell command"),
    "local_shell": ("run_command", "run a shell command"),
    "exec": ("run_command", "run a shell command"),
    "container.exec": ("run_command", "run a shell command"),
    "apply_patch": ("write_file", "patch a file"),
    "update_plan": ("other", "update a plan"),
}

# A bash command that only reads is not a decision point worth a human's time.
READ_ONLY = re.compile(
    r"^\s*(cat|ls|head|tail|wc|grep|rg|find|which|echo|pwd|date|git\s+(status|log|diff|"
    r"show|branch)|node\s+--version|python3?\s+--version)\b")
def identity_tokens(extra=()):
    """Identity strings that must never be stored: who ran the session, and where.

    Derived rather than hard-coded, because hard-coding one developer's name would leave
    the gate open for everyone else. Sources: the local account name, the home directory
    name, the git author name and email configured in this repository, and anything the
    caller passes. Tokens shorter than three characters are ignored, since a two-letter
    string would match ordinary words.
    """
    import getpass
    import subprocess
    tokens = set()
    for value in (getpass.getuser(), Path.home().name):
        if value and len(value) >= 3:
            tokens.add(value.lower())
    for key in ("user.name", "user.email"):
        try:
            out = subprocess.run(["git", "config", "--get", key], cwd=HERE.parent.parent,
                                 capture_output=True, text=True, timeout=5)
            value = (out.stdout or "").strip()
            if len(value) >= 3:
                tokens.add(value.lower())
                if "@" in value:
                    tokens.add(value.split("@")[0].lower())
        except (OSError, subprocess.SubprocessError):
            pass
    for value in extra or ():
        if value and len(value) >= 3:
            tokens.add(value.lower())
    # A local account shows up under more than one name. "harikp" also appears as
    # "Hari" in anything the account owner writes by hand, and a live case body read
    # "Thanks, Hari" straight through the gate until this was added. So for each token,
    # also drop the alphabetic run before any digit, and its capitalised form.
    # Only the alphabetic run of a token, never its pieces. An earlier version split
    # identities on punctuation and picked up "gmail" and "com" out of the email local
    # part, which would have dropped ordinary prose mentioning either word.
    derived = set()
    for token in list(tokens):
        if "@" in token:
            continue
        alpha = re.match(r"[a-z]+", token.lower())
        if alpha and len(alpha.group(0)) >= 3:
            derived.add(alpha.group(0))
            derived.add(alpha.group(0).capitalize())
    tokens |= derived
    # A given name is not derivable. "harikp" does not reveal "Hari", and a live case body
    # read "Thanks, Hari" through the gate until that was noticed. Pass it with --identity.
    return sorted(tokens, key=len, reverse=True)


IDENTITY_TOKENS = identity_tokens()


def identity_reject_reason(text):
    """Drop anything naming the local account, the home directory, or the git identity."""
    lowered = text.lower()
    for token in IDENTITY_TOKENS:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", lowered):
            return f"identity:{token}"
    return None


# Third-party and relationship detection. These FLAG, they never drop. A relationship term is
# a high-confidence lexical match; a capitalised name-like token is a heuristic that will also
# fire on ordinary capitalised words, so it must not remove a case on its own. The judgement
# whose data this is belongs to a person, not to this script.
RELATIONSHIP_TERMS = re.compile(
    r"\bmy\s+(?:wife|husband|partner|spouse|girlfriend|boyfriend|fiancee|"
    r"mother|mum|mom|father|dad|parents?|sister|brother|cousin|uncle|aunt|"
    r"son|daughter|kids?|children|niece|nephew|grandmother|grandfather|grandma|grandpa|"
    r"in-?law|boss|manager|colleague|co-?worker|friend|neighbou?r|teammate|mentor|"
    r"therapist|doctor|landlord|tenant|client|customer|patient|student|teacher)\b", re.I)
# A capitalised word that is not sentence-initial-only. Deliberately noisy.
NAME_LIKE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
# A capitalised word is only treated as a name when it sits in a slot a person would
# occupy. Without this the flag fired on 833 of 1,337 candidates, and 810 of those were
# ordinary technical prose: Kaggle, Laya, Benchmark, React, Vue. A flag that fires on two
# thirds of a batch is not a screen.
PERSON_CONTEXT = re.compile(
    r"\b(?:to|with|from|ask|email|e-mail|mail|message|text|call|tell|ask|cc|bcc|hi|hey|"
    r"dear|thanks|thank|for|about|regarding|and|or|but|@)\s+[A-Z][a-z]{2,}\b")
# A capitalised token immediately after another capitalised token, as in a full name.
NAME_PAIR = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
# Words that are capitalised for reasons other than being a name.
NAME_STOPWORDS = {
    "The","This","That","Please","Thanks","Update","Run","Use","Check","Read","Write",
    "Add","Remove","Delete","Install","Push","Commit","Rebase","Merge","Deploy","Build",
    "Test","Note","Warning","Error","Failed","Cannot","Could","Would","Should","When",
    "Where","What","Which","Who","Why","How","If","And","But","For","With","From","Then",
    "Also","Just","Only","Now","Here","There","Monday","Tuesday","Wednesday","Thursday",
    "Friday","Saturday","Sunday","January","February","March","April","May","June","July",
    "August","September","October","November","December",
    # Domain vocabulary that a coding session capitalises constantly.
    "Kaggle","Laya","Git","GitHub","GitLab","Readme","README","API","CLI","JSON","YAML",
    "Markdown","Python","JavaScript","TypeScript","Rust","React","Vue","Angular","Svelte",
    "Bootstrap","Tailwind","Node","Npm","Docker","Linux","MacOS","Windows","Chrome",
    "Safari","Firefox","Edge","Benchmark","Explore","Mode","State","Web","Developer",
    "Designing","Kerala","School","University","Model","Data","File","Files","Path","Paths",
    "Repo","Repository","Test","Tests","Config","Setup","Build","Release","Draft","Review",
    "Session","Sessions","Agent","Agents","Tool","Tools","Server","Client","Provider",
}


def privacy_flags(text):
    """Return review flags for third-party or relationship content. Never a drop."""
    flags = []
    if RELATIONSHIP_TERMS.search(text):
        flags.append("relationship_term")
    # Heuristic, and reported as such. A capitalised word counts as a name only in a slot a
    # person occupies, or as part of a capitalised pair. The local account's own name is
    # already dropped by the identity gate and must not double-count here.
    candidates = {w for w in NAME_LIKE.findall(text) if w not in NAME_STOPWORDS}
    candidates -= {t for t in IDENTITY_TOKENS}
    if candidates and (PERSON_CONTEXT.search(text) or NAME_PAIR.search(text)):
        flags.append("named_individual")
    return flags


MAX_TURN_CHARS = 600
MIN_TURN_CHARS = 12
MAX_ACTION_CHARS = 300


def digest(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def redact_reject_reason(text):
    """Return a rejection reason, or None if the text is safe to store."""
    for name, pattern in CREDENTIAL_PATTERNS.items():
        if re.search(pattern, text):
            return f"credential:{name}"
    for domain in EMAIL.findall(text):
        if domain.lower() not in EMAIL_ALLOWLIST:
            return "email:external"
    if HOME_PATH.search(text):
        return "home_path"
    if LONG_DIGITS.search(text) or PHONEISH.search(text):
        return "long_digits"
    if IPV4.search(text):
        return "ip_address"
    return identity_reject_reason(text)


def clean(text):
    """Collapse whitespace, clip length, and replace any home directory prefix."""
    text = HOME_PATH.sub("/home/<user>", text)
    text = " ".join(text.split())
    if len(text) > MAX_TURN_CHARS:
        text = text[:MAX_TURN_CHARS].rstrip() + " [clipped]"
    return text


def read_jsonl(path):
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def pi_turns(path):
    """Yield (user_turn, tool_name, tool_args) from a pi session log."""
    last_user = ""
    for record in read_jsonl(path):
        if record.get("type") != "message":
            continue
        message = record.get("message") or {}
        content = message.get("content") or []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and message.get("role") == "user":
                text = part.get("text") or ""
                if len(text.strip()) >= MIN_TURN_CHARS:
                    last_user = text
            elif part.get("type") == "toolCall":
                yield last_user, part.get("name") or "", part.get("arguments") or {}


def codex_turns(path):
    """Yield (user_turn, tool_name, tool_args) from a codex session log."""
    last_user = ""
    for record in read_jsonl(path):
        payload = record.get("payload") or {}
        kind = payload.get("type")
        if record.get("type") == "response_item" and kind == "message":
            if payload.get("role") == "user":
                for part in payload.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        if len(part["text"].strip()) >= MIN_TURN_CHARS:
                            last_user = part["text"]
        elif record.get("type") == "event_msg" and kind == "item_completed":
            item = payload.get("item") or {}
            if item.get("type") == "UserMessage":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        if len(part["text"].strip()) >= MIN_TURN_CHARS:
                            last_user = part["text"]
        elif record.get("type") == "response_item" and kind and "call" in str(kind):
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            yield last_user, payload.get("name") or "", arguments or {}


def action_target(name, args):
    """Best-effort destination, without copying a whole command line.

    For a message send the body is part of the action's shape, not incidental detail: the
    Mode 1 failures all script one identical body per recipient, so dropping the body
    would hide the very feature being targeted. Subject and body are clipped hard.
    """
    if not isinstance(args, dict):
        args = {"raw": str(args)}
    if name in ("send_batch", "send_email"):
        to = args.get("to") or args.get("recipient") or []
        body = str(args.get("body", ""))
        subject = str(args.get("subject", ""))
        one = "one message per recipient" if args.get("one_per_recipient") else ""
        return f"to {to}; subject {subject}; body {body}; {one}"[:MAX_ACTION_CHARS]
    for key in ("path", "file_path", "filePath", "target_file", "url", "recipient", "to"):
        if args.get(key):
            return str(args[key])[:120]
    command = args.get("command") or args.get("cmd") or args.get("raw") or ""
    command = str(command)
    if len(command) > MAX_ACTION_CHARS:
        command = command[:MAX_ACTION_CHARS] + " [clipped]"
    return command


def build_candidate(session_kind, session_key, user_turn, tool, args, index,
                   provenance="pending_consent", decision_source="scripted"):
    kind, verb = (PI_DECISION_TOOLS if session_kind == "pi" else CODEX_DECISION_NAMES).get(
        tool, (None, None))
    if kind is None:
        return None, "not_a_decision_tool"
    command = str(args.get("command") or args.get("cmd") or "") if isinstance(args, dict) else ""
    if READ_ONLY.match(command):
        return None, "read_only_command"
    if not user_turn:
        return None, "no_preceding_user_turn"

    task_text = clean(user_turn)
    # Paths are redacted, not treated as a drop, matching the redaction checklist in
    # review_kit/README.md: home directories reveal machine and account layout and are
    # replaced, while credentials and personal content are dropped outright below.
    target = clean(action_target(tool, args))
    description = f"{verb}: {target}" if target else verb
    description = " ".join(description.split())[:MAX_ACTION_CHARS]
    if not description:
        return None, "no_action_description"

    # The redaction gate runs on exactly the text that would be stored.
    for text in (task_text, description, target):
        reason = redact_reject_reason(text or "")
        if reason:
            return None, reason

    case_id = f"cand-{digest(session_key, index, tool)[:20]}"
    provenance_notes = (
        "LOCAL CANDIDATE collected from a local agent session log. Not reviewed, not "
        "labelled, and not consented. Requires the subject's explicit approval before any "
        "use, and two independent human reviewers before it may carry a label. Collected "
        "by review_kit/collect_from_sessions.py; nothing left the machine."
    )
    rehearsal = provenance == "scenario"
    if rehearsal:
        provenance_notes = (
            "SCRIPTED REHEARSAL SCENARIO. The user turn was written for the harness, so the "
            "grant is fictional and this is NOT a witnessed production incident and NOT a "
            "consented capture. The tool call was scripted, not decided by a model. Covers "
            "the Mode 1 outward-recipient shape. Requires the subject's explicit approval and "
            "two independent human reviewers before it may carry a label. Nothing left the "
            "machine: the mock tool writes to a local outbox."
        )
    flags = privacy_flags(f"{task_text} {description}")
    if flags:
        flag_note = (" FLAGGED FOR MANUAL REVIEW (privacy): "
                     + ", ".join(flags)
                     + ". The stored text may name a relationship or a third party. "
                       "That is a judgement about whose data this is, not a secret, so it is "
                       "flagged rather than dropped. Exclude from any shared batch until a "
                       "person clears it.")
    else:
        flag_note = ""
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "split": "dev",
        "group": {
            "session_id": f"{session_kind}-sess-{digest(session_key)[:12]}",
            "task_id": f"{session_kind}-task-{digest(session_key, index)[:12]}",
            "pair_group_id": None,
        },
        "stratum": "S6",
        "state": {
            "authorized_task": task_text,
            "permission": {"granted": False, "scope": [], "revoked": False, "source": "direct_user"},
            "events": [
                {"index": 0, "role": "user", "source": "direct_user", "content": task_text},
                {"index": 1, "role": "assistant", "source": "assistant",
                 "content": f"{session_kind} session proposed: {description}"},
            ],
            "proposed_action": {"kind": kind, "description": description,
                                "destination": target or "unspecified"},
            "context": {"history_complete": True, "omitted_events": 0, "truncation_reason": None},
        },
        "labels": None,
        "provenance": {
            "origin": provenance,
            "data_origin": provenance,
            "approval_state": "pending_consent",
            "consent_id": None,
            "redaction": "applied",
            "notes": provenance_notes + flag_note,
        },
    }
    case["provenance"]["manual_review"] = "privacy" if flags else "none"
    if flags:
        case["provenance"]["review_flags"] = flags
    # decision_source only distinguishes a rehearsal from a real agent decision, so it is
    # meaningful on a scenario and absent on a real capture.
    if rehearsal:
        case["provenance"]["decision_source"] = decision_source
    # annotations and adjudication are omitted rather than emitted empty. An empty list
    # reads to the schema as a reviewed case with too few labels; absence reads as a
    # candidate that has not been reviewed, which is what it is.
    case["stratum"] = "S6" if kind in ("send_message", "network_request", "write_file",
                                        "run_command", "delete", "publish") else "S5"
    return case, "kept"


def collect(pi_root, codex_root, max_sessions, limit, provenance="pending_consent",
            tool_names=None, decision_source="scripted"):
    stats = Counter()
    shapes = Counter()
    kept = []
    sources = []
    if pi_root and Path(pi_root).is_dir():
        sources += [("pi", p) for p in sorted(Path(pi_root).rglob("*.jsonl"))]
    if codex_root and Path(codex_root).is_dir():
        sources += [("codex", p) for p in sorted(Path(codex_root).rglob("*.jsonl"))]
    for kind, path in sources[:max_sessions]:
        session_key = str(path)
        reader = pi_turns if kind == "pi" else codex_turns
        try:
            for index, (user_turn, tool, args) in enumerate(reader(path)):
                stats["decision_points_seen"] += 1
                if tool_names and tool not in tool_names:
                    stats["tool_not_in_harness"] += 1
                    continue
                case, reason = build_candidate(kind, session_key, user_turn, tool, args, index,
                                               provenance=provenance,
                                               decision_source=decision_source)
                stats[reason] += 1
                if case:
                    if case["provenance"].get("review_flags"):
                        stats["flagged_for_manual_review"] += 1
                    shapes[case["state"]["proposed_action"]["kind"]] += 1
                    kept.append(case)
                    if limit and len(kept) >= limit:
                        return kept, stats, shapes, len(sources)
        except (OSError, UnicodeDecodeError):
            stats["unreadable_file"] += 1
    return kept, stats, shapes, len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pi-root", default=str(Path.home() / ".pi" / "agent" / "sessions"))
    parser.add_argument("--codex-root", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--out", type=Path, default=HERE / "draft_batch.jsonl")
    parser.add_argument("--max-sessions", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="stop after N candidates")
    parser.add_argument("--scenario-root", default=None,
                        help="collect from a scripted-scenario session root and tag every case "
                             "as origin=scenario, never as a real consented capture")
    parser.add_argument("--decision-source", default="scripted",
                        choices=["scripted", "model"],
                        help="scripted for a rehearsal whose tool call came from scenarios.json, "
                             "model for a real agent decision")
    parser.add_argument("--identity", action="append", default=[],
                        help="an extra identity string to drop on, repeatable")
    parser.add_argument("--tools", default=None,
                        help="comma-separated tool names to keep, for scenario roots")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts and write nothing")
    args = parser.parse_args()

    global IDENTITY_TOKENS
    if args.identity:
        IDENTITY_TOKENS = identity_tokens(args.identity)
    provenance = "pending_consent"
    pi_root, codex_root = args.pi_root, args.codex_root
    tools = None
    if args.scenario_root:
        provenance, pi_root, codex_root, tools = "scenario", args.scenario_root, "", {
            t.strip() for t in args.tools.split(",")} if args.tools else None
    if provenance == "scenario":
        decision_source = args.decision_source
    else:
        decision_source = "model"
    kept, stats, shapes, seen_files = collect(pi_root, codex_root, args.max_sessions,
                                              args.limit, provenance=provenance, tool_names=tools,
                                              decision_source=decision_source)
    print(f"LOCAL ONLY: read {seen_files} session logs under disk, wrote nothing else.",
          file=sys.stderr)
    print(f"  decision points seen : {stats['decision_points_seen']}", file=sys.stderr)
    dropped = {k: v for k, v in sorted(stats.items())
               if k not in ("decision_points_seen", "kept")}
    for reason, count in dropped.items():
        print(f"  dropped {reason:<28} {count}", file=sys.stderr)
    identity_drops = {k: v for k, v in stats.items() if k.startswith("identity:")}
    print(f"  kept candidates                {stats['kept']}", file=sys.stderr)
    print(f"  FLAGGED for manual review      {stats['flagged_for_manual_review']} "
          f"(privacy; excluded from any shared batch until cleared)", file=sys.stderr)
    if identity_drops:
        print("  dropped on identity strings   ", dict(identity_drops), file=sys.stderr)
    print(f"  identity tokens in force      {IDENTITY_TOKENS}", file=sys.stderr)
    print("  by action kind:", dict(shapes), file=sys.stderr)
    print(f"  all candidates are origin={provenance}, decision_source={decision_source}, "
          f"carry no label and no annotation.", file=sys.stderr)
    if provenance == "scenario":
        if decision_source == "scripted":
            print("  SCRIPTED REHEARSAL: the user turns are fictional and no tool call here was "
                  "decided by a model. Not consented, not witnessed.", file=sys.stderr)
        else:
            print("  REAL AGENT DECISION under a user instruction, but the user turn is a "
                  "scripted scenario, so the grant is fictional. Not consented, not witnessed.",
                  file=sys.stderr)

    if args.dry_run:
        print("  dry run: nothing written", file=sys.stderr)
        return 0
    with args.out.open("w") as handle:
        for case in kept:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    print(f"  wrote {len(kept)} candidates to {args.out}", file=sys.stderr)
    print("  NOTHING HAS LEFT THIS MACHINE. Review before copying anywhere.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
