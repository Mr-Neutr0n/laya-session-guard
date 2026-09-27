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
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "private_key_block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    "openai_style_key": r"\bsk-[A-Za-z0-9]{20,}",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "bearer_header": r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}",
    "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
    "hex_blob": r"\b[0-9a-fA-F]{40,}\b",
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
    return None


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
    """Best-effort destination, without copying a whole command line."""
    if not isinstance(args, dict):
        args = {"raw": str(args)}
    for key in ("path", "file_path", "filePath", "target_file", "url", "recipient", "to"):
        if args.get(key):
            return str(args[key])[:120]
    command = args.get("command") or args.get("cmd") or args.get("raw") or ""
    command = str(command)
    if len(command) > MAX_ACTION_CHARS:
        command = command[:MAX_ACTION_CHARS] + " [clipped]"
    return command


def build_candidate(session_kind, session_key, user_turn, tool, args, index):
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
            "origin": "pending_consent",
            "data_origin": "pending_consent",
            "approval_state": "pending_consent",
            "consent_id": None,
            "redaction": "applied",
            "notes": provenance_notes,
        },
    }
    # annotations and adjudication are omitted rather than emitted empty. An empty list
    # reads to the schema as a reviewed case with too few labels; absence reads as a
    # candidate that has not been reviewed, which is what it is.
    case["stratum"] = "S6" if kind in ("send_message", "network_request", "write_file",
                                        "run_command", "delete", "publish") else "S5"
    return case, "kept"


def collect(pi_root, codex_root, max_sessions, limit):
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
                case, reason = build_candidate(kind, session_key, user_turn, tool, args, index)
                stats[reason] += 1
                if case:
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
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts and write nothing")
    args = parser.parse_args()

    kept, stats, shapes, seen_files = collect(args.pi_root, args.codex_root,
                                              args.max_sessions, args.limit)
    print(f"LOCAL ONLY: read {seen_files} session logs under disk, wrote nothing else.",
          file=sys.stderr)
    print(f"  decision points seen : {stats['decision_points_seen']}", file=sys.stderr)
    dropped = {k: v for k, v in sorted(stats.items())
               if k not in ("decision_points_seen", "kept")}
    for reason, count in dropped.items():
        print(f"  dropped {reason:<28} {count}", file=sys.stderr)
    print(f"  kept candidates                {stats['kept']}", file=sys.stderr)
    print("  by action kind:", dict(shapes), file=sys.stderr)
    print(f"  all candidates are origin=pending_consent, carry no label and no annotation.",
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
