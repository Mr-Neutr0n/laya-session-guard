"""Write a pi-format session log from the scripted scenarios, with no model call.

This is a PIPELINE REHEARSAL. The user turn and the tool call are both read from
scenarios.json, so nothing here is an agent decision and nothing is a real request. It
exists so the harness and the collector can be tested end to end without spending API
tokens or installing anything into your pi configuration.

The output is written to a directory the collector reads with --scenario-root, so it is
tagged origin=scenario and can never be confused with a real captured session.

Usage:
    python3 review_kit/scenarios/emit_rehearsal_log.py
"""

import json
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mock_broadcast import OUTBOX, send_batch  # noqa: E402


def main():
    spec = json.loads((HERE / "scenarios.json").read_text())
    out_dir = HERE / "rehearsal-sessions"
    out_dir.mkdir(exist_ok=True)
    if OUTBOX.exists():
        OUTBOX.unlink()
    written = []
    for scenario in spec["scenarios"]:
        session_id = f"rehearsal-{scenario['id']}"
        log = out_dir / f"{session_id}.jsonl"
        stamp = time.time()
        records = [
            {"type": "session", "id": session_id, "version": "rehearsal",
             "timestamp": stamp, "cwd": str(HERE)},
            {"type": "message", "id": str(uuid.uuid4()), "parentId": None, "timestamp": stamp,
             "message": {"role": "user", "timestamp": int(stamp * 1000),
                         "content": [{"type": "text", "text": scenario["user_turn"]}]}},
        ]
        # The tool call is scripted, not decided. Say so in the transcript.
        records.append({
            "type": "message", "id": str(uuid.uuid4()), "parentId": None, "timestamp": stamp,
            "message": {"role": "assistant", "timestamp": int(stamp * 1000), "content": [
                {"type": "thinking",
                 "text": "REHEARSAL: this tool call is scripted from scenarios.json, not "
                         "chosen by a model. It exists to exercise the Mode 1 shape."},
                {"type": "toolCall", "id": str(uuid.uuid4()), "name": scenario["tool"],
                 "arguments": scenario["arguments"]}]}})
        with log.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        receipt = send_batch(**scenario["arguments"])
        written.append((session_id, receipt["count"]))
    print(f"wrote {len(written)} rehearsal session logs to {out_dir}")
    for session_id, count in written:
        print(f"  {session_id:<34} {count} recipients queued to the local outbox")
    print(f"  outbox: {OUTBOX}  (nothing was sent to any recipient)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
