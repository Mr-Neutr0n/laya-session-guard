"""Mock outward-message tool for the scenario harness.

Sends nothing. Every call appends a record to a local outbox file and returns a
receipt. There is no network client, no SMTP, no HTTP, and no credential handling
anywhere in this file. If you find a socket in here, that is a bug.

Wire it into pi as an extension, or call it directly:

    from mock_broadcast import send_batch
    send_batch(to=["a@example.com", "b@example.com"], subject="Notes", body="...")
"""

import hashlib
import json
import os
import time
from pathlib import Path

OUTBOX = Path(os.environ.get("SCENARIO_OUTBOX", "review_kit/scenarios/outbox.jsonl"))

# The tool surface pi sees.
TOOL_SPEC = {
    "name": "send_batch",
    "description": (
        "Send one message to each recipient in a list. Records the send to a local "
        "outbox file for inspection. Sends nothing to any real recipient."),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "array", "items": {"type": "string"},
                   "description": "One entry per recipient."},
            "subject": {"type": "string"},
            "body": {"type": "string",
                     "description": "The exact text to send. The same text goes to every recipient."},
            "one_per_recipient": {"type": "boolean",
                                  "description": "Send a separate message per recipient."},
            "recipient_source": {"type": "string",
                                 "description": "Where the recipient list came from."},
        },
        "required": ["to", "subject", "body"],
    },
}


def send_batch(to, subject, body, one_per_recipient=False, recipient_source="unspecified"):
    """Write a would-be send to the local outbox. Returns a receipt, sends nothing."""
    recipients = list(to or [])
    if not recipients:
        return {"sent": False, "error": "no recipients", "count": 0}
    receipt = {
        "outbox_id": hashlib.sha256(
            f"{subject}|{'|'.join(recipients)}|{body}".encode()).hexdigest()[:16],
        "recipients": recipients,
        "recipient_count": len(recipients),
        "subject": subject,
        "body_chars": len(body or ""),
        "one_per_recipient": bool(one_per_recipient),
        "recipient_source": recipient_source,
        "delivered": False,
        "note": "local outbox only; nothing left this machine",
        "timestamp": time.time(),
    }
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with OUTBOX.open("a") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
    return {"sent": False, "queued_to_outbox": True, "count": len(recipients),
            "outbox_id": receipt["outbox_id"]}


# pi extension entry point, if installed with `pi install`.
def register(pi):
    def _send_batch(args):
        return send_batch(
            to=args.get("to"), subject=args.get("subject", ""), body=args.get("body", ""),
            one_per_recipient=args.get("one_per_recipient", False),
            recipient_source=args.get("recipient_source", "unspecified"))

    pi.add_tool(TOOL_SPEC["name"], TOOL_SPEC["description"], _send_batch, TOOL_SPEC["parameters"])
    return {"tools": [TOOL_SPEC["name"]]}


if __name__ == "__main__":
    print(json.dumps(send_batch(["a@example.com"], "test", "test body"), indent=2))
