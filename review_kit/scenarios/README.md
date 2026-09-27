# Mode 1 scenario harness

A local harness for the one failure shape the collector cannot find on its own: an
outward message to a recipient list the user never named.

**These are scripted rehearsal scenarios. They are not witnessed production incidents, and they
are not consented captures.** The user turn in every scenario was written in `scenarios.json`, so
the grant is fictional. Cases produced here are tagged `origin: scenario`, never
`pending_consent`, and the validator rejects a scenario that arrives with a label, annotations, or
a consent id.

## What is here

| File | Role |
| --- | --- |
| `mock_broadcast.py` | The mock `send_batch` tool. Appends to `outbox.jsonl` and **sends nothing**. No network client anywhere in it. |
| `scenarios.json` | Five scripted situations: one recipient proposed as many, a benign clarification with no injection, an explicit multi-recipient grant, a revoked grant, and a contact-list expansion. |
| `tool_prompt.txt` | The tool signature handed to the agent. |
| `run_scenarios.sh` | One command to run the scenarios through pi. **Read the warning inside it first.** |
| `emit_rehearsal_log.py` | Writes a pi-format session log from the scenarios with **no model call**, so the pipeline can be rehearsed for free. |

## Two ways to produce cases

**Rehearsal, free and local.** `emit_rehearsal_log.py` then the collector with
`--scenario-root`. The tool call is scripted, not decided by a model, so this exercises the
extraction, redaction, schema, and validator path and nothing about model behaviour.

**Real agent decisions, costs tokens.** `run_scenarios.sh` makes one real pi model call per
scenario on your provider account, and its sessions land in pi's session store alongside your real
work sessions. The runner's own warning says this. If you use it, collect with
`--scenario-root <dir>` so those sessions are tagged as scenarios and kept out of
`draft_batch.jsonl`.

## Coverage

The five scenarios cover all five Mode 1 features, which the 1,612 real local candidates do not:
outward direction, recipient expansion, per-item loop, scripted body, and enumeration source. The
real draft covers four of five and has zero per-item loop.

## Not committed

`rehearsal-sessions/`, `outbox.jsonl`, and `review_kit/draft_mode1.jsonl` are gitignored. The
harness is committed; its output is not.
