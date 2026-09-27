#!/bin/sh
# Run the scenarios through pi with the mock broadcast tool wired in.
#
# The tool is loaded for this run only, with --extension. Nothing is installed globally,
# no settings file is touched, and no message reaches any recipient: send_batch appends
# to scenarios/outbox.jsonl and returns a receipt.
#
# Each scenario costs one real model call on your pi provider account.
#
# Session logs are written to SCENARIO_SESSION_DIR so they do NOT land beside your real
# work sessions. Collect them afterwards with --scenario-root, which tags them as
# scenarios rather than as real captures.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SESSION_DIR=${SCENARIO_SESSION_DIR:-$HERE/real-sessions}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
command -v pi >/dev/null 2>&1 || { echo "pi not on PATH" >&2; exit 2; }
mkdir -p "$SESSION_DIR"

TOTAL=0
for id in $(python3 -c "
import json
d=json.load(open('$HERE/scenarios.json'))
print(' '.join(s['id'] for s in d['scenarios']))"); do
    TOTAL=$((TOTAL+1))
    prompt=$(python3 -c "
import json
d=json.load(open('$HERE/scenarios.json'))
print([x for x in d['scenarios'] if x['id']=='$id'][0]['user_turn'])")
    echo "--- scenario $TOTAL: $id" >&2
    SCENARIO_OUTBOX="$HERE/outbox.jsonl" \
    pi --print \
       --extension "$HERE/broadcast_extension.ts" \
       --no-builtin-tools \
       --session-dir "$SESSION_DIR" \
       --name "scenario-$id" \
       -- "$prompt" \
       > "$OUT/$id.txt" 2> "$OUT/$id.err" \
       || echo "  (pi exited non-zero; see below)" >&2
    echo "  reply: $(head -c 160 "$OUT/$id.txt" | tr '\n' ' ')" >&2
    [ -s "$OUT/$id.err" ] && head -3 "$OUT/$id.err" >&2
    echo "$id|$OUT/$id.txt" >> "$OUT/replies.txt"
done

echo "ran $TOTAL scenarios; sessions in $SESSION_DIR; outbox $HERE/outbox.jsonl" >&2
echo "collect with:" >&2
echo "  python3 review_kit/collect_from_sessions.py --scenario-root $SESSION_DIR --out review_kit/draft_mode1.jsonl" >&2
