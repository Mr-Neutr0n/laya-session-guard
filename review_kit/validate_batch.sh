#!/bin/sh
# Validate a batch of human-review cases and explain what to fix.
#
# Usage:
#   review_kit/validate_batch.sh path/to/cases.jsonl [more.jsonl ...]
#   review_kit/validate_batch.sh data/review/examples/clean.jsonl --min-test-per-stratum 0
#
# Every extra argument is passed through to scripts/validate_review_set.py.
#
# Exit status is the validator's, and it is preserved deliberately. An earlier
# version of a gate in this repo piped the validator into another command, and
# under `set -e` a pipeline reports only its last status, so a failed validation
# passed silently. This script never pipes the validator and never masks its
# status. Do not add a pipe here.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
VALIDATOR="$ROOT/scripts/validate_review_set.py"

if [ "$#" -lt 1 ]; then
    cat >&2 <<'USAGE'
usage: review_kit/validate_batch.sh <cases.jsonl> [more.jsonl ...] [validator options]

Pass every split in one invocation. Leakage is a cross-split property, so
validating dev, calib, and test separately cannot detect a case that appears in
more than one of them.
USAGE
    exit 2
fi

case "$1" in
    -h|--help)
        exec python3 "$VALIDATOR" --help
        ;;
esac

if [ ! -f "$VALIDATOR" ]; then
    echo "error: validator not found at $VALIDATOR" >&2
    exit 2
fi

status=0
python3 "$VALIDATOR" "$@" || status=$?

if [ "$status" -eq 0 ]; then
    echo
    echo "Batch accepted. No errors. Warnings, if any, are coverage notes and do not block."
    exit 0
fi

# Actionable follow-up for the codes a human will actually hit first.
cat >&2 <<'GUIDANCE'

The batch was rejected. Most common causes, in the order they are worth checking:

  REPLACE-in-the-required-key
      A required field is still the placeholder from review_kit/case_template.jsonl.
      Grep your batch for REPLACE- and fix every hit before resubmitting.

  consent.missing / consent.redaction
      A real capture needs a real consent id and redaction marked applied. A case
      cannot be stored with the raw capture, only with the redacted one.

  redaction.credential / redaction.email
      An unredacted secret or a real address is still in the text. Remove the
      original, do not just mask it in the case file.

  annotation.duplicate_reviewer / annotation.blank_reviewer
      Two distinct humans, and not a blank reviewer id. A second pass by one person
      is not a second opinion.

  annotation.disagreement / annotation.adjudicator
      Reviewers disagreed and the case still says unanimous, or the adjudicator is
      one of the two annotators. A third human who was not one of them decides.

  context.truncation_reason / context.omitted_events
      history_complete and omitted_events disagree. If the history is truncated say
      how, and count the omitted events. If it is complete, the count must be zero.

  leakage.session_id / leakage.task_id / leakage.pair_group_id / leakage.text
      A group appears in more than one split. Move the whole group to one split, or
      give the second copy a new session.

  permission.source / permission.scope / permission.revoked
      The recorded grant cannot support the decision. Only a direct_user turn can
      grant anything, a grant with an empty scope authorizes nothing, and a revoked
      grant cannot be labelled allow.

  shortcut.review_confounded
      Every review case in dev and calib shares one history_complete value. Add
      review cases at the other value, or include incomplete cases that are
      decisively allow or block.

  shortcut.incomplete_is_review
      Every incomplete case is review, with no counterexample. That is the shortcut
      the rejected model learned. Add incomplete cases labelled allow or block.

  coverage.test_stratum (warning, does not block)
      A priority stratum is thin in the sealed test split. Worth fixing before a
      training run, not before storing the batch.
GUIDANCE

exit "$status"
