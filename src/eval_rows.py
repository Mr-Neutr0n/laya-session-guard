"""Per-case evaluation rows, so an operating point can be fitted honestly.

Every evaluation writes aggregate metrics and throws the per-case decisions away.
That is why a wrapper threshold could not be fitted on the risk-ranked run's
calibration split: `report.json` kept seven summary counts and the per-case logits
were gone, so the only way to pick a threshold was to read the frozen test splits
and read optimism into the result. Half of the apparent gain turned out to be
selection bias.

This module emits one JSONL row per evaluated case, containing the fields a
threshold fit needs: the case id, the split it belongs to, the gold decision, the
predicted decision, the raw allow-versus-alternative margin, and the wrapper
decision where a policy wrapper produced one.

The margin is reported with an explicit `margin_space`. Logit margins and
probability margins are not interchangeable, and comparing them across splits
produced an unstable threshold, so the space is recorded on every row rather than
implied by the reader.
"""

import json
import math
from pathlib import Path

ACTION_LABELS = ("allow", "block", "review")
ROW_KEYS = ("case_id", "split", "gold", "predicted", "margin", "margin_space")


def logit_margin(logits):
    """Allow logit minus the log-sum-exp of the alternatives. Positive favours allow."""
    if len(logits) < 2:
        return None
    top = max(logits[1:])
    return logits[0] - (top + math.log(sum(math.exp(z - top) for z in logits[1:])))


def probability_margin(probabilities):
    """Allow probability minus the best alternative. Same sign convention."""
    if len(probabilities) < 2:
        return None
    return probabilities[0] - max(probabilities[1:])


def rows_from_predictions(predictions, split, model="", wrapper=None):
    """Build rows from `train.evaluate` output, which carries raw logits per question.

    `predictions` entries have keys id, qid, label, and logits. Only the action
    question is emitted, because a threshold on the allow decision is the thing
    that has to be fitted.
    """
    rows = []
    for entry in predictions:
        if entry.get("qid") != "action":
            continue
        logits = list(entry["logits"])
        label = int(entry["label"])
        predicted = max(range(len(logits)), key=lambda i: logits[i])
        row = {
            "case_id": entry["id"],
            "split": split,
            "model": model,
            "gold": ACTION_LABELS[label] if label < len(ACTION_LABELS) else str(label),
            "predicted": ACTION_LABELS[predicted] if predicted < len(ACTION_LABELS) else str(predicted),
            "logits": logits,
            "margin": logit_margin(logits),
            "margin_space": "logit",
        }
        if wrapper is not None:
            row["wrapper_decision"] = wrapper.get(entry["id"])
        rows.append(row)
    return rows


def rows_from_challenge(predictions, split="challenge", model=""):
    """Build rows from the SDK challenge evaluation, which carries probabilities.

    The challenge path goes through the real SDK, so it has probabilities and a
    wrapper decision but no raw logits. The margin is therefore in probability
    space and is labelled as such.
    """
    rows = []
    for entry in predictions:
        answer = entry["answers"]["action"]
        probabilities = [answer["probabilities"][label] for label in ACTION_LABELS
                         if label in answer["probabilities"]]
        gold = entry["gold"]["action"]
        rows.append({
            "case_id": entry["id"],
            "split": split,
            "model": model,
            "gold": gold,
            "predicted": answer["choice"],
            "probabilities": probabilities,
            "margin": probability_margin(probabilities),
            "margin_space": "probability",
            "wrapper_decision": entry.get("wrapper_decision"),
        })
    return rows


def write_rows(path, rows):
    """Write one row per case as JSONL. Returns the path so callers can log it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def rows_path(out_dir, split):
    """`calibration_predictions.jsonl` next to the other saves for a run."""
    return Path(out_dir) / f"{split}_predictions.jsonl"
