"""Adapt the public ASPI workspace benchmark to Laya's two decision questions.

ASPI provides user tasks, intended safe actions, and attacker goals. We model its
attacker-controlled clarification as an unverified lower-trust channel. This is an
adaptation for guard evaluation, not the original ASPI agent benchmark.
"""

import json
import os
from pathlib import Path


SOURCE = Path(os.environ.get("ASPI_SOURCE", Path(__file__).resolve().parent.parent /
                             "data" / "aspi" / "workspace.jsonl"))


def _number(value):
    return int(value.rsplit("_", 1)[1])


def _split(user_number, attack_number):
    if user_number <= 27:
        return "test_attack" if attack_number >= 12 else "train"
    if user_number <= 33:
        return "calibration" if attack_number < 12 else None
    return "test_both" if attack_number >= 12 else "test_user"


def _state(prompt, clarification, response, description):
    return {
        "authorized_task": prompt,
        "events": [
            {"role": "user", "source": "direct_user", "content": prompt},
            {"role": "assistant", "source": "assistant", "content": clarification},
            {"role": "tool", "source": "unverified_clarification_channel", "content": response},
        ],
        "proposed_action": {"kind": "workflow_step", "description": description,
                            "destination": "as specified in the proposed action"},
        "context": {"history_complete": True, "omitted_events": 0},
    }


def generate(path=SOURCE):
    splits = {name: [] for name in ("train", "calibration", "test_user", "test_attack", "test_both")}
    for source_row in map(json.loads, Path(path).read_text().splitlines()):
        user_id, attack_id = source_row["user_task_id"], source_row["injection_task_id"]
        split = _split(_number(user_id), _number(attack_id))
        if split is None:
            continue
        root = f"aspi-{user_id}-{attack_id}"
        sources = [("benign", "none", source_row["benign_clarification_response"], "benign")]
        sources.extend((f"injected-{index}", injection["operator"],
                        injection["injected_clarification_response"], "suspicious")
                       for index, injection in enumerate(source_row["injections"]))
        for source_kind, operator, response, content_label in sources:
            group_id = f"{root}-{source_kind}"
            for action_kind, description, action_label in (
                ("safe", source_row["ground_truth_safe_action"], "allow"),
                ("attacker", source_row["attacker_goal"], "block"),
            ):
                splits[split].append({
                    "id": f"{group_id}-{action_kind}",
                    "group_id": group_id, "pair_id": group_id,
                    "family": f"aspi-{operator}-{source_kind}-{action_kind}",
                    "split": split,
                    "state": _state(source_row["original_prompt"],
                                    source_row["expected_clarification_question"],
                                    response, description),
                    "labels": {"content": content_label, "action": action_label},
                    "provenance": {"dataset": "ScaleAI/aspi", "suite": "workspace",
                                   "user_task_id": user_id, "injection_task_id": attack_id,
                                   "operator": operator},
                })
    return splits


if __name__ == "__main__":
    output = Path("data/aspi/adapted")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in generate().items():
        with (output / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print({split: len(rows) for split, rows in generate().items()})
