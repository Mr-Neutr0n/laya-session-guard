"""Export the exact router prompt as a standalone artifact.

The trained checkpoint is only reproducible with the prompt it was trained on:
the eight criteria carry curated descriptions, and replacing them with bare
category codes changes the model's behaviour. This writes the prompt, the
context budget and the label order to one JSON file so the Hub repository is
self-contained and the model card snippet can be checked against it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_router_data import LABELS
from router_task import INSTRUCTIONS, router_question

OUTPUT = Path(__file__).resolve().parents[1] / "router_prompt.json"


def prompt_artifact():
    question = router_question(LABELS)
    return {
        "name": "laya-arxiv-topic-router",
        "base_model": "convaiinnovations/laya",
        "base_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "task": "Eight-way arXiv subject-area routing from title and truncated abstract.",
        "input": "A state with exactly two fields, title and abstract. The abstract is "
                 "truncated to its first 100 words. No category, licence, author, date or "
                 "identifier may be present.",
        "labels": list(LABELS),
        "question": {
            "type": question["t"],
            "instructions": INSTRUCTIONS,
            "criteria": {label: question["crit"][label] for label in LABELS},
        },
        "budget": {"max_len": 640, "head_max_len": 256, "abstract_words": 100},
        "usage_note": ("The criteria descriptions are load-bearing. A caller that replaces them "
                       "with bare category codes, as an earlier version of the model card did, "
                       "gets materially worse answers from the same weights."),
    }


if __name__ == "__main__":
    print(json.dumps(prompt_artifact(), indent=2, sort_keys=True) + "\n")
