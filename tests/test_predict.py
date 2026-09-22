"""Regressions for the review wrapper. No model weights or ML runtime needed."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict


def complete_state():
    return {
        "authorized_task": "Inspect the test report and describe its results.",
        "events": [
            {"role": "user", "source": "direct_user", "content": "Inspect the recorded test results. Do not run tests."},
            {"role": "tool", "source": "build_log", "content": "The last recorded suite passed."},
        ],
        "proposed_action": {"kind": "review_test_report", "description": "Summarize the existing test report."},
        "context": {"history_complete": True, "omitted_events": 0},
    }


def output(choice="allow", confidence=.95):
    remaining = (1 - confidence) / 2
    return {"answers": {
        "content": {"choice": "benign", "probabilities": {"benign": .9, "suspicious": .1}},
        "action": {"choice": choice, "probabilities": {
            label: confidence if label == choice else remaining
            for label in ("allow", "block", "review")}},
    }}


def agent(result=None):
    return SimpleNamespace(tok=object(), cfg={"max_len": 1024, "head_max_len": 192},
                           predict=Mock(return_value=output() if result is None else result))


class PredictTests(unittest.TestCase):
    def assert_schema_review(self, state):
        model = agent()
        with patch.object(predict, "session_windows") as window:
            result = predict.inspect(model, state)
        self.assertEqual(result["decision"], "review")
        model.predict.assert_not_called()
        window.assert_not_called()

    def test_empty_or_missing_authoritative_structure_never_calls_model(self):
        invalid = [None, [], {}, {"authorized_task": "  "}]
        for key in ("authorized_task", "events", "proposed_action", "context"):
            state = complete_state()
            del state[key]
            invalid.append(state)
        for key, value in (("events", []), ("events", "log"), ("proposed_action", {})):
            state = complete_state()
            state[key] = value
            invalid.append(state)
        for state in invalid:
            with self.subTest(state=state):
                self.assert_schema_review(state)

    def test_explicit_complete_history_attestation_is_required(self):
        for metadata in ({}, {"history_complete": False}, {"history_complete": 1},
                         {"history_complete": "true"}, None):
            with self.subTest(metadata=metadata):
                state = complete_state()
                state["context"] = metadata
                self.assert_schema_review(state)

    def test_invalid_event_source_role_or_content_never_calls_model(self):
        invalid = [None, "raw text", {},
                   {"role": "tool", "source": None, "content": "note"},
                   {"role": "tool", "source": 42, "content": "note"},
                   {"role": "invented_authority", "source": "file", "content": "note"},
                   {"role": "tool", "source": "file", "content": {"text": "note"}}]
        for event in invalid:
            with self.subTest(event=event):
                state = complete_state()
                state["events"].append(event)
                self.assert_schema_review(state)

    def test_nonfinite_or_invalid_probability_vectors_are_reviewed(self):
        vectors = [
            {"allow": float("nan"), "block": float("nan"), "review": float("nan")},
            {"allow": float("inf"), "block": 0., "review": 0.},
            {"allow": .9, "block": -.1, "review": .2},
            {"allow": .9, "block": .4, "review": .2},
            {"allow": "0.9", "block": .05, "review": .05},
            {"allow": .95, "block": .05},
        ]
        for probabilities in vectors:
            with self.subTest(probabilities=probabilities):
                result = output()
                result["answers"]["action"]["probabilities"] = probabilities
                model = agent(result)
                state = complete_state()
                with patch.object(predict, "session_windows", return_value=[state]):
                    self.assertEqual(predict.inspect(model, state)["decision"], "review")

    def test_choice_must_match_probability_maximum_and_known_schema(self):
        malformed = []
        result = output("block")
        result["answers"]["action"]["choice"] = "allow"
        malformed.append(result)
        result = output()
        result["answers"]["action"]["choice"] = "execute"
        malformed.append(result)
        result = output()
        result["answers"]["content"]["probabilities"]["benign"] = float("nan")
        malformed.append(result)
        for result in malformed:
            with self.subTest(result=result):
                state = complete_state()
                with patch.object(predict, "session_windows", return_value=[state]):
                    self.assertEqual(predict.inspect(agent(result), state)["decision"], "review")

    def test_complete_bounded_sessions_preserve_allow_and_block_judgments(self):
        for decision in ("allow", "block"):
            with self.subTest(decision=decision):
                model = agent(output(decision))
                state = complete_state()
                with patch.object(predict, "session_windows", return_value=[state]):
                    result = predict.inspect(model, state)
                self.assertEqual(result["decision"], decision)
                self.assertEqual(result["window_count"], 1)
                model.predict.assert_called_once_with(state, predict.QUESTIONS)

    def test_uncertain_model_result_and_multiple_windows_require_review(self):
        state = complete_state()
        with patch.object(predict, "session_windows", return_value=[state]):
            result = predict.inspect(agent(output(confidence=.7)), state)
        self.assertEqual(result["decision"], "review")
        model = agent(output("allow"))
        with patch.object(predict, "session_windows", return_value=[state, copy.deepcopy(state)]):
            result = predict.inspect(model, state)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(model.predict.call_count, 2)

    def test_context_overflow_or_model_runtime_failure_requires_review(self):
        state = complete_state()
        model = agent()
        with patch.object(predict, "session_windows", side_effect=ValueError("event too long")):
            self.assertEqual(predict.inspect(model, state)["decision"], "review")
        model.predict.assert_not_called()
        model.predict.side_effect = RuntimeError("inference failed")
        with patch.object(predict, "session_windows", return_value=[state]):
            self.assertEqual(predict.inspect(model, state)["decision"], "review")


if __name__ == "__main__":
    unittest.main()
