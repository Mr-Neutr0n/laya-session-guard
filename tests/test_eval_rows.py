"""Tests for the per-case evaluation row instrumentation.

The point of this instrumentation is that a wrapper threshold can be fitted on
calibration instead of on the frozen test splits. These tests prove that every
evaluated case produces exactly one row carrying the fields a threshold fit needs.

The fixture in `test_fixture_rows_match_evaluated_cases` is derived from the real
downloaded artifacts in outputs/, not invented: the logits are copied from
outputs/risk/laya-risk-ranked/predictions-test_both.json and the challenge
probabilities from outputs/followup/followup-original/challenge.json. The test is
skipped if those files are absent rather than falling back to made-up numbers.
"""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_rows import (ROW_KEYS, logit_margin, probability_margin,  # noqa: E402
                       rows_from_challenge, rows_from_predictions,
                       rows_path, write_rows)

RISK_PREDICTIONS = ROOT / "outputs" / "risk" / "laya-risk-ranked" / "predictions-test_both.json"
FOLLOWUP_CHALLENGE = ROOT / "outputs" / "followup" / "followup-original" / "challenge.json"


def real_action_predictions():
    """Action-question rows from the real risk-ranked test_both predictions."""
    entries = json.loads(RISK_PREDICTIONS.read_text())
    return [e for e in entries if e["qid"] == "action"]


def real_challenge_predictions():
    return json.loads(FOLLOWUP_CHALLENGE.read_text())["predictions"]


class WriteRowsContractTests(unittest.TestCase):
    """`write_rows` returns a path. Callers must not treat that return value as the rows.

    Binding `rows = write_rows(...)` and then calling `len(rows)` passed every unit test
    in this repository and then failed the Kaggle run at its final line with
    "TypeError: object of type 'PosixPath' has no len()". The reason it survived locally is
    that ./verify compiles and unit-tests, and none of these entry points is ever executed
    without a GPU and the laya runtime, so the mistake was invisible until it cost a 39-minute
    run. These tests close that gap statically.
    """

    ENTRY_POINTS = ("train_risk.py", "train.py", "evaluate_challenge.py")

    def test_write_rows_returns_a_path_not_the_rows(self):
        rows = [{"case_id": "a", "split": "dev", "gold": "allow", "predicted": "allow",
                 "margin": 1.0, "margin_space": "logit"}]
        with tempfile.TemporaryDirectory() as tmp:
            returned = write_rows(Path(tmp) / "out.jsonl", rows)
        self.assertIsInstance(returned, Path)
        self.assertEqual(returned.name, "out.jsonl")

    def test_no_entry_point_measures_the_length_of_a_write_rows_return(self):
        for name in self.ENTRY_POINTS:
            with self.subTest(script=name):
                path = ROOT / "src" / name
                tree = ast.parse(path.read_text())
                bound = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                        func = node.value.func
                        called = getattr(func, "id", None) or getattr(func, "attr", None)
                        if called == "write_rows":
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    bound[target.id] = node.lineno
                offenders = []
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "len" and node.args
                            and isinstance(node.args[0], ast.Name)
                            and node.args[0].id in bound):
                        offenders.append((node.lineno, node.args[0].id, bound[node.args[0].id]))
                self.assertEqual(offenders, [], f"{name}: len() applied to a write_rows "
                                                f"return bound at lines {bound}")

    def test_evaluate_challenge_separates_the_two_row_variables(self):
        """A source-level guard on the exact defect, kept alongside the AST check.

        The file has a pre-existing local named `rows` that holds the input cases, and
        len(rows) on that is correct, so this asserts on the new names rather than on the
        bare string.
        """
        source = (ROOT / "src" / "evaluate_challenge.py").read_text()
        self.assertIn("challenge_rows = rows_from_challenge(", source)
        self.assertIn("path = write_rows(", source)
        self.assertIn("len(challenge_rows)", source)


class MarginTests(unittest.TestCase):
    def test_logit_margin_matches_hand_computation(self):
        # allow dominates, so the margin is large and positive.
        self.assertAlmostEqual(logit_margin([5.0, 1.0, 1.0]), 5.0 - 1.6931471805599454)
        # block dominates, so the margin is negative.
        self.assertLess(logit_margin([1.0, 5.0, 1.0]), 0)
        # near tie.
        self.assertAlmostEqual(logit_margin([0.0, 0.0, 0.0]), -0.6931471805599454, places=6)

    def test_logit_margin_is_scale_equivariant(self):
        """Temperature scaling shifts the logit gap; the relative ordering is what matters."""
        a, b = [3.0, 1.0, 0.5], [3.0, 2.9, 2.8]
        self.assertGreater(logit_margin(a), logit_margin(b))

    def test_probability_margin_sign_convention_matches_logit(self):
        self.assertAlmostEqual(probability_margin([0.7, 0.2, 0.1]), 0.5)
        self.assertLess(probability_margin([0.2, 0.5, 0.3]), 0)

    def test_margin_is_none_when_there_is_no_alternative(self):
        self.assertIsNone(logit_margin([1.0]))
        self.assertIsNone(probability_margin([1.0]))


class PredictionRowTests(unittest.TestCase):
    def setUp(self):
        if not RISK_PREDICTIONS.exists():
            self.skipTest(f"missing real artifact {RISK_PREDICTIONS}")
        self.entries = real_action_predictions()
        self.rows = rows_from_predictions(self.entries, "test_both", model="risk-ranked")

    def test_one_row_per_evaluated_action_case(self):
        self.assertEqual(len(self.rows), len(self.entries))
        self.assertEqual(len({r["case_id"] for r in self.rows}), len(self.entries))

    def test_every_row_has_the_required_keys(self):
        for row in self.rows:
            for key in ROW_KEYS:
                self.assertIn(key, row, f"{row.get('case_id')} missing {key}")
            self.assertIn(row["split"], {"dev", "calib", "test", "test_user", "test_attack",
                                         "test_both", "old_challenge", "challenge"})

    def test_gold_and_predicted_come_from_the_evaluation_not_from_inference(self):
        for entry, row in zip(self.entries, self.rows):
            self.assertEqual(row["case_id"], entry["id"])
            self.assertEqual(row["logits"], list(entry["logits"]))
            expected = max(range(len(entry["logits"])), key=lambda i: entry["logits"][i])
            self.assertEqual(row["predicted"], ("allow", "block", "review")[expected])
            self.assertEqual(row["margin_space"], "logit")
            self.assertEqual(row["model"], "risk-ranked")

    def test_content_question_rows_are_not_emitted(self):
        """A threshold is fitted on the action decision, so content rows are excluded.

        Every case is evaluated on both questions under the same case_id, so the
        discriminating test is the row count against the action-entry count, not
        disjointness of the id sets.
        """
        all_entries = json.loads(RISK_PREDICTIONS.read_text())
        action_entries = [e for e in all_entries if e["qid"] == "action"]
        self.assertGreater(len(action_entries), 0)
        self.assertEqual(len(all_entries), 2 * len(action_entries),
                         "fixture should hold one content and one action row per case")
        self.assertEqual(len(self.rows), len(action_entries),
                         "content rows must not be emitted")
        for entry, row in zip(action_entries, self.rows):
            self.assertEqual(entry["qid"], "action")
            self.assertEqual(row["case_id"], entry["id"])

    def test_margin_is_finite_and_sign_agrees_with_the_prediction(self):
        for row in self.rows:
            self.assertIsInstance(row["margin"], float)
            self.assertEqual(row["margin"] > 0, row["predicted"] == "allow")

    def test_write_rows_produces_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_rows(Path(tmp) / "calibration_predictions.jsonl", self.rows)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), len(self.rows))
            for line in lines:
                parsed = json.loads(line)
                for key in ROW_KEYS:
                    self.assertIn(key, parsed)

    def test_rows_path_names_the_calibration_file_as_agreed(self):
        self.assertEqual(rows_path("/out", "calibration").name,
                         "calibration_predictions.jsonl")
        self.assertEqual(rows_path(Path("/out"), "test_attack").parent, Path("/out"))


class ChallengeRowTests(unittest.TestCase):
    def setUp(self):
        if not FOLLOWUP_CHALLENGE.exists():
            self.skipTest(f"missing real artifact {FOLLOWUP_CHALLENGE}")
        self.entries = real_challenge_predictions()
        self.rows = rows_from_challenge(self.entries, model="outputs/followup/followup-original")

    def test_one_row_per_challenge_case(self):
        self.assertEqual(len(self.rows), len(self.entries))

    def test_challenge_rows_use_probability_space_and_carry_the_wrapper(self):
        for row, entry in zip(self.rows, self.entries):
            self.assertEqual(row["margin_space"], "probability")
            self.assertIn("wrapper_decision", row)
            self.assertEqual(row["wrapper_decision"], entry["wrapper_decision"])
            self.assertEqual(row["gold"], entry["gold"]["action"])
            self.assertEqual(row["predicted"], entry["answers"]["action"]["choice"])

    def test_probability_margin_agrees_with_the_sign_of_the_choice(self):
        for row in self.rows:
            if row["predicted"] == "allow":
                self.assertGreaterEqual(row["margin"], 0)
            else:
                self.assertLess(row["margin"], 0)

    def test_probabilities_are_recorded_in_allow_block_review_order(self):
        for row, entry in zip(self.rows, self.entries):
            expected = [entry["answers"]["action"]["probabilities"][label]
                        for label in ("allow", "block", "review")]
            self.assertEqual(row["probabilities"], expected)


class WrapperPassThroughTests(unittest.TestCase):
    def test_wrapper_decision_is_attached_when_supplied(self):
        if not RISK_PREDICTIONS.exists():
            self.skipTest("missing real artifact")
        entries = real_action_predictions()
        mapping = {e["id"]: "review" for e in entries}
        rows = rows_from_predictions(entries, "calibration", wrapper=mapping)
        self.assertTrue(all(r["wrapper_decision"] == "review" for r in rows))

    def test_wrapper_key_is_absent_when_not_supplied(self):
        if not RISK_PREDICTIONS.exists():
            self.skipTest("missing real artifact")
        rows = rows_from_predictions(real_action_predictions(), "calibration")
        self.assertTrue(all("wrapper_decision" not in r for r in rows))


if __name__ == "__main__":
    unittest.main()
