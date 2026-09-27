"""Guards on the Kaggle training entry point.

Two failures found by running the real kernel, neither visible locally:

  1. A `continue` in the evaluation loop silently dropped the 24-case diagnostic from
     report["finetuned"], so a published metric disappeared from the report while the
     code compiled and every unit test passed.
  2. The pipeline is not bit-reproducible. Two runs with the same seed and the same
     data produced different per-case decisions, so any published count needs a
     variance statement.

These are checked statically against the source because train_risk.py cannot be
executed without a GPU and the laya runtime.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_RISK = ROOT / "src" / "train_risk.py"
EVALUATED = ("calibration", "test_user", "test_attack", "test_both", "old_challenge")


class EvaluationLoopTests(unittest.TestCase):
    def setUp(self):
        self.source = TRAIN_RISK.read_text()
        self.tree = ast.parse(self.source)

    def _loop_body(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.For):
                continue
            target = ast.unparse(node.iter)
            if "calibration" in target and "old_challenge" in target:
                return node
        self.fail("evaluation loop over the evaluated splits not found")

    def test_every_evaluated_split_is_summarised_into_the_report(self):
        """A `continue` before the summary silently drops a published metric."""
        loop = self._loop_body()
        summarise = [ast.unparse(n) for n in ast.walk(loop)
                     if isinstance(n, ast.Assign)]
        summary_lines = [s for s in summarise if "final[" in s]
        self.assertTrue(summary_lines, "no split is being summarised")
        for name in EVALUATED:
            self.assertNotRegex(self.source, rf"if name == \"{name}\":\s*\n\s*continue",
                                f"{name} is skipped before being summarised")

    def test_no_early_continue_in_the_evaluation_loop(self):
        loop = self._loop_body()
        for node in ast.walk(loop):
            if isinstance(node, ast.Continue):
                self.fail("the evaluation loop still has a bare `continue`; it previously "
                          "skipped old_challenge and dropped a published metric")

    def test_per_case_predictions_are_written_for_every_split(self):
        self.assertIn('(OUT / f"predictions-{name}.json")', self.source)

    def test_split_names_are_the_evaluated_set(self):
        loop = self._loop_body()
        listed = ast.literal_eval(ast.unparse(loop.iter))
        self.assertEqual(sorted(listed), sorted(EVALUATED))

    def test_report_summarises_the_whole_final_mapping(self):
        """final is written into the report wholesale, so nothing is dropped downstream."""
        self.assertIn('"finetuned": final', self.source)


class ReproducibilityTests(unittest.TestCase):
    """The pipeline is seeded but not bit-reproducible. Pin the observation."""

    def test_no_deterministic_algorithms_flag_is_set(self):
        """The reason it is not reproducible, stated rather than implied."""
        source = TRAIN_RISK.read_text()
        self.assertNotIn("use_deterministic_algorithms", source)
        self.assertNotIn("cudnn.deterministic", source)
        for seed in ("random.seed(42)", "np.random.seed(42)", "torch.manual_seed(42)"):
            self.assertIn(seed, source, "seeding must stay explicit")

    def test_seed_is_still_42(self):
        self.assertIn("torch.manual_seed(42)", TRAIN_RISK.read_text())


if __name__ == "__main__":
    unittest.main()
