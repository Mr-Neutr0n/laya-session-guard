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

    def test_determinism_flags_are_set(self):
        """Set because two same-seed runs diverged. Assert the fix, not the absence."""
        source = TRAIN_RISK.read_text()
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=True)", source)
        self.assertIn("torch.backends.cudnn.deterministic = True", source)
        self.assertIn("torch.backends.cudnn.benchmark = False", source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", source,
                      "cuBLAS workspace config must be set before CUDA initialises")
        for seed in ("random.seed(42)", "np.random.seed(42)", "torch.manual_seed(42)"):
            self.assertIn(seed, source, "seeding must stay explicit")

    def test_determinism_caveats_are_documented_not_oversold(self):
        """Determinism is not bit-identity across stacks, and the code must say so."""
        source = TRAIN_RISK.read_text()
        for caveat in ("warn_only", "fp16", "same GPU, driver"):
            self.assertIn(caveat, source, f"caveat missing: {caveat}")

    def test_run_records_its_determinism_configuration(self):
        source = TRAIN_RISK.read_text()
        self.assertIn('"determinism": {', source)
        self.assertIn('"caveat"', source)

    def test_seed_is_still_42(self):
        self.assertIn("torch.manual_seed(42)", TRAIN_RISK.read_text())


class NotebookSummaryTests(unittest.TestCase):
    """The notebook's prose must not assert per-run numbers.

    Hard-coded counts are why this write-up contradicted its own output twice: once by
    quoting a single run, once by quoting a two-run range that a third run fell outside.
    The fix is that the run generates its own table and the prose carries only
    observations about repeatability, explicitly not bounds on any run.
    """

    def setUp(self):
        import json
        self.notebook = ROOT / "kaggle" / "risk" / "risk.ipynb"
        if not self.notebook.exists():
            self.skipTest("notebook not generated")
        self.cells = json.loads(self.notebook.read_text())["cells"]

    def test_markdown_declares_that_its_numbers_are_generated(self):
        markdown = "".join(self.cells[0]["source"])
        self.assertIn("No per-run count appears in this prose", markdown)

    def test_markdown_does_not_claim_a_range_bounds_any_run(self):
        markdown = "".join(self.cells[0]["source"])
        self.assertIn("is **not a bound on this or any future run**", markdown)
        self.assertIn("before determinism was enabled", markdown,
                      "the observed spread must be attributed to the pre-fix runs")

    def test_markdown_states_the_verdict_is_unaffected_in_every_run(self):
        markdown = "".join(self.cells[0]["source"])
        self.assertIn("verdict is unaffected in every run", markdown)
        self.assertIn("REJECTED FOR DEPLOYMENT" if "REJECTED FOR DEPLOYMENT" in markdown
                      else "rejected for deployment", markdown)

    def test_notebook_has_a_cell_that_generates_the_table_from_the_report(self):
        code = "".join("".join(c["source"]) for c in self.cells if c["cell_type"] == "code")
        self.assertIn("EXECUTED RUN SUMMARY", code)
        self.assertIn("report.json", code)
        self.assertIn("false_allow", code)
        self.assertIn("determinism", code)

    def test_summary_cell_runs_after_training(self):
        """Otherwise it would render the previous run's numbers, or none."""
        sources = ["".join(c["source"]) for c in self.cells if c["cell_type"] == "code"]
        train = next(i for i, s in enumerate(sources) if "train_risk.py" in s)
        summary = next(i for i, s in enumerate(sources) if "EXECUTED RUN SUMMARY" in s)
        self.assertGreater(summary, train, "summary cell must run after the training cell")

    def test_every_code_cell_is_valid_python(self):
        """A generated cell with a syntax error costs a 35-minute GPU run.

        v6 went to ERROR on exactly this: an apostrophe in a print statement in the
        summary cell made the emitted cell source an unterminated string. Nothing local
        executed the notebook, so nothing local noticed.
        """
        import ast
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell["source"]))
        self.assertGreaterEqual(len([c for c in self.cells if c["cell_type"] == "code"]), 2)

    def test_generated_cells_contain_no_escaped_quotes(self):
        """The generator emits cell source verbatim, so an escape here becomes a bug there."""
        for cell in self.cells:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            self.assertNotIn("\\'", source, f"escaped quote in a generated cell: {source[:80]!r}")

    def test_markdown_carries_no_hard_coded_per_run_false_allow_total(self):
        """A bare 'allowed N of M' is the failure mode; the spread sentence is allowed."""
        markdown = "".join(self.cells[0]["source"])
        import re
        # any occurrence must sit inside the explicitly-attributed spread sentence
        spread = "66, 58, and 33 of 176"
        stripped = markdown.replace(spread, "").replace(
            "of 66, 58, and 33 of 176", "")
        self.assertNotRegex(stripped, r"allowed (?:between )?\d+ of \d+",
                            "markdown still asserts a per-run false-allow count")


if __name__ == "__main__":
    unittest.main()
