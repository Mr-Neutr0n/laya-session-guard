"""Tests for the temperature-grid patch.

src/train.py is not modified: it carries uncommitted user work excluded from this
branch by request, so the correction ships as patches/temperature-grid.patch. These
tests keep the patch honest, because a patch that silently stops applying is worse
than no patch.
"""

import ast
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "src" / "train.py"
PATCH = ROOT / "patches" / "temperature-grid.patch"


def applied_copy():
    """Return (src/train.py with the patch applied, temp dir to clean up)."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").mkdir()
    shutil.copy(TRAIN, tmp / "src" / "train.py")
    result = subprocess.run(["git", "apply", str(PATCH)], cwd=tmp,
                            capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise AssertionError(f"patch did not apply: {result.stderr}")
    return (tmp / "src" / "train.py").read_text(), tmp


class PatchTests(unittest.TestCase):
    def test_patch_applies_cleanly_to_the_current_train_py(self):
        """Fails loudly if train.py drifts and the patch goes stale."""
        result = subprocess.run(["git", "apply", "--check", str(PATCH)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"patch is stale against src/train.py: {result.stderr}")

    def test_patched_source_parses_and_keeps_the_optional_argument(self):
        source, tmp = applied_copy()
        try:
            ast.parse(source)
            node = next(n for n in ast.walk(ast.parse(source))
                        if isinstance(n, ast.FunctionDef) and n.name == "fit_temperatures")
            self.assertEqual([a.arg for a in node.args.args], ["rows", "diagnostics"])
            self.assertEqual(len(node.args.defaults), 1, "diagnostics must be optional")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_patched_grid_is_log_spaced_and_reaches_below_the_old_floor(self):
        source, tmp = applied_copy()
        try:
            self.assertIn("np.geomspace(0.05, 20.0, 401)", source)
            self.assertNotIn("torch.linspace(.5,5.,181)", source)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_patched_fit_records_edge_hits(self):
        source, tmp = applied_copy()
        try:
            for marker in ('"edge"', '"truncated": edge is not None', "grid_index"):
                self.assertIn(marker, source, f"edge recording lost: {marker}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_working_tree_train_py_is_untouched(self):
        """The patch exists precisely because this file must not be modified."""
        result = subprocess.run(["git", "diff", "--stat", "--", "src/train.py"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotIn("fit_temperatures", result.stdout)

    def test_patch_is_documented_as_unapplied(self):
        text = PATCH.read_text()
        self.assertIn("Not applied to src/train.py", text)
        self.assertIn("git apply patches/temperature-grid.patch", text)


class GridBehaviourTests(unittest.TestCase):
    """Why the change matters, demonstrated on the stored logits that do exist."""

    def setUp(self):
        path = ROOT / "outputs" / "laya-session-guard" / "predictions.json"
        if not path.exists():
            self.skipTest("pilot predictions unavailable")
        data = json.loads(path.read_text())
        self.finetuned = [(r["label"], r["logits"])
                          for r in data["finetuned"] if r["qid"] == "action"]

    def test_narrow_and_wide_grids_disagree_on_a_stored_split(self):
        import math

        def fit(rows, grid):
            def ce(logits, gold, t):
                z = [v / t for v in logits]
                top = max(z)
                e = [math.exp(v - top) for v in z]
                s = sum(e)
                return -math.log(max(e[gold] / s, 1e-12))
            losses = [sum(ce(l, g, t) for g, l in rows) / len(rows) for t in grid]
            return float(grid[int(min(range(len(losses)), key=lambda i: losses[i]))])

        narrow = [0.5 + 4.5 * i / 180 for i in range(181)]
        wide = [0.05 * (20.0 / 0.05) ** (i / 400) for i in range(401)]
        self.assertNotAlmostEqual(fit(self.finetuned, narrow), fit(self.finetuned, wide), places=3)

    def test_edge_markers_are_load_bearing_not_decorative(self):
        """The pilot's fine-tuned action optimum falls below even the wide floor.

        So widening the grid alone would still report a silent truncation, and the
        edge marker is the half of the change that actually prevents one.
        """
        import math

        def argmin_t(rows, grid):
            def ce(logits, gold, t):
                z = [v / t for v in logits]
                top = max(z)
                e = [math.exp(v - top) for v in z]
                s = sum(e)
                return -math.log(max(e[gold] / s, 1e-12))
            losses = [sum(ce(l, g, t) for g, l in rows) / len(rows) for t in grid]
            return int(min(range(len(losses)), key=lambda i: losses[i])), len(losses)

        wide = [0.05 * (20.0 / 0.05) ** (i / 400) for i in range(401)]
        index, size = argmin_t(self.finetuned, wide)
        self.assertIn(index, (0, size - 1),
                      "if this ever becomes interior, the edge-marker argument changes")


if __name__ == "__main__":
    unittest.main()
