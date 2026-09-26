"""Durable guards for the calibration-transform audit.

The audit in scripts/audit_calibration.py found the arXiv bug class absent: the
temperature is applied to logits, not to already-normalised probabilities. These
tests pin that, because the bug is invisible in output and would only show up as a
flat loss and a fitted value stuck on the first grid point.
"""

import json
import math
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "src" / "train.py"
TRAIN_RISK = ROOT / "src" / "train_risk.py"
JEV = ROOT / "src" / "evaluate_jev.py"
RUN = ROOT / "outputs" / "risk" / "laya-risk-ranked"
AUDIT = ROOT / "scripts" / "audit_calibration.py"


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def action_rows():
    if not (RUN / "predictions-test_attack.json").exists():
        raise unittest.SkipTest("stored predictions unavailable")
    rows = json.loads((RUN / "predictions-test_attack.json").read_text())
    return [(r["label"], [float(x) for x in r["logits"]])
            for r in rows if r["qid"] == "action"]


class TransformOrderTests(unittest.TestCase):
    """The transform must be applied before normalisation, not after."""

    def test_train_applies_temperature_to_logits_then_softmax(self):
        source = TRAIN.read_text()
        self.assertIn('z = torch.tensor([r["logits"] for r in sel]) / temperatures[qid]',
                      source)
        self.assertIn("p = z.softmax(-1).numpy()", source)

    def test_fit_temperatures_divides_logits_not_probabilities(self):
        source = TRAIN.read_text()
        self.assertIn("cross_entropy(z/t,y)", source)
        # The fit must read the raw logits field, never a probability field.
        self.assertNotIn("probabilities", source)

    def test_no_file_divides_a_probability_by_a_temperature(self):
        """The precise form of the bug: a probability scaled by a fitted temperature.

        Reading or validating a probability is legitimate and happens in predict.py, so
        this looks for an actual rescale rather than the mere word.
        """
        pattern = re.compile(r"(prob\w*|\bp\b|\bp_)\s*[/]\s*(t|temp\w*|temperature\w*)\b")
        for path in (TRAIN, TRAIN_RISK, JEV, ROOT / "src" / "evaluate_challenge.py",
                     ROOT / "src" / "predict.py", ROOT / "src" / "eval_rows.py"):
            source = path.read_text()
            hit = pattern.search(source)
            self.assertIsNone(hit, f"{path.name} appears to scale a probability by a "
                                   f"temperature: {hit.group(0) if hit else ''}")

    def test_jev_applies_no_temperature(self):
        source = JEV.read_text()
        self.assertNotIn("temperat", source.lower())
        # The only transform is a renormalisation that undoes 2-decimal rounding.
        self.assertIn("value/sum(raw.values())", source)


class WrapperCalibrationDependencyTests(unittest.TestCase):
    """The wrapper thresholds a calibrated probability, so its numbers are not
    temperature-invariant even though the model's own argmax decisions are."""

    def test_wrapper_defers_on_a_calibrated_probability_threshold(self):
        source = (ROOT / "src" / "predict.py").read_text()
        self.assertIn('max(action["probabilities"].values()) < .8', source)
        self.assertIn("provisional 0.8 review threshold", source)

    def test_the_threshold_is_hardcoded_not_fitted(self):
        """A provisional constant on a calibrated output, so it inherits the fit's error."""
        source = (ROOT / "src" / "predict.py").read_text()
        self.assertNotIn("temperat", source.lower())
        self.assertIn("provisional", source)


class IdentityPropertyTests(unittest.TestCase):
    """Divide by T then renormalise by the new sum is the identity. Prove it, so the
    difference from the bug is documented rather than assumed."""

    def test_renormalise_by_new_sum_is_the_identity(self):
        p = [0.70, 0.22, 0.08]
        for t in (0.25, 0.5, 1.0, 2.0, 7.5):
            scaled = [v / t for v in p]
            out = [v / sum(scaled) for v in scaled]
            for got, want in zip(out, p):
                self.assertAlmostEqual(got, want, places=12)

    def test_softmax_after_scaling_is_not_the_identity(self):
        """The neighbouring transform, which must not be confused with the bug."""
        p = [0.70, 0.22, 0.08]
        out = softmax([v / 2.0 for v in p])
        self.assertGreater(max(abs(a - b) for a, b in zip(out, p)), 1e-3)


class DecisionInvarianceTests(unittest.TestCase):
    """Argmax is invariant to positive temperature scaling, so no guard number moves."""

    def test_argmax_is_unchanged_by_temperature_on_stored_logits(self):
        rows = action_rows()
        for t in (0.5, 1.0, 5.0):
            changed = sum(1 for _g, l in rows
                          if max(range(len(l)), key=lambda i: l[i])
                          != max(range(len(l)), key=lambda i: l[i] / t))
            self.assertEqual(changed, 0, f"T={t} changed {changed} argmax decisions")

    def test_loss_curve_on_the_repo_path_is_not_flat(self):
        """A flat curve is the signature of the identity bug."""
        rows = action_rows()
        losses = []
        for i in range(181):
            t = 0.5 + 4.5 * i / 180
            losses.append(sum(-math.log(max(softmax([v / t for v in l])[g], 1e-12))
                              for g, l in rows) / len(rows))
        spread = abs(losses[0] - losses[-1])
        self.assertGreater(spread, 1e-6, "loss is flat across the grid: identity-shaped")

    def test_published_temperature_is_the_first_grid_point(self):
        """Pinned because it is the symptom to re-check, and it is unverifiable locally.

        The calibration logits were never persisted, so whether 0.5 is a truncated
        optimum or a flat surface cannot be settled from stored artifacts. If a future
        run persists calibration rows this assertion is the thing to revisit.
        """
        if not (RUN / "report.json").exists():
            self.skipTest("risk run report unavailable")
        temps = json.loads((RUN / "report.json").read_text())["temperatures"]
        self.assertEqual(temps["action"], 0.5)
        self.assertEqual(0.5, 0.5)  # first point of torch.linspace(0.5, 5.0, 181)


class AuditScriptTests(unittest.TestCase):
    def test_audit_script_runs_and_its_claims_hold(self):
        result = subprocess.run([sys.executable, str(AUDIT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for claim in ("is the identity for every T",
                      "Non-flat, so the identity transform is NOT present",
                      "NOT DETERMINABLE",
                      "UNAFFECTED",
                      "argmax differs between T=1 and T=0.5 on 0 of"):
            self.assertIn(claim, result.stdout, f"audit lost the claim: {claim}")


if __name__ == "__main__":
    unittest.main()
