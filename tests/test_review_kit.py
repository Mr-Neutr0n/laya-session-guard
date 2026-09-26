"""Tests for the human-review collection kit.

The kit is the thing that unblocks the human data path, so its own correctness
matters more than usual: a template that does not validate, or a gate script that
returns success on a broken batch, would send real collection work down a false
path.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "review_kit"
SCRIPT = KIT / "validate_batch.sh"
TEMPLATE = KIT / "case_template.jsonl"
EXAMPLES = KIT / "examples.jsonl"
VALIDATOR = ROOT / "scripts" / "validate_review_set.py"
STOPGAP = ROOT / "data" / "s6_s7_synthetic" / "cases.jsonl"


def run_batch(*args):
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class TemplateTests(unittest.TestCase):
    """The shipped templates must validate against the current schema, as shipped."""

    def test_case_template_validates_clean_with_placeholders_allowed(self):
        result = run_batch(str(TEMPLATE), "--min-test-per-stratum", "0", "--allow-placeholders")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0 errors", result.stdout)

    def test_case_template_is_refused_as_a_batch(self):
        """A template is a shape, not a case. It must not pass the gate unedited."""
        result = run_batch(str(TEMPLATE), "--min-test-per-stratum", "0")
        self.assertEqual(result.returncode, 1, "an unedited template must not be accepted")
        self.assertIn("template.unedited", result.stdout)

    def test_consented_template_placeholder_consent_id_is_refused(self):
        """The exact case that got through in the second campaign."""
        rows = TEMPLATE.read_text()
        self.assertIn("REPLACE-WITH-YOUR-CONSENT-ID", rows)
        result = run_batch(str(TEMPLATE), "--min-test-per-stratum", "0")
        self.assertIn("template.unedited", result.stdout)

    def test_empty_batch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "empty.jsonl"
            batch.write_text("")
            result = run_batch(str(batch))
        self.assertEqual(result.returncode, 1, "an empty batch must not validate")
        self.assertIn("batch.empty", result.stdout)

    def test_examples_validate_clean(self):
        result = run_batch(str(EXAMPLES), "--min-test-per-stratum", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0 errors", result.stdout)

    def test_every_template_case_matches_the_schema_version_in_use(self):
        current = json.loads((ROOT / "data" / "review" / "case_schema_v1.json").read_text())
        for path in (TEMPLATE, EXAMPLES):
            for case in rows(path):
                self.assertEqual(case["schema_version"], current["properties"]["schema_version"]["const"],
                                 f"{path.name}:{case['case_id']}")

    def test_templates_cover_the_strata_the_models_fail(self):
        strata = {case["stratum"] for case in rows(EXAMPLES)}
        for stratum in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9"):
            self.assertIn(stratum, strata, f"examples should demonstrate {stratum}")
        template_strata = {case["stratum"] for case in rows(TEMPLATE)}
        self.assertIn("S6", template_strata)
        self.assertIn("S8", template_strata)

    def test_templates_contain_a_consented_capture_and_synthetic_cases(self):
        origins = {case["provenance"]["origin"] for case in rows(TEMPLATE)}
        self.assertIn("consented_session", origins)
        self.assertIn("synthetic_template", origins)
        for case in rows(TEMPLATE) + rows(EXAMPLES):
            expected = "consented" if case["provenance"]["origin"] == "consented_session" else "synthetic"
            self.assertEqual(case["provenance"].get("data_origin"), expected, case["case_id"])

    def test_review_class_appears_at_both_completeness_values_in_each_file(self):
        for path in (TEMPLATE, EXAMPLES):
            flags = {case["state"]["context"]["history_complete"]
                     for case in rows(path) if case["labels"]["action"] == "review"}
            self.assertEqual(flags, {True, False}, f"{path.name} review cases share one flag value")

    def test_synthetic_templates_carry_no_credentials(self):
        for path in (TEMPLATE, EXAMPLES):
            blob = path.read_text()
            for marker in ("AKIA", "ghp_", "BEGIN RSA PRIVATE KEY", "xoxb-"):
                self.assertNotIn(marker, blob, f"{path.name} contains {marker}")

    def test_consented_template_uses_an_obvious_placeholder_consent_id(self):
        """A template must not carry something that reads like a real consent record."""
        consented = [c for c in rows(TEMPLATE) if c["provenance"]["origin"] == "consented_session"]
        self.assertTrue(consented)
        for case in consented:
            self.assertIn("REPLACE", case["provenance"]["consent_id"], case["case_id"])
            self.assertEqual(case["provenance"]["redaction"], "applied", case["case_id"])


class GateScriptTests(unittest.TestCase):
    """The gate must accept good input and, just as importantly, reject bad input."""

    def test_gate_accepts_a_copy_of_the_synthetic_examples(self):
        if not STOPGAP.exists():
            self.skipTest("synthetic stopgap not generated")
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "cases.jsonl"
            shutil.copy(STOPGAP, batch)
            result = run_batch(str(batch))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Batch accepted", result.stdout)
        self.assertIn("0 errors", result.stdout)

    def test_gate_rejects_a_broken_batch(self):
        """Proves the gate is not vacuous, and that status is not masked."""
        cases = rows(EXAMPLES)
        cases[0]["state"]["permission"]["source"] = "none"
        cases[0]["state"]["permission"]["granted"] = True
        cases[1]["annotations"][1]["reviewer_id"] = cases[1]["annotations"][0]["reviewer_id"]
        cases[2]["provenance"]["notes"] = "key AKIAIOSFODNN7EXAMPLE"
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "bad.jsonl"
            batch.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
            result = run_batch(str(batch), "--min-test-per-stratum", "0")
        self.assertEqual(result.returncode, 1, "a broken batch must not pass")
        self.assertIn("permission.source", result.stdout)
        self.assertIn("annotation.duplicate_reviewer", result.stdout)
        self.assertIn("redaction.credential", result.stdout)

    def test_gate_prints_actionable_guidance_on_failure(self):
        cases = rows(EXAMPLES)
        cases[0]["case_id"] = "Not_A_Valid_Id"
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "bad.jsonl"
            batch.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
            result = run_batch(str(batch), "--min-test-per-stratum", "0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("REPLACE-", result.stderr)
        self.assertIn("consent.missing", result.stderr)
        self.assertIn("leakage.session_id", result.stderr)

    def test_gate_never_pipes_the_validator(self):
        """A pipeline reports only its last status, which is how a gate silently passes."""
        source = SCRIPT.read_text()
        lines = [line.strip() for line in source.splitlines()
                 if "validate_review_set.py" in line and not line.strip().startswith("#")]
        for line in lines:
            self.assertNotIn("|", line, f"validator invocation is piped: {line!r}")
        self.assertIn("|| status=$?", source)

    def test_missing_arguments_gives_usage_and_nonzero(self):
        result = run_batch()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("one invocation", result.stderr)

    def test_missing_file_is_reported_not_silently_accepted(self):
        result = run_batch("/nonexistent/batch.jsonl")
        self.assertNotEqual(result.returncode, 0)

    def test_extra_arguments_reach_the_validator(self):
        result = run_batch(str(EXAMPLES))
        self.assertIn("coverage.test_stratum", result.stdout,
                      "--min-test-per-stratum should default to 20 and warn")


class ReadmeTests(unittest.TestCase):
    """The README is the interface for a human, so its required content is asserted."""

    def setUp(self):
        self.text = (KIT / "README.md").read_text()
        self.flat = " ".join(self.text.split())

    def test_consent_wording_is_marked_as_draft_and_not_legal_advice(self):
        self.assertIn("DRAFT FOR YOUR REVIEW, NOT LEGAL ADVICE", self.flat.upper())
        self.assertIn("I am not a lawyer", self.flat)

    def test_consent_template_covers_use_redaction_withdrawal_and_publication(self):
        for phrase in ("consent to", "redacted", "withdraw", "published", "keep ownership"):
            self.assertIn(phrase, self.flat.lower(), f"consent wording missing {phrase}")

    def test_redaction_checklist_is_present_and_actionable(self):
        for item in ("Credentials", "Personal data", "Paths", "Fictional names"):
            self.assertIn(item, self.flat)
        self.assertIn("- [ ]", self.text, "checklist should be a real checkbox list")

    def test_readme_states_the_three_things_to_provide(self):
        for heading in ("Case source", "Consent", "Redaction"):
            self.assertIn(heading, self.flat)

    def test_readme_documents_the_command_and_its_exit_behaviour(self):
        self.assertIn("review_kit/validate_batch.sh", self.flat)
        self.assertIn("Exit 0 means no errors", self.flat)
        self.assertIn("Pass every split in one invocation", self.flat)

    def test_readme_lists_the_fields_a_human_must_decide(self):
        for field in ("state.permission.scope", "state.permission.revoked",
                      "state.permission.source", "labels.action",
                      "annotations[].action", "adjudication", "provenance.consent_id",
                      "provenance.data_origin", "group.session_id"):
            self.assertIn(field, self.flat, f"decision table missing {field}")

    def test_readme_documents_allow_placeholders(self):
        self.assertIn("--allow-placeholders", self.flat)
        self.assertIn("do not pass the gate as shipped", self.flat)

    def test_readme_refuses_to_describe_synthetic_as_human_evidence(self):
        flat = self.flat.lower()
        self.assertIn("human-reviewed evidence", flat)
        self.assertIn("a specification of a failure shape, not evidence", flat)

    def test_readme_points_at_the_preregistered_run(self):
        self.assertIn("section 10", self.flat)
        self.assertIn("do not push", self.flat.lower())


if __name__ == "__main__":
    unittest.main()
