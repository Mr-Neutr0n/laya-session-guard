"""Tests for the human-review dataset validator.

Every fixture is hand-written and synthetic. No real session is read, and no label
here is treated as ground truth about a model: the fixtures exist only to exercise the
validator, and the tests assert which defect codes fire.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "data" / "review" / "examples"
VALIDATOR = ROOT / "scripts" / "validate_review_set.py"


def run_validator(name, *extra):
    """Run the validator as a subprocess, the way ./verify does."""
    command = [sys.executable, str(VALIDATOR), str(EXAMPLES / name),
               "--min-test-per-stratum", "0", *extra]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def codes(name, *extra):
    _, output = run_validator(name, *extra)
    found = set()
    for line in output.splitlines():
        if line.startswith(("ERROR", "WARNING")):
            found.add(line.split(":")[1].strip() if line.count(":") > 1 else line)
    return found


def rows(name):
    return [json.loads(line) for line in (EXAMPLES / name).read_text().splitlines() if line.strip()]


class SchemaTests(unittest.TestCase):
    def test_clean_fixture_passes(self):
        code, output = run_validator("clean.jsonl")
        self.assertEqual(code, 0, output)
        self.assertIn("0 errors", output)

    def test_only_the_designed_defect_declares_a_bad_version(self):
        """Every fixture targets schema 1.0.0 except the case written to violate it."""
        offenders = [(path.name, case["case_id"], case["schema_version"])
                     for path in EXAMPLES.glob("*.jsonl") for case in rows(path.name)
                     if case["schema_version"] != "1.0.0"]
        self.assertEqual(offenders, [("defect_consent.jsonl", "cons-bad-version", "0.9.0")])

    def test_annotations_are_human_only(self):
        """The role enum has no model member, so an LLM label is unrepresentable."""
        schema = json.loads((ROOT / "data" / "review" / "case_schema_v1.json").read_text())
        role = schema["properties"]["annotations"]["items"]["properties"]["role"]
        self.assertEqual(role["enum"], ["human"])

    def test_no_llm_generated_origin_exists(self):
        schema = json.loads((ROOT / "data" / "review" / "case_schema_v1.json").read_text())
        origin = schema["properties"]["provenance"]["properties"]["origin"]["enum"]
        self.assertNotIn("llm_generated", origin)
        self.assertIn("consented_session", origin)

    def test_action_enum_is_allow_block_review(self):
        schema = json.loads((ROOT / "data" / "review" / "case_schema_v1.json").read_text())
        self.assertEqual(schema["$defs"]["action"]["enum"], ["allow", "block", "review"])


class AnnotationTests(unittest.TestCase):
    def test_duplicate_reviewer_is_rejected(self):
        self.assertIn("annotation.duplicate_reviewer", codes("defect_annotations.jsonl"))

    def test_disagreement_requires_adjudication(self):
        found = codes("defect_annotations.jsonl")
        self.assertIn("annotation.disagreement", found)
        self.assertIn("annotation.adjudicator", found)

    def test_adjudicator_must_not_be_an_annotator(self):
        self.assertIn("annotation.adjudicator", codes("defect_annotations.jsonl"))

    def test_label_must_match_adjudication(self):
        self.assertIn("annotation.label_mismatch", codes("defect_annotations.jsonl"))

    def test_single_annotation_fails_the_schema(self):
        self.assertIn("schema.minItems", codes("defect_annotations.jsonl"))

    def test_clean_fixture_uses_a_third_reviewer_for_disagreement(self):
        """fix-dev-03 is the one clean case where the two reviewers disagree."""
        case = next(c for c in rows("clean.jsonl") if c["case_id"] == "fix-dev-03")
        self.assertEqual(len({a["action"] for a in case["annotations"]}), 2)
        self.assertEqual(case["adjudication"]["status"], "adjudicated")
        self.assertNotIn(case["adjudication"]["adjudicator_id"],
                         [a["reviewer_id"] for a in case["annotations"]])


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.found = codes("defect_leakage.jsonl")

    def test_session_leak_is_rejected(self):
        self.assertIn("leakage.session_id", self.found)

    def test_task_leak_is_rejected(self):
        self.assertIn("leakage.task_id", self.found)

    def test_pair_group_leak_is_rejected(self):
        self.assertIn("leakage.pair_group_id", self.found)

    def test_duplicate_text_leak_is_rejected(self):
        self.assertIn("leakage.text", self.found)

    def test_clean_fixture_has_no_leakage(self):
        self.assertEqual(codes("clean.jsonl"), set())


class ShortcutTests(unittest.TestCase):
    def test_one_way_incomplete_is_review_shortcut_is_rejected(self):
        """The real historical defect. It runs one way only, so review_confounded misses it.

        In the risk-ranked run's 2,920-case train split all 216 incomplete-history cases were
        review with no counterexample, while 40 review cases had complete history. The global
        one-bit rate stays below the gate, and review appears at both flag values, so neither
        earlier check fires.
        """
        path = EXAMPLES / "defect_shortcut_incomplete.jsonl"
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), str(path), "--min-test-per-stratum", "0"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("shortcut.incomplete_is_review", result.stdout)
        # The global rate is under the gate, so this check is the one doing the work.
        self.assertIn("single-bit accuracy on dev+calib: 81.8%", result.stdout)
        self.assertNotIn("shortcut.review_confounded", result.stdout)

    def test_clean_fixture_keeps_incomplete_cases_that_are_not_review(self):
        """The counterexample direction must be present, or the new check is pure noise."""
        fitting = [c for c in rows("clean.jsonl") if c["split"] in ("dev", "calib")]
        incomplete = [c for c in fitting if not c["state"]["context"]["history_complete"]]
        self.assertTrue(incomplete)
        self.assertNotIn("shortcut.incomplete_is_review", codes("clean.jsonl"))

    def test_confounded_review_class_is_rejected(self):
        self.assertIn("shortcut.review_confounded", codes("defect_shortcut.jsonl"))

    def test_single_bit_accuracy_gate_fires_when_lowered(self):
        """The confounded check is threshold-free; the accuracy gate is a second line of defence.

        The fixture is only six cases, so its one-bit accuracy is 83.3% and does not clear
        the default 95% gate. Lowering the gate proves the gate is wired to the real number
        rather than to the presence of the word 'review'.
        """
        _, output = run_validator("defect_shortcut.jsonl")
        self.assertIn("single-bit accuracy on dev+calib: 83.3%", output)
        self.assertIn("shortcut.single_bit_accuracy", codes("defect_shortcut.jsonl",
                                                            "--shortcut-accuracy-gate", "0.80"))

    def test_clean_fixture_decorrelates_the_flag(self):
        self.assertNotIn("shortcut.review_confounded", codes("clean.jsonl"))

    def test_clean_fixture_has_review_at_both_flag_values(self):
        """This is the exact condition the earlier Laya training data violated."""
        fitting = [c for c in rows("clean.jsonl") if c["split"] in ("dev", "calib")]
        flags = {c["state"]["context"]["history_complete"]
                 for c in fitting if c["labels"]["action"] == "review"}
        self.assertEqual(flags, {True, False})

    def test_clean_fixture_keeps_decisive_incomplete_cases(self):
        """Decorrelation must not forbid incomplete-history cases that are clearly block."""
        fitting = [c for c in rows("clean.jsonl") if c["split"] in ("dev", "calib")]
        incomplete_non_review = [c for c in fitting
                                 if not c["state"]["context"]["history_complete"]
                                 and c["labels"]["action"] != "review"]
        self.assertTrue(incomplete_non_review)
        _, output = run_validator("clean.jsonl")
        self.assertNotIn("shortcut", output.replace("shortcut-accuracy-gate", ""))

    def test_missing_truncation_reason_is_rejected(self):
        self.assertIn("context.truncation_reason", codes("defect_consent.jsonl"))


class ConsentTests(unittest.TestCase):
    def setUp(self):
        self.found = codes("defect_consent.jsonl")

    def test_consented_session_requires_a_consent_id(self):
        self.assertIn("consent.missing", self.found)

    def test_consented_session_requires_redaction(self):
        self.assertIn("consent.redaction", self.found)

    def test_unredacted_credential_is_rejected(self):
        self.assertIn("redaction.credential", self.found)

    def test_unsupported_schema_version_is_rejected(self):
        found = self.found
        self.assertTrue({"schema.const", "schema.unsupported_version"} & found)

    def test_unknown_key_is_rejected(self):
        self.assertIn("schema.additionalProperties", self.found)

    def test_label_outside_the_enum_is_rejected(self):
        self.assertIn("schema.enum", self.found)

    def test_llm_provenance_and_llm_annotator_are_unrepresentable(self):
        self.assertIn("schema.enum", self.found)

    def test_fictional_domains_do_not_warn(self):
        self.assertNotIn("redaction.email", self.found)


class CoverageTests(unittest.TestCase):
    def test_short_sealed_strata_warn_without_failing(self):
        """Coverage is a warning, not an error: a small set is still schema-valid."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), str(EXAMPLES / "clean.jsonl"),
             "--min-test-per-stratum", "20"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("coverage.test_stratum", result.stdout)

    def test_zero_minimum_is_silent(self):
        _, output = run_validator("clean.jsonl")
        self.assertNotIn("coverage.test_stratum", output)


class AdversarialTests(unittest.TestCase):
    """Bypasses found by attacking the validator, one fixture per weakness.

    Every case here got past an earlier version of the validator. Each test states the
    code that now catches it, so a future change cannot quietly reopen the hole.
    """

    # case_id -> the error code that must now reject it
    REGRESSIONS = {
        "adv-reviewer-case-variant": "annotation.duplicate_reviewer",
        "adv-reviewer-space-variant": "annotation.duplicate_reviewer",
        "adv-blank-reviewer": "annotation.blank_reviewer",
        "adv-adjudicator-case-variant": "annotation.adjudicator",
        "adv-cred-in-rationale": "redaction.credential",
        "adv-cred-in-notes": "redaction.credential",
        "adv-cred-in-event-list": "redaction.credential",
        "adv-blank-consent-id": "consent.missing",
        "adv-real-email-consented": "redaction.email",
        "adv-omitted-but-complete": "context.omitted_events",
        "adv-omitted-zero-incomplete": "context.omitted_events",
        "adv-session-case-variant": "leakage.session_id",
        "adv-task-case-variant": "leakage.task_id",
        "adv-pair-case-variant": "leakage.pair_group_id",
        "adv-session-tab-variant": "leakage.session_id",
        "adv-revoked-but-allow": "permission.revoked",
        "adv-source-none-granted": "permission.source",
        "adv-granted-empty-scope": "permission.scope",
        "adv-nonuser-source": "permission.source",
    }

    # Deliberately not rejected as errors; see the limitations section of the protocol.
    KNOWN_WARNINGS = ["adv-fingerprint-evade"] + [f"adv-stratum-shortcut-{i:02d}" for i in range(6)]

    @classmethod
    def setUpClass(cls):
        cls.adversarial = {}
        for path in (EXAMPLES / "adversarial").glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                if case["provenance"]["notes"].startswith("ADVERSARIAL"):
                    cls.adversarial[case["case_id"]] = case

    def validate_with(self, *cases):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            for case in cases:
                handle.write(json.dumps(case, sort_keys=True) + "\n")
            path = Path(handle.name)
        try:
            return subprocess.run(
                [sys.executable, str(VALIDATOR), str(path), "--min-test-per-stratum", "0"],
                capture_output=True, text=True)
        finally:
            path.unlink()

    def test_every_regression_fixture_exists(self):
        """Every bypass has a fixture, and the only non-rejected cases are the known ones."""
        self.assertTrue(set(self.REGRESSIONS) <= set(self.adversarial))
        self.assertEqual(sorted(set(self.adversarial) - set(self.REGRESSIONS)),
                         sorted(self.KNOWN_WARNINGS))

    def test_each_reproduced_bypass_is_now_rejected(self):
        clean = rows("clean.jsonl")
        for case_id, code in sorted(self.REGRESSIONS.items()):
            with self.subTest(case=case_id):
                result = self.validate_with(*clean, self.adversarial[case_id])
                self.assertEqual(result.returncode, 1, f"{case_id} still passes:\n{result.stdout}")
                self.assertIn(code, result.stdout,
                              f"{case_id} was rejected, but not by the expected check {code}")

    def test_stratum_local_shortcut_is_rejected_at_file_level(self):
        """The per-stratum shortcut needs its stratum population, so it is tested as a file."""
        path = EXAMPLES / "adversarial" / "adv_stratum_shortcut.jsonl"
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), str(path), "--min-test-per-stratum", "0"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("shortcut.stratum_single_bit", result.stdout)
        # The global rate stays well under the gate, so only the per-stratum check can fire.
        self.assertIn("single-bit accuracy on dev+calib: 75.0%", result.stdout)

    def test_single_label_stratum_is_not_a_false_positive(self):
        """A stratum with one label scores 100% by construction and must not trip the gate."""
        cases = []
        for index in range(5):
            case = copy.deepcopy(next(c for c in rows("clean.jsonl") if c["stratum"] == "S1"))
            case["case_id"] = f"single-label-s1-{index:02d}"
            case["group"] = {"session_id": f"single-s1-{index:02d}",
                             "task_id": f"single-t1-{index:02d}", "pair_group_id": None}
            case["state"]["authorized_task"] = f"Run the parser tests for shard {index:02d}."
            case["state"]["proposed_action"]["description"] = f"Run the parser suite for shard {index:02d}"
            case["state"]["proposed_action"]["destination"] = f"test runner {index:02d}"
            cases.append(case)
        self.assertEqual({c["labels"]["action"] for c in cases}, {"allow"})
        result = self.validate_with(*cases)
        self.assertNotIn("shortcut.stratum_single_bit", result.stdout)
        self.assertNotIn("shortcut.single_bit_accuracy:", result.stdout)

    def test_tiny_slice_does_not_trip_the_global_gate(self):
        """A one-case dev slice scores 100% one-bit by construction and must only warn."""
        case = copy.deepcopy(next(c for c in rows("clean.jsonl") if c["split"] == "dev"))
        result = self.validate_with(case)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("shortcut.single_bit_accuracy_small", result.stdout)

    def test_reworded_twin_is_warned_not_silently_passed(self):
        """Documented limitation: semantic near-duplication is a human call, so it warns."""
        clean = rows("clean.jsonl")
        result = self.validate_with(*clean, self.adversarial["adv-fingerprint-evade"])
        self.assertIn("leakage.text_near", result.stdout)
        self.assertNotIn("leakage.text:", result.stdout)

    def test_cross_split_leak_is_caught_across_two_files(self):
        """Leakage is a cross-split property, so per-split invocations cannot see it."""
        dev = [self.adversarial["adv-session-case-variant"]]
        test = [c for c in rows("clean.jsonl") if c["split"] == "test"]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as a, \
             tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as b:
            for case in dev:
                a.write(json.dumps(case, sort_keys=True) + "\n")
            for case in test:
                b.write(json.dumps(case, sort_keys=True) + "\n")
            paths = (Path(a.name), Path(b.name))
        try:
            for path in paths:
                alone = subprocess.run(
                    [sys.executable, str(VALIDATOR), str(path), "--min-test-per-stratum", "0"],
                    capture_output=True, text=True)
                self.assertEqual(alone.returncode, 0,
                                 "a single split has nothing to leak across; it should pass")
            together = subprocess.run(
                [sys.executable, str(VALIDATOR), *[str(p) for p in paths],
                 "--min-test-per-stratum", "0"], capture_output=True, text=True)
            self.assertEqual(together.returncode, 1, together.stdout)
            self.assertIn("leakage.session_id", together.stdout)
        finally:
            for path in paths:
                path.unlink()

    def test_malformed_case_is_reported_not_crashed(self):
        """A type-invalid case used to raise AttributeError and print a traceback."""
        cases = rows("clean.jsonl")
        for field, value in (("annotations", "reviewer-a said allow"),
                             ("state.events", "a single event"),
                             ("provenance", ["consented_session"])):
            case = copy.deepcopy(cases[0])
            case["case_id"] = "malformed-probe"
            if field == "state.events":
                case["state"]["events"] = value
            else:
                case[field] = value
            result = self.validate_with(*cases, case)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            self.assertIn("schema.type", result.stdout)
            self.assertIn("checks.skipped", result.stdout)

    def test_blank_group_key_is_rejected(self):
        cases = rows("clean.jsonl")
        case = copy.deepcopy(cases[0])
        case["case_id"] = "blank-group-key"
        case["group"]["session_id"] = "   "
        result = self.validate_with(*cases, case)
        self.assertEqual(result.returncode, 1)
        self.assertIn("leakage.blank_session_id", result.stdout)

    def test_normalization_fails_safe(self):
        """Ids differing only in punctuation merge, which over-reports rather than misses."""
        from importlib import util
        spec = util.spec_from_file_location("vrs", VALIDATOR)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for left, right in (("reviewer-a", " Reviewer-A "), ("s-01", "s_01"),
                            ("fix-s06\t", "fix-s06"), ("Reviewer A", "reviewer-a")):
            self.assertEqual(module.normalize_key(left), module.normalize_key(right), (left, right))
        self.assertNotEqual(module.normalize_key("s-01"), module.normalize_key("s-02"))


class ValidatorContractTests(unittest.TestCase):
    def test_missing_file_exits_nonzero(self):
        result = subprocess.run([sys.executable, str(VALIDATOR), str(EXAMPLES / "absent.jsonl")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such file", result.stdout)

    def test_malformed_json_is_reported_not_raised(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("{not json\n")
            path = Path(handle.name)
        try:
            result = subprocess.run([sys.executable, str(VALIDATOR), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("json.parse", result.stdout)
        finally:
            path.unlink()

    def test_validator_exits_zero_on_the_clean_set_without_a_pipe(self):
        """Guards the gate itself: a caller must not pipe this away under set -e."""
        result = subprocess.run([sys.executable, str(VALIDATOR), str(EXAMPLES / "clean.jsonl"),
                                 "--min-test-per-stratum", "0"], capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout)


if __name__ == "__main__":
    unittest.main()
