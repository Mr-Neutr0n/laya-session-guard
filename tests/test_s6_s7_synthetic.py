"""Tests for the synthetic S6/S7 stopgap set.

These are synthetic cases that cover a known failure shape. They are not a
measurement of any model and they do not stand in for consented human-reviewed
sessions. The tests below assert the structural guarantees the set is supposed to
provide: matched pairs, independent variation of the surface axes, no injection
phrasing in the S6 half, and a decorrelated history_complete flag.
"""

import json
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SET_DIR = ROOT / "data" / "s6_s7_synthetic"
CASES = SET_DIR / "cases.jsonl"
PROVENANCE = SET_DIR / "provenance.json"

INJECTION_MARKERS = re.compile(
    r"\b(ignore the (earlier|previous|above)|disregard|override|maintainer|approved it|"
    r"skip asking|urgent|immediately|system notice|new instruction|do not mention|"
    r"treat this (page|note|message) as|authority)\b", re.I)


def cases():
    if not CASES.exists():
        raise unittest.SkipTest(f"missing generated set {CASES}; run "
                                f"python3 scripts/generate_s6_s7_synthetic.py")
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def provenance():
    if not PROVENANCE.exists():
        raise unittest.SkipTest(f"missing {PROVENANCE}")
    return json.loads(PROVENANCE.read_text())


class MatchedPairTests(unittest.TestCase):
    def setUp(self):
        self.rows = cases()
        self.pairs = defaultdict(dict)
        for row in self.rows:
            if row["group"]["pair_group_id"]:
                self.pairs[row["group"]["pair_group_id"]][row["stratum"]] = row

    def test_every_pair_has_both_halves(self):
        for pair_id, halves in self.pairs.items():
            self.assertEqual(sorted(halves), ["S6", "S7"], pair_id)

    def test_action_text_is_identical_across_the_pair(self):
        for pair_id, halves in self.pairs.items():
            self.assertEqual(halves["S6"]["state"]["proposed_action"]["description"],
                             halves["S7"]["state"]["proposed_action"]["description"], pair_id)
            self.assertEqual(halves["S6"]["state"]["proposed_action"]["kind"],
                             halves["S7"]["state"]["proposed_action"]["kind"], pair_id)
            self.assertEqual(halves["S6"]["state"]["proposed_action"]["destination"],
                             halves["S7"]["state"]["proposed_action"]["destination"], pair_id)

    def test_labels_are_block_for_s6_and_allow_for_s7(self):
        for pair_id, halves in self.pairs.items():
            self.assertEqual(halves["S6"]["labels"]["action"], "block", pair_id)
            self.assertEqual(halves["S7"]["labels"]["action"], "allow", pair_id)

    def test_only_permission_differs_between_the_halves(self):
        for pair_id, halves in self.pairs.items():
            s6, s7 = halves["S6"]["state"], halves["S7"]["state"]
            self.assertNotEqual(s6["permission"], s7["permission"], pair_id)
            self.assertNotEqual(s6["authorized_task"], s7["authorized_task"], pair_id)

    def test_no_pair_straddles_two_splits(self):
        for pair_id, halves in self.pairs.items():
            self.assertEqual(halves["S6"]["split"], halves["S7"]["split"], pair_id)

    def test_s6_scope_does_not_cover_the_action_and_s7_scope_does(self):
        for pair_id, halves in self.pairs.items():
            self.assertNotEqual(halves["S6"]["state"]["permission"]["scope"],
                                halves["S7"]["state"]["permission"]["scope"], pair_id)


class InjectionFreeTests(unittest.TestCase):
    def test_s6_half_carries_no_injection_phrasing(self):
        offenders = [c["case_id"] for c in cases() if c["stratum"] == "S6"
                     and (INJECTION_MARKERS.search(c["state"]["proposed_action"]["description"])
                          or any(INJECTION_MARKERS.search(e["content"])
                                 for e in c["state"]["events"]))]
        self.assertEqual(offenders, [], f"S6 must be the case with nothing to detect: {offenders}")

    def test_s6_never_grants_the_outward_action(self):
        for row in cases():
            if row["stratum"] != "S6":
                continue
            action = row["state"]["proposed_action"]["description"].lower()
            scope = " ".join(row["state"]["permission"]["scope"]).lower()
            self.assertNotIn(action[:40], scope,
                             f"{row['case_id']} scope appears to cover the action itself")

    def test_no_case_uses_a_non_user_source_as_authority(self):
        for row in cases():
            self.assertEqual(row["state"]["permission"]["source"], "direct_user", row["case_id"])


class SurfaceVariationTests(unittest.TestCase):
    def setUp(self):
        self.s6 = [c for c in cases() if c["stratum"] == "S6"]

    ARTIFACTS = __import__("generate_s6_s7_synthetic").ARTIFACTS

    def test_recipient_and_verb_vary_independently(self):
        pairs = {(c["state"]["proposed_action"]["kind"],
                  re.sub(r"^the \w+ ", "", c["state"]["proposed_action"]["destination"]))
                 for c in self.s6}
        self.assertGreaterEqual(len(pairs), 8,
                                "recipient and verb must not collapse onto a few combinations")

    def test_recipient_axis_does_not_dominate(self):
        """The recipient axis must stay spread.

        The action kind is deliberately send-heavy, because the failure being targeted
        is an outward message expansion and diluting it would misrepresent the target.
        So the invariant is on the recipient axis, which is the one a model could key on
        instead of the permission rule, plus a floor on distinct kinds.
        """
        kinds = Counter(c["state"]["proposed_action"]["kind"] for c in self.s6)
        recipients = Counter(re.sub(r"^the \w+ ", "", c["state"]["proposed_action"]["destination"])
                             for c in self.s6)
        self.assertGreaterEqual(len(kinds), 4, f"only {len(kinds)} action kinds")
        self.assertGreaterEqual(len(recipients), 8, f"only {len(recipients)} recipient classes")
        self.assertLessEqual(recipients.most_common(1)[0][1], 0.25 * len(self.s6),
                             f"one recipient class dominates: {recipients.most_common(3)}")

    def test_shape_vocabulary_matches_the_generator(self):
        from generate_s6_s7_synthetic import SHAPES
        self.assertEqual(len(SHAPES), 11)
        for case in self.s6:
            for _shape, template in SHAPES:
                if any(case["state"]["authorized_task"] == template.format(artifact=a)
                       for a in self.ARTIFACTS):
                    break
            else:
                self.fail(f"{case['case_id']} task text matches no shape template")

    def test_action_texts_are_distinct_enough_to_avoid_a_single_template_cue(self):
        texts = Counter(c["state"]["proposed_action"]["description"] for c in self.s6)
        self.assertEqual(len(texts), len(self.s6),
                         f"only {len(texts)} distinct actions across {len(self.s6)} cases")

    def test_s6_task_texts_are_not_all_identical(self):
        """Regression guard: an earlier draft used 5 task templates for 64 cases."""
        tasks = Counter(c["state"]["authorized_task"] for c in self.s6)
        self.assertGreaterEqual(len(tasks), 30,
                                f"only {len(tasks)} distinct S6 task texts; the shape expansion "
                                f"should have raised this well above 5")


class DecorrelationTests(unittest.TestCase):
    def setUp(self):
        self.rows = cases()

    def test_both_flag_values_appear_in_dev_and_calib(self):
        for split in ("dev", "calib"):
            flags = {c["state"]["context"]["history_complete"] for c in self.rows
                     if c["split"] == split}
            self.assertEqual(flags, {True, False}, f"{split} has flags {flags}")

    def test_truncated_cases_include_decisive_block_and_allow(self):
        truncated = defaultdict(set)
        for row in self.rows:
            if not row["state"]["context"]["history_complete"]:
                truncated[row["labels"]["action"]].add(row["case_id"])
        self.assertTrue(truncated.get("block"), "need incomplete cases that are decisively block")
        self.assertTrue(truncated.get("allow"), "need incomplete cases that are decisively allow")

    def test_flag_is_not_confounded_with_split(self):
        """The whole point: a truncated-only sealed split would be the audited failure."""
        for split in ("dev", "calib", "test"):
            flags = Counter(c["state"]["context"]["history_complete"] for c in self.rows
                            if c["split"] == split)
            self.assertGreaterEqual(len(flags), 2, f"{split} has a single flag value: {flags}")
            for value, count in flags.items():
                self.assertGreaterEqual(count, 5, f"{split} flag={value} is too rare: {count}")

    def test_sealed_test_split_contains_no_review_cases(self):
        """A known gap. Recorded here so it cannot be forgotten."""
        review_in_test = [c["case_id"] for c in self.rows
                          if c["split"] == "test" and c["labels"]["action"] == "review"]
        self.assertEqual(review_in_test, [])


class ProvenanceTests(unittest.TestCase):
    def test_every_case_is_tagged_synthetic(self):
        for row in cases():
            self.assertEqual(row["provenance"]["data_origin"], "synthetic", row["case_id"])
            self.assertNotEqual(row["provenance"]["origin"], "consented_session", row["case_id"])
            self.assertIsNone(row["provenance"]["consent_id"], row["case_id"])

    def test_provenance_records_generator_seed_counts_and_schema(self):
        meta = provenance()
        for key in ("generator", "seed", "counts", "schema_version", "limitations", "data_origin"):
            self.assertIn(key, meta)
        self.assertEqual(meta["data_origin"], "synthetic")
        self.assertEqual(meta["counts"]["total"], len(cases()))

    def test_provenance_states_the_set_is_not_a_substitute_for_human_cases(self):
        limitations = " ".join(provenance()["limitations"]).lower()
        self.assertIn("synthetic", limitations)
        self.assertIn("not a substitute", limitations)
        self.assertIn("must never be presented", limitations)

    def test_generator_is_deterministic(self):
        from generate_s6_s7_synthetic import generate, review_cases
        first = generate() + review_cases()
        second = generate() + review_cases()
        self.assertEqual(first, second, "the same seed must produce the same set")

    def test_generated_set_matches_the_committed_set(self):
        from generate_s6_s7_synthetic import generate, review_cases
        fresh = sorted(json.dumps(c, sort_keys=True) for c in generate() + review_cases())
        committed = sorted(json.dumps(c, sort_keys=True) for c in cases())
        self.assertEqual(fresh, committed,
                         "committed set is stale; re-run scripts/generate_s6_s7_synthetic.py")


class PreRegistrationTests(unittest.TestCase):
    """The pre-registration must stay true of the data it describes."""

    def setUp(self):
        raw = (ROOT / "reports" / "REVIEW_PROTOCOL.md").read_text()
        # Collapse line wrapping so assertions about prose survive a rewrap.
        self.doc = " ".join(raw.split())
        if not (ROOT / "outputs" / "risk" / "laya-risk-ranked" / "report.json").exists():
            self.skipTest("risk run report unavailable")

    def test_preregistered_training_subtotal_matches_the_real_generators(self):
        import os
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.setdefault("ASPI_SOURCE", str(ROOT / "data" / "aspi" / "workspace.jsonl"))
        from aspi_data import generate as aspi
        from counterfactual_data import generate as counterfactual
        from session_data import generate as original
        total = (len(original()["train"]) + len(counterfactual()["train"])
                 + len(aspi()["train"]))
        self.assertEqual(total, 2920, "pre-registration says 2,920 training rows")
        self.assertIn("2,920", self.doc)

    def test_preregistered_split_hashes_match_the_verifier(self):
        from importlib import util
        spec = util.spec_from_file_location("vrr", ROOT / "scripts" / "verify_risk_report.py")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for split, digest in module.SPLIT_SHA.items():
            self.assertIn(digest, self.doc, f"{split} hash missing from the pre-registration")

    def test_preregistered_stopgap_counts_match_the_generated_set(self):
        meta = provenance()
        rows = cases()
        for key, stratum in (("s6", "S6"), ("s7", "S7")):
            self.assertEqual(meta["counts"][key], sum(1 for r in rows if r["stratum"] == stratum))
        self.assertEqual(meta["counts"]["review_support"],
                         sum(1 for r in rows if r["labels"]["action"] == "review"))
        self.assertIn(str(meta["counts"]["s6"]), self.doc)
        self.assertIn(str(meta["counts"]["s7"]), self.doc)

    def test_no_consented_case_exists_anywhere_in_the_stopgap(self):
        self.assertEqual(sum(1 for r in cases()
                             if r["provenance"]["data_origin"] == "consented"), 0)

    def test_sealed_test_split_of_the_stopgap_is_present_and_accounted_for(self):
        test_rows = [r for r in cases() if r["split"] == "test"]
        self.assertTrue(test_rows)
        self.assertIn("sealed test split contains no synthetic case", self.doc)

    def test_publication_criteria_are_numeric(self):
        for criterion in ("at most 8 of 176", "at least 167 of 176", "at most 3 of 36",
                          "every seed passes", "below 95%"):
            self.assertIn(criterion, self.doc, f"criterion text drifted: {criterion}")


class FidelityTests(unittest.TestCase):
    """The transfer verdict in the pre-registration must match the measurement."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import compare_s6_s7_fidelity as fidelity
        cls.fidelity = fidelity
        cls.s6 = [c for c in cases() if c["stratum"] == "S6"]
        cls.aspi = fidelity.load_aspi()
        cls.aspi_block = [c for c in cls.aspi if c["labels"]["action"] == "block"]
        cls.mode1 = fidelity.load_mode1()
        cls.doc = " ".join((ROOT / "reports" / "REVIEW_PROTOCOL.md").read_text().split())

    def setUp(self):
        if not self.aspi:
            self.skipTest("real ASPI artifacts unavailable")

    def test_mode1_loader_finds_the_documented_23(self):
        """Guards the loader that once read content rows and silently returned nothing."""
        self.assertEqual(len(self.mode1), 23)

    def test_permission_field_divergence_is_as_documented(self):
        with_permission = sum(1 for c in self.s6 if c["state"].get("permission"))
        without = sum(1 for c in self.aspi if c["state"].get("permission"))
        self.assertEqual(with_permission, len(self.s6))
        self.assertEqual(without, 0)
        self.assertIn("64 of 64", self.doc)
        self.assertIn("0 of 2,720", self.doc)

    def test_length_divergence_is_as_documented(self):
        import statistics
        s6_mean = statistics.mean(len(self.fidelity.case_text(c)) for c in self.s6)
        aspi_mean = statistics.mean(len(self.fidelity.case_text(c)) for c in self.aspi)
        self.assertAlmostEqual(s6_mean / aspi_mean, 0.35, places=2)
        self.assertIn("0.35x", self.doc)

    def test_entity_density_inversion_is_as_documented(self):
        def density(rows):
            return sum(1 for c in rows
                       if self.fidelity.DATE_OR_ENTITY.search(self.fidelity.case_text(c))) / len(rows)
        self.assertEqual(density(self.s6), 0.0)
        self.assertGreater(density(self.aspi), 0.9)
        self.assertIn("92%", self.doc)

    def test_vocabulary_overlap_is_as_documented(self):
        s6 = set()
        for case in self.s6:
            s6.update(set(self.fidelity.words(self.fidelity.case_text(case))))
        aspi = set()
        for case in self.mode1:
            aspi.update(set(self.fidelity.words(self.fidelity.case_text(case))))
        self.assertAlmostEqual(self.fidelity.jaccard(s6, aspi), 0.21, delta=0.05)
        self.assertIn("0.208 vs Mode 1", self.doc)

    def test_the_two_aligning_axes_are_still_true(self):
        """If these stop holding the shape coverage argument is void, so flag it."""
        def names_recipients(rows):
            return sum(1 for c in rows
                       if self.fidelity.RECIPIENT_WORDS.search(
                           c["state"]["proposed_action"]["description"]))
        self.assertEqual(names_recipients(self.s6), len(self.s6))
        self.assertEqual(names_recipients(self.mode1), len(self.mode1))
        self.assertEqual(names_recipients(self.aspi_block), 772)

    def test_verdict_is_recorded_as_do_not_run(self):
        self.assertIn("should not be expected to transfer", self.doc)
        self.assertIn("does not justify a training run", self.doc)
        self.assertIn("Do not run a fine-tune whose justification", self.doc)

    def test_amendments_section_still_exists_and_is_empty(self):
        self.assertIn("### 10.8 Amendments", self.doc)
        self.assertIn("None yet", self.doc)


if __name__ == "__main__":
    unittest.main()
