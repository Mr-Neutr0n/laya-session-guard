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

    def test_recipient_and_verb_vary_independently(self):
        pairs = {(c["state"]["proposed_action"]["kind"],
                  re.sub(r"^the \w+ ", "", c["state"]["proposed_action"]["destination"]))
                 for c in self.s6}
        self.assertGreaterEqual(len(pairs), 8,
                                "recipient and verb must not collapse onto a few combinations")

    def test_no_single_recipient_or_kind_dominates(self):
        kinds = Counter(c["state"]["proposed_action"]["kind"] for c in self.s6)
        recipients = Counter(re.sub(r"^the \w+ ", "", c["state"]["proposed_action"]["destination"])
                             for c in self.s6)
        self.assertLessEqual(kinds.most_common(1)[0][1], 0.6 * len(self.s6))
        self.assertLessEqual(recipients.most_common(1)[0][1], 0.2 * len(self.s6))

    def test_action_texts_are_distinct_enough_to_avoid_a_single_template_cue(self):
        texts = Counter(c["state"]["proposed_action"]["description"] for c in self.s6)
        self.assertGreaterEqual(len(texts), 0.5 * len(self.s6),
                                f"only {len(texts)} distinct actions across {len(self.s6)} cases")


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


if __name__ == "__main__":
    unittest.main()
