import collections
import json
import tempfile
import unittest
from pathlib import Path

from src.session_data import QUESTIONS, generate, write_dataset


class SessionDataTests(unittest.TestCase):
    def test_schema_balance_and_unique_states(self):
        data = generate()
        self.assertEqual({k: len(v) for k, v in data.items()}, {"train": 480, "calibration": 96, "test": 144})
        seen = set()
        for split, records in data.items():
            for row in records:
                self.assertEqual(row["split"], split)
                self.assertEqual(set(row["labels"]), set(QUESTIONS))
                for key, label in row["labels"].items():
                    self.assertIn(label, QUESTIONS[key]["criteria"])
                serialized = json.dumps(row["state"], sort_keys=True)
                self.assertNotIn(serialized, seen)
                seen.add(serialized)
                self.assertNotIn("labels", row["state"])
                self.assertTrue(all("role" in e and "source" in e for e in row["state"]["events"]))
            for key in QUESTIONS:
                counts = collections.Counter(r["labels"][key] for r in records)
                self.assertEqual(len(set(counts.values())), 1)

    def test_groups_and_families_do_not_cross_splits(self):
        seen_groups, seen_families = set(), set()
        for records in generate().values():
            groups = {row["group_id"] for row in records}
            families = {row["family"] for row in records}
            self.assertFalse(groups & seen_groups)
            self.assertFalse(families & seen_families)
            seen_groups |= groups
            seen_families |= families

    def test_authorization_counterfactual_and_source_action_separation(self):
        for records in generate().values():
            groups = collections.defaultdict(list)
            for row in records:
                groups[row["group_id"]].append(row)
            for rows in groups.values():
                authorized, denied, ignored, followed = rows[:4]
                self.assertEqual(authorized["state"]["proposed_action"], denied["state"]["proposed_action"])
                self.assertEqual(authorized["state"]["events"][-1], denied["state"]["events"][-1])
                self.assertEqual([authorized["labels"]["action"], denied["labels"]["action"]], ["allow", "block"])
                self.assertEqual(ignored["state"]["events"], followed["state"]["events"])
                self.assertEqual(ignored["labels"]["content"], followed["labels"]["content"])
                self.assertNotEqual(ignored["labels"]["action"], followed["labels"]["action"])

    def test_revocation_and_incomplete_context(self):
        for records in generate().values():
            for row in records:
                if row["id"].endswith("permission_revoked"):
                    self.assertEqual(row["labels"]["action"], "block")
                    self.assertEqual(row["state"]["events"][-1]["role"], "user")
                if not row["state"]["context"]["history_complete"]:
                    self.assertEqual(row["labels"]["action"], "review")

    def test_output_is_deterministic(self):
        self.assertEqual(generate(), generate())
        with tempfile.TemporaryDirectory() as directory:
            metadata = write_dataset(directory)
            self.assertIn("synthetic", metadata["source"])
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            write_dataset(directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})

    def test_coding_workflows_and_sources_are_varied(self):
        for records in generate().values():
            actions = {r["state"]["proposed_action"]["kind"] for r in records}
            self.assertTrue({"edit_configuration", "execute_test_suite", "install_dependency", "remove_fixture", "publish_build_artifact", "update_documentation"} <= actions)
            sources = {e["source"] for r in records for e in r["state"]["events"] if e["role"] == "tool"}
            self.assertEqual(sources, {"repository_file", "issue_thread", "build_log", "package_documentation"})


if __name__ == "__main__":
    unittest.main()
