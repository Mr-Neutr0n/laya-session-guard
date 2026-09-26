"""Check the public benchmark adaptation and held-out split boundaries."""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aspi_data import generate


class AspiDataTests(unittest.TestCase):
    def test_pairs_and_split_boundaries(self):
        splits = generate()
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {"train": 1864, "calibration": 456, "test_user": 432,
                          "test_attack": 352, "test_both": 72})
        seen_ids = set()
        for name, rows in splits.items():
            pairs = defaultdict(list)
            for row in rows:
                self.assertNotIn(row["id"], seen_ids)
                seen_ids.add(row["id"])
                pairs[row["pair_id"]].append(row)
                user_id = int(row["provenance"]["user_task_id"].rsplit("_", 1)[1])
                attack_id = int(row["provenance"]["injection_task_id"].rsplit("_", 1)[1])
                if name == "train":
                    self.assertLessEqual(user_id, 27)
                    self.assertLess(attack_id, 12)
                elif name == "test_user":
                    self.assertGreaterEqual(user_id, 34)
                    self.assertLess(attack_id, 12)
                elif name == "test_attack":
                    self.assertLessEqual(user_id, 27)
                    self.assertGreaterEqual(attack_id, 12)
                elif name == "test_both":
                    self.assertGreaterEqual(user_id, 34)
                    self.assertGreaterEqual(attack_id, 12)
            for members in pairs.values():
                self.assertEqual({r["labels"]["action"] for r in members}, {"allow", "block"})
                self.assertEqual(len(members), 2)
                a, b = members
                self.assertEqual(a["state"]["events"], b["state"]["events"])


if __name__ == "__main__":
    unittest.main()
