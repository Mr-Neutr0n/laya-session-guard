"""Invariants that keep training and racing on the same legal graph."""
import json
import unittest
from pathlib import Path

from wikirace.graph import distances_to, load_graph, reverse_graph, shortlist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/wikirace"


class WikiraceDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph()
        cls.splits = {split: [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]
                      for split in ("train", "calibration", "test")}

    def test_target_disjoint_and_choices_legal(self):
        target_sets = {split: {row["target"] for row in rows} for split, rows in self.splits.items()}
        self.assertFalse(target_sets["train"] & target_sets["calibration"])
        self.assertFalse(target_sets["train"] & target_sets["test"])
        self.assertFalse(target_sets["calibration"] & target_sets["test"])
        for rows in self.splits.values():
            for row in rows:
                self.assertEqual(row["candidates"], shortlist(self.graph, row["current"], row["target"]))
                self.assertTrue(all(v in self.graph["links"][row["current"]] for v in row["candidates"]))
                self.assertIn(row["label"], range(len(row["candidates"])))

    def test_heldout_races_reachable_and_out_of_training_targets(self):
        incoming = reverse_graph(self.graph)
        train_targets = {row["target"] for row in self.splits["train"]}
        races = json.loads((DATA / "eval_races.json").read_text())
        self.assertEqual(len(races), 60)
        for race in races:
            self.assertNotIn(race["target"], train_targets)
            self.assertEqual(distances_to(incoming, race["target"])[race["start"]], race["shortest_hops"])

    def test_explore_shortlist_refills_after_excluding_visited_links(self):
        graph = {"titles": ["Start"] + [f"Page {n}" for n in range(1, 15)] + ["Goal"],
                 "links": [list(range(1, 15))] + [[] for _ in range(15)],
                 "indegree": [0] + [1] * 14 + [0]}
        baseline = shortlist(graph, 0, 15)
        explore = shortlist(graph, 0, 15, exclude=(0, 1, 2))
        self.assertEqual(len(baseline), 12)
        self.assertEqual(len(explore), 12)
        self.assertTrue(set(explore).isdisjoint({0, 1, 2}))
        self.assertEqual(set(explore), set(range(3, 15)))


if __name__ == "__main__":
    unittest.main()
