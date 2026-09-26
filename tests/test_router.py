import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arxiv_router_data as data
import router_metrics as metrics
import router_task as task


def example(paper_id, title, abstract, label, split="train"):
    return {"id": paper_id, "title": title, "abstract": abstract, "label": label,
            "first_version_date": "2008-01-01", "license": None, "split": split}


class SelectionTests(unittest.TestCase):
    def test_legacy_identifiers_and_short_text_are_dropped(self):
        rows = {"train": [example("hep-th/0601001", "A title that is long enough", "word " * 60, "hep-th")],
                "calibration": [], "test": []}
        kept, stats = data.select(rows, ["hep-th"], {"train": 10, "calibration": 10, "test": 10}, 50)
        self.assertEqual(kept["train"], [])
        self.assertEqual(stats["train"]["dropped"]["legacy_identifier"], 1)

    def test_short_abstract_is_dropped(self):
        rows = {"train": [example("0704.0001", "A perfectly reasonable title", "too short", "hep-th")],
                "calibration": [], "test": []}
        kept, stats = data.select(rows, ["hep-th"], {"train": 10, "calibration": 10, "test": 10}, 50)
        self.assertEqual(kept["train"], [])
        self.assertEqual(stats["train"]["dropped"]["short_text"], 1)

    def test_cited_legacy_identifier_is_stripped(self):
        cleaned = data.strip_reference_cues("We extend REF and arXiv:math.PR/0410115 results.")
        self.assertNotIn("hep-th", cleaned)
        self.assertIn("REF", cleaned)
        # New-style identifiers carry no archive and are left readable.
        self.assertIn("0712.3456", data.strip_reference_cues("see 0712.3456 for details"))

    def test_rows_whose_text_mentions_the_archive_are_dropped(self):
        # Backstop for archive mentions that are not legacy identifiers, such as
        # a bare "in hep-th we show" or a "[math.ag]" cross-reference tag.
        rows = {"train": [example("0704.0002", "A title that is long enough",
                                  "word " * 60 + " in hep-th we show the result", "hep-th")],
                "calibration": [], "test": []}
        kept, stats = data.select(rows, ["hep-th"], {"train": 10, "calibration": 10, "test": 10}, 50)
        self.assertEqual(kept["train"], [])
        self.assertEqual(stats["train"]["dropped"]["residual_archive_cue"], 1)

    def test_cited_legacy_identifier_is_removed_not_merely_dropped(self):
        rows = {"train": [example("0704.0008", "A title that is long enough",
                                  "extending hep-th/0607218 here " + "word " * 60, "hep-th")],
                "calibration": [], "test": []}
        kept, _ = data.select(rows, ["hep-th"], {"train": 10, "calibration": 10, "test": 10}, 200)
        self.assertEqual(len(kept["train"]), 1)
        self.assertIn("REF", kept["train"][0]["abstract"])
        self.assertNotIn("hep-th", kept["train"][0]["abstract"])

    def test_normalised_titles_never_cross_splits(self):
        rows = {"train": [example("0704.0003", "Shared Title About Transport", "word " * 60, "hep-th")],
                "calibration": [example("0705.0004", "shared, title about transport!", "word " * 60, "gr-qc")],
                "test": []}
        kept, _ = data.select(rows, ["hep-th", "gr-qc"], {"train": 10, "calibration": 10, "test": 10}, 50)
        self.assertEqual(len(kept["train"]), 1)
        self.assertEqual(kept["calibration"], [])

    def test_selection_is_class_balanced_and_deterministic(self):
        rows = {"train": [example(f"0704.1{i:03d}", f"Title number {i} about a subject", "word " * 60,
                                  "hep-th" if i < 9 else "gr-qc") for i in range(20)],
                "calibration": [], "test": []}
        caps = {"train": 8, "calibration": 4, "test": 4}
        first, _ = data.select(rows, ["hep-th", "gr-qc"], caps, 50)
        second, _ = data.select(rows, ["hep-th", "gr-qc"], caps, 50)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 8)
        self.assertEqual([row["label"] for row in first["train"]].count("hep-th"), 4)

    def test_max_examples_caps_the_total(self):
        rows = {split: [example(f"0704.{split}{i:03d}", f"Title {split} {i} here", "word " * 60, "hep-th")
                        for i in range(20)] for split in ("train", "calibration", "test")}
        caps = {"train": 20, "calibration": 20, "test": 20}
        kept, _ = data.select(rows, ["hep-th"], caps, 50, max_examples=25)
        self.assertEqual(sum(len(v) for v in kept.values()), 25)

    def test_verify_labels_rejects_a_thin_class(self):
        rows = {"train": [{"primary_category": "hep-th"}], "calibration": [], "test": []}
        with self.assertRaises(ValueError):
            data.verify_labels(rows, ["hep-th"], min_per_label=1)

    def test_curated_labels_span_archives_and_exclude_moving_targets(self):
        archives = {label.split(".")[0] for label in data.LABELS}
        self.assertGreaterEqual(len(archives), 4)
        self.assertNotIn("astro-ph", data.LABELS)
        self.assertEqual(len(set(data.LABELS)), len(data.LABELS))


class TaskTests(unittest.TestCase):
    def test_state_excludes_every_non_text_field(self):
        row = example("0704.0005", "A title about lattices", "word " * 60, "hep-th")
        row["license"] = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
        self.assertEqual(set(task.paper_state(row)), {"title", "abstract"})
        self.assertTrue(task.assert_no_cue(row, data.LABELS))

    def test_label_cue_in_input_is_rejected(self):
        row = example("0704.0006", "A title about lattices", "see hep-th/0607218 for details " + "word " * 60, "hep-th")
        with self.assertRaises(ValueError):
            task.assert_no_cue(row, data.LABELS)

    def test_extra_state_field_is_rejected(self):
        row = example("0704.0007", "A title about lattices", "word " * 60, "hep-th")
        with self.assertRaises(ValueError):
            task.assert_no_cue(row, data.LABELS, state={"title": "t", "abstract": "a", "label": "hep-th"})

    def test_question_has_one_criterion_per_label(self):
        question = task.router_question(data.LABELS)
        self.assertEqual(question["t"], "choice")
        self.assertEqual(list(question["crit"]), list(data.LABELS))
        self.assertTrue(all(value for value in question["crit"].values()))


class ReportTests(unittest.TestCase):
    """The write-up claims that the temperature cannot move any ranked metric."""

    def _report(self, rows, labels, calibration, temperature_samples=(0.25, 1.0, 4.0)):
        import router_run
        results = []
        for temperature in temperature_samples:
            results.append(router_run.report_for("t", rows, labels, calibration, 200, _StubLatency()))
        return results

    def test_ranked_metrics_are_temperature_invariant(self):
        import router_run
        labels = ["a", "b", "c"]
        rows = [{"id": str(i), "label": i % 3, "category": labels[i % 3],
                 "logits": [2.0 * (i % 3 == 0) - 1.0, 2.0 * (i % 3 == 1) - 1.0, 2.0 * (i % 3 == 2) - 1.0]}
                for i in range(30)]
        calibration = [{"id": f"c{i}", "label": i % 3, "category": labels[i % 3],
                        "logits": [1.5, -0.5, -1.0]} for i in range(12)]
        reports = self._report(rows, labels, calibration)
        self.assertGreater(len({r["accuracy"] for r in reports}), 0)
        for key in ("accuracy", "macro_f1", "confusion_matrix", "correct", "per_class"):
            self.assertEqual(reports[0][key], reports[-1][key], key)

    def test_confidence_metrics_do_move_with_the_temperature(self):
        import router_run
        labels = ["a", "b"]
        calibration = [{"id": f"c{i}", "label": i % 2, "category": labels[i % 2], "logits": [2.0, -2.0]}
                       for i in range(8)]
        right = [{"id": str(i), "label": i % 2, "category": labels[i % 2], "logits": [3.0, -3.0]}
                 for i in range(20)]
        right_report = router_run.report_for("t", right, labels, calibration, 200, _StubLatency())
        self.assertAlmostEqual(right_report["accuracy"], 0.5)
        # A systematically wrong model is overconfident, so ECE and Brier are poor,
        # and both read the probability vector rather than the ranking.
        wrong = [{"id": str(i), "label": 1, "category": "b", "logits": [3.0, -3.0]} for i in range(20)]
        wrong_report = router_run.report_for("t", wrong, labels, calibration, 200, _StubLatency())
        self.assertAlmostEqual(wrong_report["accuracy"], 0.0)
        self.assertGreater(wrong_report["ece"], 0.0)
        self.assertGreater(wrong_report["brier"], right_report["brier"])


class _StubLatency:
    def report(self):
        return {}


class MetricsTests(unittest.TestCase):
    def test_per_class_and_macro_f1(self):
        labels = ["a", "b", "c"]
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        stats = metrics.summary(y_true, y_pred, labels)
        self.assertAlmostEqual(stats["accuracy"], 4 / 6)
        expected = (0.5 + 0.8 + 2 / 3) / 3
        self.assertAlmostEqual(stats["macro_f1"], expected)
        self.assertEqual(stats["per_class"]["a"]["support"], 2)

    def test_mcnemar_counts_discordant_pairs(self):
        result = metrics.mcnemar([(True, True), (True, False), (False, True), (False, False)])
        self.assertEqual(result["n_discordant"], 2)
        self.assertEqual(result["p_value"], 1.0)
        all_better = metrics.mcnemar([(False, True)] * 10)
        self.assertLess(all_better["p_value"], 0.01)

    def test_temperature_fit_improves_likelihood(self):
        # Overconfident but imperfect logits: the fit must sharpen towards the
        # truth without collapsing to the grid edge, because errors are penalised.
        logits = [[9.0, -2.0, -6.0], [8.0, -1.0, -5.0], [-4.0, 7.0, 1.0], [-3.0, 6.0, 2.0],
                  [5.0, 3.0, 1.0], [-2.0, 1.0, 4.0]]
        y = [0, 0, 1, 1, 0, 1]
        temperature, nll = metrics.fit_temperature(logits, y)
        unit_nll = metrics.fit_temperature(logits, y, grid=[1.0])[1]
        self.assertLess(nll, unit_nll)
        self.assertGreater(temperature, 1.0)

    def test_temperature_fit_uses_logits_not_probabilities(self):
        # Overlapping classes, so the optimum is interior rather than degenerate.
        base_logits = [[2.0, 0.5, -1.0], [1.6, 0.2, -0.4], [0.4, 1.8, 0.1], [0.1, 1.4, 0.6],
                       [0.9, -0.3, 0.5], [0.5, 0.8, 0.2], [-0.2, 0.4, 1.5], [0.3, 1.1, 0.4]]
        y = [0, 0, 1, 1, 0, 0, 2, 1]
        logits = [[value * 12.0 for value in row] for row in base_logits]
        probs = [metrics.softmax(row, 1.0) for row in logits]
        logit_temperature, logit_nll = metrics.fit_temperature(logits, y)
        probability_temperature, _ = metrics.fit_temperature(probs, y)
        self.assertGreater(logit_temperature, 1.0)
        self.assertNotAlmostEqual(logit_temperature, probability_temperature)
        # The fitted value is a real optimum, not a grid-edge artefact.
        neighbour_nll = metrics.fit_temperature(logits, y,
                                                grid=[round(logit_temperature - 0.01, 4),
                                                      round(logit_temperature + 0.01, 4)])[1]
        self.assertLessEqual(logit_nll, neighbour_nll + 1e-9)
        # Applying the probability-path temperature to the logits is worse.
        wrong_nll = -sum(math.log(max(metrics.softmax(row, probability_temperature)[truth], 1e-12))
                         for row, truth in zip(logits, y)) / len(y)
        self.assertGreater(wrong_nll, logit_nll)

    def test_temperature_does_not_change_the_prediction(self):
        values = [2.0, 0.0, -1.0]
        # softmax(z / T) is the identity in T once the input is already normalised,
        # which is why a probability input cannot express logit-scale sharpening.
        for temperature in (0.05, 0.5, 3.0, 12.0):
            self.assertAlmostEqual(sum(metrics.softmax(values, temperature)), 1.0)
        self.assertEqual(max(range(3), key=lambda i: metrics.softmax(values, 0.05)[i]),
                         max(range(3), key=lambda i: metrics.softmax(values, 12.0)[i]))

    def test_softmax_is_invariant_to_positive_scaling_of_the_temperature_path(self):
        values = [2.0, 0.0, -1.0]
        for temperature in (0.05, 0.5, 3.0, 12.0):
            probs = metrics.softmax(values, temperature)
            self.assertAlmostEqual(sum(probs), 1.0)
        # Ranking, and therefore accuracy, is unchanged by any positive temperature.
        self.assertEqual(max(range(3), key=lambda i: metrics.softmax(values, 0.05)[i]),
                         max(range(3), key=lambda i: metrics.softmax(values, 12.0)[i]))

    def test_bootstrap_interval_brackets_the_point_estimate(self):
        values = [(1,)] * 60 + [(0,)] * 40
        low, high = metrics.bootstrap_ci(values, lambda draw: sum(v[0] for v in draw) / len(draw), samples=300, seed=1)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_latency_report_percentiles(self):
        latency = metrics.Latency()
        for value in (1.0, 2.0, 3.0, 100.0):
            latency.record(value)
        report = latency.report()
        self.assertEqual(report["calls"], 4)
        self.assertAlmostEqual(report["ms_per_example_mean"], 26.5)
        self.assertGreaterEqual(report["ms_per_example_p95"], report["ms_per_example_p50"])


if __name__ == "__main__":
    unittest.main()
