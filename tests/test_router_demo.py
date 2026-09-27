import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import router_demo
from build_demo_view import build


class DemoCliTests(unittest.TestCase):
    def test_revision_is_pinned_to_the_published_commit(self):
        self.assertEqual(router_demo.DEFAULT_MODEL, "hxrikp/laya-arxiv-topic-router")
        self.assertRegex(router_demo.DEFAULT_REVISION, r"^[0-9a-f]{40}$")

    def test_abstract_is_truncated_to_the_released_budget(self):
        long_text = " ".join(["word"] * 500)
        self.assertEqual(len(router_demo.clean(long_text, 100).split()), 100)
        self.assertEqual(router_demo.clean("  a   b \n c "), "a b c")
        self.assertEqual(router_demo.clean(""), "")

    def test_demo_reuses_the_released_context_budget(self):
        # The published metrics were computed under this budget; a different one
        # would make the demo's numbers incomparable with the write-up.
        from router_task import build_items  # noqa: F401  (import check only)
        self.assertEqual(router_demo.MAX_LEN, 640)
        self.assertEqual(router_demo.HEAD_MAX_LEN, 256)
        self.assertEqual(router_demo.MAX_WORDS, 100)


class DemoViewTests(unittest.TestCase):
    def _fixtures(self, tmp):
        predictions = {"model": "hxrikp/laya-arxiv-topic-router",
                       "revision": "1c2bb313b6a94b674c2e2c57483760eb5ccab244",
                       "device": "cpu", "max_abstract_words": 100, "max_len": 640,
                       "head_max_len": 256, "labels": ["a", "b"],
                       "predictions": [{"id": "x1", "title": "T <b> & \"q\"",
                                        "tokens": 10, "latency_ms_per_example": 1.0,
                                        "distribution": {"a": 0.75, "b": 0.25},
                                        "top_label": "a", "top_probability": 0.75,
                                        "margin_over_runner_up": 0.5}]}
        cases = [{"id": "x1", "kind": "real", "title": "T", "abstract": "abs </script>",
                  "gold_primary_category": "zz.other"}]
        (tmp / "predictions.json").write_text(json.dumps(predictions))
        (tmp / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
        return tmp / "predictions.json", tmp / "cases.jsonl"

    def test_view_is_self_contained_and_parses(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            predictions, cases = self._fixtures(tmp)
            target, count = build(predictions, cases, tmp / "index.html")
            page = target.read_text()
            self.assertEqual(count, 1)
            # No network: a file:// page cannot rely on anything being fetched.
            self.assertNotIn("http://", page)
            self.assertNotIn("https://", page)
            start = page.index('type="application/json">') + len('type="application/json">')
            end = page.index("</script>", start)
            payload = json.loads(page[start:end].replace("<\\/", "</"))
            self.assertEqual(payload["cases"][0]["gold_primary_category"], "zz.other")
            # The payload travels as JSON and is escaped once, in the browser, at
            # render time. A title carrying markup must reach the esc() helper and
            # must never be interpolated raw into innerHTML.
            self.assertEqual(payload["cases"][0]["title"], 'T <b> & "q"')
            self.assertIn("esc(c.title)", page)
            self.assertNotIn("${c.title}", page)
            self.assertIn("${c.abstract}", page.replace("esc(c.abstract)", "${c.abstract}"))
            # A closing script tag inside case data cannot terminate the payload.
            self.assertIn("abs <\\/script>", page)

    def test_view_marks_a_gold_label_outside_the_option_set(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            predictions, cases = self._fixtures(tmp)
            target, _ = build(predictions, cases, tmp / "index.html")
            page = target.read_text()
            # The "not one of the eight options" verdict is decided in the browser,
            # so the builder must ship the gold label and the option set.
            self.assertIn("zz.other", page)
            self.assertIn('"labels"', page)


if __name__ == "__main__":
    unittest.main()
