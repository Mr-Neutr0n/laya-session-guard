"""The browser starts both model lanes together, so checkpoint loads must serialize."""
import concurrent.futures
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.model_adapter import ModelAdapter


class RaceHistoryTest(unittest.TestCase):
    """Explore tells the model where it has been; Benchmark must match the evaluated input."""

    def test_explore_sends_recent_articles_and_benchmark_sends_none(self):
        seen = {}

        class FakeAgent:
            def predict(self, model_state, questions):
                seen.update(model_state)
                labels = questions["next_link"]["criteria"]
                return {"answers": {"next_link": {"choice": "link_0",
                    "probabilities": {name: (1.0 if name == "link_0" else 0.0) for name in labels}}}}

        graph = {"titles": ["A", "B", "C", "Goal"], "links": [[1, 2], [3], [], []],
                 "indegree": [0, 1, 1, 1]}
        with tempfile.TemporaryDirectory() as temp:
            adapter = ModelAdapter()
            adapter._paths = {name: str(Path(temp, name)) for name in ("base", "tuned")}
            for path in adapter._paths.values():
                Path(path).mkdir()
            with patch.dict(sys.modules, {"laya": types.SimpleNamespace(load=lambda path, device: FakeAgent())}):
                adapter.predict("base", graph, 2, 3, [1], visited=[0, 1, 2])
                self.assertEqual(seen["recent_articles"], ["A", "B", "C"])
                adapter.predict("base", graph, 2, 3, [1])
                self.assertEqual(seen["recent_articles"], [])


class ConcurrentCheckpointLoadTest(unittest.TestCase):
    def test_two_lanes_load_without_overlapping_runtime_initialization(self):
        active = 0
        gate = threading.Barrier(2)
        lock = threading.Lock()

        class FakeAgent:
            def predict(self, model_state, questions):
                labels = questions["next_link"]["criteria"]
                return {"answers": {"next_link": {"choice": "link_0",
                    "probabilities": {name: (1.0 if name == "link_0" else 0.0) for name in labels}}}}

        def fake_load(path, device):
            nonlocal active
            with lock:
                active += 1
                if active > 1:
                    raise RuntimeError("overlapping checkpoint loads")
            time.sleep(.03)
            with lock:
                active -= 1
            return FakeAgent()

        graph = {"titles": ["Start", "Goal"], "links": [[1], []], "indegree": [0, 1]}
        with tempfile.TemporaryDirectory() as temp:
            adapter = ModelAdapter()
            adapter._paths = {name: str(Path(temp, name)) for name in ("base", "tuned")}
            for path in adapter._paths.values():
                Path(path).mkdir()

            def run(name):
                gate.wait()
                return adapter.predict(name, graph, 0, 1, [1])

            with patch.dict(sys.modules, {"laya": types.SimpleNamespace(load=fake_load)}):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(run, name) for name in ("base", "tuned")]
                    outcomes = [future.result(timeout=2) for future in futures]
            self.assertEqual([result["index"] for result in outcomes], [1, 1])


if __name__ == "__main__":
    unittest.main()
