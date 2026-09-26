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
