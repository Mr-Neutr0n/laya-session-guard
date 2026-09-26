"""The race handler must send history in Explore and nothing in Benchmark."""
import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import server as wikirace_server  # noqa: E402

RaceServer = wikirace_server.RaceServer


class RecordingModels:
    def __init__(self, graph):
        self.graph = graph
        self.seen = []

    def status(self):
        return {"base": {"available": True}, "tuned": {"available": True}}

    def predict(self, name, graph, current, target, options, visited=()):
        self.seen.append((name, current, target, tuple(visited)))
        return {"choice": graph["titles"][options[0]], "index": options[0],
                "probabilities": {graph["titles"][i]: 1.0 / len(options) for i in options},
                "model_ms": 1.0}


class RaceHandlerTest(unittest.TestCase):
    def setUp(self):
        from wikirace.graph import load_graph

        self.graph = load_graph()
        # Start is a real article with outgoing links; pick a second article as the target.
        start = next(i for i, links in enumerate(self.graph["links"]) if len(links) >= 3)
        target = next(i for i in self.graph["links"][start] if i != start)
        self.start, self.target = start, target
        self.models = RecordingModels(self.graph)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), wikirace_server.Handler)
        self.httpd.graph = self.graph
        self.httpd.races = []
        self.httpd.report = None
        self.httpd.models = self.models
        self.httpd.timings = {"base": [], "tuned": []}
        self.httpd.timings_lock = threading.Lock()
        self.httpd.timing_summary = RaceServer.timing_summary.__get__(self.httpd)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def step(self, payload):
        connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=5)
        connection.request("POST", "/api/step", json.dumps(payload),
                           {"Content-Type": "application/json"})
        body = json.loads(connection.getresponse().read())
        connection.close()
        return body

    def test_explore_sends_visited_history_and_benchmark_sends_none(self):
        visited = [self.start, self.target]
        explore = self.step({"model": "base", "current": self.target, "target": self.start,
                             "visited": visited, "mode": "explore"})
        self.assertEqual(explore["outcome"], "move")
        self.assertEqual(self.models.seen[-1], ("base", self.target, self.start, (self.start, self.target)))
        benchmark = self.step({"model": "base", "current": self.target, "target": self.start,
                               "visited": visited})
        self.assertEqual(benchmark["outcome"], "move")
        self.assertEqual(self.models.seen[-1], ("base", self.target, self.start, ()))

    def test_benchmark_offers_repeats_and_explore_excludes_visited_links(self):
        from wikirace.graph import shortlist

        here = self.target
        offered = shortlist(self.graph, here, self.start)
        gone = offered[0]
        gone_title = self.graph["titles"][gone]
        repeated = self.step({"model": "base", "current": here, "target": self.start, "visited": [here]})
        self.assertIn(gone_title, [item["title"] for item in repeated["candidates"]])
        explored = self.step({"model": "base", "current": here, "target": self.start,
                              "visited": [gone, here], "mode": "explore"})
        titles = [item["title"] for item in explored["candidates"]]
        self.assertNotIn(gone_title, titles)
        # Excluding one offered link refills the menu with the next best link.
        self.assertEqual(titles, [self.graph["titles"][i] for i in shortlist(self.graph, here, self.start, exclude=(gone, here))])


if __name__ == "__main__":
    unittest.main()
