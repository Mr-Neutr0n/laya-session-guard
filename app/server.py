"""Local Wikiracing app. Run with python3 app/server.py."""

import argparse
import json
import math
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wikirace.graph import load_graph, shortlist  # noqa: E402
from model_adapter import ModelAdapter, ModelUnavailable  # noqa: E402

GRAPH_FILE = ROOT / "data/wikirace/graph.json"
RACES_FILE = ROOT / "data/wikirace/eval_races.json"
REPORT_FILE = ROOT / "outputs/wikirace/tuned/report.json"


def percentile(values, rate):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * rate
    low = math.floor(position)
    high = math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 2)


class RaceServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, Handler)
        self.graph = load_graph(GRAPH_FILE)
        self.races = json.loads(RACES_FILE.read_text()) if RACES_FILE.exists() else []
        self.report = json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else None
        self.models = ModelAdapter()
        self.timings = {"base": [], "tuned": []}
        self.timings_lock = Lock()

    def timing_summary(self):
        with self.timings_lock:
            return {name: {"n": len(values), "p50_ms": percentile(values, .50),
                           "p95_ms": percentile(values, .95)}
                    for name, values in self.timings.items()}


class Handler(BaseHTTPRequestHandler):
    def json_response(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/races":
            graph = self.server.graph
            races = [{**row, "start_title": graph["titles"][row["start"]],
                      "target_title": graph["titles"][row["target"]]}
                     for row in self.server.races]
            self.json_response(200, {"source": "Archived Wikispeedia article graph",
                                     "article_count": len(graph["titles"]), "races": races,
                                     "models": self.server.models.status(),
                                     "report": self.server.report,
                                     "timing": self.server.timing_summary()})
            return
        if route == "/api/timing":
            self.json_response(200, self.server.timing_summary())
            return
        name = "index.html" if route == "/" else route.lstrip("/")
        if name not in {"index.html", "style.css", "app.js", "favicon.svg"}:
            self.send_error(404)
            return
        file = (APP / name).resolve()
        if APP not in file.parents or not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/step":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 16384:
                raise ValueError("Request body must be at most 16 KB")
            request = json.loads(self.rfile.read(size))
            graph = self.server.graph
            count = len(graph["titles"])
            current, target = request["current"], request["target"]
            visited = request.get("visited", [])
            name = request["model"]
            mode = request.get("mode", "benchmark")
            if (type(current) is not int or type(target) is not int or
                    not 0 <= current < count or not 0 <= target < count or
                    not isinstance(visited, list) or len(visited) > 32 or
                    any(type(v) is not int or not 0 <= v < count for v in visited) or
                    name not in ("base", "tuned") or mode not in ("benchmark", "explore") or
                    (mode == "explore" and (not visited or visited[-1] != current or
                                            len(visited) != len(set(visited)) or len(visited) > 16))):
                raise ValueError("Invalid race state")
            options = shortlist(graph, current, target,
                                exclude=visited if mode == "explore" else ())
            if not options:
                self.json_response(200, {"outcome": "dead_end", "candidates": [],
                                         "reason": "No unvisited outgoing links" if mode == "explore"
                                                   else "No outgoing links"})
                return
            # Benchmark keeps the evaluated input: no race history. Explore tells the model where it has been.
            result = self.server.models.predict(name, graph, current, target, options,
                                                visited if mode == "explore" else ())
            with self.server.timings_lock:
                self.server.timings[name].append(result["model_ms"])
            self.json_response(200, {"outcome": "move", "current": current, "target": target,
                                     "candidates": [{"index": index, "title": graph["titles"][index],
                                                     "probability": result["probabilities"][graph["titles"][index]]}
                                                    for index in options],
                                     **result, "timing": self.server.timing_summary()})
        except ModelUnavailable as error:
            self.json_response(503, {"error": str(error), "code": "model_unavailable"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.json_response(400, {"error": str(error), "code": "bad_request"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = RaceServer((args.host, args.port))
    print(f"Wikiracing at http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()
