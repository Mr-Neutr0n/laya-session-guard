"""Replay all 60 held-out races locally and compare with the recorded Kaggle traces.

Needs a running app/server.py and both checkpoints. Benchmark rules only: 8 hops,
repeats allowed, no race history, which is exactly what the Kaggle evaluation used.
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def step(base, payload):
    request = Request(f"{base}/api/step", method="POST", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.load(response)


def play(base, model, route, hop_limit=8):
    current, visited, trail, times = route["start"], [route["start"]], [], []
    for _ in range(hop_limit):
        reply = step(base, {"model": model, "current": current, "target": route["target"],
                            "visited": visited})
        if reply["outcome"] == "dead_end":
            break
        current = reply["index"]
        trail.append(current)
        times.append(reply["model_ms"])
        if current == route["target"]:
            break
        visited.append(current)
    return {"path": trail, "success": current == route["target"], "hops": len(trail), "times": times}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    with urlopen(f"{base}/api/races") as response:
        races = json.load(response)["races"]
    report_gpu = json.loads((ROOT / "outputs/wikirace/tuned/report.json").read_text())["gpu"]
    recorded = json.loads((ROOT / "outputs/wikirace/tuned/race_traces.json").read_text())
    for model, key in [("base", "base"), ("tuned", "tuned")]:
        wins = agree = same_hops = 0
        differences = []
        for index, route in enumerate(races):
            result = play(base, model, route)
            record = recorded[key][index]
            assert (route["start"], route["target"]) == (record["start"], record["target"])
            wins += result["success"]
            agree += result["success"] == record["success"]
            same_hops += result["hops"] == record["hops"]
            if result["success"] != record["success"]:
                differences.append({"race": index, "local": result["success"],
                                    "kaggle": record["success"], "local_hops": result["hops"],
                                    "kaggle_hops": record["hops"],
                                    "local_path": result["path"], "kaggle_path": record["path"]})
        recorded_times = [t for r in recorded[key] for t in r["latency_ms"]]
        print(f"{model:5s} local wins {wins}/60 | Kaggle wins {sum(r['success'] for r in recorded[key])}/60"
              f" | outcome agreement {agree}/60 | identical hop count {same_hops}/60"
              f" | Kaggle p50 {sorted(recorded_times)[len(recorded_times)//2]:.1f} ms on {report_gpu}")
        for difference in differences:
            print("  differs:", json.dumps(difference))
    with urlopen(f"{base}/api/timing") as response:
        print("local timing", json.load(response))


if __name__ == "__main__":
    main()
