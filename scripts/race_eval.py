"""Replay held-out races locally and compare with the recorded Kaggle traces.

Needs a running app/server.py and both checkpoints. Benchmark rules only: 8 hops, repeats
allowed, no race history, which is exactly what the Kaggle evaluation used.

    python3 scripts/race_eval.py [--races data/wikirace/eval_races_240.json]
"""
import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

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


def paired_interval(differences, seed=42, draws=5000):
    rng = random.Random(seed)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    return tuple(float(np.percentile(samples, p)) for p in (2.5, 97.5))


def ceiling(races, graph, hop_limit=8):
    """Best case for this shortlist: hop only through links the model is actually offered."""
    import hashlib
    import math
    import re

    def terms(title):
        return set(re.findall(r"[a-z0-9]+", title.lower())) - {"the", "of", "and", "in", "a"}

    def shortlist(current, target, limit=12):
        titles, links = graph["titles"], graph["links"]
        available = list(links[current])
        if target in available:
            selected = [target] + [v for v in available if v != target][:limit - 1]
            return sorted(selected, key=lambda v: hashlib.sha256(f"{current}:{target}:{v}".encode()).digest())
        target_terms = terms(titles[target])

        def score(v):
            t = terms(titles[v])
            overlap = len(t & target_terms) / max(1, len(t | target_terms))
            return (4 * overlap + 0.10 * math.log1p(graph["indegree"][v]), -v)

        selected = sorted(available, key=score, reverse=True)[:limit]
        return sorted(selected, key=lambda v: hashlib.sha256(f"{current}:{target}:{v}".encode()).digest())

    results = []
    for route in races:
        start, target = route["start"], route["target"]
        seen, queue, reachable = {start}, deque([(start, 0)]), False
        while queue and not reachable:
            node, depth = queue.popleft()
            if depth >= hop_limit:
                continue
            for candidate in shortlist(node, target):
                if candidate == target:
                    reachable = True
                    break
                if candidate not in seen:
                    seen.add(candidate)
                    queue.append((candidate, depth + 1))
        results.append(reachable)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default=BASE)
    parser.add_argument("--races", default=str(ROOT / "data/wikirace/eval_races.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="both", choices=["both", "base", "tuned"])
    args = parser.parse_args()
    races = json.loads(Path(args.races).read_text())
    published = json.loads((ROOT / "data/wikirace/eval_races.json").read_text())
    recorded = json.loads((ROOT / "outputs/wikirace/tuned/race_traces.json").read_text())
    reachable = ceiling(races, json.loads((ROOT / "data/wikirace/graph.json").read_text()))
    print(f"races {len(races)} | shortlist permits a route within 8 moves for {sum(reachable)}")
    results = {}
    for model, key in [("base", "base"), ("tuned", "tuned")]:
        if args.model != "both" and args.model != model:
            continue
        outcomes = [play(args.base, model, route) for route in races]
        results[key] = outcomes
        wins = sum(outcome["success"] for outcome in outcomes)
        agreement = same_hops = 0
        for index, outcome in enumerate(outcomes):
            if index < len(published):
                assert (races[index]["start"], races[index]["target"]) == (published[index]["start"], published[index]["target"])
                record = recorded[key][index]
                agreement += outcome["success"] == record["success"]
                same_hops += outcome["hops"] == record["hops"]
        times = [t for outcome in outcomes for t in outcome["times"]]
        print(f"{model:5s} wins {wins}/{len(races)}"
              + (f" | vs Kaggle on the shared 60: outcome agreement {agreement}/60,"
                 f" identical hop count {same_hops}/60" if len(races) >= 60 else "")
              + f" | local p50 {sorted(times)[len(times)//2]:.0f} ms over {len(times)} moves")
    if len(results) == 2:
        delta = [int(t["success"]) - int(b["success"]) for b, t in zip(results["base"], results["tuned"])]
        lo, hi = paired_interval(delta)
        won = sum(not b["success"] and t["success"] for b, t in zip(results["base"], results["tuned"]))
        lost = sum(b["success"] and not t["success"] for b, t in zip(results["base"], results["tuned"]))
        print(f"paired win difference (tuned minus base) {np.mean(delta):+.1%}; "
              f"95% paired bootstrap interval [{lo:.1%}, {hi:.1%}]")
        print(f"tuned won {won} races base lost and lost {lost} that base won")
    else:
        delta = lo = hi = won = lost = None
    if args.output:
        payload = {"races": str(args.races), "n": len(races), "ceiling": int(sum(reachable)),
                   "base_wins": sum(r["success"] for r in results["base"]) if "base" in results else None,
                   "tuned_wins": sum(r["success"] for r in results["tuned"]) if "tuned" in results else None,
                   "paired_win_difference": float(np.mean(delta)) if delta is not None else None,
                   "interval": [lo, hi] if delta is not None else None,
                   "won": won, "lost": lost,
                   "paths": {k: [r["path"] for r in v] for k, v in results.items()}}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print("wrote", args.output)


if __name__ == "__main__":
    main()
