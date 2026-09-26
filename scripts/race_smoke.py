"""Drive real /api/step races the way app/app.js does, and assert the mode rules hold.

Local smoke test. Needs a running app/server.py and both checkpoints:

    LAYA_DEVICE=cpu .venv/bin/python app/server.py &
    .venv/bin/python scripts/race_smoke.py http://127.0.0.1:8765 [route_index]
"""
import json
import sys
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"
HOP_LIMIT = {"benchmark": 8, "explore": 16}


def post(base, payload):
    request = Request(f"{base}/api/step", method="POST",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except Exception as error:  # HTTPError carries the parsed body
        return getattr(error, "code", 0), json.loads(error.read())


def race(base, model, route, mode):
    """Mirror raceModel() in app/app.js."""
    current, visited, trail, times = route["start"], [route["start"]], [], []
    for _ in range(HOP_LIMIT[mode]):
        status, step = post(base, {"model": model, "current": current, "target": route["target"],
                                   "visited": visited, "mode": mode})
        assert status == 200, (status, step)
        if step["outcome"] == "dead_end":
            return {"outcome": "dead_end", "reason": step.get("reason"), "trail": trail,
                    "hops": len(trail), "ms_mean": None}
        assert step["current"] == current
        by_index = {item["index"]: item for item in step["candidates"]}
        assert step["choice"] == by_index[step["index"]]["title"]
        assert abs(sum(item["probability"] for item in step["candidates"]) - 1.0) < 1e-6
        if mode == "explore":
            assert step["index"] not in visited, ("revisit", step["index"], trail)
        times.append(step["model_ms"])
        current = step["index"]
        trail.append(current)
        if current == route["target"]:
            break
        visited.append(current)
    won = current == route["target"]
    repeats = len(trail) != len(set(trail))
    return {"outcome": "won" if won else "hop_limit", "trail": trail, "hops": len(trail),
            "repeats": repeats, "ms_mean": round(sum(times) / len(times), 1) if times else None}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    route_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for mode in ("benchmark", "explore"):
        for model in ("base", "tuned"):
            with urlopen(f"{base}/api/races") as response:
                route = json.load(response)["races"][route_index]
            result = race(base, model, route, mode)
            limit = HOP_LIMIT[mode]
            assert result["hops"] <= limit, result
            if mode == "explore":
                assert not result["repeats"], result
            print(f"route {route_index} {route['start_title']} -> {route['target_title']}")
            print(f"  {mode:9s} {model:5s} {result['outcome']:9s} hops={result['hops']}/{limit} "
                  f"repeats={result['repeats']} mean_ms={result['ms_mean']}")
    with urlopen(f"{base}/api/timing") as response:
        print("timing", json.load(response))


if __name__ == "__main__":
    main()
