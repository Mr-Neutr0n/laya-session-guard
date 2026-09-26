"""Derive a larger held-out race set with the same procedure and seed as build_data.py.

The 60 published races are the first 60 entries of the same deterministic ordering, so the
larger set is a strict superset and the two measurements stay comparable.

    PYTHONPATH=. python3 scripts/build_wikirace_races.py --count 240
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from wikirace.build_data import SEED, SOURCE, rows
from wikirace.graph import distances_to, load_graph, reverse_graph

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/wikirace"
PUBLISHED = OUTPUT / "eval_races.json"


def human_paths(ids, links):
    """Same filter as wikirace/build_data.py: only fully verifiable finished paths."""
    paths = []
    for parts in rows("paths_finished.tsv"):
        path = [ids[x] for x in parts[3].split(";") if x in ids and x != "<"]
        if len(path) >= 2 and path[-1] != path[0] and all(b in links[a] for a, b in zip(path, path[1:])):
            paths.append(path)
    return paths


def test_races(count):
    raw_titles = [parts[0] for parts in rows("articles.tsv")]
    ids = {title: i for i, title in enumerate(raw_titles)}
    graph = load_graph(OUTPUT / "graph.json")
    assert len(graph["titles"]) == len(ids), "graph.json and articles.tsv disagree on article count"
    incoming = reverse_graph(graph)
    paths = human_paths(ids, graph["links"])
    random.seed(SEED)
    targets = sorted({path[-1] for path in paths})
    random.shuffle(targets)
    n = len(targets)
    test_targets = set(targets[int(n * .9):])
    seen, races, cache = set(), [], {}
    for path in paths:
        target = path[-1]
        if target not in test_targets:
            continue
        if target not in cache:
            cache[target] = distances_to(incoming, target)
        start = path[0]
        if not 2 <= cache[target].get(start, 99) <= 5 or (start, target) in seen:
            continue
        seen.add((start, target))
        races.append({"start": start, "target": target, "shortest_hops": cache[target][start]})
    races.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['start']}:{r['target']}".encode()).hexdigest())
    return races, targets, test_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()
    races, targets, test_targets = test_races(args.count)
    published = json.loads(PUBLISHED.read_text())
    assert races[:len(published)] == published, "ordering diverged from the published race set"
    print(f"test targets {len(test_targets)} of {len(targets)} | eligible test races {len(races)}")
    print(f"published 60 are a prefix of the ordering: {races[:60] == published}")
    if len(races) < args.count:
        print(f"requested {args.count} but only {len(races)} eligible races exist")
    selection = races[:args.count]
    print(f"selected {len(selection)} races; shortest hops "
          f"{min(r['shortest_hops'] for r in selection)} to {max(r['shortest_hops'] for r in selection)}")
    print("new routes beyond the published 60:", len(selection) - len(published))
    destination = OUTPUT / f"eval_races_{len(selection)}.json"
    destination.write_text(json.dumps(selection, indent=2))
    print("wrote", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
