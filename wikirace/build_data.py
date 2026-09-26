"""Build a deterministic Wikispeedia graph and target-disjoint training splits."""
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import unquote

from wikirace.graph import distances_to, reverse_graph, shortlist, state, question

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/wikirace/source/wikispeedia_paths-and-graph"
OUTPUT = ROOT / "data/wikirace"
SEED = 42


def rows(name):
    for line in (SOURCE / name).read_text().splitlines():
        if line and not line.startswith("#"):
            yield line.split("\t")


def build_graph():
    raw_titles = [parts[0] for parts in rows("articles.tsv")]
    ids = {title: i for i, title in enumerate(raw_titles)}
    links = [set() for _ in ids]
    for source, target in rows("links.tsv"):
        if source in ids and target in ids and source != target:
            links[ids[source]].add(ids[target])
    links = [sorted(v) for v in links]
    indegree = [0] * len(ids)
    for neighbors in links:
        for v in neighbors:
            indegree[v] += 1
    graph = {"titles": [unquote(x).replace("_", " ") for x in raw_titles], "links": links, "indegree": indegree,
             "source": "Stanford SNAP Wikispeedia paths-and-graph archive"}
    (OUTPUT / "graph.json").write_text(json.dumps(graph, separators=(",", ":")))
    return graph, ids


def build():
    random.seed(SEED)
    graph, ids = build_graph()
    incoming = reverse_graph(graph)
    human_paths = []
    for parts in rows("paths_finished.tsv"):
        path = [ids[x] for x in parts[3].split(";") if x in ids and x != "<"]
        if len(path) >= 2 and path[-1] != path[0] and all(b in graph["links"][a] for a, b in zip(path, path[1:])):
            human_paths.append(path)
    targets = sorted(set(path[-1] for path in human_paths))
    random.shuffle(targets)
    n = len(targets)
    split_targets = {"train": set(targets[:int(n*.8)]), "calibration": set(targets[int(n*.8):int(n*.9)]),
                     "test": set(targets[int(n*.9):])}
    rows_by_split = {key: [] for key in split_targets}
    races_by_split = {key: [] for key in split_targets}
    seen_races = {key: set() for key in split_targets}
    seen_decisions = {key: set() for key in split_targets}
    dist_cache = {}
    for path in human_paths:
        target = path[-1]
        split = next(key for key, values in split_targets.items() if target in values)
        if target not in dist_cache:
            dist_cache[target] = distances_to(incoming, target)
        distances = dist_cache[target]
        start = path[0]
        if 2 <= distances.get(start, 99) <= 5 and (start, target) not in seen_races[split]:
            races_by_split[split].append({"start": start, "target": target, "shortest_hops": distances[start]})
            seen_races[split].add((start, target))
        for step, current in enumerate(path[:-1]):
            key = (current, target)
            if key in seen_decisions[split] or distances.get(current, 99) > 5:
                continue
            candidates = shortlist(graph, current, target)
            best_distance = min(distances.get(v, 99) for v in candidates)
            if best_distance >= distances.get(current, 99):
                continue
            optimal = [i for i, v in enumerate(candidates) if distances.get(v, 99) == best_distance]
            human_next = path[step + 1]
            if human_next in candidates and candidates.index(human_next) in optimal:
                best_index = candidates.index(human_next)
            else:
                best_index = min(optimal, key=lambda i: hashlib.sha256(f"label:{current}:{target}:{candidates[i]}".encode()).digest())
            rows_by_split[split].append({"id": f"{split}-{len(rows_by_split[split])}", "current": current,
                                         "target": target, "candidates": candidates, "label": best_index,
                                         "state": state(graph, current, target), "question": question(graph, candidates)})
            seen_decisions[split].add(key)
    for split, values in rows_by_split.items():
        random.shuffle(values)
        cap = {"train": 6000, "calibration": 600, "test": 600}[split]
        values = values[:cap]
        with (OUTPUT / f"{split}.jsonl").open("w") as f:
            for row in values:
                f.write(json.dumps(row) + "\n")
    # Freeze a manageable race set from held-out targets.
    races = sorted(races_by_split["test"], key=lambda r: hashlib.sha256(f"{SEED}:{r['start']}:{r['target']}".encode()).hexdigest())[:60]
    (OUTPUT / "eval_races.json").write_text(json.dumps(races, indent=2))
    archive = OUTPUT / "source/wikispeedia_paths-and-graph.tar.gz"
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    report = {"seed": SEED, "articles": len(graph["titles"]), "links": sum(map(len, graph["links"])),
              "valid_human_paths": len(human_paths), "target_counts": {k: len(v) for k, v in split_targets.items()},
              "decision_counts": {k: min(len(v), {"train": 6000, "calibration": 600, "test": 600}[k]) for k, v in rows_by_split.items()},
              "eval_races": len(races), "source_url": "https://snap.stanford.edu/data/wikispeedia.html",
              "source_archive_sha256": archive_sha256}
    (OUTPUT / "provenance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
