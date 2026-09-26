"""Load the archived Wikispeedia graph and construct legal model choices."""
import json
import hashlib
import math
import re
from collections import deque
from pathlib import Path

GRAPH_PATH = Path(__file__).resolve().parents[1] / "data/wikirace/graph.json"


def load_graph(path=GRAPH_PATH):
    return json.loads(Path(path).read_text())


def terms(title):
    return set(re.findall(r"[a-z0-9]+", title.lower())) - {"the", "of", "and", "in", "a"}


def shortlist(graph, current, target, limit=12):
    """Rank only legal outgoing links, without consulting distance to target."""
    titles, links = graph["titles"], graph["links"]
    available = links[current]
    if target in available:
        selected = [target] + [v for v in available if v != target][:limit - 1]
        return sorted(selected, key=lambda v: hashlib.sha256(f"{current}:{target}:{v}".encode()).digest())
    target_terms = terms(titles[target])
    indegree = graph["indegree"]

    def score(v):
        t = terms(titles[v])
        overlap = len(t & target_terms) / max(1, len(t | target_terms))
        # Hubs are useful for getting out of narrow topics, but do not dominate.
        return (4 * overlap + 0.10 * math.log1p(indegree[v]), -v)

    selected = sorted(available, key=score, reverse=True)[:limit]
    return sorted(selected, key=lambda v: hashlib.sha256(f"{current}:{target}:{v}".encode()).digest())


def question(graph, candidates):
    return {"type": "choice", "instructions": "Which linked article is the best next step toward the target article?", "criteria": {f"link_{i}": graph["titles"][v] for i, v in enumerate(candidates)}}


def state(graph, current, target, visited=()):
    return {"current_article": graph["titles"][current], "target_article": graph["titles"][target], "recent_articles": [graph["titles"][v] for v in tuple(visited)[-3:]]}


def reverse_graph(graph):
    incoming = [[] for _ in graph["titles"]]
    for u, neighbors in enumerate(graph["links"]):
        for v in neighbors:
            incoming[v].append(u)
    return incoming


def distances_to(incoming, target):
    distances = {target: 0}
    queue = deque([target])
    while queue:
        v = queue.popleft()
        for u in incoming[v]:
            if u not in distances:
                distances[u] = distances[v] + 1
                queue.append(u)
    return distances
