"""Bounded session handling. Never silently discard transcript content."""
import json


def checked_sequence(tok, state, question, max_len=1024, head_max_len=192):
    from laya.common import build_sequence
    q = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
    full, markers = build_sequence(tok, state, q, 1_000_000, head_max_len)
    if len(full) > max_len:
        raise ValueError(f"Session requires {len(full)} tokens, budget {max_len}; use windowed review")
    if len(markers) != len(question["criteria"]):
        raise ValueError("Question options were truncated")
    return full, markers


def session_windows(state, tok, questions, max_len=1024, head_max_len=192):
    """Repeat task/action and retain complete ordered events. Oversized events fail closed."""
    def fits(value):
        try:
            for q in questions.values():
                checked_sequence(tok, value, q, max_len, head_max_len)
            return True
        except ValueError:
            return False
    if fits(state):
        return [state]
    base = {k: v for k, v in state.items() if k != "events"}
    base["context_partitioned"] = True
    base["events"] = []
    if not fits(base):
        raise ValueError("Task and proposed action exceed the context budget")
    windows, current = [], dict(base)
    current["events"] = []
    for event in state.get("events", []):
        candidate = dict(current, events=current["events"] + [event])
        if fits(candidate):
            current = candidate
        else:
            if current["events"]:
                windows.append(current)
            current = dict(base, events=[event])
            if not fits(current):
                raise ValueError("A transcript event exceeds the budget; human review required")
    if current["events"]:
        windows.append(current)
    return windows
