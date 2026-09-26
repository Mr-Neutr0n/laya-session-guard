"""Lazy Laya inference. Local checkpoint paths come from environment variables."""

import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

from wikirace.graph import question, state

QUESTION_ID = "next_link"


class ModelUnavailable(Exception):
    pass


class ModelAdapter:
    def __init__(self):
        self._agents = {}
        self._locks = {"base": Lock(), "tuned": Lock()}
        self._load_lock = Lock()
        self._paths = {
            "base": os.environ.get("LAYA_BASE_MODEL", "outputs/wikirace/base"),
            "tuned": os.environ.get("LAYA_TUNED_MODEL", "outputs/wikirace/tuned"),
        }
        self.device = os.environ.get("LAYA_DEVICE", "cpu")

    def status(self):
        return {name: {"available": Path(path).is_dir(), "path": path}
                for name, path in self._paths.items()}

    def predict(self, name, graph, current, target, options, visited=()):
        if name not in self._paths:
            raise ValueError("Unknown model")
        path = Path(self._paths[name])
        if not path.is_dir():
            raise ModelUnavailable(f"{name.title()} checkpoint is not on this machine")
        model_state = state(graph, current, target)
        questions = {QUESTION_ID: question(graph, options)}
        labels = questions[QUESTION_ID]["criteria"]
        with self._locks[name]:
            if name not in self._agents:
                with self._load_lock:
                    os.environ.setdefault("USE_TF", "0")
                    os.environ.setdefault("USE_FLAX", "0")
                    try:
                        import laya
                        self._agents[name] = laya.load(str(path), device=self.device)
                    except Exception as error:
                        raise ModelUnavailable(f"Could not load {name} checkpoint: {type(error).__name__}: {error}") from error
            start = perf_counter()
            try:
                answer = self._agents[name].predict(model_state, questions)["answers"][QUESTION_ID]
            except Exception as error:
                raise ModelUnavailable(f"{name.title()} inference failed: {type(error).__name__}: {error}") from error
            model_ms = (perf_counter() - start) * 1000
        choice = answer["choice"]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(labels) or choice not in labels:
            raise ModelUnavailable("Model returned a link outside the candidate list")
        values = list(probabilities.values())
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 for value in values):
            raise ModelUnavailable("Model returned invalid probabilities")
        total = sum(values)
        if total <= 0:
            raise ModelUnavailable("Model returned zero probability mass")
        probabilities = {graph["titles"][index]: float(probabilities[f"link_{i}"]) / total
                         for i, index in enumerate(options)}
        return {
            "choice": graph["titles"][options[list(labels).index(choice)]],
            "index": options[list(labels).index(choice)],
            "probabilities": probabilities,
            "model_ms": round(model_ms, 2),
        }
