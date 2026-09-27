"""Classification metrics, latency summary, and paired comparison. Pure stdlib.

`verify` runs the tests with the system interpreter, which has no numpy and no
torch, so nothing here may import them.
"""
import math
import random
import time


def confusion(y_true, y_pred, n_classes):
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for truth, guess in zip(y_true, y_pred):
        matrix[truth][guess] += 1
    return matrix


def per_class(matrix, labels):
    n = len(labels)
    out = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(n))
        actual = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"support": actual, "predicted": predicted,
                      "precision": precision, "recall": recall, "f1": f1}
    return out


def summary(y_true, y_pred, labels, temperatures=None, probabilities=None):
    matrix = confusion(y_true, y_pred, len(labels))
    classes = per_class(matrix, labels)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    total = sum(sum(row) for row in matrix)
    result = {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(v["f1"] for v in classes.values()) / len(labels),
        "macro_precision": sum(v["precision"] for v in classes.values()) / len(labels),
        "macro_recall": sum(v["recall"] for v in classes.values()) / len(labels),
        "per_class": classes,
        "confusion_matrix": matrix,
        "labels": list(labels),
    }
    if probabilities is not None and temperatures is not None:
        n_classes = len(labels)
        brier = 0.0
        ece = 0.0
        bins = [[] for _ in range(10)]
        for index, (row, truth) in enumerate(zip(probabilities, y_true)):
            scaled = [p / temperatures for p in row]
            total_p = sum(scaled) or 1.0
            probs = [p / total_p for p in scaled]
            brier += sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))
            guess = max(range(n_classes), key=lambda i: probs[i])
            confidence = probs[guess]
            bins[min(int(confidence * 10), 9)].append((confidence, float(guess == truth)))
        result["brier"] = brier / len(probabilities)
        result["ece"] = sum(len(b) / len(probabilities) * abs(
            sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b))
            for b in bins if b)
    return result


def softmax(values, temperature):
    scaled = [value / temperature for value in values]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def fit_temperature(logits, y_true, grid=None, min_temperature=0.01, max_temperature=20.0):
    """Pick one scalar temperature by negative log likelihood on held-out logits.

    The input must be raw logits. Temperature scaling is defined on unnormalised
    scores: softmax(z / T). Feeding an already-normalised probability vector
    instead fits the log-probability family, and since p / T renormalises back
    to p, the transform being fitted is the identity. That path can still return
    a numerically chosen T, but it has no relationship to the model's logit
    scale, and applying the value it returns to the real logits does not
    minimise the calibration loss. The floor is deliberately low, so a
    confidently miscalibrated model is not silently clipped at the grid edge.
    """
    grid = grid or [round(min_temperature + 0.01 * i, 4)
                    for i in range(int((max_temperature - min_temperature) / 0.01) + 1)]
    best, best_loss = 1.0, float("inf")
    for temperature in grid:
        nll = 0.0
        for values, truth in zip(logits, y_true):
            probs = softmax(values, temperature)
            nll -= math.log(max(probs[truth], 1e-12))
        loss = nll / len(logits)
        if loss < best_loss - 1e-12:
            best, best_loss = temperature, loss
    return best, best_loss


def bootstrap_ci(values, statistic, samples=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over examples. `values` is a list of per-example tuples."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(draw))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * samples) - 1)]
    hi = stats[min(samples - 1, int((1 - alpha / 2) * samples))]
    return (lo, hi)


def accuracy_statistic(draw):
    return sum(1 for truth, _ in draw if truth == 1) / len(draw)


def mcnemar(rows):
    """Exact-ish paired test on the shared held-out set.

    `rows` is a list of (base_correct, tuned_correct) booleans.
    """
    b_only = sum(1 for b, t in rows if b and not t)
    t_only = sum(1 for b, t in rows if t and not b)
    n = b_only + t_only
    if n == 0:
        return {"base_only_correct": 0, "tuned_only_correct": 0, "p_value": 1.0, "n_discordant": 0}
    # Two-sided exact binomial test against p = 0.5.
    k = min(b_only, t_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"base_only_correct": b_only, "tuned_only_correct": t_only,
            "n_discordant": n, "p_value": min(1.0, 2 * tail)}


class Latency:
    """Wall-clock accounting for inference.

    `record` takes one call's duration and how many examples it covered, because a
    batched call is not a per-example measurement. The first call of a pass is a cold
    start and is excluded from the steady-state means.
    """

    def __init__(self):
        self.samples = []
        self.items = []
        self.started = time.perf_counter()

    def time_call(self, function, *args, **kwargs):
        start = time.perf_counter()
        value = function(*args, **kwargs)
        self.record((time.perf_counter() - start) * 1000.0)
        return value

    def record(self, milliseconds, examples=1):
        self.samples.append(milliseconds)
        self.items.append(examples)

    def report(self):
        if not self.samples:
            return {}
        per_example = [ms / max(1, n) for ms, n in zip(self.samples, self.items)]
        steady = per_example[1:] if len(per_example) > 1 else per_example
        steady_calls = self.samples[1:] if len(self.samples) > 1 else self.samples
        ordered = sorted(per_example)

        def pct(values, q):
            return values[min(len(values) - 1, max(0, int(q * len(values))))]

        return {"calls": len(self.samples), "examples": sum(self.items),
                "total_seconds": sum(self.samples) / 1000.0,
                "ms_per_example_mean": sum(steady) / len(steady),
                "ms_per_example_p50": pct(ordered, 0.50),
                "ms_per_example_p95": pct(ordered, 0.95),
                "ms_per_example_including_cold_start": sum(per_example) / len(per_example),
                "ms_per_call_mean": sum(steady_calls) / len(steady_calls),
                "cold_start_excluded": len(self.samples) > 1,
                "wall_seconds_including_load": time.perf_counter() - self.started}
