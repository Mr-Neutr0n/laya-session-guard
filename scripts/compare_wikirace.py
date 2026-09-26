"""Render a paired, reproducible base versus tuned Wikiracing report."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/wikirace/tuned"
REPORT = ROOT / "reports/WIKIRACE_RESULTS.md"


def interval(differences, seed=42):
    rng = random.Random(seed)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(5000)]
    return tuple(round(float(np.percentile(samples, p)), 4) for p in (2.5, 97.5))


def main():
    report = json.loads((OUTPUT / "report.json").read_text())
    if report.get("status") != "completed":
        raise ValueError("No completed Kaggle training report")
    traces = json.loads((OUTPUT / "race_traces.json").read_text())
    decisions = json.loads((OUTPUT / "decision_predictions.json").read_text())
    base_races, tuned_races = traces["base"], traces["tuned"]
    assert [(x["start"], x["target"]) for x in base_races] == [(x["start"], x["target"]) for x in tuned_races]
    assert [x["id"] for x in decisions["base"]] == [x["id"] for x in decisions["tuned"]]
    heldout_rows = {row["id"]: row for row in (json.loads(line) for line in
                    (ROOT / "data/wikirace/test.jsonl").read_text().splitlines())}
    race_delta = [int(t["success"]) - int(b["success"]) for b, t in zip(base_races, tuned_races)]
    new_wins = sum(not b["success"] and t["success"] for b, t in zip(base_races, tuned_races))
    lost_wins = sum(b["success"] and not t["success"] for b, t in zip(base_races, tuned_races))
    decision_delta = [int(max(range(len(t["logits"])), key=t["logits"].__getitem__) == t["label"]) -
                      int(max(range(len(b["logits"])), key=b["logits"].__getitem__) == b["label"])
                      for b, t in zip(decisions["base"], decisions["tuned"])]
    b, t = report["base"], report["tuned"]
    indirect = [(base, tuned) for base, tuned in zip(decisions["base"], decisions["tuned"])
                if heldout_rows[base["id"]]["target"] not in heldout_rows[base["id"]]["candidates"]]
    def correct(row):
        return max(range(len(row["logits"])), key=row["logits"].__getitem__) == row["label"]
    indirect_base = sum(correct(base) for base, _ in indirect) / len(indirect)
    indirect_tuned = sum(correct(tuned) for _, tuned in indirect) / len(indirect)
    choice_ci = interval(decision_delta)
    race_ci = interval(race_delta)
    lines = ["# Wikiracing results", "", "Source: archived Stanford Wikispeedia graph. The same 600 held-out choices and 60 held-out races were used for both checkpoints. Test targets do not occur in training or calibration.", "",
             "| Measure | Base Laya | Fine-tuned Laya |", "| --- | ---: | ---: |",
             f"| Choice accuracy, 600 decisions | {b['decisions']['top1_accuracy']:.1%} | {t['decisions']['top1_accuracy']:.1%} |",
             f"| Choice accuracy without direct target, {len(indirect)} decisions | {indirect_base:.1%} | {indirect_tuned:.1%} |",
             f"| Choice Brier score | {b['decisions']['brier']:.3f} | {t['decisions']['brier']:.3f} |",
             f"| Choice ECE, 10 bins | {b['decisions']['ece_10_bin']:.3f} | {t['decisions']['ece_10_bin']:.3f} |",
             f"| Races won within 8 hops, 60 races | {b['races']['wins']} ({b['races']['win_rate']:.1%}) | {t['races']['wins']} ({t['races']['win_rate']:.1%}) |",
             f"| Model latency p50, T4 | {b['races']['latency_ms_p50']:.1f} ms | {t['races']['latency_ms_p50']:.1f} ms |",
             f"| Model latency p95, T4 | {b['races']['latency_ms_p95']:.1f} ms | {t['races']['latency_ms_p95']:.1f} ms |", "",
             f"Paired choice-accuracy difference (tuned minus base): {sum(decision_delta)/len(decision_delta):+.1%}; 95% paired bootstrap interval [{choice_ci[0]:.1%}, {choice_ci[1]:.1%}].",
             f"Paired race-win difference: {sum(race_delta)/len(race_delta):+.1%}; 95% paired bootstrap interval [{race_ci[0]:.1%}, {race_ci[1]:.1%}]. The fine-tune won {new_wins} races that base lost and lost {lost_wins} that base won.", "",
             "The shortlist permits a path within eight moves for 55 of 60 races. Model latency measures warm GPU inference including batch preparation and transfer, but excluding tokenization, checkpoint load, graph lookup, browser rendering, and live network requests. The game uses a fixed historical subset of Wikipedia, so these results do not establish performance on the current full site.", "",
             f"Training took {report['training_seconds']:.1f} seconds on {report['gpu']}. Base revision: `{report['base_revision']}`. See `outputs/wikirace/tuned/report.json`, `race_traces.json`, and `decision_predictions.json` for per-case results."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
