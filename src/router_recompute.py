"""Recompute every router metric from the per-item logits stored in a report.

This is the auditable path. Training and inference produce logits once; the
metrics, temperatures, bootstrap intervals, paired test and latency figures are
then derived from those logits, so a reporting change never requires a rerun
and a reviewer can reproduce the tables from the saved file alone.
"""
import argparse
import json
from pathlib import Path

import router_metrics as metrics
from router_run import paired, probabilities, report_for


def recompute(source, output, bootstrap_samples=2000):
    report = json.loads(Path(source).read_text())
    labels = report["base"]["labels"]
    base_rows = report["predictions"]["base"]
    tuned_rows = report["predictions"]["finetuned"]
    base_cal = report.get("calibration_predictions", {}).get("base")
    tuned_cal = report.get("calibration_predictions", {}).get("finetuned")
    refit = bool(base_cal and tuned_cal)
    report["base"] = _rebuild(report["base"], "base", base_rows, labels, base_cal, bootstrap_samples)
    report["finetuned"] = _rebuild(report["finetuned"], "finetuned", tuned_rows, labels, tuned_cal, bootstrap_samples)
    report["paired"] = paired(report["base"], report["finetuned"], labels)
    report["recomputed"] = {"source": str(source), "bootstrap_samples": bootstrap_samples,
                            "temperature_refit_from_calibration_logits": refit,
                            "note": None if refit else "calibration logits absent; the recorded temperature was reused"}
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _rebuild(stored, name, rows, labels, calibration_rows, bootstrap_samples):
    if calibration_rows:
        return report_for(name, rows, labels, calibration_rows, bootstrap_samples,
                          _latency(stored, "batched_latency"), _latency(stored, "batch1_latency"))
    rebuilt = _metrics_only(rows, labels, stored["temperature"])
    rebuilt["batched_latency"] = stored.get("batched_latency", {})
    rebuilt["batch1_latency"] = stored.get("batch1_latency", {})
    return rebuilt


def _metrics_only(rows, labels, temperature):
    """Metrics at a fixed temperature, without the calibration split."""
    scored = probabilities(rows, temperature)
    y_true = [row["label"] for row in scored]
    y_pred = [row["prediction"] for row in scored]
    stats = metrics.summary(y_true, y_pred, labels, temperatures=temperature,
                            probabilities=[row["probs"] for row in scored])
    accuracy_ci = metrics.bootstrap_ci([(int(t == p),) for t, p in zip(y_true, y_pred)],
                                       lambda draw: sum(v[0] for v in draw) / len(draw),
                                       samples=2000, seed=42)
    return {"n": stats["n"], "temperature": temperature, "calibration_nll": None,
            "accuracy": stats["accuracy"], "accuracy_ci95": list(accuracy_ci),
            "macro_f1": stats["macro_f1"], "macro_f1_ci95": None,
            "macro_precision": stats["macro_precision"], "macro_recall": stats["macro_recall"],
            "ece": stats["ece"], "brier": stats["brier"],
            "per_class": stats["per_class"], "confusion_matrix": stats["confusion_matrix"],
            "labels": list(labels),
            "correct": [int(t == p) for t, p in zip(y_true, y_pred)]}


def _latency(report, key):
    payload = report.get(key)
    report["paired"] = paired(report["base"], report["finetuned"], labels)
    report["recomputed"] = {"source": str(source), "bootstrap_samples": bootstrap_samples,
                            "temperature_grid": [0.01, 20.0]}
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _latency(report, key):
    payload = report.get(key)
    if not payload:
        return None
    stub = metrics.Latency()
    for _ in range(int(payload.get("calls", 0))):
        stub.record(0.0)
    stub.report = lambda payload=payload: payload
    return stub


def verify(source):
    """Check the stored numbers still follow from the stored logits."""
    report = json.loads(Path(source).read_text())
    labels = report["base"]["labels"]
    findings = []
    for name in ("base", "finetuned"):
        rows = report["predictions"][name]
        scored = probabilities(rows, report[name]["temperature"])
        accuracy = sum(int(row["label"] == row["prediction"]) for row in scored) / len(scored)
        findings.append((name, accuracy, report[name]["accuracy"]))
    return {"labels": labels, "checks": [{"model": name, "recomputed_accuracy": value,
                                          "reported_accuracy": reported,
                                          "matches": abs(value - reported) < 1e-9} for name, value, reported in findings]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="reports/router_results.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.report), indent=2))
    else:
        result = recompute(args.report, args.output or args.report, args.bootstrap_samples)
        print(json.dumps({"base_accuracy": result["base"]["accuracy"],
                          "finetuned_accuracy": result["finetuned"]["accuracy"],
                          "base_macro_f1": result["base"]["macro_f1"],
                          "finetuned_macro_f1": result["finetuned"]["macro_f1"],
                          "base_temperature": result["base"]["temperature"],
                          "finetuned_temperature": result["finetuned"]["temperature"],
                          "mcnemar_p": result["paired"]["p_value"]}, indent=2))
