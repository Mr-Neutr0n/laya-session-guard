"""Compare the frozen 24-case challenge predictions from three Laya runs."""
import argparse
import json
from pathlib import Path


def summarize(report):
    predictions = report["predictions"]
    total = len(predictions)
    return {
        "content": sum(p["answers"]["content"]["choice"] == p["gold"]["content"] for p in predictions) / total,
        "action": sum(p["answers"]["action"]["choice"] == p["gold"]["action"] for p in predictions) / total,
        "suspicious": (sum(p["answers"]["content"]["choice"] == "suspicious" for p in predictions
                           if p["gold"]["content"] == "suspicious"),
                       sum(p["gold"]["content"] == "suspicious" for p in predictions)),
        "block": (sum(p["answers"]["action"]["choice"] == "block" for p in predictions
                      if p["gold"]["action"] == "block"),
                  sum(p["gold"]["action"] == "block" for p in predictions)),
        "false_allow": sum(p["answers"]["action"]["choice"] == "allow" and p["gold"]["action"] != "allow"
                           for p in predictions),
        "wrapper_false_allow": report["wrapper"]["nonallow_cases_incorrectly_allowed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/followup")
    args = parser.parse_args()
    root = Path(args.output_root)
    paths = {"Previous 3 epochs": Path("reports/challenge_results.json"),
             "Original data, 8 epochs": root / "followup-original" / "challenge.json",
             "Counterfactual data, 8 epochs": root / "followup-counterfactual" / "challenge.json"}
    print("| Run | Content | Action | Suspicious recall | Block recall | Raw false allows | Wrapper false allows |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, path in paths.items():
        report = json.loads(path.read_text())
        assert report["n_sessions"] == 24
        value = summarize(report)
        caught, suspicious = value["suspicious"]
        blocked, block = value["block"]
        print(f"| {name} | {value['content']:.1%} | {value['action']:.1%} | {caught}/{suspicious} | "
              f"{blocked}/{block} | {value['false_allow']} | {value['wrapper_false_allow']} |")


if __name__ == "__main__":
    main()
