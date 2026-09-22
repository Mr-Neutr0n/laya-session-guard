"""Compare frozen per-session outputs without making API calls or tuning prompts."""
import hashlib
import json
from pathlib import Path
import random
from evaluate_jev import metrics


def main():
    paths={"base_laya":"reports/challenge_base_results.json",
           "finetuned_laya":"reports/challenge_results.json",
           "jev":"reports/jev_challenge_results.json"}
    reports={name:json.loads(Path(path).read_text()) for name,path in paths.items()}
    expected=[json.loads(line) for line in Path("data/challenge.jsonl").read_text().splitlines()]
    for report in reports.values():
        assert [(r["id"],r["gold"]) for r in report["predictions"]]==[(r["id"],r["labels"]) for r in expected]
    result={"n_sessions":len(expected),"synthetic_only":True,
        "dataset_sha256":hashlib.sha256(Path("data/challenge.jsonl").read_bytes()).hexdigest(),
        "decision_policy":"Raw returned choice, no confidence threshold or wrapper",
        "models":{name:{"metrics":metrics(report["predictions"]),
            "raw_nonallow_cases_incorrectly_allowed":sum(p["answers"]["action"]["choice"]=="allow" and p["gold"]["action"]!="allow" for p in report["predictions"])} for name,report in reports.items()},
        "paired_jev_minus_finetuned":{}}
    for qid in ("content","action"):
        differences=[int(j["answers"][qid]["choice"]==j["gold"][qid])-int(f["answers"][qid]["choice"]==f["gold"][qid])
            for j,f in zip(reports["jev"]["predictions"],reports["finetuned_laya"]["predictions"])]
        rng=random.Random(42)
        draws=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(10000))
        result["paired_jev_minus_finetuned"][qid]={"accuracy_difference":sum(differences)/len(differences),
            "jev_only_correct":differences.count(1),"finetuned_only_correct":differences.count(-1),
            "paired_bootstrap_95_percentile_interval":[draws[249],draws[9749]],"bootstrap_seed":42,"bootstrap_samples":10000}
    result["limitations"]=["Intervals resample only these 24 synthetic cases, not a representative real-session population",
        "Original labels remain frozen, including a debatable block-versus-review case",
        "API latency includes network; no fair hardware-normalized latency comparison",
        "Probability calibration differs across models; the primary comparison is raw label correctness"]
    Path("reports/jev_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
