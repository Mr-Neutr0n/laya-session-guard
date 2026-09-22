"""Paired Jev evaluation using unchanged Laya questions and synthetic states."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import time
from datetime import datetime, timezone
from session_data import QUESTIONS

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def read_key(env_file):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and env_file:
        for line in Path(env_file).expanduser().read_text().splitlines():
            line = line.strip().removeprefix("export ")
            name, separator, value = line.partition("=")
            if separator and name.strip() == "TYPESAFE_API_KEY":
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    key = parts[0]
    if not key:
        raise ValueError("TYPESAFE_API_KEY is missing")
    return key


def validate(answers):
    for qid, question in QUESTIONS.items():
        answer = answers[qid]
        p = answer["probabilities"]
        if set(p) != set(question["criteria"]) or answer["choice"] not in p:
            raise ValueError("Unexpected answer labels")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
            raise ValueError("Invalid probabilities")
        # Jev returns probabilities rounded to two decimals. Three entries may
        # legitimately sum to 0.99 or 1.01. Keep raw values in the saved output.
        if abs(sum(p.values()) - 1) > .005 * len(p) + 1e-9 or p[answer["choice"]] != max(p.values()):
            raise ValueError("Inconsistent probability distribution")


def metrics(rows):
    result = {}
    for qid, q in QUESTIONS.items():
        labels = list(q["criteria"])
        matrix = [[0] * len(labels) for _ in labels]
        brier, bins = [], [[] for _ in range(10)]
        for row in rows:
            gold = row["gold"][qid]; answer = row["answers"][qid]
            matrix[labels.index(gold)][labels.index(answer["choice"])]+=1
            raw = answer["probabilities"]
            p = {label:value/sum(raw.values()) for label,value in raw.items()}
            brier.append(sum((p[label] - int(label == gold))**2 for label in labels))
            confidence=max(p.values())
            bins[min(9,int(confidence*10))].append((confidence,int(answer["choice"]==gold)))
        result[qid] = {"n":len(rows),"accuracy":sum(matrix[i][i] for i in range(len(labels)))/len(rows),
            "labels":labels,"confusion_matrix_true_rows":matrix,
            "recall":{label:matrix[i][i]/max(1,sum(matrix[i])) for i,label in enumerate(labels)},
            "precision":{label:matrix[i][i]/max(1,sum(r[i] for r in matrix)) for i,label in enumerate(labels)},
            "brier":statistics.mean(brier),
            "ece":sum(len(b)/len(rows)*abs(statistics.mean(p for p,c in b)-statistics.mean(c for p,c in b)) for b in bins if b)}
    return result


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/challenge.jsonl")
    parser.add_argument("--output",default="reports/jev_challenge_results.json")
    parser.add_argument("--model",default="jev-1.13.0")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    data_path = Path(args.data); output=Path(args.output)
    rows=[json.loads(line) for line in data_path.read_text().splitlines()]
    cache_path=output.with_suffix(".jsonl")
    cached={}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row=json.loads(line);cached[row["request_sha256"]]=row
    session=requests.Session()
    session.headers.update({"Authorization":"Bearer "+read_key(args.env_file),"Content-Type":"application/json"})
    predictions=[];attempts=0
    for index,row in enumerate(rows):
        payload={"state":row["state"],"model":args.model,"questions":QUESTIONS}
        request_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        if request_hash in cached:
            prediction=dict(cached[request_hash],gold=row["labels"],id=row["id"])
            validate(prediction["answers"])
        else:
            started=time.perf_counter()
            for attempt in range(3):
                attempts+=1
                response=session.post(ENDPOINT,json=payload,timeout=(10,45),allow_redirects=False)
                if response.status_code==200:
                    break
                if response.status_code not in {429,500,502,503,504,529} or attempt==2:
                    raise RuntimeError(f"Jev returned HTTP {response.status_code}; response body omitted")
                time.sleep(2**attempt)
            body=response.json(); validate(body["answers"])
            if body.get("model") != args.model:
                raise ValueError("Returned model does not match requested pinned version")
            prediction={"id":row["id"],"gold":row["labels"],"answers":body["answers"],
                "model":body["model"],"usage":body.get("usage",{}),
                "latency_ms":1000*(time.perf_counter()-started),"request_sha256":request_hash,
                "observed_at":datetime.now(timezone.utc).isoformat()}
            with cache_path.open("a") as f:f.write(json.dumps(prediction,allow_nan=False)+"\n")
        predictions.append(prediction)
        if (index+1)%12==0 or index+1==len(rows):print(f"Evaluated {index+1}/{len(rows)}",flush=True)
    report={"endpoint":ENDPOINT,"requested_model":args.model,"n_sessions":len(rows),"synthetic_only":True,
        "data_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),"questions":QUESTIONS,
        "request_attempts_this_run":attempts,"metrics":metrics(predictions),
        "probability_metrics":"Raw probabilities preserved; renormalized to sum one only for Brier/ECE. Returned choices unchanged.",
        "raw_nonallow_cases_incorrectly_allowed":sum(p["answers"]["action"]["choice"]=="allow" and p["gold"]["action"]!="allow" for p in predictions),
        "usage":{key:sum(p["usage"].get(key,0) for p in predictions) for key in ("input_tokens","output_tokens")},
        "end_to_end_latency_median_ms":statistics.median(p["latency_ms"] for p in predictions),
        "predictions":predictions}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"predictions","questions"}},indent=2),flush=True)

if __name__=="__main__":main()
