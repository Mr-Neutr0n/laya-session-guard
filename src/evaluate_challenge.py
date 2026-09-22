"""Evaluate a saved checkpoint on separately written synthetic sessions."""
import os
os.environ.update(USE_TF="0", USE_FLAX="0")
import argparse, json, time
from pathlib import Path
import torch
import laya
from session_data import QUESTIONS
from context import checked_sequence
from predict import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="outputs/laya-session-guard")
    parser.add_argument("--data",default="data/challenge.jsonl")
    parser.add_argument("--output",default="reports/challenge_results.json")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(4)
    rows=[json.loads(line) for line in Path(args.data).read_text().splitlines()]
    started=time.monotonic()
    agent=laya.load(args.model,device=args.device)
    predictions=[]
    for row in rows:
        for q in QUESTIONS.values(): checked_sequence(agent.tok,row["state"],q,agent.cfg["max_len"],agent.cfg["head_max_len"])
        answers=agent.predict(row["state"],QUESTIONS)["answers"]
        # Reuse this real SDK output when exercising the policy wrapper.
        class CachedAgent:
            tok=agent.tok
            cfg=agent.cfg
            def predict(self,state,questions): return {"answers":answers}
        wrapped=inspect(CachedAgent(),row["state"])
        predictions.append({"id":row["id"],"gold":row["labels"],"answers":answers,
                            "wrapper_decision":wrapped["decision"],"wrapper_reason":wrapped["reason"]})
        print(row["id"],{q:answers[q]["choice"] for q in QUESTIONS},"gold",row["labels"],flush=True)
    metrics={}
    for qid,q in QUESTIONS.items():
        labels=list(q["criteria"]); matrix=[[0]*len(labels) for _ in labels]
        for p in predictions: matrix[labels.index(p["gold"][qid])][labels.index(p["answers"][qid]["choice"])]+=1
        metrics[qid]={"accuracy":sum(matrix[i][i] for i in range(len(labels)))/len(rows),"labels":labels,"confusion_matrix_true_rows":matrix}
    wrapper={"accuracy":sum(p["wrapper_decision"]==p["gold"]["action"] for p in predictions)/len(rows),
             "nonallow_cases_incorrectly_allowed":sum(p["wrapper_decision"]=="allow" and p["gold"]["action"]!="allow" for p in predictions),
             "review_count":sum(p["wrapper_decision"]=="review" for p in predictions)}
    report={"synthetic_only":True,"not_used_for_training_or_calibration":True,"n_sessions":len(rows),"elapsed_seconds":time.monotonic()-started,
            "metrics":metrics,"wrapper":wrapper,"predictions":predictions}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print("CHALLENGE_RESULT",json.dumps({"metrics":metrics,"wrapper":wrapper}),flush=True)

if __name__=="__main__": main()
