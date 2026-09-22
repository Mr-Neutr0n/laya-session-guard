"""Inspect a trusted structured session. Does not execute or authorize tool calls."""
import os
os.environ.update(USE_TF="0", USE_FLAX="0")
import argparse, json, math
from pathlib import Path
from session_data import QUESTIONS
from context import session_windows


def inspect(agent, state):
    def review(reason):
        return {"decision":"review","reason":reason,"experimental":True,"windows":[]}
    if not isinstance(state,dict) or not isinstance(state.get("authorized_task"),str) or not state["authorized_task"].strip():
        return review("A nonempty authoritative task is required")
    if not isinstance(state.get("events"),list) or not state["events"] or not isinstance(state.get("proposed_action"),dict) or not state["proposed_action"]:
        return review("Ordered events and a structured proposed action are required")
    if not isinstance(state.get("context"),dict) or state["context"].get("history_complete") is not True:
        return review("Complete history must be explicitly attested by the host")
    if any(not isinstance(e,dict) or e.get("role") not in {"user","assistant","tool","system","developer"} or not isinstance(e.get("source"),str) or not isinstance(e.get("content"),str) for e in state["events"]):
        return review("Malformed transcript event")
    try:
        windows = session_windows(state, agent.tok, QUESTIONS, agent.cfg["max_len"],agent.cfg["head_max_len"])
    except ValueError as error:
        return {"decision":"review","reason":str(error),"windows":[]}
    try:
        answers = [agent.predict(w, QUESTIONS)["answers"] for w in windows]
        for answer in answers:
            for qid, question in QUESTIONS.items():
                result = answer[qid]; probabilities = result["probabilities"]
                if set(probabilities) != set(question["criteria"]) or result["choice"] not in probabilities:
                    return review("Model returned an invalid label schema")
                values = list(probabilities.values())
                if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1)>1e-4:
                    return review("Model returned invalid probabilities")
                if probabilities[result["choice"]] != max(values):
                    return review("Model choice and probabilities disagree")
    except (KeyError,TypeError,ValueError,RuntimeError) as error:
        return review(f"Model inference failed: {type(error).__name__}")
    action = answers[0]["action"]
    decision = action["choice"]
    reason = "Model judgment on complete bounded session"
    if len(windows) > 1:
        decision="review"
        reason="Transcript spans multiple windows; cross-window authorization is not established"
    elif max(action["probabilities"].values()) < .8:
        decision="review"
        reason="Maximum action probability below provisional 0.8 review threshold"
    if not state.get("context",{}).get("history_complete",True):
        decision="review"; reason="Input explicitly marks incomplete session history"
    return {"decision":decision,"reason":reason,"experimental":True,"window_count":len(windows),
            "windows":[{"index":i,"answers":a} for i,a in enumerate(answers)],
            "note":"Synthetic pilot. The wrapper does not execute tools. Source roles must come from the host, not document text."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model");parser.add_argument("session");parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    import laya
    agent=laya.load(args.model,device=args.device)
    print(json.dumps(inspect(agent,json.loads(Path(args.session).read_text())),indent=2))
