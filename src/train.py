"""Full supervised Laya fine-tune, disjoint calibration, and paired base evaluation."""
import os
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
import json, math, random, time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.common import build_model
from laya.agent import _fix_tokenizer_config
from session_data import generate, QUESTIONS
from context import checked_sequence

REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
OUT = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/laya-session-guard"))


def collate(items, pad, device):
    n, length = len(items), max(len(x["ids"]) for x in items)
    k = max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    pos = torch.zeros((n, k), dtype=torch.long)
    mask = torch.zeros((n, k), dtype=torch.bool)
    for i, row in enumerate(items):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"])
        att[i, :len(row["ids"])] = 1
        pos[i, :len(row["markers"])] = torch.tensor(row["markers"])
        mask[i, :len(row["markers"])] = True
    return [x.to(device) for x in (ids, att, pos, mask, torch.zeros(n, dtype=torch.long))]


def evaluate(model, items, tok, device):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(items), 8):
            rows = items[start:start+8]
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(*collate(rows, tok.pad_token_id, device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite evaluation logits")
            for row, z in zip(rows, logits.float().cpu()):
                result.append({"id": row["id"], "qid": row["qid"], "family": row["family"],
                               "label": row["label"], "logits": z[:len(row["markers"])].tolist()})
    return result


def metrics(rows, temperatures):
    report = {}
    for qid, q in QUESTIONS.items():
        sel = [r for r in rows if r["qid"] == qid]
        z = torch.tensor([r["logits"] for r in sel]) / temperatures[qid]
        p = z.softmax(-1).numpy()
        y = np.array([r["label"] for r in sel]); pred = p.argmax(-1)
        labels = list(q["criteria"]); cm = np.zeros((len(labels),len(labels)),dtype=int)
        for a,b in zip(y,pred): cm[a,b] += 1
        confidence = p.max(-1); correct = pred == y
        ece = 0.
        bin_ids = np.minimum((confidence * 10).astype(int),9)
        for bin_id in range(10):
            mask = bin_ids == bin_id
            if mask.any(): ece += mask.mean() * abs(confidence[mask].mean()-correct[mask].mean())
        report[qid] = {"n":len(sel),"accuracy":float(correct.mean()),"ece":float(ece),
            "brier":float(((p-np.eye(len(labels))[y])**2).sum(-1).mean()),
            "labels":labels,"confusion_matrix_true_rows":cm.tolist(),
            "recall":{label:float(cm[i,i]/max(1,cm[i].sum())) for i,label in enumerate(labels)},
            "precision":{label:float(cm[i,i]/max(1,cm[:,i].sum())) for i,label in enumerate(labels)}}
        if qid == "content": report[qid]["benign_false_positive_rate"] = float(cm[0,1]/max(1,cm[0].sum()))
    return report


def fit_temperatures(rows):
    result = {}
    for qid in QUESTIONS:
        sel = [r for r in rows if r["qid"] == qid]
        z = torch.tensor([r["logits"] for r in sel]); y = torch.tensor([r["label"] for r in sel])
        candidates = torch.linspace(.5,5.,181)
        losses = [torch.nn.functional.cross_entropy(z/t,y).item() for t in candidates]
        result[qid] = float(candidates[np.argmin(losses)])
    return result


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "This job requires Kaggle GPU; refusing CPU training"
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    print("GPU",torch.cuda.get_device_name(0),"visible",torch.cuda.device_count(), flush=True)
    model_dir = snapshot_download("convaiinnovations/laya", revision=REVISION,
        allow_patterns=["model.safetensors","rl_agent_config.json","encoder/config.json","tokenizer/*"])
    _fix_tokenizer_config(model_dir)
    cfg = json.loads(Path(model_dir,"rl_agent_config.json").read_text())
    cfg.update(max_len=1024, head_max_len=192, amp_dtype="fp16", temperature=[1.,1.,1.],temperature_by_options={})
    tok = AutoTokenizer.from_pretrained(Path(model_dir,"tokenizer"))
    splits = generate(); prepared = {}
    for name, records in splits.items():
        prepared[name] = []
        with (OUT / f"{name}.jsonl").open("w") as f:
            for row in records:
                f.write(json.dumps(row)+"\n")
                for qid,q in QUESTIONS.items():
                    ids,markers = checked_sequence(tok,row["state"],q,cfg["max_len"],cfg["head_max_len"])
                    prepared[name].append(dict(ids=ids,markers=markers,id=row["id"],family=row["family"],qid=qid,
                        label=list(q["criteria"]).index(row["labels"][qid])))
    print("Prepared",{k:len(v) for k,v in prepared.items()},"max tokens",max(len(x["ids"]) for rows in prepared.values() for x in rows),flush=True)
    model = build_model(cfg,encoder_dir=str(Path(model_dir,"encoder")))
    model.encoder.config.reference_compile = False
    model.load_state_dict(load_file(str(Path(model_dir,"model.safetensors"))),strict=True)
    model.to(device)
    baseline = evaluate(model,prepared["test"],tok,device)
    baseline_cal = evaluate(model,prepared["calibration"],tok,device)
    base_temp = fit_temperatures(baseline_cal)
    (OUT/"baseline.json").write_text(json.dumps(metrics(baseline,base_temp),indent=2))
    print("BASELINE",metrics(baseline,base_temp),flush=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    # The SDK's act head is not an authorization classifier. Freeze and ignore it.
    for p in model.act_head.parameters(): p.requires_grad_(False)
    enc = [p for n,p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n,p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params":enc,"lr":2e-5},{"params":head,"lr":5e-5}],weight_decay=.01)
    epochs, batch_size, accum = 3, 4, 4
    steps_per_epoch = math.ceil(len(prepared["train"])/batch_size)
    updates = epochs * math.ceil(steps_per_epoch/accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda s: min((s+1)/max(1,int(updates*.06)), max(0.,(updates-s)/max(1,updates-int(updates*.06)))))
    scaler = torch.amp.GradScaler("cuda")
    history=[]; started=time.time()
    for epoch in range(epochs):
        model.train(); rows=list(prepared["train"]); random.shuffle(rows)
        opt.zero_grad(set_to_none=True); total_loss=0.
        for step,start in enumerate(range(0,len(rows),batch_size)):
            chunk=rows[start:start+batch_size]
            group_size=min(accum,steps_per_epoch-(step//accum)*accum)
            with torch.autocast("cuda",dtype=torch.float16):
                z,_=model(*collate(chunk,tok.pad_token_id,device))
                loss=torch.nn.functional.cross_entropy(z.float(),torch.tensor([r["label"] for r in chunk],device=device))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss/group_size).backward(); total_loss+=loss.item()
            if (step+1)%accum==0 or step+1==steps_per_epoch:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            if (step+1)%30==0: print(f"epoch {epoch+1}/{epochs} batch {step+1}/{steps_per_epoch} loss {loss.item():.4f} elapsed {time.time()-started:.0f}s",flush=True)
        history.append({"epoch":epoch+1,"loss":total_loss/steps_per_epoch})
        save_file({k:v.detach().half().cpu().contiguous() for k,v in model.state_dict().items()},str(OUT/"model.safetensors"))
        tok.save_pretrained(OUT/"tokenizer"); model.encoder.config.save_pretrained(OUT/"encoder")
        (OUT/"rl_agent_config.json").write_text(json.dumps(cfg,indent=2))
        print("SAVED",history[-1],flush=True)
    del opt,scaler,sched; torch.cuda.empty_cache()
    calibration=evaluate(model,prepared["calibration"],tok,device)
    temperatures=fit_temperatures(calibration)
    test=evaluate(model,prepared["test"],tok,device)
    # Both questions are choice, but have different option counts and calibration buckets.
    cfg["temperature_by_options"]={"choice:2":temperatures["content"],"choice:3-5":temperatures["action"]}
    cfg["fine_tuned"]=True; cfg["model_name"]="laya-session-guard-synthetic-pilot"
    (OUT/"rl_agent_config.json").write_text(json.dumps(cfg,indent=2))
    report={"status":"completed","synthetic_only":True,"seed":42,"model_revision":REVISION,
        "gpu":torch.cuda.get_device_name(0),"training_seconds":time.time()-started,"history":history,
        "split_sizes":{k:len(v) for k,v in splits.items()},"temperatures":temperatures,
        "baseline_temperatures":base_temp,"baseline":metrics(baseline,base_temp),
        "finetuned":metrics(test,temperatures),"finetuned_uncalibrated":metrics(test,{k:1. for k in QUESTIONS}),
        "limitations":["Synthetic template labels; no real session validation","English only","No production authorization claim","Single seed","Long sessions require review when partitioned"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    (OUT/"predictions.json").write_text(json.dumps({"base":baseline,"finetuned":test},indent=2))
    (OUT/"questions.json").write_text(json.dumps(QUESTIONS,indent=2))
    import importlib.metadata
    (OUT/"versions.json").write_text(json.dumps({k:importlib.metadata.version(k) for k in ["torch","laya","transformers","safetensors","huggingface_hub"]},indent=2))
    print("COMPLETED",json.dumps(report),flush=True)

if __name__ == "__main__": main()
