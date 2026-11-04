"""Exploratory (not pre-registered): audit every failure of the two strong agents (flash, g38) on every dataset, to see
whether the Haiku dataset is harder or noisier. Same judge and prompt as audit.py. Output audit_strong.jsonl."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import audit, dataset

HERE = Path(__file__).resolve().parent
PRE = f"{dataset.V}_" if dataset.V else ""  # SYNTH_V=v2 for the hard round
jobs = []
for g in audit.GENS:
    items = {it["uid"]: it for it in dataset.cases(g)}
    for a in ("flash", "g38"):
        for l in open(HERE / "runs" / f"{PRE}{g}_{a}.jsonl"):
            r = json.loads(l)
            if r.get("reward") == 0.0:
                jobs.append((g, a, items[r["task_id"]]))


def run(j):
    g, a, it = j
    try:
        res = audit.ask(audit.item(it, a))
    except Exception as e:
        res = {"cause": "error", "detail": str(e)[:200]}
    return {"gen": g, "agent": a, "uid": it["uid"], "level": it["plan"]["level"], "behavior": it["plan"]["behavior"], **res}


with ThreadPoolExecutor(4) as ex, open(HERE / f"{PRE}audit_strong.jsonl", "w") as f:
    for r in ex.map(run, jobs):
        f.write(json.dumps(r) + "\n")
print(len(jobs), "episodes audited")
