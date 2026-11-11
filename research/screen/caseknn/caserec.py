"""Exploratory, no API calls: does the customer's database record predict which case of a family fails, before running?
Features of the case's customer (counts of accounts, cards, deposits, mandates, requests, loans, insurance, record size)
plus the family's seeded-probe rate; gradient boosting trained leave one family out; within family x agent AUC."""
import json, sys, collections
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(S / "tib"))
from common import load_tasks, DOM
ALL = {t["id"]: t for t in load_tasks()}
DB = json.load(open(DOM / "db.json"))["customers"]


def outcomes(p):
    return {json.loads(l)["task_id"]: 1 - int(json.loads(l)["reward"] >= 1) for l in open(p) if json.loads(l).get("reward") is not None}


def feats(cid):
    c = DB[cid]; f = {}
    for k, v in c.items():
        if isinstance(v, list): f[f"n_{k}"] = len(v)
        elif isinstance(v, dict): f[f"n_{k}"] = len(v)
    f["size"] = len(json.dumps(c))
    return f


rows = []
for base, sel, seeded in ((S / "tib", json.load(open(S / "tib/selection.json")), "seeded_{a}.jsonl"), (S / "combo_rep", json.load(open(S / "combo_rep/selection_rep.json")), "seeded_{a}.jsonl")):
    for a in ("g38", "flash"):
        ro, po = outcomes(base / f"runs/real_{a}.jsonl"), outcomes(base / "runs" / seeded.format(a=a))
        for f in sel["test_families"]:
            pr = [v for k, v in po.items() if f"_{f}_" in k]; rate = (sum(pr) + .5) / (len(pr) + 1)
            for tid in sel["real_tasks"][f]:
                if tid not in ro: continue
                cid = ALL[tid]["initial_state"]["initialization_data"]["agent_data"]["active_customer"]
                rows.append({"agent": a, "fam": f, "fail": ro[tid], "probe_rate": rate, "is_g38": int(a == "g38"), **feats(cid)})
D = pd.DataFrame(rows).fillna(0)
X = [c for c in D.columns if c not in ("agent", "fam", "fail")]
s = pd.Series(np.nan, index=D.index)
for f in D.fam.unique():
    tr, te = D[D.fam != f], D[D.fam == f]
    s[te.index] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, random_state=0).fit(tr[X], tr.fail).predict_proba(te[X])[:, 1]
v, w = [], []
for (a, f), g in D.groupby(["agent", "fam"]):
    if 0 < g.fail.mean() < 1 and s[g.index].nunique() > 1:
        v.append(roc_auc_score(g.fail, s[g.index])); w.append(len(g))
print(f"features {X}")
print(f"within family x agent AUC from customer records + probe rate: {np.average(v, weights=w):.3f} over {len(v)} cells; pooled AUC {roc_auc_score(D.fail, s):.3f} (probe rate alone {roc_auc_score(D.fail, D.probe_rate):.3f})")
