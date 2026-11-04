import json, collections, sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(S / "tib"))
from common import load_tasks
ALL = {t["id"]: t for t in load_tasks()}
rng = np.random.default_rng(5)


def text(t):
    i = t["user_scenario"]["instructions"]
    return " ".join(str(i.get(k, "")) for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions")) if isinstance(i, dict) else str(i)


def outcomes(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("reward") is not None: d[r["task_id"]] = 1 - int(r["reward"] >= 1)
    return d


rows = []
for base, sel, seeded in ((S / "tib", json.load(open(S / "tib/selection.json")), json.load(open(S / "tib/seeded/probes.json"))),
                          (S / "combo_rep", json.load(open(S / "combo_rep/selection_rep.json")), json.load(open(S / "combo_rep/probes_seeded.json")))):
    for a in ("g38", "flash"):
        ro, po = outcomes(base / "runs" / f"real_{a}.jsonl"), outcomes(base / "runs" / f"seeded_{a}.jsonl")
        for f in sel["test_families"]:
            reals = [ALL[i] for i in sel["real_tasks"][f] if i in ro]
            probes = [p for p in seeded if f"_{f}_" in p["id"] and p["id"] in po]
            if len(probes) < 3 or not reals: continue
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([text(t) for t in reals + probes])
            R, P = vec.transform([text(t) for t in reals]), vec.transform([text(t) for t in probes])
            sim = (R @ P.T).toarray()
            pf = np.array([po[p["id"]] for p in probes]); fam_rate = (pf.sum() + .5) / (len(pf) + 1)
            for j, t in enumerate(reals):
                w = sim[j] + 1e-6
                knn = (w * pf).sum() / w.sum()
                rows.append((a, f, ro[t["id"]], fam_rate, 0.5 * knn + 0.5 * fam_rate))
import pandas as pd
D = pd.DataFrame(rows, columns=["agent", "fam", "fail", "fam_rate", "knn"])


def wauc(d, col):
    v, w = [], []
    for (a, f), g in d.groupby(["agent", "fam"]):
        if 0 < g.fail.mean() < 1 and g[col].nunique() > 1:
            v.append(roc_auc_score(g.fail, g[col])); w.append(len(g))
    return np.average(v, weights=w), len(v)


m, k = wauc(D, "knn")
fams = D.fam.unique(); bs = []
for _ in range(2000):
    pick = rng.choice(fams, len(fams))
    dd = pd.concat([D[D.fam == f].assign(fam=f"{f}#{i}") for i, f in enumerate(pick)])
    bs.append(wauc(dd, "knn")[0])
print(f"cases {len(D)}; within family x agent AUC of nearest-probe prediction {m:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] over {k} cells")
for col in ("fam_rate", "knn"):
    print(f"{col:9s} pooled AUC {roc_auc_score(D.fail, D[col]):.3f}  Brier {np.mean((D[col] - D.fail) ** 2):.4f}")
