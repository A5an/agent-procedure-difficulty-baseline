import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, json, re, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
R=ZR.REPO + "/data"
t=pd.read_csv(f"{R}/sopbench/tasks.csv"); it=pd.read_csv(f"{R}/sopbench/irt/1d_1pl/items.csv").rename(columns={"Unnamed: 0":"case_id"})
st={json.loads(l)['case_id']:json.loads(l)['text'] for l in open(f"{R}/sopbench/statements.jsonl")}
t=t.merge(it[["case_id","b"]],on="case_id")
t["msg"]=t.case_id.map(lambda c: re.search(r"Customer message: (.*?)\nAvailable tools", st[c], re.S).group(1))
def gib(tok):
    sw=sum(1 for a,b in zip(tok,tok[1:]) if a.isalpha() and b.isalpha() and a.islower()!=b.islower())
    return sw
def f(m):
    toks=re.findall(r"[A-Za-z0-9]{10,}",m)
    return max([gib(x) for x in toks],default=0)
t["gib"]=t.msg.map(f)
t["gibf"]=(t.gib>=4).astype(int)
print(pd.crosstab(t.gibf,t.should_succeed))
# within-procedure AUC for refuse
aucs=[]
for p,g in t.groupby("procedure_id"):
    if g.should_succeed.nunique()==2 and g.gib.nunique()>1:
        aucs.append(roc_auc_score(1-g.should_succeed,g.gib))
print("mean within-proc AUC gib->refuse",np.mean(aucs),len(aucs))
print("pooled AUC", roc_auc_score(1-t.should_succeed,t.gib))
# among refuse, gib cases easier?
r=t[t.should_succeed==0]
print("refuse: b by gibf", r.groupby("gibf").b.agg(['mean','count']))
print("exec: b mean", t[t.should_succeed==1].b.mean())
