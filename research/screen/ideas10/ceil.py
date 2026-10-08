import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, json, numpy as np
from scipy.stats import spearmanr
R=ZR.REPO + "/data"
def load(b):
    t=pd.read_csv(f"{R}/{b}/tasks.csv"); it=pd.read_csv(f"{R}/{b}/irt/1d_1pl/items.csv").rename(columns={"Unnamed: 0":"case_id"})
    st={json.loads(l)['case_id']:json.loads(l)['text'] for l in open(f"{R}/{b}/statements.jsonl")}
    t=t.merge(it[["case_id","b"]],on="case_id"); t["text"]=t.case_id.map(st); return t
def pooled(t,col):
    num=den=0; out={}
    for d,g in t.groupby("domain"):
        if g[col].nunique()<2: r=0
        else: r=spearmanr(g[col],g.b).correlation
        n=len(g); w=n*(n-1)/2; num+=r*w; den+=w; out[d]=round(r,3)
    return round(num/den,3), out
s=load("sopbench"); t=load("tau2")
for name,t_ in [("sop",s),("tau2",t)]:
    t_["proc_mean"]=t_.groupby("procedure_id").b.transform("mean")
    t_["text_mean"]=t_.groupby("text").b.transform("mean")
    print(name,"proc-mean oracle",pooled(t_,"proc_mean"))
    print(name,"identical-text oracle",pooled(t_,"text_mean"))
s["pl"]=s.groupby(["procedure_id","should_succeed"]).b.transform("mean")
print("sop proc x label oracle",pooled(s,"pl"))
print("sop label only",pooled(s,"should_succeed"))
s["neg"]=-s.should_succeed
print("sop label only (refuse=hard?)",pooled(s,"neg"))
# variance decomposition within domain
for name,t_ in [("sop",s)]:
    t_["bd"]=t_.b-t_.groupby("domain").b.transform("mean")
    tot=(t_.bd**2).sum(); within=((t_.b-t_.proc_mean)**2).sum(); wl=((t_.b-t_.pl)**2).sum()
    print("share within-proc",within/tot, "share within proc x label", wl/tot)
