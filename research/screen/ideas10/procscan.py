"""Raw single-feature procedure-level scan on SOPBench (exploratory, full-data b): mean feature per procedure vs mean b, within domain, pooled by pairs."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, sys
from scipy.stats import spearmanr
from ceil import load
S=ZR.SCREEN
s=load("sopbench").set_index("case_id")
def tab(p): d=pd.read_csv(p); return d.set_index(d.columns[0])
srcs={"sm":tab(S+"/ideas7/small_model/sm_sopbench.csv"),"cmp":tab(S+"/ideas3/compile_consistency/cons_sopbench.csv"),
      "plan":tab(S+"/ideas3/plan_tools/features/plan3_sopbench.csv"),"rub":tab(S+"/rubric/features/rub_proc_sopbench.csv"),
      "gap":tab(S+"/ideas9/gap/gap_sopbench.csv"),"adele":tab(ZR.REPO + "/data/sopbench/features/adele.csv")}
extra=[a for a in sys.argv[1:]]
for e in extra: srcs[e.split("/")[-1].split(".")[0]]=tab(e)
rows=[]
def prho(x):
    pm=pd.DataFrame({"x":x,"b":s.b,"d":s.domain,"p":s.procedure_id}).dropna().groupby(["d","p"]).mean(numeric_only=True).reset_index()
    num=den=0; per={}
    for d,g in pm.groupby("d"):
        n=len(g); w=n*(n-1)/2
        r=spearmanr(g.x,g.b).correlation if g.x.nunique()>1 else 0
        r=0 if r!=r else r; num+=r*w; den+=w; per[d]=r
    return num/den, per
def crho(x):
    df=pd.DataFrame({"x":x,"b":s.b,"d":s.domain}).dropna(); num=den=0
    for d,g in df.groupby("d"):
        n=len(g); w=n*(n-1)/2; r=spearmanr(g.x,g.b).correlation if g.x.nunique()>1 else 0; r=0 if r!=r else r; num+=r*w; den+=w
    return num/den
for k,t in srcs.items():
    for c in t.columns:
        x=pd.to_numeric(t[c],errors="coerce").reindex(s.index)
        if x.notna().sum()<100: continue
        p,per=prho(x); rows.append((k,c,round(p,3),round(crho(x),3),min(per.values()).__round__(2),sum(v>0 for v in per.values())))
r=pd.DataFrame(rows,columns=["src","feat","proc_rho","case_rho","min_dom","n_pos_dom"]).sort_values("proc_rho")
pd.set_option("display.width",200); pd.set_option("display.max_rows",300)
print(r.to_string(index=False))
