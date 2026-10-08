import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, sys
from scipy.special import logit
from scipy.stats import spearmanr
from ceil import load
s=load("sopbench").set_index("case_id")
def pr(path,col):
    m=pd.read_csv(path); u=m.drop_duplicates(["seed","fold","case_id"]).assign(l=lambda x:-logit(x[col].clip(1e-4,1-1e-4)))
    return u
def cellrho(u, mask=None):
    num=den=0
    for (sd,f),g in u.groupby(["seed","fold"]):
        g=g.join(s[["domain","b","should_succeed"]],on="case_id")
        if mask is not None: g=g[g.should_succeed==mask]
        for d,h in g.groupby("domain"):
            n=len(h)
            if n<3 or h.l.nunique()<2: continue
            w=n*(n-1)/2; num+=spearmanr(h.l,h.b).correlation*w; den+=w
    return num/den
S=ZR.SCREEN
for sch in ["new_procedures","new_domain"]:
    for col in ["adele","t0_clad","g1_clad_gap"]:
        u=pr(f"{S}/ideas9/gap/out/sopbench/obs_{sch}.csv.gz",col)
        print(sch,col,"all %.3f exec %.3f refuse %.3f"%(cellrho(u),cellrho(u,1),cellrho(u,0)))
