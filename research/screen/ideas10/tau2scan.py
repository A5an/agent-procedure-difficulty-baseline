import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np
from scipy.special import logit
from scipy.stats import spearmanr
from ceil import load
S=ZR.SCREEN
t=load("tau2").set_index("case_id")
def tab(p): d=pd.read_csv(p); return d.set_index(d.columns[0])
for sch in ["new_procedures","new_domain"]:
    m=pd.read_csv(f"{S}/ideas9/gap/out/tau2/obs_{sch}.csv.gz")
    for col in ["adele","t0_clad","g1_clad_gap"]:
        u=m.drop_duplicates(["seed","fold","case_id"]).assign(l=lambda x:-logit(x[col].clip(1e-4,1-1e-4)))
        # within (seed, fold, domain) as protocol, here: average over seeds per case then within domain
        d=u.groupby("case_id").l.mean()
        print(sch,col,{dm:round(spearmanr(d.reindex(g.index),g.b).correlation,3) for dm,g in t.groupby("domain")})
srcs={"plan":tab(S+"/ideas3/plan_tools/features/plan3_tau2.csv"),"gap":tab(S+"/ideas9/gap/gap_tau2.csv"),"adele":tab(ZR.REPO + "/data/tau2/features/adele.csv")}
import glob
for p in glob.glob(S+"/ideas4/*/feat*tau2*.csv")+glob.glob(S+"/ideas4/*/*scn*tau2*.csv"): srcs[p.split("/")[-2]+":"+p.split("/")[-1]]=tab(p)
t["len"]=t.text.str.len()
rows=[]
for k,tb in list(srcs.items())+[("text",t[["len"]])]:
    for c in tb.columns:
        x=pd.to_numeric(tb[c],errors="coerce").reindex(t.index)
        if x.notna().sum()<200: continue
        rows.append((k[:30],c)+tuple(round(spearmanr(x.reindex(g.index),g.b,nan_policy="omit").correlation,2) for dm,g in t.groupby("domain")))
r=pd.DataFrame(rows,columns=["src","feat","airline","retail","telecom"]).sort_values("retail")
pd.set_option("display.width",200); pd.set_option("display.max_rows",300); print(r.to_string(index=False))
