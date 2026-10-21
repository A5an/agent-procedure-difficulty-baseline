import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
S=ZR.SCREEN
H=sys.argv[1]
t=pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv").set_index("case_id")
f=pd.read_csv(f"{S}/ideas6/msg_facts/{H}").set_index("case_id"); eo=pd.read_csv(f"{S}/ideas6/emu_feat/emu_case_features.csv").set_index("case_id")
d=f.join(t[["domain","should_succeed","procedure_id"]]).join(eo[["p_dry","p_pre","p_plan","p_rub"]])
print(len(d),"cases, perform share",d.should_succeed.mean().round(3))
def auc(x,y): 
    try: return roc_auc_score(y,x)
    except: return np.nan
def within(col,sgn=1):
    r=[];w=[]
    for k,g in d.groupby("domain"):
        if g.should_succeed.nunique()==2: r.append(auc(sgn*g[col],g.should_succeed)); w.append(len(g))
    return np.average(r,weights=w)
def procw(col,sgn=1):
    r=[];w=[]
    for k,g in d.groupby("procedure_id"):
        if g.should_succeed.nunique()==2 and g[col].nunique()>1: r.append(auc(sgn*g[col],g.should_succeed)); w.append(len(g))
    return np.average(r,weights=w) if r else np.nan
for col,sgn in [("p_perf",1),("entropy",1),("n_prior",1),("n_pinned",1),("frac_prior",1),("forced_refuse",-1),("frac_pinned_refusal",-1),("p_dry",-1),("p_pre",-1),("p_plan",-1),("p_rub",-1)]:
    d2=d.copy(); d2[col]=d2[col].fillna(d2[col].mean())
    d_=d; 
    print(f"{col:22s} AUC all {auc(sgn*d2[col],d2.should_succeed):.3f}  within-domain {within(col,sgn):.3f}  within-procedure {procw(col,sgn):.3f}")
print(d.groupby("should_succeed")[["p_perf","n_pinned","n_prior","forced_refuse","entropy","err_rate","n_facts"]].mean().round(3))
