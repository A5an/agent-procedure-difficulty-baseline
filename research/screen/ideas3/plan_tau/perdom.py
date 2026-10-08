import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, json
from scipy.special import logit
from scipy.stats import spearmanr
REPO=ZR.REPO
res={}
for b in ["tau2","sopbench"]:
    it=pd.read_csv(f"{REPO}/data/{b}/irt/1d_1pl/items.csv",index_col=0)["b"]
    for sc in ["new_procedures","new_domain"]:
        o=pd.read_csv(f"out/run/{b}/obs_{sc}.csv.gz"); cols=[c for c in ["adele","plan","plan_cat_adele","plan_grp_adele"] if c in o.columns]
        for c in cols:
            o["_l"]=-logit(o[c].clip(1e-4,1-1e-4))
        g=o.groupby(["seed","fold","case_id"])[[ *cols,"_l"]].mean() if False else None
        out={}
        for c in cols:
            o["_l"]=-logit(o[c].clip(1e-4,1-1e-4))
            pc=o.groupby(["seed","case_id"])["_l"].mean().reset_index()
            pc["dom"]=pc.case_id.str.split("/").str[0]; pc["b"]=pc.case_id.map(it)
            out[c]={d: round(float(np.nanmean([spearmanr(x.b,x._l)[0] for _,x in xx.groupby("seed")])),3) for d,xx in pc.groupby("dom")}
        res[f"{b}|{sc}"]=out
json.dump(res,open("perdomain.json","w"),indent=1)
for k,v in res.items(): print(k); [print(" ",c,d) for c,d in v.items()]
