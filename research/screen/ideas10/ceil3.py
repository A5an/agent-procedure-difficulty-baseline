import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, json, re, numpy as np
from ceil import load, pooled
s=load("sopbench")
s["pol"]=s.text.map(lambda x: re.search(r"Policy for this action:\n(.*?)\nCustomer message:", x, re.S).group(1))
# leave-one-out mean of other cases with same policy text; fallback to LOO procedure mean
def loo(df,key):
    g=df.groupby(key).b; sm=g.transform("sum"); n=g.transform("count")
    return np.where(n>1,(sm-df.b)/(n-1).clip(lower=1),np.nan)
s["pol_loo"]=loo(s,"pol"); s["proc_loo"]=loo(s,"procedure_id")
s["pred"]=s.pol_loo.fillna(s.proc_loo)
print("LOO policy-text oracle (fallback LOO proc)",pooled(s,"pred"))
print("LOO proc oracle",pooled(s,"proc_loo"))
m=s.pol_loo.notna()
print("only cases with shared text", m.sum(), pooled(s[m],"pol_loo"), "vs proc loo on same", pooled(s[m],"proc_loo"))
# constraint-set from raw data: use number of constraints
R=ZR.DATA + "/SOPBench-d2622008/data"
rows=[]
for dom in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
    d=json.load(open(f"{R}/{dom}_tasks.json"))
    for act,lst in d.items():
        for i,e in enumerate(lst):
            rows.append(dict(case_id=f"{dom}/{act}/{i}",cons=json.dumps(e["constraints"]),ncons=json.dumps(e["constraints"]).count('"single"')))
c=pd.DataFrame(rows); s=s.merge(c,on="case_id",how="left")
print("matched",s.cons.notna().sum())
print("distinct constraint sets",s.cons.nunique())
s["cons_loo"]=loo(s,"cons"); s["pred2"]=s.cons_loo.fillna(s.proc_loo)
print("LOO constraint-set oracle",pooled(s,"pred2"))
print("ncons alone (gold count of checks)",pooled(s,"ncons"))
from scipy.stats import spearmanr
print("pol text length", pooled(s.assign(pl=s.pol.str.len()),"pl"))
