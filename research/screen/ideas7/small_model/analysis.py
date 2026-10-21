import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np
from scipy.stats import spearmanr
R=ZR.REPO + "/data/"
S=ZR.SCREEN + "/"
f=pd.read_csv("sm_sopbench.csv",index_col=0)
t=pd.read_csv(R+"sopbench/tasks.csv",index_col=0)
b=pd.read_csv(R+"sopbench/irt/1d_1pl/items.csv",index_col=0)["b"]
d=f.join(t[["domain","procedure_id","should_succeed"]]).join(b)
for n in ["sonnet","gemini"]:
    r=pd.read_csv(S+f"ideas5/smart_{n}/real_results.csv",index_col=0); d["fail_"+n]=1-r["success"]
cols=[c for c in f.columns if c!="sm_nchk"]
def within(d,col,tg,grp="procedure_id"):
    out=[];w=[]
    for _,g in d.groupby(grp):
        g=g.dropna(subset=[col,tg])
        if len(g)>=6 and g[col].nunique()>1 and g[tg].nunique()>1:
            out.append(spearmanr(g[col],g[tg])[0]);w.append(len(g))
    return (np.average(out,weights=w) if out else np.nan,len(out))
rows=[]
for c in cols:
    r={"feat":c}
    for tg in ["b","fail_sonnet","fail_gemini"]:
        r[tg+"_proc"],r[tg+"_n"]=within(d,c,tg)
        r[tg+"_dom"]=within(d,c,tg,"domain")[0]
    # label-conditioned
    for lab in [0,1]:
        r[f"b_proc_lab{lab}"]=within(d[d.should_succeed==lab],c,"b")[0]
    r["rho_label"]=spearmanr(d[c],d.should_succeed,nan_policy="omit")[0]
    rows.append(r)
o=pd.DataFrame(rows).set_index("feat").round(3); o.to_csv("analysis_sop.csv"); print(o.to_string())
print("n real", d.fail_sonnet.notna().sum(), d.fail_gemini.notna().sum())
