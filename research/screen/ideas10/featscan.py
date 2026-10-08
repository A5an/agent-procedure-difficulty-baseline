"""Raw within-domain Spearman of a feature table with b on each benchmark, plus within-procedure on SOPBench. usage: featscan.py <prefix> (reads <prefix>_<bench>.csv)"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pandas as pd, numpy as np
from scipy.stats import spearmanr
R=ZR.REPO + "/data"
pre=sys.argv[1]
def wd(df,col,key):
    num=den=0
    for k,g in df.groupby(key):
        n=len(g)
        if n<3 or g[col].nunique()<2: continue
        w=n*(n-1)/2; num+=spearmanr(g[col],g.b).correlation*w; den+=w
    return num/den if den else np.nan
rows=[]
for b in ["sopbench","tau2","tauk_banking"]:
    t=pd.read_csv(f"{R}/{b}/tasks.csv"); t["b"]=t.case_id.map(pd.read_csv(f"{R}/{b}/irt/1d_1pl/items.csv",index_col=0).b)
    f=pd.read_csv(f"{pre}_{b}.csv").set_index("case_id"); t=t.join(f,on="case_id")
    for c in f.columns:
        r={"feat":c,"bench":b,"within_dom":round(wd(t,c,"domain"),3)}
        if b=="sopbench": r["within_proc"]=round(wd(t,c,"procedure_id"),3)
        if b=="tau2":
            for d,g in t.groupby("domain"): r[d]=round(spearmanr(g[c],g.b).correlation,2) if g[c].nunique()>1 else np.nan
        rows.append(r)
d=pd.DataFrame(rows)
pd.set_option("display.width",220)
print(d.pivot_table(index="feat",columns="bench",values="within_dom").round(3).join(d[d.bench=="sopbench"].set_index("feat")[["within_proc"]]).join(d[d.bench=="tau2"].set_index("feat")[["airline","retail","telecom"]]))
