"""Raw (zero-shot, no fitting) score vs b: within-domain Spearman with bootstrap over procedures, paired vs length and human_time."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pandas as pd, numpy as np
from scipy.stats import spearmanr
R=ZR.REPO + "/data"
def load(b, extra):
    t=pd.read_csv(f"{R}/{b}/tasks.csv"); t["b"]=t.case_id.map(pd.read_csv(f"{R}/{b}/irt/1d_1pl/items.csv",index_col=0).b)
    for n in ["length","human_time"]:
        f=pd.read_csv(f"{R}/{b}/features/{n}.csv").set_index("task_id"); t[n]=t.case_id.map(f.iloc[:,0])
    for name,(path,col) in extra.items(): t[name]=t.case_id.map(pd.read_csv(path).set_index("case_id")[col])
    return t
def wd(t,col,idx=None):
    t=t if idx is None else t.iloc[idx]; t=t[t[col].notna()]; num=den=0
    for d,g in t.groupby("domain"):
        n=len(g)
        if n<3 or g[col].nunique()<2: continue
        w=n*(n-1)/2; num+=spearmanr(g[col],g.b).correlation*w; den+=w
    return num/den
def report(b, extra, cols, nboot=2000, seed=1):
    t=load(b, extra); procs=t.procedure_id.unique(); gi={p:np.where(t.procedure_id==p)[0] for p in procs}
    rng=np.random.default_rng(seed); D={c:[] for c in cols}
    for _ in range(nboot):
        idx=np.concatenate([gi[p] for p in rng.choice(procs,len(procs))])
        for c in cols: D[c].append(wd(t,c,idx))
    out={}
    for c in cols:
        d=np.array(D[c]); s=f"{b:12s} {c:14s} {wd(t,c):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        for r in ["length","human_time"]:
            if r!=c and r in cols: dd=d-np.array(D[r]); s+=f" vs {r} {wd(t,c)-wd(t,r):+.3f} [{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}]"
        print(s); out[c]=wd(t,c)
    return out
if __name__=="__main__":
    F=ZR.SCREEN + "/ideas10/fc/fc_{}.csv"
    for b in ["tauk_banking","tau2","sopbench"]:
        report(b, {"fc_diff":(F.format(b),"fc_diff"),"fc_sum_p":(F.format(b),"fc_sum_p")}, ["fc_diff","fc_sum_p","length","human_time"])
