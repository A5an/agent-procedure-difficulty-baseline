import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0,"."); sys.path.insert(0,ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd, json
from scipy.stats import rankdata
from cpulock import cpu_lock
import diag as DG
FAM={
 "ADELE":lambda c:c.startswith("ad."),
 "RUBRIC":lambda c:c.startswith("rubp."),
 "PLAN":lambda c:c.startswith("plan3."),
 "CMP":lambda c:c in("cons.len_mean","cons.branch_mean"),
 "TEXTCOUNT":lambda c:c.startswith("txt."),
 "LEN+HT":lambda c:c in("len.log_chars","ht.log_minutes"),
 "ENT":lambda c:c.startswith("ent."),
 "SMALLMODEL":lambda c:c.startswith("sm."),
 "DRYRUN":lambda c:c.startswith("dry."),
}
FAM["c_lad"]=lambda c:FAM["ADELE"](c) or FAM["RUBRIC"](c) or FAM["PLAN"](c) or FAM["CMP"](c)
FAM["c_lad-ADELE"]=lambda c:FAM["c_lad"](c) and not FAM["ADELE"](c)
FAM["c_lad-RUBRIC"]=lambda c:FAM["c_lad"](c) and not FAM["RUBRIC"](c)
FAM["c_lad-PLAN"]=lambda c:FAM["c_lad"](c) and not FAM["PLAN"](c)
FAM["c_lad-CMP"]=lambda c:FAM["c_lad"](c) and not FAM["CMP"](c)
def ridge(X,y,a=10.0):
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1
    Z=(X-mu)/sd; ym=y.mean()
    w=np.linalg.solve(Z.T@Z+a*np.eye(Z.shape[1]),Z.T@(y-ym)); return mu,sd,w,ym
def pred(m,X): mu,sd,w,ym=m; return ((X-mu)/sd)@w+ym
def spr(a,b):
    if np.ptp(a)==0: return np.nan
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def run(mode,bench,t,X,seed=0,K=5):
    """mode raw: features as is, target b; within: features and b centred by procedure; within_lab: centred by procedure x label (sop)"""
    g=t.procedure_id; b=t.b.copy(); X=X.copy()
    X=X.fillna(X.median()).fillna(0)
    if mode=="within": grp=g
    elif mode=="within_lab": grp=g+"|"+t.should_succeed.astype(str)
    if mode!="raw":
        keep=grp.map(grp.value_counts())>=2
        X=X[keep].groupby(grp[keep]).transform(lambda v:v-v.mean()); b=b[keep]-b[keep].groupby(grp[keep]).transform("mean"); t=t[keep]; g=g[keep]
    X=X.loc[:,X.std()>1e-9]
    if X.shape[1]==0: return {}
    rng=np.random.default_rng(seed); ug=g.unique(); rng.shuffle(ug); fold={p:i%K for i,p in enumerate(ug)}
    f=g.map(fold).values; P=np.zeros(len(b))
    for k in range(K):
        tr=f!=k; te=f==k
        m=ridge(X.values[tr],b.values[tr]); P[te]=pred(m,X.values[te])
    out={}
    for d in sorted(t.domain.unique()):
        mk=(t.domain==d).values
        if mk.sum()>=8: out[d]=spr(P[mk],b.values[mk])
    return out
def main():
    res={}
    for bench in ["sopbench","tau2"]:
        t,X,meta,_,_=DG.load_bench(bench)
        for fam,fn in FAM.items():
            cols=[c for c in X.columns if fn(c)]
            if not cols: continue
            if bench=="tau2" and fam=="CMP": continue
            for mode in (["raw","within","within_lab"] if bench=="sopbench" else ["raw","within"]):
                rs=[run(mode,bench,t,X[cols],seed=s) for s in range(3)]
                dd={d:np.nanmean([r.get(d,np.nan) for r in rs]) for d in rs[0]}
                res[(bench,fam,mode)]=dict(n_feat=len(cols),mean=float(np.nanmean(list(dd.values()))) if dd else np.nan,dom=dd)
    pd.to_pickle(res,"famcv.pkl")
    rows=[]
    for (b,f,m),v in res.items(): rows.append(dict(bench=b,fam=f,mode=m,nf=v["n_feat"],rho=v["mean"]))
    df=pd.DataFrame(rows).pivot_table(index=["bench","fam","nf"],columns="mode",values="rho").round(3)
    print(df.to_string())
if __name__=="__main__":
    with cpu_lock("diag"): main()
