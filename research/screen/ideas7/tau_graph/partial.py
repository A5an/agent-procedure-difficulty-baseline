import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np
from scipy.stats import rankdata, spearmanr, t as tdist
S=ZR.SCREEN
b=pd.read_csv(ZR.REPO + "/data/tau2/irt/1d_1pl/items.csv",index_col=0)["b"]
plan=pd.read_csv(S+"/ideas3/plan_tools/features/plan3_tau2.csv",index_col=0)
out=[]
def partial(x,y,z):
    rx,ry,rz=[rankdata(v) for v in (x,y,z)]
    def res(a): 
        A=np.c_[np.ones(len(rz)),rz]; return a-A@np.linalg.lstsq(A,a,rcond=None)[0]
    r=np.corrcoef(res(rx),res(ry))[0,1]; n=len(x); tt=r*np.sqrt((n-3)/(1-r*r)); p=2*tdist.sf(abs(tt),n-3); return r,p
for src in ["gem","son","avg"]:
    G=pd.read_csv(f"features/graph_{src}_tau2.csv",index_col=0)
    G["plan_calls"]=plan.plan_calls; G["plan_writes"]=plan.plan_writes
    for dom in ["airline","retail","telecom"]:
        ix=[i for i in G.index if i.startswith(dom)]; D=G.loc[ix]; bb=b.loc[ix]
        for c in D.columns:
            if D[c].std()<1e-9: continue
            rho=spearmanr(D[c],bb)[0]
            pr,pp=(np.nan,np.nan) if c=="plan_calls" else partial(D[c].values,bb.values,D["plan_calls"].values)
            out.append(dict(src=src,domain=dom,feat=c,n=len(ix),rho=rho,partial_rho=pr,p=pp))
R=pd.DataFrame(out); R.to_csv("partial_spearman.csv",index=False)
pd.set_option("display.width",200)
for src in ["avg","gem","son"]:
    print(src); print(R[R.src==src].pivot(index="feat",columns="domain",values="partial_rho").round(2).join(R[(R.src==src)].pivot(index="feat",columns="domain",values="rho").round(2),rsuffix="_raw"))
print(R[(R.src=="avg")&(R.p<0.05)])
