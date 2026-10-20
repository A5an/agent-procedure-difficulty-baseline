import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, numpy as np, pandas as pd
sys.path.insert(0,ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
from scipy.stats import spearmanr
R=ZR.REPO + "/data/"
with cpu_lock("psychometric"):
  for bench in ["sopbench","tau2","tauk_banking"]:
    t=pd.read_csv(R+bench+"/tasks.csv"); a=pd.read_csv(R+bench+"/features/adele.csv")
    it=pd.read_csv(R+bench+"/irt/1d_1pl/items.csv",index_col=0)
    a=a.rename(columns={a.columns[0]:"case_id"}); d=t.merge(a,on="case_id").set_index("case_id"); d["b"]=it["b"]
    d=d.dropna(subset=["b"]); sc=[c for c in a.columns if c!="case_id"]
    grp=d.groupby("procedure_id")
    tot=d["b"].var(); within=(d["b"]-grp["b"].transform("mean")).var()
    print(f"\n{bench} n={len(d)} procs={d.procedure_id.nunique()} within-proc share of IRT b variance={within/tot:.2f}")
    rows=[]
    for c in sc:
      x=d[c]; vt=x.var()
      if vt==0: continue
      xw=x-grp[c].transform("mean"); bw=d["b"]-grp["b"].transform("mean")
      sh=xw.var()/vt
      r=spearmanr(xw,bw).correlation if xw.std()>0 else np.nan
      rows.append((c,sh,r))
    r=pd.DataFrame(rows,columns=["scale","within_share","rho_within_proc"]).sort_values("rho_within_proc")
    print("mean within-proc share of ADeLe scale variance: %.2f"%r.within_share.mean())
    print("mean |rho| within-proc: %.3f  best:"%r.rho_within_proc.abs().mean(), r.iloc[[0,-1]].round(3).values.tolist())
