import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os; os.environ["PYTHONHASHSEED"]="0"
import sys; sys.path.insert(0,ZR.SCREEN)
import fs, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
tasks=pd.read_csv(fs.REPO/"data/tau2/tasks.csv"); ids=tasks.case_id.tolist()
def folds(scheme,seed=0):
    if scheme=="new_procedures":
        it=GroupKFold(5,shuffle=True,random_state=seed).split(ids,groups=tasks.procedure_id)
    else:
        doms=sorted(tasks.domain.unique()); it=[((tasks.domain!=d).to_numpy().nonzero()[0],(tasks.domain==d).to_numpy().nonzero()[0]) for d in doms]
    return [([ids[i] for i in a],[ids[i] for i in b]) for a,b in it]
def top(case,pool,k):
    dom=fs.DOM[case]; cand=[c for c in pool if fs.DOM[c]==dom] or list(pool)
    s=fs.EX[[fs.EIDX[c] for c in cand]]@fs.EX[fs.EIDX[case]]
    return tuple(cand[i] for i in np.argsort(-s,kind="stable")[:k])
P=fs.PROC
for k in (3,4,5):
  keys=set(); 
  for scheme in ["new_procedures","new_domain"]:
    for tr,te in folds(scheme):
        trs=set(tr)
        for t in te: keys.add((t,top(t,trs,k)))
        for t in tr:
            pool=[c for c in tr if P[c]!=P[t]] if scheme=="new_procedures" else [c for c in tr if fs.DOM[c]==fs.DOM[t] and P[c]!=P[t]]
            keys.add((t,top(t,pool,k)))
  print(k,len(keys))
