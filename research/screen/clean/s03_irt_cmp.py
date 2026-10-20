import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd
from scipy.stats import spearmanr
R=ZR.REPO + "/data/sopbench/irt/1d_1pl/"
old=pd.read_csv(R+"items.csv",index_col=0)
oa=pd.read_csv(R+"abilities.csv",index_col=0)
print(old.columns.tolist(), oa.columns.tolist())
t=pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv").set_index("case_id")
for v in "AB":
    D=f"data_{v}/sopbench/irt/1d_1pl/"
    n=pd.read_csv(D+"items.csv",index_col=0); na=pd.read_csv(D+"abilities.csv",index_col=0)
    c=old.index.intersection(n.index)
    print(v,len(n),"rho beta",spearmanr(old.b[c],n.b[c])[0],"nan",n.b.isna().sum())
    ca=oa.index.intersection(na.index); print(" rho theta",spearmanr(oa.iloc[:,0][ca],na.iloc[:,0][ca])[0])
    # within-domain
    import numpy as np
    print(" within-domain rho:",{d:round(spearmanr(old.b[c][t.loc[c].domain==d],n.b[c][t.loc[c].domain==d])[0],3) for d in sorted(t.domain.unique())})
    print(" old sd/new sd", old.b.std(), n.b.std())
