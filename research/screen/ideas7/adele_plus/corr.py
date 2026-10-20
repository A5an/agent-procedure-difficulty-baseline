import numpy as np, pandas as pd, lib
from scipy.stats import spearmanr
from lib import DEMANDS
res = {}
for v in ["A", "A0", "B", "C"]:
    rs = []
    for b in ["sopbench", "tau2"]:
        o = pd.read_csv(lib.REPO / "data" / b / "features" / "adele.csv").set_index("task_id"); n = pd.read_csv(lib.HERE / "features" / f"{v}_{b}.csv").set_index("task_id")
        t = pd.read_csv(lib.REPO / "data" / b / "tasks.csv").set_index("case_id")
        r = {}
        for d in DEMANDS:
            x, y = o[d], n.loc[o.index, d]
            r[d] = spearmanr(x, y)[0] if x.nunique() > 1 and y.nunique() > 1 else np.nan
        # within-domain: demean by domain
        rw = {}
        dm = t.loc[o.index, "domain"]
        for d in DEMANDS:
            x, y = o[d] - o[d].groupby(dm).transform("mean"), n.loc[o.index, d] - n.loc[o.index, d].groupby(dm.values).transform("mean").values
            rw[d] = spearmanr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan
        rs.append((b, r, rw, n.loc[o.index].nunique().mean(), n.loc[o.index].std().mean(), o.std().mean()))
    res[v] = rs
for v, rs in res.items():
    for b, r, rw, nu, sd, osd in rs:
        print(v, b, "mean spearman vs original", round(np.nanmean(list(r.values())), 3), "within-domain", round(np.nanmean(list(rw.values())), 3), "mean sd new", round(sd, 2), "orig", round(osd, 2))
    print("  per scale (sop)", {d: round(x, 2) for d, x in rs[0][1].items()})
