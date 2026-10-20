"""Is Sonnet's D just text length? Within-domain Spearman, and partial Spearman of D given length (rank residuals)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
R = ZR.REPO + "/data"
P = pd.read_csv(ZR.SCREEN + "/ideas10/panel/panel_scores.csv", index_col=0)
def res(a, b):
    a, b = rankdata(a), rankdata(b); b1 = np.c_[np.ones_like(b), b]
    return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
for b in ["tau2", "tauk_banking"]:
    L = pd.read_csv(f"{R}/{b}/features/length.csv", index_col=0)
    col = L.columns[0]
    X = P[P.bench == b].join(L[[col]], how="inner").dropna(subset=["sonnet", "y", col])
    out = {"D~y": [], "len~y": [], "D~len": [], "D~y|len": []}; w = []
    for d, g in X.groupby("dom"):
        if len(g) < 5: continue
        out["D~y"].append(spearmanr(g.sonnet, g.y)[0]); out["len~y"].append(spearmanr(g[col], g.y)[0])
        out["D~len"].append(spearmanr(g.sonnet, g[col])[0])
        out["D~y|len"].append(np.corrcoef(res(g.sonnet, g[col]), res(g.y, g[col]))[0, 1]); w.append(len(g))
    print(b, len(X), "length column:", col, {k: round(float(np.average(v, weights=w)), 3) for k, v in out.items()})
