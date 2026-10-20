"""Pre-harness analysis: within-procedure Spearman with b, AUC for perform/refuse label. Analysis only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
import prep, skipf
R = ZR.REPO + "/data/sopbench"
b = pd.read_csv(f"{R}/irt/1d_1pl/items.csv", index_col=0).b
t = pd.read_csv(f"{R}/tasks.csv").set_index("case_id")
F = {k: pd.read_csv(f"fam_{k}.csv", index_col=0) for k in ["nli", "ret", "act", "bnd"]}
# skip-weighted NLI columns with leave-one-domain-out skip model (analysis only)
parts = []
for d in t.domain.unique():
    tr = t.index[t.domain != d]; te = t.index[t.domain == d]; parts.append(skipf.build(tr, te))
F["nli"] = F["nli"].join(pd.concat(parts))
D = pd.concat(F.values(), axis=1).join(t[["procedure_id", "domain", "should_succeed"]]).join(b)
rows = []
for fam, df in F.items():
    for col in df.columns:
        x = D[col]
        # within-procedure Spearman with b
        sp, n = [], 0
        for p, g in D.groupby("procedure_id"):
            g = g[[col, "b"]].dropna()
            if len(g) >= 8 and g[col].nunique() > 1:
                r = spearmanr(g[col], g.b)[0]
                if np.isfinite(r): sp.append(r)
        # AUC for label (perform=1): pooled, within-procedure mean, within-domain mean
        y = D.should_succeed; ok = x.notna()
        auc = roc_auc_score(y[ok], x[ok]) if ok.sum() > 10 and x[ok].nunique() > 1 else np.nan
        ap = []
        for p, g in D.groupby("procedure_id"):
            g = g[[col, "should_succeed"]].dropna()
            if g.should_succeed.nunique() == 2 and g[col].nunique() > 1 and g.should_succeed.value_counts().min() >= 2: ap.append(roc_auc_score(g.should_succeed, g[col]))
        ad = []
        for p, g in D.groupby("domain"):
            g = g[[col, "should_succeed"]].dropna()
            if g.should_succeed.nunique() == 2 and g[col].nunique() > 1: ad.append(roc_auc_score(g.should_succeed, g[col]))
        pool = spearmanr(x[ok], D.b[ok])[0]
        rows.append(dict(family=fam, col=col, n_proc=len(sp), rho_within_proc=np.mean(sp) if sp else np.nan, rho_pooled=pool, auc_pooled=auc,
                         auc_within_proc=np.mean(ap) if ap else np.nan, n_proc_auc=len(ap), auc_within_dom=np.mean(ad) if ad else np.nan))
R_ = pd.DataFrame(rows); R_.to_csv("analysis.csv", index=False)
pd.set_option("display.width", 200); print(R_.round(3).to_string(index=False))
