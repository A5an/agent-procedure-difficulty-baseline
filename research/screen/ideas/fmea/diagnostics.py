import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import build_features as BF, prompts as P
HERE = Path(__file__).parent; REPO = BF.REPO
A = BF.load("A"); out = {}
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else np.nan
# 1. SOPBench label AUC
t = pd.read_csv(REPO / "data/sopbench/tasks.csv")
t["refuse"] = 1 - t.should_succeed
for c in ("acts_when_should_refuse", "refuses_when_should_act", "skips_required_check"):
    t["occ_" + c] = t.case_id.map(lambda k: BF.occ_cat(A[k], c))
res = {}
for c in ("acts_when_should_refuse", "refuses_when_should_act", "skips_required_check"):
    col = "occ_" + c
    r = {"overall": auc(t.refuse, t[col])}
    for d, g in t.groupby("domain"): r[d] = auc(g.refuse, g[col])
    r["mean_within_domain"] = float(np.nanmean([v for k, v in r.items() if k not in ("overall", "mean_within_domain")]))
    res[c] = r
# difference acts - refuses
t["diff"] = t.occ_acts_when_should_refuse - t.occ_refuses_when_should_act
r = {"overall": auc(t.refuse, t["diff"])}
for d, g in t.groupby("domain"): r[d] = auc(g.refuse, g["diff"])
r["mean_within_domain"] = float(np.nanmean([v for k, v in r.items() if k != "overall"]))
res["acts_minus_refuses"] = r
out["label_auc_sopbench"] = res
# 2. best FMEA column vs beta within domain, and vs ADeLe
corr = {}
for b in ("sopbench", "tau2"):
    f = pd.read_csv(HERE / f"features/fmea_{b}.csv").set_index("case_id")
    tk = pd.read_csv(REPO / f"data/{b}/tasks.csv").set_index("case_id")
    beta = pd.read_csv(REPO / f"data/{b}/irt/1d_1pl/items.csv", index_col=0).b
    ad = pd.read_csv(REPO / f"data/{b}/features/adele.csv"); ad = ad.set_index(ad.columns[0])
    df = f.join(tk.domain).join(beta.rename("beta"))
    rows = {}
    for c in f.columns:
        rs = []; w = []
        for d, g in df.groupby("domain"):
            if g[c].nunique() > 1 and len(g) > 5: rs.append(spearmanr(g[c], g.beta)[0]); w.append(len(g))
        rows[c] = float(np.average(rs, weights=w)) if rs else np.nan
    best = sorted(rows, key=lambda c: -abs(rows[c] if rows[c] == rows[c] else 0))
    top = best[0]
    # correlation with ADeLe inside domain (weighted mean abs and max)
    ac = {}
    for a in ad.columns:
        rs = []; w = []
        for d, g in df.groupby("domain"):
            x = ad.loc[g.index, a]
            if g[top].nunique() > 1 and x.nunique() > 1: rs.append(spearmanr(g[top], x)[0]); w.append(len(g))
        ac[a] = float(np.average(rs, weights=w)) if rs else np.nan
    amax = max(ac, key=lambda a: abs(ac[a]) if ac[a] == ac[a] else 0)
    corr[b] = {"within_domain_rho_with_beta": {c: rows[c] for c in best}, "best": top,
               "best_vs_adele": ac, "max_abs_adele_corr": [amax, ac[amax]],
               "mean_abs_adele_corr": float(np.nanmean(np.abs(list(ac.values()))))}
    # R2 of best column on ADeLe (pooled, linear) = restatement check
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    y = df[top].values; X = ad.loc[df.index].values
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), X, y, cv=5)
    corr[b]["cv_r2_of_best_from_adele"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
out["beta_corr"] = corr
json.dump(out, open(HERE / "diagnostics.json", "w"), indent=1)
print(json.dumps({k: (v if k != "beta_corr" else {b: {kk: vv for kk, vv in x.items() if kk not in ("best_vs_adele",)} for b, x in v.items()}) for k, v in out.items()}, indent=1)[:6000])
