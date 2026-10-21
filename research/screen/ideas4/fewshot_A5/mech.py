import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json
os.environ.setdefault("PYTHONHASHSEED","0")
S = ZR.SCREEN
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import a5
T = a5.TASKS["tau2"]; IDS = T.index.tolist()
b = pd.read_csv(a5.REPO / "data/tau2/irt/1d_1pl/items.csv", index_col=0)["b"]
G = pd.DataFrame(a5._GOLD["tau2"], index=["g_len","g_writes","g_all"]).T  # diagnostics only, after prediction
P = a5.PLAN3["tau2"]
def folds(scheme, seed):
    if scheme == "new_procedures":
        it = GroupKFold(5, shuffle=True, random_state=seed).split(IDS, groups=T.procedure_id)
    else:
        doms = sorted(T.domain.unique()); it = [((T.domain != d).to_numpy().nonzero()[0], (T.domain == d).to_numpy().nonzero()[0]) for d in doms]
    return [([IDS[i] for i in x], [IDS[i] for i in y]) for x, y in it]
rows = []
for scheme, seeds in [("new_procedures", range(5)), ("new_domain", [0])]:
    for sd in seeds:
        for tr, te in folds(scheme, sd):
            A = a5.stage_a("tau2", tr, te); D = a5.stage_a("tau2", tr, te, demean=True)
            for c in te:
                rows.append(dict(scheme=scheme, seed=sd, case=c, dom=T.domain[c], b=b[c], A_len=A.loc[c,"len"], A_w=A.loc[c,"writes"], A_all=A.loc[c,"all"],
                                 Adm_w=D.loc[c,"writes"], Adm_all=D.loc[c,"all"], **{k: P.loc[c,k] for k in P.columns}, **G.loc[c].to_dict()))
d = pd.DataFrame(rows); d.to_pickle("mech_rows.pkl")
cols = ["A_len","A_w","A_all","Adm_w","Adm_all","g_len","g_writes","g_all","plan_calls","plan_writes","plan_reads","plan_rules","plan_refuse","plan_transfer","plan_facts","plan_ambig"]
out = []
for scheme, d0 in d.groupby("scheme"):
    for dom, x in list(d0.groupby("dom")) + [("pooled_within(mean of domains)", None)]:
        if x is None:
            r = {c: np.mean([np.nanmean([spearmanr(g2[c], g2.b)[0] for _, g2 in gs.groupby("dom") if g2[c].nunique() > 1]) for _, gs in d0.groupby("seed")]) for c in cols}
            out.append(dict(scheme=scheme, dom="MEAN3", **r)); continue
        r = {c: np.mean([spearmanr(g2[c], g2.b)[0] if g2[c].nunique() > 1 else np.nan for _, g2 in x.groupby("seed")]) for c in cols}
        out.append(dict(scheme=scheme, dom=dom, n=len(x[x.seed == x.seed.iloc[0]]), **r))
R = pd.DataFrame(out).round(2); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(R.T.to_string()); R.to_csv("mech_domain_rho.csv", index=False)
# A_w vs gold writes within domain
print("\nA_w vs gold writes within domain (rho, mean over seeds), and A_all vs gold all, A_len vs gold len")
for scheme, d0 in d.groupby("scheme"):
    for dom, x in d0.groupby("dom"):
        f = lambda a, g: np.mean([spearmanr(g2[a], g2[g])[0] for _, g2 in x.groupby("seed")])
        print(scheme, dom, "w", round(f("A_w","g_writes"),2), "all", round(f("A_all","g_all"),2), "len", round(f("A_len","g_len"),2),
              "| mean A_w", round(x.A_w.mean(),2), "gold w", round(x.g_writes.mean(),2), "sd A_w", round(x.A_w.std(),2))
# domain structure: how much of A_w variance is between-domain
for c in ["A_w","A_all","A_len","plan_calls","plan_writes"]:
    x = d[(d.scheme=="new_procedures")&(d.seed==0)]
    print(c, "between-domain share of variance", round(x.groupby("dom")[c].transform("mean").var()/x[c].var(),2))
# domain mean difficulty
print(d[(d.scheme=="new_procedures")&(d.seed==0)].groupby("dom").b.agg(["mean","std","count"]).round(2))
# stage-1 coefficients (descriptive; full-data fit, gold of all tasks, NOT used as any feature)
from sklearn.linear_model import Ridge
X = P.loc[IDS].to_numpy(float); Z = (X - X.mean(0)) / (X.std(0)+1e-9)
for j, nm in enumerate(["len","writes","all"]):
    m = Ridge(alpha=100).fit(Z, G.loc[IDS].to_numpy(float)[:, j]); print("coef", nm, dict(zip(P.columns, m.coef_.round(2))))
