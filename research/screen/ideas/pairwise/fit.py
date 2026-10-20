"""Bradley-Terry fit with position bias and L2, plus feature CSVs. Texts only, no outcomes."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from pathlib import Path
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
LAM = 0.1   # L2 penalty on scores (fixed before the run)

def load_comparisons():
    P = pd.read_csv(HERE/"pairs.csv").fillna({"repeat_of": ""})
    import llmpairs as LP   # identical prompts (cases with identical text) share one cached answer, so map by prompt key
    C = {}
    for l in open(HERE/"cache.jsonl"):
        r = json.loads(l)
        if r.get("ok"): C[r["key"]] = r
    P["key"] = [LP.key(LP.prompt(r.bench, r.domain, r.a, r.b)) for r in P.itertuples()]
    P["harder"] = P.key.map(lambda k: C[k]["harder"] if k in C else None)
    P["conf"] = P.key.map(lambda k: C[k]["confidence"] if k in C else np.nan)
    P = P[P.harder.notna()].copy()
    P["y_soft"] = np.where(P.harder == "A", P.conf, 1 - P.conf)
    P["y_hard"] = (P.harder == "A").astype(float)
    return P

def bt_fit(a, b, y, n, lam=LAM):
    """a,b index arrays, y = P(A harder). logit = s_a - s_b + gamma. Returns (s, gamma)."""
    def f(th):
        s, g = th[:n], th[n]
        z = s[a] - s[b] + g
        p = 1/(1+np.exp(-z))
        eps = 1e-9
        loss = -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)).sum() + lam*(s**2).sum()
        r = p - y
        gs = np.zeros(n); np.add.at(gs, a, r); np.add.at(gs, b, -r)
        return loss, np.concatenate([gs + 2*lam*s, [r.sum()]])
    res = minimize(f, np.zeros(n+1), jac=True, method="L-BFGS-B")
    return res.x[:n], res.x[n]

def fit_domain(P, ids, col):
    idx = {c: i for i, c in enumerate(ids)}
    a = P.a.map(idx).values; b = P.b.map(idx).values
    return bt_fit(a, b, P[col].values, len(ids))

def zs(x):
    x = np.asarray(x, float); sd = x.std()
    return (x - x.mean())/sd if sd > 0 else x*0

def scores(P, col):
    """dict bench -> DataFrame(case_id, domain, score z-scored in domain), plus gammas"""
    out, gam = {}, {}
    for bench in ["sopbench", "tau2"]:
        t = pd.read_csv(REPO/"data"/bench/"tasks.csv")
        rows = []
        for dom, g in t.groupby("domain"):
            ids = list(g.case_id)
            Q = P[(P.bench == bench) & (P.domain == dom) & (P.repeat_of == "")]
            s, gm = fit_domain(Q, ids, col)
            gam[(bench, dom)] = gm
            rows.append(pd.DataFrame({"case_id": ids, "domain": dom, "bt": zs(s)}))
        out[bench] = pd.concat(rows)
    return out, gam

if __name__ == "__main__":
    P = load_comparisons()
    print(len(P), "comparisons, primary", (P.repeat_of == "").sum())
    (HERE/"features").mkdir(exist_ok=True)
    for name, col in [("bt", "y_soft"), ("bt_hard", "y_hard")]:
        S, gam = scores(P, col)
        for bench, d in S.items():
            d[["case_id", "bt"]].rename(columns={"bt": name}).to_csv(HERE/"features"/f"{name}_{bench}.csv", index=False)
        print(name, {f"{k[0]}/{k[1]}": round(v, 3) for k, v in gam.items()})
    # concat with adele
    for bench in ["sopbench", "tau2"]:
        bt = pd.read_csv(HERE/"features"/f"bt_{bench}.csv").set_index("case_id")
        ad = pd.read_csv(REPO/"data"/bench/"features"/"adele.csv"); ad = ad.set_index(ad.columns[0])
        pd.concat([bt, ad], axis=1).reset_index().rename(columns={"index": "case_id"}).to_csv(HERE/"features"/f"bt_cat_adele_{bench}.csv", index=False)
