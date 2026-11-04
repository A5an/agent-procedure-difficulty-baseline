"""Exp 5, robustness of the TauIndianBank result (exploratory): fewer probes per family (3 of 6, 10 random subsets),
and a logistic-regression monitor instead of gradient boosting. Primary costs, router seeded+cal with M2.

  python robust.py -> robust.out (stdout)
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import combo as C

REAL, PROBE, FAMS, AGS = C.REAL, C.PROBE, C.FAMS, C.AGS
F, h, q = 20, 5, 0.99
H = h + F * (1 - q)
rs = np.random.default_rng(7)


def build(P, model_fn):
    oof = pd.Series(np.nan, index=P.index)
    for g in FAMS:
        a_, b_ = P[P.fam != g], P[P.fam == g]
        oof[b_.index] = model_fn().fit(a_[C.FEATS], a_.fail).predict_proba(b_[C.FEATS])[:, 1]
    s = pd.Series(model_fn().fit(P[C.FEATS], P.fail).predict_proba(REAL[C.FEATS])[:, 1], index=REAL.index)
    cur = {a: C.curve(oof[P.agent == a].values, P[P.agent == a].fail.values) for a in AGS}
    return {f: {"score": s[REAL.fam == f], "curve": cur} for f in FAMS}


def estimate(P):
    e = {}
    for a in AGS:
        raw = {f: C.rate(P[P.src == "seeded"], a, f) for f in FAMS}
        e[a] = {}
        for f in FAMS:
            o = [g for g in FAMS if g != f]
            e[a][f] = float(np.clip(raw[f] + np.mean([C.real_rate[a][g] for g in o]) - np.mean([raw[g] for g in o]), 0.01, 0.99))
    return e


logreg = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
full_est = estimate(PROBE)
for name, fn in (("gradient boosting", C.gbm), ("logistic regression", logreg)):
    r = C.route(full_est, build(PROBE, fn), F, h, q)
    m, lo, hi = C.boot(r.loss.values - H)
    print(f"all 6+6 probes, monitor {name:20s}: {r.loss.mean():.2f}  vs always human {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")

res = []
for k in range(10):
    keep = []
    for (a, f, src), g in PROBE.groupby(["agent", "fam", "src"]):
        keep += list(rs.choice(g.index, min(3, len(g)), replace=False))
    P = PROBE.loc[sorted(keep)].reset_index(drop=True)
    r = C.route(estimate(P), build(P, C.gbm), F, h, q)
    res.append(r.loss.values)
    print(f"subset {k}: 3+3 probes per family and agent: {r.loss.mean():.2f}")
L = np.mean(res, 0)
m, lo, hi = C.boot(L - H)
print(f"3+3 probes, mean over 10 subsets: {L.mean():.2f}  vs always human {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
