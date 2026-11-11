"""Routing per procedure with intervals (leave-one-domain-out predictions, no API calls).

Options per procedure: flash, g38, sonnet, human. Predicted loss = API cost + F * predicted failure (human: h + F(1-q)).
Realized loss = the same formula with the real per-procedure failure rate (benchmark mix 0.35 perform / 0.65 refuse).
Paired bootstrap over procedures (2000 draws) for differences in mean realized loss; predictions averaged over 10
random probe subsets where n < all.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import eval2 as M

API = {"flash": 0.008, "g38": 0.021, "sonnet": 0.05}
AGS = list(API)
rng = np.random.default_rng(5)
REAL_FAIL = {a: M.obs_proc(M.AG.index(a)) for a in AGS}
P = len(M.PROCS)
PRED = {}
for n in (2, 4, 8, None):
    for s in (range(10) if n else [0]):
        up = {a: M.unit_preds(M.AG.index(a), n, s) for a in AGS}
        for m in ("raw", "calib", "mf"):
            PRED[(n, s, m)] = {a: M.proc_vec(up[a][m]) for a in AGS}


def realized(F, h, q, choice_fn):
    L = np.vstack([API[a] + F * REAL_FAIL[a] for a in AGS] + [np.full(P, h + F * (1 - q))])
    return L, L[choice_fn(L), np.arange(P)] if choice_fn else None


out = {}
for F, h, q in ((20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)):
    key = f"F={F} h={h} q={q}"
    L = np.vstack([API[a] + F * REAL_FAIL[a] for a in AGS] + [np.full(P, h + F * (1 - q))])
    per = {}
    for i, o in enumerate(AGS + ["human"]): per[f"always {o}"] = L[i]
    per["oracle"] = L.min(0)
    for n in (2, 4, 8, None):
        for m in ("raw", "calib", "mf"):
            vals = []
            for s in (range(10) if n else [0]):
                pr = PRED[(n, s, m)]
                PL = np.vstack([API[a] + F * pr[a] for a in AGS] + [np.full(P, h + F * (1 - q))])
                vals.append(L[PL.argmin(0), np.arange(P)])
            per[f"{m} n={n or 'all'}"] = np.mean(vals, 0)
    best_const = min((k for k in per if k.startswith("always")), key=lambda k: per[k].mean())
    res = {k: float(v.mean()) for k, v in per.items()}
    res["best_constant"] = best_const
    def diff(a, b):
        d = per[a] - per[b]; bs = [d[rng.integers(0, P, P)].mean() for _ in range(2000)]
        return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    for n in ("2", "4", "8", "all"):
        res[f"mf n={n} - best constant"] = diff(f"mf n={n}", best_const)
        res[f"mf n={n} - raw n={n}"] = diff(f"mf n={n}", f"raw n={n}")
        res[f"mf n={n} - calib n={n}"] = diff(f"mf n={n}", f"calib n={n}")
        res[f"mf n={n} - raw n=all"] = diff(f"mf n={n}", "raw n=all")
        g = per[best_const].mean() - per["oracle"].mean()
        res[f"mf n={n} share of oracle gain"] = float((per[best_const].mean() - per[f"mf n={n}"].mean()) / g)
    out[key] = res
    print("\n" + key)
    for k, v in res.items():
        print("  ", k, (round(v, 3) if isinstance(v, float) else (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v)))
json.dump(out, open(os.path.join(HERE, "routing.json"), "w"), indent=1)
