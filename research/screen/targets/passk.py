"""E6: pass^k from the baseline's out-of-fold P, independent attempts vs beta-binomial with phi fitted on other folds."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, comb

RUN = ZR.SCREEN + "/targets/run"
KS = [2, 3, 4]


def bb_ll(s, n, mu, phi):
    a, b = mu * phi, (1 - mu) * phi
    return (gammaln(s + a) + gammaln(n - s + b) - gammaln(n + phi) + gammaln(phi) - gammaln(a) - gammaln(b)
            + gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1))


def fit_phi(s, n, mu):
    r = minimize_scalar(lambda lp: -bb_ll(s, n, mu, np.exp(lp)).sum(), bounds=(np.log(0.05), np.log(1e4)), method="bounded")
    return float(np.exp(r.x))


def passk_bb(mu, phi, k):
    out = np.ones_like(mu)
    for i in range(k):
        out *= (mu * phi + i) / (phi + i)
    return out


def cell_table(bench, scheme):
    o = pd.read_csv(f"{RUN}/{bench}/obs_{scheme}.csv.gz", usecols=["seed", "fold", "agent", "case_id", "y", "adele"])
    g = o.groupby(["seed", "fold", "agent", "case_id"]).agg(s=("y", "sum"), n=("y", "size"), p=("adele", "first")).reset_index()
    g["p"] = g.p.clip(1e-4, 1 - 1e-4)
    return g


def run(bench, scheme):
    g = cell_table(bench, scheme)
    rows, phis = [], []
    parts = []
    for (seed, fold), te in g.groupby(["seed", "fold"]):
        tr = g[(g.seed == seed) & (g.fold != fold)]
        phi = fit_phi(tr.s.values, tr.n.values, tr.p.values)
        phis.append(phi)
        te = te.copy(); te["phi"] = phi
        parts.append(te)
    d = pd.concat(parts)
    res = {"phi_mean": float(np.mean(phis)), "phi_min": float(np.min(phis)), "phi_max": float(np.max(phis)),
           "n_cells": int(len(d) / d.seed.nunique())}
    for k in KS:
        m = d[d.n >= k]
        # observed: fraction of the k-subsets of the n attempts that are all successes (expected value of the indicator)
        frac = comb(m.s.values, k) / comb(m.n.values, k)
        for name, pk in (("indep", m.p.values ** k), ("betabin", passk_bb(m.p.values, m.phi.values, k))):
            brier = pk ** 2 - 2 * pk * frac + frac  # mean over k-subsets of (pk - indicator)^2
            res[f"k{k}_{name}"] = {"brier": float(brier.mean()), "mean_pred": float(pk.mean()), "mean_obs": float(frac.mean())}
        # paired bootstrap over tasks (clustered), seed-averaged cells
        bi = m.p.values ** k; bb = passk_bb(m.p.values, m.phi.values, k)
        diff = (bb ** 2 - 2 * bb * frac + frac) - (bi ** 2 - 2 * bi * frac + frac)
        tdf = pd.DataFrame({"t": m.case_id.values, "d": diff}).groupby("t").d.agg(["sum", "count"])
        rng = np.random.default_rng(0); sums, cnts = tdf["sum"].values, tdf["count"].values
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(tdf), len(tdf)); bs.append(sums[ix].sum() / cnts[ix].sum())
        res[f"k{k}_diff_bb_minus_indep"] = {"mean": float(diff.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        # calibration by deciles of predicted pass^k for the better one is overkill; give 4 bins for both
        for name, pk in (("indep", bi), ("betabin", bb)):
            q = pd.qcut(pk, 4, duplicates="drop")
            res[f"k{k}_{name}_bins"] = [[float(a), float(b)] for a, b in zip(pd.Series(pk).groupby(q, observed=True).mean(), pd.Series(frac).groupby(q, observed=True).mean())]
    return res


out = {}
for bench, schemes in (("tau2", ["new_procedures", "new_domain", "random_cases"]), ("tauk_banking", ["random_cases"])):
    for s in schemes:
        out[f"{bench}|{s}"] = run(bench, s)
        r = out[f"{bench}|{s}"]
        print(bench, s, "phi", round(r["phi_mean"], 2), [round(r["phi_min"], 2), round(r["phi_max"], 2)])
        for k in KS:
            print(f"  k={k} indep brier {r[f'k{k}_indep']['brier']:.4f} pred {r[f'k{k}_indep']['mean_pred']:.3f} | bb brier {r[f'k{k}_betabin']['brier']:.4f} pred {r[f'k{k}_betabin']['mean_pred']:.3f} | obs {r[f'k{k}_indep']['mean_obs']:.3f} | diff {r[f'k{k}_diff_bb_minus_indep']}")
json.dump(out, open("passk.json", "w"), indent=1)
