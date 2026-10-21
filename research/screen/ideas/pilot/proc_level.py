"""Procedure-level rho for SOPBench: prior only vs P2 (25%, 50% piloted) vs P1 adaptive k=1."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, pickle
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import pilot as P
from cpulock import cpu_lock
rows, ll, p2rows, _ = pickle.load(open(os.path.join(P.OUT, "raw.pkl"), "rb"))
def rho_proc(dom, pred, truth):
    num = den = 0.0
    for d in np.unique(dom):
        m = dom == d; n = m.sum()
        if n < 3: continue
        pairs = n * (n - 1) / 2; a, b = rankdata(pred[m]), rankdata(truth[m])
        if a.std() > 0 and b.std() > 0: num += np.corrcoef(a, b)[0, 1] * pairs
        den += pairs
    return num / den
def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
def build(scheme):
    """returns dict cond -> {seed: DataFrame(proc, dom, pred, truth)} for variants A (truth all cases) and B (non-pilot)."""
    out = {}
    for pname in ("doc", "free"):
        for frac in (0.25, 0.5):
            for seed_fold in [None]:
                pass
        per = {}
        for (seed, k_, te, procs, cell, bf, mu, m0, mk, t0, S, N) in p2rows[(scheme, pname)]:
            dom = cell % 100
            df = pd.DataFrame({"seed": seed, "proc": procs, "dom": dom, "m0": m0, "mk": mk, "bf": bf, "mu": mu})
            for frac in (0.25, 0.5):
                rng = np.random.default_rng([777, seed, k_]); ch = np.zeros(len(df), bool)
                for p, ix in df.groupby("proc").indices.items():
                    n = max(1, int(np.ceil(frac * len(ix)))); n = min(n, len(ix) - 1) if len(ix) > 1 else n
                    ch[rng.choice(ix, n, replace=False)] = True
                shift = (df.mk - df.m0)[ch].groupby(df.proc[ch]).mean()
                d = df.assign(p2=df.m0 + df.proc.map(shift).fillna(0.0), chosen=ch)
                per.setdefault(frac, []).append(d)
            per.setdefault("all", []).append(df)
        out[pname] = {k: pd.concat(v) for k, v in per.items()}
    return out
def agg(d, predcol, nonpilot, truth_nonpilot):
    x = d[~d.chosen] if nonpilot else d
    t = d[~d.chosen] if truth_nonpilot else d
    g = x.groupby(["seed", "proc", "dom"])[predcol].mean().rename("pred").reset_index()
    h = t.groupby(["seed", "proc", "dom"]).bf.mean().rename("truth").reset_index()
    return g.merge(h, on=["seed", "proc", "dom"])
res = {}
with cpu_lock("pilot"):
    for scheme in ("new_domain", "new_procedures"):
        B = build(scheme); conds = {}
        for pname in ("doc", "free"):
            for frac in (0.25, 0.5):
                d = B[pname][frac]
                for var, np_, tn in (("truth_all", False, False), ("nonpilot", True, True)):
                    conds[(f"p2_{int(frac*100)}", pname, var)] = agg(d, "p2", np_, tn)
                    conds[(f"k0_matched_{int(frac*100)}", pname, var)] = agg(d, "m0" if pname == "doc" else "m0", np_, tn)
            # P1 adaptive k=1 (all cases piloted once): posterior mean of every case
            d = B[pname]["all"]; conds[("p1_adaptive_k1", pname, "truth_all")] = agg(d.assign(chosen=False), "mk", False, False)
            conds[("k0", pname, "truth_all")] = agg(d.assign(chosen=False), "m0", False, False)
        procs0 = sorted(conds[("k0", "doc", "truth_all")].query("seed==0").proc.unique())
        rng = np.random.default_rng([20261005, 0 if scheme == "new_domain" else 1])
        picks = [rng.choice(procs0, len(procs0)) for _ in range(500)]
        def point(df): return float(np.mean([rho_proc(g.dom.values, g.pred.values, g.truth.values) for _, g in df.groupby("seed")]))
        def draws(df):
            g0 = df[df.seed == 0].set_index("proc"); out = []
            for pk in picks:
                pk = [p for p in pk if p in g0.index]; s = g0.loc[pk]
                out.append(rho_proc(s.dom.values, s.pred.values, s.truth.values))
            return np.array(out)
        R = {}
        for k, df in conds.items(): R[k] = (point(df), draws(df), df.query("seed==0").proc.nunique())
        out = {}
        for var in ("truth_all", "nonpilot"):
            for pname in ("doc", "free"):
                for frac in (25, 50):
                    a, ref = R[(f"p2_{frac}", pname, var)], R[(f"k0_matched_{frac}", pname, var)]
                    refdoc = R[(f"k0_matched_{frac}", "doc", var)]
                    out[f"{var}|{pname}|p2_{frac}"] = dict(rho=a[0], rho_ci=ci(a[1]), prior_only_same_prior=ref[0], gain_vs_own_prior=a[0] - ref[0],
                        gain_ci=ci(a[1] - ref[1]), gain_vs_doc_prior_only=a[0] - refdoc[0], gain_vs_doc_ci=ci(a[1] - refdoc[1]))
        for pname in ("doc", "free"):
            a, ref, refdoc = R[("p1_adaptive_k1", pname, "truth_all")], R[("k0", pname, "truth_all")], R[("k0", "doc", "truth_all")]
            out[f"truth_all|{pname}|p1_adaptive_k1"] = dict(rho=a[0], rho_ci=ci(a[1]), prior_only_same_prior=ref[0], gain_vs_own_prior=a[0] - ref[0],
                gain_ci=ci(a[1] - ref[1]), gain_vs_doc_prior_only=a[0] - refdoc[0], gain_vs_doc_ci=ci(a[1] - refdoc[1]))
        out["n_procedures"] = R[("k0", "doc", "truth_all")][2]
        res[scheme] = out
json.dump(res, open(os.path.join(P.OUT, "proc_level.json"), "w"), indent=1)
for sch, o in res.items():
    print(sch, "n_proc", o["n_procedures"])
    for k, v in o.items():
        if k != "n_procedures": print(f"  {k:34s} rho {v['rho']:+.3f} [{v['rho_ci'][0]:+.3f},{v['rho_ci'][1]:+.3f}] prior-only {v['prior_only_same_prior']:+.3f} gain {v['gain_vs_own_prior']:+.3f} [{v['gain_ci'][0]:+.3f},{v['gain_ci'][1]:+.3f}] vs doc-only {v['gain_vs_doc_prior_only']:+.3f} [{v['gain_vs_doc_ci'][0]:+.3f},{v['gain_vs_doc_ci'][1]:+.3f}]")
