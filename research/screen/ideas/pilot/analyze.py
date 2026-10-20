import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, pickle
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pilot as P
from pilot import rho_seed, boot, E
from cpulock import cpu_lock
OUT = P.OUT
rows, ll, p2rows, draws = pickle.load(open(os.path.join(OUT, "raw.pkl"), "rb"))
SC = [f"{b}|{s}" for b, s in P.SCEN]; SHORT = dict(zip(SC, ["sop-proc", "sop-dom", "tau2-proc", "tau2-dom"]))
KALL = [0] + P.KS
def key(scen, prior, rule, k): return (scen, prior, "none" if k == 0 else rule, k)
def ci(x): return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
cache = {}
def rho(scen, prior, rule, k, col):
    kk = (key(scen, prior, rule, k), col)
    if kk not in cache:
        df = pd.concat(rows[key(scen, prior, rule, k)])
        cache[kk] = rho_seed(df, col)
    return cache[kk]
def dr(scen, prior, rule, k, col):
    kk = key(scen, prior, rule, k)
    if col == "ref_own" and (k == 0 or False): col = "pred"
    if col == "ref_doc" and prior == "doc": col = "ref_own" if k else "pred"
    return draws[scen][(kk, col)]
def pt(scen, prior, rule, k, col):
    if col == "ref_own" and k == 0: col = "pred"
    if col == "ref_doc" and prior == "doc": col = "ref_own" if k else "pred"
    return rho(scen, prior, rule, k, col)
res = {"rho": {}, "per_scenario": {}}
def pooled(f_pt, f_dr):
    p = float(np.mean([f_pt(s) for s in SC])); d = np.mean([f_dr(s) for s in SC], axis=0)
    return p, ci(d)
R = {}
for prior in ("doc", "free"):
    for rule in P.RULES:
        for k in KALL:
            if k == 0 and rule != "adaptive": continue
            r, c = pooled(lambda s: pt(s, prior, rule, k, "pred"), lambda s: dr(s, prior, rule, k, "pred"))
            g = pooled(lambda s: pt(s, prior, rule, k, "pred") - pt(s, prior, rule, k, "ref_own"),
                       lambda s: dr(s, prior, rule, k, "pred") - dr(s, prior, rule, k, "ref_own"))
            gd = pooled(lambda s: pt(s, prior, rule, k, "pred") - pt(s, prior, rule, k, "ref_doc"),
                        lambda s: dr(s, prior, rule, k, "pred") - dr(s, prior, rule, k, "ref_doc"))
            R[(prior, rule, k)] = dict(rho=r, rho_ci=c, gain_vs_k0_own=g[0], gain_ci=g[1], gain_vs_doc_k0=gd[0], gain_vs_doc_k0_ci=gd[1],
                by_scenario={SHORT[s]: pt(s, prior, rule, k, "pred") for s in SC},
                gain_by_scenario={SHORT[s]: pt(s, prior, rule, k, "pred") - pt(s, prior, rule, k, "ref_own") for s in SC},
                target_vs_full_beta={SHORT[s]: float(np.nanmean([E.rank_arrays(g_.cell.values, g_.tgt.values, g_.bf.values)
                    for _, g_ in pd.concat(rows[key(s, prior, rule, k)]).groupby("seed")])) for s in SC})
# free vs doc and decisions
for prior in ("doc", "free"):
    for rule in P.RULES:
        if rule != "adaptive": R[(prior, rule, 0)] = R[(prior, "adaptive", 0)]
dec = {}
for rule in P.RULES:
    row = {}
    for k in KALL:
        fd = pooled(lambda s: pt(s, "free", rule, k, "pred") - pt(s, "doc", rule, k, "pred"),
                    lambda s: dr(s, "free", rule, k, "pred") - dr(s, "doc", rule, k, "pred"))
        row[k] = dict(free_minus_doc=fd[0], ci=fd[1], matches=bool(fd[1][0] <= 0 <= fd[1][1] and fd[0] >= -0.03),
                      free_k_beats_doc_k0=bool(R[("free", rule, k)]["gain_vs_doc_k0_ci"][0] > 0),
                      free_k_minus_doc_k0=R[("free", rule, k)]["gain_vs_doc_k0"], free_k_minus_doc_k0_ci=R[("free", rule, k)]["gain_vs_doc_k0_ci"])
    first = lambda f: next((k for k in KALL if f(row[k])), None)
    dec[rule] = dict(by_k=row, smallest_k_docfree_matches_doc=first(lambda r: r["matches"]),
                     smallest_k_pilots_beat_doc_only=first(lambda r: r["free_k_beats_doc_k0"]))
# log loss
LL = {}
for prior in ("doc", "free"):
    for rule in P.RULES:
        for k in KALL:
            if k == 0 and rule != "adaptive": continue
            per = []
            for s in SC:
                a = pd.DataFrame(ll[key(s, prior, rule, k)], columns=["seed", "lp", "lq", "n"]).groupby("seed").sum()
                per.append(((a.lp / a.n).mean(), (a.lq / a.n).mean()))
            LL[(prior, rule, k)] = dict(prior_logloss=float(np.mean([p[0] for p in per])), posterior_logloss=float(np.mean([p[1] for p in per])),
                                        by_scenario={SHORT[s]: dict(prior=float(p[0]), post=float(p[1])) for s, p in zip(SC, per)})
# P2
p2 = {}
sop = "sopbench|new_procedures"; rows2, draws2 = {}, {}
for frac in (0.25, 0.5):
    for pname in ("doc", "free"):
        frames = []
        for (seed, k_, te, procs, cell, bf, mu, m0, mk, t0, S, N) in p2rows[("new_procedures", pname)]:
            rng = np.random.default_rng([777, seed, k_])
            df = pd.DataFrame({"seed": seed, "cell": cell, "case_id": te, "proc": procs, "m0": m0, "mk": mk, "tgt": t0, "bf": bf, "mu": mu})
            chosen = np.zeros(len(df), bool)
            for p, ix in df.groupby("proc").indices.items():
                n = max(1, int(np.ceil(frac * len(ix)))); n = min(n, len(ix) - 1) if len(ix) > 1 else n
                chosen[rng.choice(ix, n, replace=False)] = True
            shift = (df.mk - df.m0)[chosen].groupby(df.proc[chosen]).mean()
            df["pred"] = df.m0 + df.proc.map(shift).fillna(0.0)
            df["ref_own"] = df.m0; df["ref_doc"] = df.mu
            frames.append(df[~chosen][["seed", "cell", "case_id", "proc", "pred", "tgt", "ref_doc", "ref_own", "bf"]])
        rows2[(f"sopbench|p2_{int(frac*100)}", pname, "adaptive", 1)] = frames
scen2 = sorted({k[0] for k in rows2})
with cpu_lock("pilot"):
    d2 = boot(rows2, None, scen2)
for fr in ("p2_25", "p2_50"):
    s = f"sopbench|{fr}"
    for pname in ("doc", "free"):
        kk = (s, pname, "adaptive", 1); df = pd.concat(rows2[kk])
        a, b = rho_seed(df, "pred"), rho_seed(df, "ref_own")
        dd = d2[s][(kk, "pred")] - d2[s][(kk, "ref_own")]
        p2[f"{fr}|{pname}"] = dict(rho=a, rho_prior_only_same_cases=b, gain=a - b, gain_ci=ci(dd), n_cases_per_seed=int(len(df) / df.seed.nunique()))
    # compare with P1 k=1 adaptive on sop-proc
    for pname in ("doc", "free"):
        p2[f"{fr}|{pname}"]["p1_k1_adaptive_gain_sop_proc"] = R[(pname, "adaptive", 1)]["gain_by_scenario"]["sop-proc"]
json.dump(dict(R={"|".join(map(str, k)): v for k, v in R.items()}, decisions=dec, logloss={"|".join(map(str, k)): v for k, v in LL.items()}, P2=p2,
               nboot=P.NBOOT), open(os.path.join(OUT, "results.json"), "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
# plot
fig, ax = plt.subplots(figsize=(7, 4.5)); col = dict(weakest="#c0392b", median="#7f8c8d", strongest="#2471a3", adaptive="#d97757")
for rule in P.RULES:
    for prior, ls in (("doc", "-"), ("free", "--")):
        ys = [R[(prior, rule, k)]["rho"] if (k or rule == "adaptive") else R[(prior, "adaptive", 0)]["rho"] for k in KALL]
        ax.plot(KALL, ys, ls, color=col[rule], marker="o", ms=3, label=f"{rule} ({'doc' if prior == 'doc' else 'no doc'})")
ax.set_xlabel("number of pilot agents k"); ax.set_ylabel("pooled within-domain rho"); ax.set_title("Pilot runs, pooled over four new-process scenarios")
ax.axhline(0.131, color="k", lw=0.6, ls=":"); ax.legend(fontsize=6, ncol=2); ax.set_xticks(KALL); fig.tight_layout(); fig.savefig(os.path.join(OUT, "pilot_rho_vs_k.png"), dpi=150)
# text tables
L = []
for prior in ("doc", "free"):
    L.append(f"\n{prior} prior: pooled rho [95% CI]; gain over k=0 of same prior on same target [95% CI]")
    for rule in P.RULES:
        for k in KALL:
            if k == 0 and rule != "adaptive": continue
            r = R[(prior, rule, k)]
            L.append(f"{rule:9s} k={k} rho {r['rho']:+.3f} [{r['rho_ci'][0]:+.3f},{r['rho_ci'][1]:+.3f}]  gain {r['gain_vs_k0_own']:+.3f} [{r['gain_ci'][0]:+.3f},{r['gain_ci'][1]:+.3f}]  "
                     + " ".join(f"{a}={b:+.2f}" for a, b in r["by_scenario"].items()) + f"  tgt~fullbeta " + " ".join(f"{b:.2f}" for b in r["target_vs_full_beta"].values()))
L.append("\ndecisions"); L += [f"{r}: matches at k={d['smallest_k_docfree_matches_doc']}, beats doc-only at k={d['smallest_k_pilots_beat_doc_only']}" for r, d in dec.items()]
L.append("\nfree minus doc at same k, and free_k minus doc_k0")
for rule in P.RULES:
    for k in KALL:
        r = dec[rule]["by_k"][k]
        L.append(f"{rule:9s} k={k} free-doc {r['free_minus_doc']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  free_k-doc_0 {r['free_k_minus_doc_k0']:+.3f} [{r['free_k_minus_doc_k0_ci'][0]:+.3f},{r['free_k_minus_doc_k0_ci'][1]:+.3f}]")
L.append("\nlog-loss (non-pilot agents; prior -> posterior), pooled over scenarios")
for k_, v in LL.items(): L.append(f"{'|'.join(map(str, k_)):22s} {v['prior_logloss']:.4f} -> {v['posterior_logloss']:.4f}")
L.append("\nP2"); L += [f"{k}: " + json.dumps(v) for k, v in p2.items()]
open(os.path.join(OUT, "tables.txt"), "w").write("\n".join(L)); print("\n".join(L))
