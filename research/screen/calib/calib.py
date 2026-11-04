"""PREREG.md: calibrated success probabilities from few probes. Reuses the loaders of ../metrics/metrics.py."""
import sys, json
import numpy as np
from pathlib import Path
from scipy.stats import beta as Beta
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metrics"))
src = (HERE.parent / "metrics" / "metrics.py").read_text().split("def evaluate(")[0]
ns = {"__file__": str(HERE.parent / "metrics" / "metrics.py")}; exec(src, ns)
S = ns["sets"]()


def mom(rates):
    m, v = np.mean(rates), np.var(rates)
    v = max(v, 1e-4); c = max(m * (1 - m) / v - 1, 0.5)
    return m * c, (1 - m) * c


def logit(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def run(name, D, probes_key, other=None):
    P = D["probes"][probes_key]; real = D["real"]; grp = D["group"]
    procs = sorted(p for p in real if p in P and P[p] and (other is None or (p in other and other[p])))
    y = {p: np.array(real[p]) for p in procs}
    k = {p: sum(P[p]) for p in procs}; n = {p: len(P[p]) for p in procs}
    raw = {p: (k[p] + .5) / (n[p] + 1) for p in procs}
    pred = {"RAW": raw}; post = {}
    eb = {}
    for p in procs:
        o = [q for q in procs if grp[q] != grp[p]]
        a, b = mom([k[q] / n[q] for q in o]); eb[p] = (k[p] + a) / (n[p] + a + b); post[p] = (k[p] + a, n[p] - k[p] + b)
    pred["EB"] = eb
    rc = {}
    for p in procs:
        o = [q for q in procs if grp[q] != grp[p]]
        X = np.array([logit(raw[q]) for q in o]); Y = np.array([logit(np.mean(y[q])) for q in o]); W = np.array([len(y[q]) for q in o])
        A = np.vstack([np.ones_like(X), X]).T * np.sqrt(W)[:, None]; coef = np.linalg.lstsq(A, Y * np.sqrt(W), rcond=None)[0]
        rc[p] = float(1 / (1 + np.exp(-(coef[0] + coef[1] * logit(raw[p])))))
    pred["RECAL"] = rc
    if other is not None:
        ko = {p: sum(other[p]) for p in procs}; no = {p: len(other[p]) for p in procs}
        def pool(w, p, o):
            a, b = mom([(k[q] + w * ko[q]) / (n[q] + w * no[q]) for q in o])
            return (k[p] + w * ko[p] + a) / (n[p] + w * no[p] + a + b), (k[p] + w * ko[p] + a, n[p] - k[p] + w * (no[p] - ko[p]) + b)
        pl, ppost = {}, {}
        for p in procs:
            o = [q for q in procs if grp[q] != grp[p]]
            best = min((0, .25, .5, .75, 1), key=lambda w: np.mean([np.mean((pool(w, q, [r for r in o if r != q])[0] - y[q]) ** 2) for q in o]))
            pl[p], ppost[p] = pool(best, p, o)
        pred["POOL"] = pl
    n_all = sum(len(y[p]) for p in procs)
    yc = np.concatenate([y[p] for p in procs])
    base = {p: np.mean(np.concatenate([y[q] for q in procs if grp[q] != grp[p]])) for p in procs}
    bb = np.mean((np.concatenate([np.full(len(y[p]), base[p]) for p in procs]) - yc) ** 2)
    print(f"\n=== {name} [{probes_key}], {len(procs)} procedures")
    out = {}
    for kname, pr in pred.items():
        pc = np.clip(np.concatenate([np.full(len(y[p]), pr[p]) for p in procs]), 1e-3, 1 - 1e-3)
        br = np.mean((pc - yc) ** 2); ll = -np.mean(yc * np.log(pc) + (1 - yc) * np.log(1 - pc))
        ps = np.array([pr[p] for p in procs]); w = np.array([len(y[p]) for p in procs]); q5 = np.quantile(ps, np.linspace(0, 1, 6)); ece = 0
        for i in range(5):
            m = (ps >= q5[i]) & ((ps <= q5[i + 1]) if i == 4 else (ps < q5[i + 1]))
            if m.any(): ece += w[m].sum() / w.sum() * abs(np.average(ps[m], weights=w[m]) - np.average([np.mean(y[p]) for p, mm in zip(procs, m) if mm], weights=w[m]))
        sel = []
        for t in (0.9, 0.8):
            ch = [p for p in procs if pr[p] >= t]
            sel.append(f"{sum(len(y[p]) for p in ch) / n_all:4.0%} {np.mean(np.concatenate([y[p] for p in ch])) if ch else float('nan'):4.0%}")
        out[kname] = dict(brier=br, skill=1 - br / bb, ll=ll, ece=ece)
        print(f"  {kname:6s} Brier {br:.3f} skill {1 - br / bb:+.2f} logloss {ll:.3f} ECE {ece:.3f} | p>=0.9: {sel[0]}  p>=0.8: {sel[1]}")
    sel = []
    for t in (0.9, 0.8):
        ch = [p for p in procs if 1 - Beta.cdf(t, *post[p]) >= 0.8]
        sel.append(f"{sum(len(y[p]) for p in ch) / n_all:4.0%} {np.mean(np.concatenate([y[p] for p in ch])) if ch else float('nan'):4.0%}")
    print(f"  lower-bound rule (P(rate>=t) >= 0.8 under EB): t=0.9: {sel[0]}  t=0.8: {sel[1]}")
    return out


R = {}
for name, D in S.items():
    for pk in D["probes"]:
        oth = None
        if name.startswith("TauIndian"):
            o = "flash" if "g38" in name else "g38"
            oth = S[f"TauIndianBank {o}"]["probes"][pk]
        R[f"{name} | {pk}"] = run(name, D, pk, oth)
json.dump(R, open(HERE / "calib.json", "w"), indent=1)
