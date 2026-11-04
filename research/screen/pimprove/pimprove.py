"""PREREG.md in this folder: can the same probe runs predict the family's real failure rate better than the seeded
probe pass rate? Leave one family out over the 50 TauIndianBank families.

  python pimprove.py -> pimprove.out (stdout), results.json
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
# loader copied from ../combo_rep/analyze_rep.py (importing it would run the whole routing analysis)
src = (HERE.parent / "combo_rep" / "analyze_rep.py").read_text()
ns = {"__file__": str(HERE.parent / "combo_rep" / "analyze_rep.py")}
exec(src.split("def curve(")[0], ns)
A = type("A", (), ns)
E = pd.concat([ns["E_old"], ns["E_new"]], ignore_index=True)
FAMS = ns["OLD"] + ns["NEW"]
AGS = ns["AGS"]
rng = np.random.default_rng(20261113)
REAL = E[E.src == "real"]
PROBE = E[E.src != "real"].reset_index(drop=True)


def frate(df, a, f):
    x = df[(df.agent == a) & (df.fam == f)].fail
    return x.mean() if len(x) else np.nan


real = {a: np.array([frate(REAL, a, f) for f in FAMS]) for a in AGS}
seed = {a: np.array([frate(PROBE[PROBE.src == "seeded"], a, f) for f in FAMS]) for a in AGS}
doc = {a: np.array([frate(PROBE[PROBE.src == "doc"], a, f) for f in FAMS]) for a in AGS}
p12 = {a: np.array([frate(PROBE, a, f) for f in FAMS]) for a in AGS}

# R: monitor trained on probe episodes of the other families, mean risk of this family's probe episodes
risk = {a: np.zeros(len(FAMS)) for a in AGS}
for j, f in enumerate(FAMS):
    tr, te = PROBE[PROBE.fam != f], PROBE[PROBE.fam == f]
    s = A.gbm().fit(tr[A.FEATS], tr.fail).predict_proba(te[A.FEATS])[:, 1]
    te = te.assign(s=s)
    for a in AGS:
        risk[a][j] = te[te.agent == a].s.mean()

# C: linear regression, leave one family out
comb = {}
for a in AGS:
    o = "flash" if a == "g38" else "g38"
    X = np.column_stack([seed[a], doc[a], risk[a], seed[o]])
    pred = np.zeros(len(FAMS))
    for j in range(len(FAMS)):
        m = np.ones(len(FAMS), bool); m[j] = False
        pred[j] = LinearRegression().fit(X[m], real[a][m]).predict(X[j:j + 1])[0]
    comb[a] = np.clip(pred, 0, 1)


def stats(x, y, base=None, B=4000):
    r = spearmanr(x, y).correlation; mae = np.mean(np.abs(x - y))
    rs, ds, ms = [], [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        rr = spearmanr(x[i], y[i]).correlation; rs.append(rr)
        ms.append(np.mean(np.abs(x[i] - y[i])))
        if base is not None and np.ptp(base[i]) > 0:
            ds.append(rr - spearmanr(base[i], y[i]).correlation)
    out = {"rho": r, "rho_ci": np.percentile(rs, [2.5, 97.5]).tolist(), "mae": mae, "mae_ci": np.percentile(ms, [2.5, 97.5]).tolist()}
    if ds:
        out["rho_minus_B"] = (r - spearmanr(base, y).correlation, *np.percentile(ds, [2.5, 97.5]).tolist())
    return out


# ceiling: split-half reliability of the real target (odd vs even tasks within family)
def split_half(a):
    d = REAL[REAL.agent == a].copy()
    d["k"] = d.groupby("fam").cumcount() % 2
    t = d.groupby(["fam", "k"]).fail.mean().unstack()
    r = spearmanr(t[0], t[1]).correlation
    return 2 * r / (1 + r)


RES = {}
print("\n================ procedure-level prediction, 50 TauIndianBank families, leave one family out")
for a in AGS:
    y = real[a]
    rel = split_half(a)
    print(f"\n{a}: real failure mean {y.mean():.2f}; ceiling (split-half reliability of the target) {rel:.2f}, so rho above about {np.sqrt(rel):.2f} is out of reach")
    RES[a] = {"reliability": rel}
    for name, x in (("B  seeded probe rate", seed[a]), ("   documentation probe rate", doc[a]), ("P12 all 12 probes", p12[a]),
                    ("R  monitor risk on probe episodes", risk[a]), ("C  combination", comb[a])):
        s = stats(x, y, None if name.startswith("B") else seed[a])
        RES[a][name.strip()] = s
        extra = f"  vs B {s['rho_minus_B'][0]:+.2f} [{s['rho_minus_B'][1]:+.2f}, {s['rho_minus_B'][2]:+.2f}]" if "rho_minus_B" in s else ""
        print(f"  {name:36s} rho {s['rho']:+.2f} [{s['rho_ci'][0]:+.2f}, {s['rho_ci'][1]:+.2f}]  MAE {s['mae']:.3f}{extra}")
json.dump(RES, open(HERE / "results.json", "w"), indent=1, default=float)


# descriptive, added after the main table: how noisy is a family's rate from 6 seeded probes, and what would more
# probes of the same kind buy (classical test theory: observed rho = true rho x sqrt(rel_probe x rel_target))
def split_rel(df, a, B=200):
    vals = []
    for b in range(B):
        r2 = np.random.default_rng(b)
        h = {0: [], 1: []}
        for f in FAMS:
            x = df[(df.agent == a) & (df.fam == f)].fail.values
            idx = r2.permutation(len(x)); k = len(x) // 2
            h[0].append(x[idx[:k]].mean()); h[1].append(x[idx[k:2 * k]].mean())
        r = spearmanr(h[0], h[1]).correlation
        vals.append(2 * r / (1 + r))
    return float(np.mean(vals))


print("\n================ how far more probes of the same kind could go")
for a in AGS:
    rp = split_rel(PROBE[PROBE.src == "seeded"], a)
    rt = split_rel(REAL, a)
    rho = spearmanr(seed[a], real[a]).correlation
    true = rho / np.sqrt(max(rp, 1e-6) * rt)
    def sb(r, k):
        return k * r / (1 + (k - 1) * r)
    proj = {n: true * np.sqrt(sb(rp, n / 6) * rt) for n in (12, 18, 30)}
    print(f"{a}: reliability of the 6-seeded-probe rate {rp:.2f}, of the 6-task real rate {rt:.2f}; observed rho {rho:.2f};"
          f" implied rho between true rates {min(true, 1):.2f}; projected observed rho with 12 / 18 / 30 seeded probes "
          + " / ".join(f"{min(v, 1):.2f}" for v in proj.values()))
