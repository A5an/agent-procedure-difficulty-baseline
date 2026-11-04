"""Exp 8 analysis (PREREG.md). Families complete for every set; Sonnet agent."""
import json, collections, sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
FAMS = json.load(open(HERE.parent / "tib/selection.json"))["test_families"]
rng = np.random.default_rng(20261114)


def load(name):
    d = collections.defaultdict(list); p = HERE / "runs" / f"{name}_sonnet.jsonl"
    if not p.exists(): return d
    seen = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("reward") is not None: seen[r["task_id"]] = r
    for t, r in seen.items():
        f = next((f for f in FAMS if f"_{f}_" in t), None)
        if f: d[f].append(int(r["reward"] >= 1))
    return d


real = load("real")
sets = {k: load(k) for k in ("doc", "seed", "pseudo")}
sets = {k: v for k, v in sets.items() if sum(len(x) for x in v.values()) >= 40}  # the seeded set was stopped early (DEVIATIONS.md)
fams = [f for f in FAMS if len(real[f]) >= 4 and all(len(v[f]) >= 3 for v in sets.values())]  # stopped early: families with >= 4 real and >= 3 probe episodes of every set
print(f"{len(fams)} families complete; real success {np.mean([x for f in fams for x in real[f]]):.2f}")
y = np.array([np.mean(real[f]) for f in fams]); w = np.array([len(real[f]) for f in fams])


def metrics(p, idx=None):
    i = np.arange(len(fams)) if idx is None else idx
    base = np.array([np.average(np.delete(y[i], j), weights=np.delete(w[i], j)) for j in range(len(i))])
    yc = [np.array(real[fams[k]]) for k in i]
    br = np.mean(np.concatenate([(p[k] - yy) ** 2 for k, yy in zip(i, yc)]))
    bb = np.mean(np.concatenate([(base[j] - yy) ** 2 for j, yy in enumerate(yc)]))
    return 1 - br / bb


P = {k: np.array([(sum(v[f]) + .5) / (len(v[f]) + 1) for f in fams]) for k, v in sets.items()}
for k, p in P.items():
    r = spearmanr(p, y).correlation
    rs = [spearmanr(p[i], y[i]).correlation for i in (rng.integers(0, len(fams), len(fams)) for _ in range(4000)) if np.ptp(p[i]) and np.ptp(y[i])]
    sk = metrics(p)
    dl = p >= 0.9
    cov = w[dl].sum() / w.sum(); suc = np.average(y[dl], weights=w[dl]) if dl.any() else float("nan")
    print(f"{k:7s} rho {r:+.2f} [{np.percentile(rs, 2.5):+.2f}, {np.percentile(rs, 97.5):+.2f}]  Brier skill {sk:+.2f}  mean |p-real| {np.mean(np.abs(p - y)):.2f}  delegate p>=0.9: {cov:.0%} of cases, success {suc:.0%}")
if "pseudo" in P and "doc" in P:
    d = []
    for _ in range(4000):
        i = rng.integers(0, len(fams), len(fams))
        d.append(metrics(P["pseudo"], i) - metrics(P["doc"], i))
    print(f"H2 Brier skill PSEUDO - DOC: {metrics(P['pseudo']) - metrics(P['doc']):+.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")
