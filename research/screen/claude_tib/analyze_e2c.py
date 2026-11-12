"""E2-C analysis (PREREG_E2C.md), written before the runs finished. Per Claude agent: k=2 probes with the Haiku customer
(runs/k2_<agent>.jsonl) vs the Sonnet customer (runs/k2_<agent>_simsonnet.jsonl); real tasks keep the Haiku customer.
  python3 analyze_e2c.py -> analyze_e2c.out (stdout), results_e2c.json"""
import json, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEL = json.load(open(HERE.parent / "seedk" / "selection_seedk.json")); F = sorted(SEL["families"])
B, SEED = 2000, 20261112


def fam_of(tid):
    return next(f for f in F if f"_{f}_" in tid)


def load(path):
    per = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is not None: per[r["task_id"]] = int(r["reward"] >= 1.0)
    return per


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return np.nan if rx.std() == 0 or ry.std() == 0 else float(np.corrcoef(rx, ry)[0, 1])


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


OUT = {}; rng = np.random.default_rng(SEED)
for a in ["haiku", "sonnet"]:
    real, h, s = load(HERE / f"runs/real_{a}.jsonl"), load(HERE / f"runs/k2_{a}.jsonl"), load(HERE / f"runs/k2_{a}_simsonnet.jsonl")
    both = sorted(set(h) & set(s))
    byf = collections.defaultdict(list)
    for t in both: byf[fam_of(t)].append(t)
    fams = [f for f in F if len(byf[f]) == 6 and any(fam_of(t) == f for t in real)]
    rf = {f: 1 - np.mean([v for t, v in real.items() if fam_of(t) == f]) for f in fams}
    ph = {f: 1 - np.mean([h[t] for t in byf[f]]) for f in fams}; ps = {f: 1 - np.mean([s[t] for t in byf[f]]) for f in fams}
    pm = {f: (ph[f] + ps[f]) / 2 for f in fams}
    dis = {f: np.mean([h[t] != s[t] for t in byf[f]]) for f in fams}
    point = {"disagree": float(np.mean([dis[f] for f in fams])),
             "rho_haiku_customer": spearman([ph[f] for f in fams], [rf[f] for f in fams]),
             "rho_sonnet_customer": spearman([ps[f] for f in fams], [rf[f] for f in fams]),
             "rho_mean": spearman([pm[f] for f in fams], [rf[f] for f in fams])}
    bs = collections.defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, len(fams), len(fams)); ff = [fams[i] for i in idx]
        bs["disagree"].append(np.mean([dis[f] for f in ff]))
        r1 = spearman([ph[f] for f in ff], [rf[f] for f in ff]); r2 = spearman([ps[f] for f in ff], [rf[f] for f in ff])
        r3 = spearman([pm[f] for f in ff], [rf[f] for f in ff])
        bs["rho_haiku_customer"].append(r1); bs["rho_sonnet_customer"].append(r2); bs["rho_mean"].append(r3)
        bs["sonnet-haiku"].append(r2 - r1)
    six = {"haiku_customer": [f for f in fams if ph[f] == 0], "sonnet_customer": [f for f in fams if ps[f] == 0]}
    six["both"] = [f for f in six["haiku_customer"] if f in six["sonnet_customer"]]
    real_ok = lambda fs: (sum(v for t, v in real.items() if fam_of(t) in fs), sum(1 for t in real if fam_of(t) in fs))
    res = {"families": len(fams), "probes": len(both), "point": point, "ci": {k: ci(v) for k, v in bs.items()},
           "six_of_six": {k: {"families": len(v), "real": real_ok(set(v))} for k, v in six.items()},
           "six_changed": len(set(six["haiku_customer"]) ^ set(six["sonnet_customer"]))}
    OUT[a] = res
    print(f"\n=== agent {a}: {len(fams)} families, {len(both)} probes with both customers")
    for k, v in point.items(): print(f"  {k:22s} {v:+.3f}  [{ci(bs[k])[0]:+.3f}, {ci(bs[k])[1]:+.3f}]")
    print(f"  rho sonnet - haiku customer  [{ci(bs['sonnet-haiku'])[0]:+.3f}, {ci(bs['sonnet-haiku'])[1]:+.3f}]")
    for k, v in res["six_of_six"].items():
        print(f"  6 of 6 under {k:16s}: {v['families']} families, real {v['real'][0]}/{v['real'][1]}" + (f" = {v['real'][0] / v['real'][1]:.2f}" if v['real'][1] else ""))
    print(f"  families whose 6-of-6 status changes: {res['six_changed']}")
json.dump(OUT, open(HERE / "results_e2c.json", "w"), indent=1, default=float)
