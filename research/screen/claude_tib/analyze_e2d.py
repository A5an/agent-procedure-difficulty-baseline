"""E2-D analysis (PREREG_E2D.md), written before the runs. python3 analyze_e2d.py -> analyze_e2d.out (stdout), results_e2d.json"""
import json, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEL = json.load(open(HERE.parent / "seedk" / "selection_seedk.json")); F = sorted(SEL["families"])
B, SEED = 2000, 20261113


def fam_of(tid): return next(f for f in F if f"_{f}_" in tid)


def load(path):
    per = {}
    if not Path(path).exists(): return per
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
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def rate(per, f): return float(np.mean([v for t, v in per.items() if fam_of(t) == f]))


OUT = {}; rng = np.random.default_rng(SEED)
for a in ["haiku", "sonnet"]:
    P = {"H": load(HERE / f"runs/k2_{a}.jsonl"), "S": load(HERE / f"runs/k2_{a}_simsonnet.jsonl")}
    R = {"H": load(HERE / f"runs/real_{a}.jsonl"), "S": load(HERE / f"runs/real_{a}_simsonnet.jsonl")}
    pboth = set(P["H"]) & set(P["S"]); rboth = set(R["H"]) & set(R["S"])
    fams = [f for f in F if sum(fam_of(t) == f for t in pboth) == 6 and any(fam_of(t) == f for t in rboth)]
    PH = {p: {t: v for t, v in P[p].items() if t in pboth} for p in P}; RH = {r: {t: v for t, v in R[r].items() if t in rboth} for r in R}
    pf = {p: {f: 1 - rate(PH[p], f) for f in fams} for p in "HS"}; rf = {r: {f: 1 - rate(RH[r], f) for f in fams} for r in "HS"}
    cells = {f"{p}{r}": spearman([pf[p][f] for f in fams], [rf[r][f] for f in fams]) for p in "HS" for r in "HS"}
    bs = collections.defaultdict(list)
    for _ in range(B):
        ff = [fams[i] for i in rng.integers(0, len(fams), len(fams))]
        c = {f"{p}{r}": spearman([pf[p][f] for f in ff], [rf[r][f] for f in ff]) for p in "HS" for r in "HS"}
        for k, v in c.items(): bs[k].append(v)
        bs["SS-SH"].append(c["SS"] - c["SH"]); bs["HH-HS"].append(c["HH"] - c["HS"])
        bs["matched-mismatched"].append((c["SS"] + c["HH"]) / 2 - (c["SH"] + c["HS"]) / 2)
    point = dict(cells); point["SS-SH"] = cells["SS"] - cells["SH"]; point["HH-HS"] = cells["HH"] - cells["HS"]
    point["matched-mismatched"] = (cells["SS"] + cells["HH"]) / 2 - (cells["SH"] + cells["HS"]) / 2
    change = float(np.mean([RH["H"][t] != RH["S"][t] for t in rboth]))
    succ = {r: float(np.mean(list(RH[r].values()))) for r in "HS"}
    six = {p: [f for f in fams if pf[p][f] == 0] for p in "HS"}; six["both"] = [f for f in six["H"] if f in six["S"]]
    def real_ok(fs, r): return (sum(v for t, v in RH[r].items() if fam_of(t) in fs), sum(1 for t in RH[r] if fam_of(t) in fs))
    sixtab = {f"probes {p}, real {r}": real_ok(set(six[p]), r) for p in ("H", "S", "both") for r in "HS"}
    res = {"families": len(fams), "real_tasks_both": len(rboth), "point": point, "ci": {k: ci(v) for k, v in bs.items()},
           "real_outcome_change": change, "real_success": succ, "six_counts": {k: len(v) for k, v in six.items()}, "six_real": sixtab,
           "hypothesis_SS_ge_0.55": cells["SS"] >= 0.55}
    OUT[a] = res
    print(f"\n=== agent {a}: {len(fams)} families, {len(rboth)} real tasks with both customers")
    for k in ("HH", "HS", "SH", "SS", "SS-SH", "HH-HS", "matched-mismatched"):
        print(f"  {k:20s} {point[k]:+.3f}  {ci(bs[k])}")
    print(f"  real outcome changes with the customer: {change:.3f}; real success Haiku customer {succ['H']:.3f}, Sonnet customer {succ['S']:.3f}")
    for k, (o, n) in sixtab.items(): print(f"  6 of 6 {k:24s}: {o}/{n}" + (f" = {o / n:.2f}" if n else ""))
    print(f"  hypothesis SS >= 0.55: {res['hypothesis_SS_ge_0.55']}")
json.dump(OUT, open(HERE / "results_e2d.json", "w"), indent=1, default=float)
