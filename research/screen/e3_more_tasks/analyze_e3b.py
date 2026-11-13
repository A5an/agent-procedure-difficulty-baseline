"""E3-B analysis (PREREG.md), written before the runs. python3 analyze_e3b.py -> analyze_e3b.out (stdout), results_e3b.json"""
import json, collections
from pathlib import Path
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent; SCR = HERE.parent; CT = SCR / "claude_tib" / "runs"
SEL = json.load(open(SCR / "seedk" / "selection_seedk.json"))["families"]; F = sorted(SEL)
NEW = {x["id"]: x for x in json.load(open(HERE / "new_tasks.json"))}
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


def cp_lower(k, n, a=0.05): return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


OUT = {}; rng = np.random.default_rng(SEED)
for a in ["haiku", "sonnet"]:
    old = load(CT / f"real_{a}.jsonl"); new = {t: v for t, v in load(CT / f"new_{a}.jsonl").items() if t in NEW and not NEW[t]["seed_own"]}
    comb = {**old, **new}
    byf = lambda d: {f: [v for t, v in d.items() if fam_of(t) == f] for f in F}
    O, N, Cb = byf(old), byf(new), byf(comb)
    res = {"new_tasks_scored": len(new), "real_success_old": float(np.mean(list(old.values()))), "real_success_new": float(np.mean(list(new.values()))) if new else None}
    for k in ("k2", "k5"):
        pr = load(CT / f"{k}_{a}.jsonl"); pf = {f: [v for t, v in pr.items() if fam_of(t) == f] for f in F}
        acc = [f for f in F if len(pf[f]) == 6 and all(pf[f])]
        s = sum(sum(Cb[f]) for f in acc); n = sum(len(Cb[f]) for f in acc)
        bs = []
        for _ in range(B):
            ff = [acc[i] for i in rng.integers(0, len(acc), len(acc))]
            nn = sum(len(Cb[f]) for f in ff); bs.append(sum(sum(Cb[f]) for f in ff) / nn if nn else np.nan)
        so = sum(sum(O[f]) for f in acc); no = sum(len(O[f]) for f in acc)
        res[f"six_of_six_{k}"] = {"families": len(acc), "cases": n, "success": s / n if n else None, "cp_lower_95": cp_lower(s, n),
                                   "boot_p5_families": float(np.nanpercentile(bs, 5)), "old_six_only": [so, no, so / no if no else None]}
        print(f"{a} {k} 6-of-6: {len(acc)} families, {n} cases, success {s}/{n} = {s / n:.3f}, CP lower {cp_lower(s, n):.3f}, bootstrap p5 {np.nanpercentile(bs, 5):.3f} (old six only {so}/{no})")
    fam3 = [f for f in F if len(N[f]) >= 3]
    ro = [np.mean(O[f]) for f in fam3]; rn = [np.mean(N[f]) for f in fam3]
    bsr, bsd = [], []
    for _ in range(B):
        ii = rng.integers(0, len(fam3), len(fam3)); bsr.append(spearman([ro[i] for i in ii], [rn[i] for i in ii])); bsd.append(np.mean([rn[i] - ro[i] for i in ii]))
    res["old_vs_new"] = {"families": len(fam3), "spearman": spearman(ro, rn), "spearman_ci": ci(bsr), "mean_diff_new_minus_old": float(np.mean(np.array(rn) - np.array(ro))), "diff_ci": ci(bsd)}
    print(f"{a} old six vs new tasks per family ({len(fam3)} families with >=3 new): Spearman {res['old_vs_new']['spearman']:.3f} {ci(bsr)}, mean new-old {res['old_vs_new']['mean_diff_new_minus_old']:+.3f} {ci(bsd)}")
    for k in ("k0", "k2", "k5", "foreign"):
        pr = load(CT / f"{k}_{a}.jsonl"); fams = [f for f in F if any(fam_of(t) == f for t in pr)]
        x = [1 - np.mean([v for t, v in pr.items() if fam_of(t) == f]) for f in fams]
        y_old = [1 - np.mean(O[f]) for f in fams]; y_c = [1 - np.mean(Cb[f]) for f in fams]
        bs = []
        for _ in range(B):
            ii = rng.integers(0, len(fams), len(fams)); bs.append(spearman([x[i] for i in ii], [y_c[i] for i in ii]))
        res[f"rho_{k}"] = {"old": spearman(x, y_old), "extended": spearman(x, y_c), "extended_ci": ci(bs)}
        print(f"{a} rho {k}: old {spearman(x, y_old):.3f}, extended {spearman(x, y_c):.3f} {ci(bs)}")
    OUT[a] = res
json.dump(OUT, open(HERE / "results_e3b.json", "w"), indent=1, default=float)
