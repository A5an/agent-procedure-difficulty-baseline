"""PREREG_V2.md HV1, HV2, HV4: first round vs hard round. Reads episodes.csv and v2_episodes.csv (from analyze.py)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261109)
E1 = pd.read_csv(HERE / "episodes.csv"); E1["round"] = 1
E2 = pd.read_csv(HERE / "v2_episodes.csv"); E2["round"] = 2


def boot_rate(d, B=4000):
    pr = d.groupby("pid").fail.agg(["sum", "count"])
    k = len(pr)
    v = [pr.iloc[i].sum().pipe(lambda s: s["sum"] / s["count"]) for i in [rng.integers(0, k, k) for _ in range(B)]]
    return np.percentile(v, [2.5, 97.5])


def boot_diff(a, b, B=4000):
    pa, pb = a.groupby("pid").fail.agg(["sum", "count"]), b.groupby("pid").fail.agg(["sum", "count"])
    out = []
    for _ in range(B):
        x = pa.iloc[rng.integers(0, len(pa), len(pa))].sum(); y = pb.iloc[rng.integers(0, len(pb), len(pb))].sum()
        out.append(y["sum"] / y["count"] - x["sum"] / x["count"])
    return np.percentile(out, [2.5, 97.5])


print("HV1: failure rate, first round vs hard round")
for a in ("g38", "flash"):
    for g in ("haiku", "sonnet", "opus"):
        x, y = E1[(E1.agent == a) & (E1.gen == g)], E2[(E2.agent == a) & (E2.gen == g)]
        if x.empty or y.empty:
            continue
        lo, hi = boot_diff(x, y)
        print(f"  {a:5s} {g:6s} round 1 {x.fail.mean():.3f} (n {len(x)})  hard {y.fail.mean():.3f} (n {len(y)}, {y.pid.nunique()} procs)"
              f"  diff {y.fail.mean() - x.fail.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("HV2: hard round, failure by behaviour")
for a in ("g38", "flash"):
    d = E2[E2.agent == a]
    if d.empty:
        continue
    print(f"  {a}: " + ", ".join(f"{b} {d[d.behavior == b].fail.mean():.2f} (n {len(d[d.behavior == b])})" for b in sorted(d.behavior.unique())))
    hard, easy = d[d.behavior.isin(["negotiates", "changes_mind"])], d[d.behavior.isin(["cooperative", "terse"])]
    v = []
    pids = d.pid.unique()
    for _ in range(4000):
        pick = rng.choice(pids, len(pids))
        h = pd.concat([hard[hard.pid == p] for p in pick]); e = pd.concat([easy[easy.pid == p] for p in pick])
        v.append(h.fail.mean() - e.fail.mean())
    print(f"     negotiates+changes_mind minus cooperative+terse: {hard.fail.mean() - easy.fail.mean():+.3f} "
          f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
    print("     by level: " + ", ".join(f"{l} {d[d.level == l].fail.mean():.2f}" for l in sorted(d.level.unique())))

print("HV4: split-half reliability of procedure difficulty (odd vs even cases, Spearman-Brown), per agent")
for a in ("g38", "flash"):
    for name, E in (("round 1", E1), ("hard", E2)):
        for g in ("haiku", "sonnet", "opus", None):
            d = E[(E.agent == a) & ((E.gen == g) if g else True)].copy()
            if d.empty:
                continue
            d["k"] = d.case.str[1:].astype(int) % 2
            t = d.groupby(["proc", "k"]).fail.mean().unstack().dropna()
            if t[0].std() == 0 or t[1].std() == 0:
                print(f"  {a} {name} {g or 'pooled'}: no variance"); continue
            r = spearmanr(t[0], t[1]).correlation
            print(f"  {a} {name:7s} {g or 'pooled':6s}: {2 * r / (1 + r):+.2f} ({len(t)} procedures)")
