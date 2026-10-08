"""Ablation evaluation (PREREG.md): direct rating D vs pre-mortem P1, test-retest P1 vs R, mean(P1, R)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, hashlib, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, ".."); sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
import ablate as A
cache = {}
for l in open("sonnet_cache.jsonl"): d = json.loads(l); cache[d["k"]] = d["r"]
def score(t):
    if not t: return np.nan
    d = ff.pj(t)
    try: s = min(max(float(d["p_success"]), 0.02), 0.98); return float(np.log((1 - s) / s))
    except Exception: return np.nan
P1 = {}
for b in ["tau2", "tauk_banking"]:
    P1.update(pd.read_csv(f"../fc/fc_{b}.csv").set_index("case_id").fc_diff.to_dict())
P1.update(pd.read_csv("../fresh/fresh_scores2.csv").set_index("key").FCg.to_dict())
rows = []
for (kind, key), p in A.jobs.items():
    rows.append({"key": key, "kind": kind, "v": score(cache.get(hashlib.sha256(p.encode()).hexdigest()))})
X = pd.DataFrame(rows).pivot(index="key", columns="kind", values="v"); X["P1"] = X.index.map(P1)
X["P1R"] = (X.P1.rank(pct=True) + X.R.rank(pct=True))
def bench(k): return k.split("::")[0] if "::" in k else ("tauk_banking" if k.startswith("banking") else "tau2")
X["bench"] = X.index.map(bench)
R = ZR.REPO + "/data"
tgt = {}
for b in ["tau2", "tauk_banking"]:
    t = pd.read_csv(f"{R}/{b}/tasks.csv"); t["b"] = t.case_id.map(pd.read_csv(f"{R}/{b}/irt/1d_1pl/items.csv", index_col=0).b)
    tgt.update({c: (d, v) for c, d, v in zip(t.case_id, t.domain, t.b)})
T = pd.read_csv("../fresh/targets.csv")
tgt.update({k: (b, 1 - s) for k, b, s in zip(T.key, T.bench, T.success)})
X["dom"] = X.index.map(lambda k: tgt[k][0]); X["y"] = X.index.map(lambda k: tgt[k][1])
def wd(g, c):
    num = den = 0
    for d, h in g.groupby("dom"):
        h = h.dropna(subset=[c, "y"]); n = len(h)
        if n < 3 or h[c].nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(h[c], h.y).correlation * w; den += w
    return num / den
rng = np.random.default_rng(3)
for b, g in X.groupby("bench"):
    g = g.reset_index()
    D = {c: [] for c in ["P1", "D", "R", "P1R"]}
    for _ in range(1000):
        h = g.iloc[rng.integers(0, len(g), len(g))]
        for c in D: D[c].append(wd(h, c))
    s = f"{b:12s} n={len(g)} retest rho(P1,R) {spearmanr(g.P1, g.R, nan_policy='omit').correlation:.2f}"
    for c in D:
        d = np.array(D[c]); s += f" | {c} {wd(g, c):+.3f}"
        if c != "P1": dd = d - np.array(D["P1"]); s += f" ({wd(g,c)-wd(g,'P1'):+.3f} [{np.nanpercentile(dd,2.5):+.3f},{np.nanpercentile(dd,97.5):+.3f}])"
    print(s)
X.to_csv("ablate_scores.csv")
