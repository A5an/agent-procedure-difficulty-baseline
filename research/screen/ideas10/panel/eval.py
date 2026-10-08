"""Panel evaluation (PREREG.md)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "..")
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
def score(t):
    d = ff.pj(t) if t else None
    try: s = min(max(float(d["p_success"]), 0.02), 0.98); return float(np.log((1 - s) / s))
    except Exception: return np.nan
S = {}
for b in ["tau2", "tauk_banking", "tac", "mcpmark", "drafter"]:
    for k, v in pd.read_csv(f"../fcd/d_{b}.csv").set_index("case_id").d_diff.items(): S.setdefault("sonnet", {})[k] = v
for m in ["haiku", "opus"]:
    try: S[m] = {k: score(v) for k, v in json.load(open(f"{m}_raw.json")).items()}
    except FileNotFoundError: pass
try:
    g = json.load(open("gemini_raw.json"))
    for m, d in g.items(): S[m] = {k: score(v) for k, v in d.items()}
except FileNotFoundError: pass
X = pd.DataFrame(S)
def bench(c): return c.split("::")[0] if "::" in c else ("tauk_banking" if c.startswith("banking") else "tau2")
X["bench"] = X.index.map(bench)
models = [c for c in X.columns if c != "bench"]
z = X.groupby("bench")[models].transform(lambda s: (s - s.mean()) / s.std())
X["PANEL"] = z.mean(axis=1)
X["PANEL_noopus"] = z[[m for m in models if m != "opus"]].mean(axis=1)
R = ZR.REPO + "/data"
tgt = {}
for b in ["tau2", "tauk_banking"]:
    t = pd.read_csv(f"{R}/{b}/tasks.csv"); t["b"] = t.case_id.map(pd.read_csv(f"{R}/{b}/irt/1d_1pl/items.csv", index_col=0).b)
    tgt.update({c: (d, v) for c, d, v in zip(t.case_id, t.domain, t.b)})
T = pd.read_csv("../fresh/targets.csv"); tgt.update({k: (b, 1 - s) for k, b, s in zip(T.key, T.bench, T.success)})
n = pd.read_csv("../newpop/newpop_tau2.csv").set_index("case_id")
X["dom"] = X.index.map(lambda k: tgt[k][0]); X["y"] = X.index.map(lambda k: tgt[k][1]); X["ynew"] = X.index.map(lambda k: 1 - n.newpop_success.get(k, np.nan))
def wd(g, c, y="y"):
    num = den = 0
    for d, h in g.groupby("dom"):
        h = h.dropna(subset=[c, y]); k = len(h)
        if k < 3 or h[c].nunique() < 2: continue
        w = k * (k - 1) / 2; num += spearmanr(h[c], h[y]).correlation * w; den += w
    return num / den if den else np.nan
cols = models + ["PANEL", "PANEL_noopus"]
rng = np.random.default_rng(5)
for b, g in X.groupby("bench"):
    g = g.reset_index(); D = {c: [] for c in cols}
    for _ in range(1000):
        h = g.iloc[rng.integers(0, len(g), len(g))]
        for c in cols: D[c].append(wd(h, c))
    line = f"{b:12s}"
    for c in cols:
        r = wd(g, c); line += f" | {c} {r:+.3f}"
        if c.startswith("PANEL"):
            dd = np.array(D[c]) - np.array(D["sonnet"]); line += f" (vs sonnet {r - wd(g,'sonnet'):+.3f} [{np.nanpercentile(dd,2.5):+.3f},{np.nanpercentile(dd,97.5):+.3f}])"
    print(line)
    if b == "tau2": print("   new population:", {c: round(wd(g, c, "ynew"), 3) for c in cols})
X.to_csv("panel_scores.csv")
