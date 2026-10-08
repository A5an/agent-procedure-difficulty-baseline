"""Fresh-benchmark evaluation (PREREG.md). Reads Sonnet cache directly so it can run before fcg_raw.json exists."""
import json, hashlib, re, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "../fc"); sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
src = open("fcg.py").read(); ns = {"__file__": "fcg.py"}; exec(src.split("items = json.load")[0].replace("import vx, fc as FC1", ""), ns)
items = json.load(open("items.json")); P, ENV = ns["P"], ns["ENV"]
cache = {}
for l in open("sonnet_cache.jsonl"): d = json.loads(l); cache[d["k"]] = d["r"]
gem = json.load(open("gem_raw.json"))
def minutes(t):
    for s in reversed(re.findall(r"\{[^{}]*\}", t or "")):
        try: v = float(json.loads(s)["minutes"]); return v if v > 0 else None
        except Exception: pass
rows = []
for it in items:
    k = f'{it["bench"]}::{it["id"]}'
    s = cache.get(hashlib.sha256(P.format(env=ENV[it["bench"]], text=it["text"]).encode()).hexdigest())
    m = minutes(gem[k]["HT"])
    rows.append({"key": k, "bench": it["bench"], "group": it["group"], "FCg": ff.feats(s).get("fc_diff", np.nan) if s else np.nan,
                 "GG": ff.feats(gem[k]["GG"]).get("fc_diff", np.nan), "HT": np.log(m) if m else np.nan, "length": len(it["text"])})
F = pd.DataFrame(rows)
T = pd.read_csv("targets.csv").drop(columns=["bench"]); F = F.merge(T, on="key")
F["diff"] = 1 - F.success
F["FCg_GG"] = F.groupby("bench")[["FCg", "GG"]].transform(lambda s: (s - s.mean()) / s.std()).mean(axis=1, skipna=False)
cols = ["FCg", "GG", "FCg_GG", "HT", "length"]
rng = np.random.default_rng(7)
for b, g in F.groupby("bench"):
    g = g.dropna(subset=["FCg"]).reset_index(drop=True)
    if len(g) < 10: print(b, "not enough", len(g)); continue
    D = {c: [] for c in cols}
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g)); h = g.iloc[i]
        for c in cols: D[c].append(spearmanr(h[c], h["diff"], nan_policy="omit").correlation)
    print(f"== {b}: n={len(g)}, agents median {g.n_agents.median():.0f}, mean success {g.success.mean():.2f}")
    for c in cols:
        r = spearmanr(g[c], g["diff"], nan_policy="omit").correlation; d = np.array(D[c])
        s = f"   {c:7s} rho {r:+.3f} [{np.nanpercentile(d,2.5):+.3f},{np.nanpercentile(d,97.5):+.3f}]"
        for base in ["length", "HT"]:
            if base != c: dd = d - np.array(D[base]); s += f"  vs {base} {r - spearmanr(g[base], g['diff'], nan_policy='omit').correlation:+.3f} [{np.nanpercentile(dd,2.5):+.3f},{np.nanpercentile(dd,97.5):+.3f}]"
        print(s)
    if b in ("tac", "drafter", "mcpmark"):
        num = den = 0
        for gr, h in g.groupby("group"):
            if len(h) >= 5 and h.FCg.nunique() > 1: w = len(h) * (len(h) - 1) / 2; num += spearmanr(h.FCg, h["diff"]).correlation * w; den += w
        print(f"   FCg within {b} groups (pooled by pairs): {num/den:+.3f}" if den else "")
F.to_csv("fresh_scores.csv", index=False)
