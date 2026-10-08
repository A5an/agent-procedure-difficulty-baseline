"""Round-2 evaluation (PREREG round 2): SP2, GP2, OP1 and ENS3 = mean z(FCg, SP2, GP2), on TAC and MCPMark (+drafter)."""
import json, hashlib, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
import fcg2
F = pd.read_csv("fresh_scores.csv")
def load_cache(p):
    c = {}
    try:
        for l in open(p): d = json.loads(l); c[d["k"]] = d["r"]
    except FileNotFoundError: pass
    return c
son2, opus = load_cache("sonnet_cache2.jsonl"), load_cache("opus_cache2.jsonl")
gp2 = json.load(open("gp2_raw.json"))
items = {f'{it["bench"]}::{it["id"]}': it for it in fcg2.G.items}
def fd(t): return ff.feats(t).get("fc_diff", np.nan) if t else np.nan
F["SP2"] = F.key.map(lambda k: fd(son2.get(hashlib.sha256(fcg2.p2(items[k]).encode()).hexdigest())))
F["GP2"] = F.key.map(lambda k: fd(gp2.get(k)))
F["OP1"] = F.key.map(lambda k: fd(opus.get(hashlib.sha256(fcg2.G.prompt(items[k]).encode()).hexdigest())))
z = lambda s: (s - s.mean()) / s.std()
for c in ["FCg", "SP2", "GP2"]: F["z_" + c] = F.groupby("bench")[c].transform(z)
F["ENS3"] = F[["z_FCg", "z_SP2", "z_GP2"]].mean(axis=1, skipna=False)
cols = ["FCg", "SP2", "GP2", "ENS3", "OP1", "HT", "length"]
rng = np.random.default_rng(8)
for b, g in F.groupby("bench"):
    print(f"== {b}: available " + ", ".join(f"{c} {g[c].notna().sum()}" for c in cols))
    for c in cols:
        h0 = g.dropna(subset=[c, "FCg"]).reset_index(drop=True)
        if len(h0) < 15: continue
        r = spearmanr(h0[c], h0["diff"]).correlation; rf = spearmanr(h0["FCg"], h0["diff"]).correlation
        D = []; DF = []
        for _ in range(2000):
            i = rng.integers(0, len(h0), len(h0)); h = h0.iloc[i]
            D.append(spearmanr(h[c], h["diff"]).correlation); DF.append(spearmanr(h["FCg"], h["diff"]).correlation)
        D, DF = np.array(D), np.array(DF)
        print(f"   {c:6s} n={len(h0):3d} rho {r:+.3f} [{np.nanpercentile(D,2.5):+.3f},{np.nanpercentile(D,97.5):+.3f}]  vs FCg on same cases {r-rf:+.3f} [{np.nanpercentile(D-DF,2.5):+.3f},{np.nanpercentile(D-DF,97.5):+.3f}]")
F.to_csv("fresh_scores2.csv", index=False)
