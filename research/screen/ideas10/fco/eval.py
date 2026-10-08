import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, hashlib, sys, numpy as np, pandas as pd
sys.path.insert(0, "..")
sys.path.insert(0, "../fc"); sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
import fc as FC1
from gap import inputs
from rawci import report
from scipy.stats import spearmanr
cache = {}
for l in open("opus_cache.jsonl"): d = json.loads(l); cache[d["k"]] = d["r"]
rows = []
for it in inputs():
    if it[0] == "sopbench": continue
    t = cache.get(hashlib.sha256(FC1.prompt_of(it).encode()).hexdigest())
    rows.append({"case_id": it[1], "opus": ff.feats(t).get("fc_diff", np.nan) if t else np.nan})
O = pd.DataFrame(rows).set_index("case_id")
S1 = pd.concat([pd.read_csv("../fc/fc_tau2.csv").set_index("case_id").fc_diff, pd.read_csv("../fc/fc_tauk_banking.csv").set_index("case_id").fc_diff])
O["S1"] = S1; O["avg"] = (O.opus.rank() / O.opus.notna().sum() + O.S1.rank() / len(O))  # rank average, per benchmark below
O.to_csv("opus_scores.csv")
for b, pre in [("tau2", ("airline", "retail", "telecom")), ("tauk_banking", ("banking",))]:
    sub = O[O.index.map(lambda c: c.split("/")[0].startswith(pre))].copy()
    sub["avg"] = sub.opus.rank(pct=True) + sub.S1.rank(pct=True)
    sub.to_csv(f"tmp_{b}.csv"); print(b, "opus available", sub.opus.notna().sum(), "of", len(sub))
    report(b, {c: (f"tmp_{b}.csv", c) for c in ["opus", "S1", "avg"]}, ["opus", "S1", "avg", "length", "human_time"], nboot=1000)
# new population
n = pd.read_csv("../newpop/newpop_tau2.csv").set_index("case_id")
t = pd.read_csv(ZR.REPO + "/data/tau2/tasks.csv")
t["y"] = 1 - t.case_id.map(n.newpop_success); t["opus"] = t.case_id.map(O.opus); t["S1"] = t.case_id.map(O.S1)
for c in ["opus", "S1"]:
    num = den = 0
    for d, g in t.groupby("domain"):
        g = g.dropna(subset=[c]); w = len(g) * (len(g) - 1) / 2; num += spearmanr(g[c], g.y).correlation * w; den += w
    print("new population", c, round(num / den, 3))
