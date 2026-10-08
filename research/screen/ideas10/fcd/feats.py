"""Direct-rating feature d_diff per benchmark from the ablation cache. -> d_<bench>.csv"""
import json, hashlib, sys, numpy as np, pandas as pd
sys.path.insert(0, "../ablate"); sys.path.insert(0, "..")
import ablate as A
import importlib.util
spec = importlib.util.spec_from_file_location("ff", "../fc/feats.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
cache = {}
for l in open("../ablate/sonnet_cache.jsonl"): d = json.loads(l); cache[d["k"]] = d["r"]
def score(t):
    d = ff.pj(t) if t else None
    try: s = min(max(float(d["p_success"]), 0.02), 0.98); return float(np.log((1 - s) / s))
    except Exception: return np.nan
rows = [{"case_id": k, "d_diff": score(cache.get(hashlib.sha256(p.encode()).hexdigest()))} for (kind, k), p in A.jobs.items() if kind == "D"]
D = pd.DataFrame(rows).set_index("case_id")
def bench(c): return c.split("::")[0] if "::" in c else ("sopbench" if c.count("/") == 2 else ("tauk_banking" if c.startswith("banking") else "tau2"))
for b, g in D.groupby(D.index.map(bench)):
    print(b, len(g), "missing", int(g.d_diff.isna().sum())); g.fillna(g.median()).to_csv(f"d_{b}.csv")
