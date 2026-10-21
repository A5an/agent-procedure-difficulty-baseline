"""Cache -> features/fmea_<bench>.csv and test-retest."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import prompts as P
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)

def load(variant):
    out = {}
    for l in open(HERE / "cache" / "fmea.jsonl"):
        r = json.loads(l)
        if r.get("ok"):
            v, cid = r["key"].split("|", 1)
            if v == variant: out[cid] = r["parsed"]
    return out

def row(d):
    m = d["modes"]; occ = np.array([x["occurrence"] for x in m], float)
    rpn = np.array([x["occurrence"] * x["severity"] * x["detection"] for x in m], float)
    r = {"n_modes": len(m), "sum_rpn": rpn.sum(), "max_rpn": rpn.max() if len(m) else 0,
         "mean_occurrence": occ.mean() if len(m) else 0, "max_occurrence": occ.max() if len(m) else 0}
    for c in P.CATS: r["cnt_" + c] = sum(x["category"] == c for x in m)
    for c in ("skips_required_check", "acts_when_should_refuse"):
        r["wocc_" + c] = float(sum(x["occurrence"] for x in m if x["category"] == c))
    # per-category max occurrence (diagnostics only, not a feature)
    return r

def occ_cat(d, c):
    v = [x["occurrence"] for x in d["modes"] if x["category"] == c]
    return max(v) if v else 0

if __name__ == "__main__":
    A = load("A")
    for b in ["sopbench", "tau2"]:
        ids = pd.read_csv(REPO / "data" / b / "tasks.csv").case_id.tolist()
        df = pd.DataFrame.from_dict({k: row(A[k]) for k in ids if k in A}, orient="index").reindex(ids)
        print(b, "missing", int(df.n_modes.isna().sum()))
        df = df.fillna(df.median()); df.index.name = "case_id"
        df.reset_index().to_csv(HERE / "features" / f"fmea_{b}.csv", index=False)
    B = load("B"); common = [k for k in B if k in A]
    ra = pd.DataFrame([row(A[k]) for k in common]); rb = pd.DataFrame([row(B[k]) for k in common])
    res = {"n": len(common)}
    for c in ["sum_rpn", "max_occurrence", "n_modes", "max_rpn", "mean_occurrence"]:
        res[c] = float(spearmanr(ra[c], rb[c])[0])
    print(res); json.dump(res, open(HERE / "retest.json", "w"), indent=1)
