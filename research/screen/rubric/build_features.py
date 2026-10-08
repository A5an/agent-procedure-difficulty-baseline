"""Cache -> feature CSVs (case_id first), one per feature set and benchmark."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
import prompts as P
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
BENCHES = ["sopbench", "tau2", "tauk_banking"]

def load(rubric, rep=1):
    out = {}
    for l in open(HERE / "cache" / f"{rubric}.jsonl"):
        r = json.loads(l)
        if r.get("ok"):
            rb, rp, cid = r["key"].split("|", 2)
            if int(rp) == rep:
                out[cid] = r["parsed"]
    return out

def proc_row(d):
    steps = d["steps"]
    row = {"n_steps": len(steps)}
    for t in P.STEP_TYPES:
        row[f"n_{t}"] = sum(s["type"] == t for s in steps)
    row["max_depth"] = max(s["depth"] for s in steps)
    row["refusal_conditions"] = d["refusal_conditions"]
    row["needs_hidden_data"] = d["needs_hidden_data"]
    row["p_refusal"] = float(d["p_refusal"])
    return row

def lara_row(d):
    w = np.array([1, 1, 1, 1.5, 1.0]); D = np.array([d[f"D{i}"] for i in range(1, 6)], float)
    m = float((D * w).sum() / w.sum())
    lvl = 1 if m <= 2.0 else 2 if m <= 3.0 else 3 if m <= 4.0 else 4
    if d["D4"] >= 4:
        lvl = max(lvl, 3)       # D4 floor rule of the paper
    return {**{f"D{i}": d[f"D{i}"] for i in range(1, 6)}, "lara_mean": m, "lara_level": lvl}

def rpa_row(d):
    row = {n: d[n] for n, _, _ in P.RPA_CRITERIA}
    row["rpa_total"] = sum(row.values())
    return row

def tables(rep=1):
    """dict rubric -> DataFrame indexed by case_id (all benchmarks), plus missing counts."""
    T = {}
    for rb, fn in (("proc", proc_row), ("lara", lara_row), ("rpa", rpa_row)):
        T[rb] = pd.DataFrame.from_dict({k: fn(v) for k, v in load(rb, rep).items()}, orient="index")
    return T

if __name__ == "__main__":
    T = tables()
    (HERE / "features").mkdir(exist_ok=True)
    rep = []
    for b in BENCHES:
        ids = pd.read_csv(REPO / "data" / b / "tasks.csv").case_id.tolist()
        for rb, name, cols in (("proc", "rub_proc", None),
                               ("proc", "rub_len", ["n_steps"]),
                               ("proc", "rub_types", [f"n_{t}" for t in P.STEP_TYPES]),
                               ("lara", "rub_lara", None), ("rpa", "rub_rpa", None)):
            df = T[rb].reindex(ids)
            miss = int(df.iloc[:, 0].isna().sum())
            df = df.fillna(df.median())          # unsupervised median imputation, no outcomes used
            if cols:
                df = df[cols]
            df.index.name = "case_id"
            df.reset_index().to_csv(HERE / "features" / f"{name}_{b}.csv", index=False)
            rep.append((b, name, miss, df.shape[1]))
    for r in rep: print(r)
