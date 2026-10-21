"""E3 within-agent held-out log-loss and AUC, steps_mirt vs baseline, paired bootstrap over procedures (seed 0, 300 draws)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
RUN = ZR.SCREEN + "/e3/run"
TASKS = ZR.REPO + "/data/{}/tasks.csv"
COLS = ["adele", "steps_mirt", "steps_mirt_l10", "adele_steps"]
rng = np.random.default_rng(7)

def stats(m):
    out = {}
    y = m.y.values
    for c in COLS:
        p = np.clip(m[c].values, 1e-4, 1 - 1e-4)
        out[c + "|ll"] = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        aucs, ws = [], []
        for _, g in m.groupby(["fold", "agent"]):
            yy = g.y.values; n1 = yy.sum(); n0 = len(yy) - n1
            if n1 and n0:
                r = rankdata(g[c].values); aucs.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)); ws.append(len(yy))
        out[c + "|aucw"] = float(np.average(aucs, weights=ws))
    return out

res = {}
for b in ("sopbench", "tau2"):
    proc = pd.read_csv(TASKS.format(b)).set_index("case_id").procedure_id
    for s in ("new_procedures", "new_domain"):
        m = pd.read_csv(f"{RUN}/{b}/obs_{s}.csv.gz"); m = m[m.seed == 0].reset_index(drop=True)
        m["proc"] = proc.loc[m.case_id].values
        pt = stats(m)
        idx = {p: g.index.values for p, g in m.groupby("proc")}; procs = list(idx)
        bs = [stats(m.iloc[np.concatenate([idx[p] for p in rng.choice(procs, len(procs))])]) for _ in range(300)]
        r = {"point": pt}
        for c in COLS[1:]:
            for k in ("ll", "aucw"):
                d = [x[f"{c}|{k}"] - x[f"adele|{k}"] for x in bs]
                r[f"{c}-adele|{k}"] = {"diff": pt[f"{c}|{k}"] - pt[f"adele|{k}"], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        res[f"{b}|{s}"] = r
        print(b, s, {k: round(v, 4) for k, v in pt.items()})
        for k, v in r.items():
            if k != "point": print("   ", k, round(v["diff"], 4), [round(x, 4) for x in v["ci"]])
json.dump(res, open(RUN + "/within.json", "w"), indent=1)
