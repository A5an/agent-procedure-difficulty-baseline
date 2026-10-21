import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import harness as H
from cpulock import cpu_lock
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import run_baseline as RB
import cm
from cm import *
tk = pd.read_csv(f"{REPO}/data/sopbench/tasks.csv")
rows = []
with cpu_lock("conditions_quality"):
    for scheme in ["new_procedures", "new_domain"]:
        seeds = [0] if scheme == "new_domain" else list(range(RB.N_SEEDS))
        for seed in seeds:
            for k, (tr, te) in enumerate(RB.splits(tk, scheme, seed)):
                m = CondModel(tr)
                o = OUT[OUT.case_id.isin(set(te))]
                # step type of condition (rule tree, NULL only)
                otr = OUT[OUT.case_id.isin(set(tr))]
                for name, sub, ncol in [("perf", o[o.should_succeed == 1], "perf"), ("fail", o[o.blocking == 1], "fail")]:
                    agg = sub.groupby(["unit", "idx"]).skip.agg(["mean", "size"]).reset_index()
                    agg["row"] = [ROW[(u, i)] for u, i in zip(agg.unit, agg.idx)]
                    agg["st"] = cd.loc[agg.row, "step_type"].values
                    trs = (otr[otr.should_succeed == 1] if name == "perf" else otr[otr.blocking == 1])
                    trs = trs.merge(cd[["unit", "idx", "step_type"]], on=["unit", "idx"])
                    tm = trs.groupby("step_type").skip.mean()
                    gm = trs.skip.mean()
                    for _, r in agg.iterrows():
                        rows.append(dict(scheme=scheme, seed=seed, fold=k, target=name, y=r["mean"], w=r["size"], p_text=m.pred[name][r.row],
                                         p_global=gm, p_type=tm.get(r.st, gm), step_type=r.st))
D = pd.DataFrame(rows)
D.to_csv("cond_quality_rows.csv", index=False)
res = {}
def wr2(d, col):
    sse = (d.w * (d.y - d[col]) ** 2).sum(); sst = (d.w * (d.y - d.p_global) ** 2).sum()
    return 1 - sse / sst
for (sc, tg), g in D.groupby(["scheme", "target"]):
    r = {}
    rhos = {c: np.mean([spearmanr(h[c], h.y)[0] if h[c].nunique() > 1 else 0 for _, h in g.groupby(["seed", "fold"])]) for c in ["p_text", "p_type"]}
    r["rho_text"] = rhos["p_text"]; r["rho_type_null"] = rhos["p_type"]
    r["wR2_vs_global_text"] = float(np.mean([wr2(h, "p_text") for _, h in g.groupby("seed")]))
    r["wR2_vs_global_type"] = float(np.mean([wr2(h, "p_type") for _, h in g.groupby("seed")]))
    r["n_cond_rows_per_seed"] = int(len(g) / g.seed.nunique())
    res[f"{sc}|{tg}"] = r
    print(sc, tg, {a: round(b, 3) for a, b in r.items()})
json.dump(res, open("cond_quality.json", "w"), indent=1)
