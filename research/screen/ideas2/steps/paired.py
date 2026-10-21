"""Paired bootstrap (procedures resampled) of differences in the protocol rho between any two columns, SOPBench scenarios,
same resampling scheme as the repository evaluator (seed 0 predictions, one generator per benchmark)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
import harness as H
E_ = H.E
import metrics_extra as X
OUT = ZR.SCREEN + "/ideas2/steps/out"
tasks = pd.read_csv(E_.DATA / "sopbench" / "tasks.csv")
tasks["b"] = tasks.case_id.map(pd.read_csv(E_.DATA / "sopbench" / "irt/1d_1pl/items.csv", index_col=0).b)
SCEN = ["new_procedures", "new_domain"]
COLS = ["adele", "cnt_geo", "cnt_logit", "cnt_logit_a", "typ_logit", "typ_logit_a", "typ_prod", "cnt_prod",
        "typ_logit+adele", "cnt_logit+adele", "typ_logit_a+adele"]
PAIRS = [("typ_logit", "cnt_logit"), ("typ_logit_a", "cnt_logit_a"), ("typ_logit", "cnt_geo"), ("typ_prod", "cnt_prod"),
         ("typ_logit", "adele"), ("typ_logit_a", "adele"), ("typ_logit+adele", "adele"), ("typ_logit+adele", "cnt_logit+adele"),
         ("cnt_logit", "adele"), ("typ_logit_a", "typ_logit")]


def prep(scheme):
    m = E_.load.__wrapped__(None, None) if False else None
    m = pd.read_csv(f"{OUT}/sopbench/obs_{scheme}.csv.gz")
    c = lambda x: np.clip(x, 1e-4, 1 - 1e-4)
    from scipy.special import expit, logit
    for combo in [("typ_logit", "adele"), ("cnt_logit", "adele"), ("typ_logit_a", "adele")]:
        m["+".join(combo)] = expit(np.mean([logit(c(m[k])) for k in combo], axis=0))
    diffs = {c_: E_.difficulty(m, c_) for c_ in COLS}
    per_proc = {}
    for c_ in COLS:
        x = diffs[c_][diffs[c_].seed == 0].merge(tasks[["case_id", "domain", "procedure_id", "b"]], on="case_id")
        x["cell"] = x.fold.astype(str) + "|" + x.domain
        per_proc[c_] = {p: (g.cell.values, g.l.values, g.b.values) for p, g in x.groupby("procedure_id")}
    pt = {c_: np.nanmean([E_.rank_cells(g, tasks)[0] for _, g in diffs[c_].groupby("seed")]) for c_ in COLS}
    return per_proc, pt


res = {}
data = {s: prep(s) for s in SCEN}
rng = np.random.default_rng([20260928, 0])
procs = list(data["new_procedures"][0]["adele"])
assert procs == list(data["new_domain"][0]["adele"])
draws = {s: {c: [] for c in COLS} for s in SCEN}
for _ in range(1000):
    pick = rng.choice(procs, len(procs))
    for s in SCEN:
        pp = data[s][0]
        for c in COLS:
            cell = np.concatenate([pp[c][p][0] for p in pick]); l = np.concatenate([pp[c][p][1] for p in pick]); b = np.concatenate([pp[c][p][2] for p in pick])
            draws[s][c].append(E_.rank_arrays(cell, l, b))
for s in SCEN:
    for c in COLS:
        draws[s][c] = np.array(draws[s][c])
    res[s] = {"rho": data[s][1]}
out = {"point": {s: data[s][1] for s in SCEN}, "pairs": {}}
for a, b in PAIRS:
    for key, fn in [(s, lambda s_: (data[s_][1][a] - data[s_][1][b], draws[s_][a] - draws[s_][b])) for s in SCEN] + \
                   [("mean_of_two", None)]:
        if key == "mean_of_two":
            pt = np.mean([data[s][1][a] - data[s][1][b] for s in SCEN]); d = np.mean([draws[s][a] - draws[s][b] for s in SCEN], axis=0)
        else:
            pt, d = fn(key)
        out["pairs"][f"{a} - {b} | {key}"] = [float(pt), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
ci = {}
for c in COLS:
    pts = np.mean([data[s][1][c] for s in SCEN]); d = np.mean([draws[s][c] for s in SCEN], axis=0)
    ci[c] = [float(pts), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
out["mean_of_two"] = ci
json.dump(out, open("paired.json", "w"), indent=1)
for k, v in out["mean_of_two"].items():
    print(f"{k:22s} mean rho {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]  np {out['point']['new_procedures'][k]:+.3f} nd {out['point']['new_domain'][k]:+.3f}")
for k, v in out["pairs"].items():
    print(f"{k:55s} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")
