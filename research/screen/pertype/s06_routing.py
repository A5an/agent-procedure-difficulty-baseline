"""Step 4: offline routing on held-out tasks. For each held-out task, pick the agent with the highest predicted P.
Realised success = that agent's outcome on the task (SOPBench: 1 attempt; tau2: share of attempts). Bootstrap over procedures."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(ZR.REPO)
RUN = Path("run/sopbench")
rng = np.random.default_rng(11)
ty = pd.read_csv("types.csv").set_index("case_id")

def route(obs, tasks, methods, label):
    """obs: rows seed, fold, agent, case_id, y, <methods>. returns per-task vectors per strategy (averaged over seeds)."""
    dom = tasks.set_index("case_id").domain
    proc = tasks.set_index("case_id").procedure_id
    Rm = obs.groupby(["agent", "case_id"]).y.mean().unstack()           # agent x case mean outcome (all seeds same y)
    agents = list(Rm.index)
    allcases = list(Rm.columns)
    vec = {}
    for (seed, fold), g in obs.groupby(["seed", "fold"]):
        test = sorted(g.case_id.unique())
        train = [c for c in allcases if c not in set(test)]
        Rtr = Rm[train]
        best_train = Rtr.mean(1).idxmax()
        for t in test:
            col = Rm[t]
            res = {"oracle": col.max(), "random_agent": col.mean(), "single_best_train": col[best_train]}
            d = dom[t]
            trd = [c for c in train if dom[c] == d]
            res["best_per_domain_train"] = col[Rtr[trd].mean(1).idxmax()] if len(trd) else np.nan
            gt = g[g.case_id == t].set_index("agent")
            for m in methods:
                res[m] = col[gt[m].idxmax()]
            for k, v in res.items():
                vec.setdefault(k, {}).setdefault(t, []).append(v)
    ids = sorted(set(c for v in vec["oracle"] for c in [v]))
    ids = sorted(vec["oracle"])
    V = {k: np.array([np.nanmean(v[t]) if len(v.get(t, [])) else np.nan for t in ids]) for k, v in vec.items()}
    V["single_best_full"] = np.full(len(ids), Rm.mean(1).max()) * 0 + np.array([Rm.loc[Rm.mean(1).idxmax(), t] for t in ids])
    pr = np.array([proc[t] for t in ids])
    up = list(dict.fromkeys(pr)); ix = {p: np.flatnonzero(pr == p) for p in up}
    draws = [np.concatenate([ix[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
    base = "adele"
    out = {}
    for k, v in V.items():
        ok = ~np.isnan(v)
        bs = [np.nanmean(v[d]) for d in draws]
        r = {"mean": float(np.nanmean(v)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_tasks": int(ok.sum())}
        if k != base:
            ds = [np.nanmean(v[d] - V[base][d]) for d in draws]
            r["minus_adele"] = float(np.nanmean(v - V[base])); r["minus_adele_ci"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
        for ref in ["best_per_domain_train", "single_best_train", "single_best_full"]:
            if k != ref and not np.all(np.isnan(V[ref])):
                dd = [np.nanmean(v[d] - V[ref][d]) for d in draws]
                r["minus_" + ref] = float(np.nanmean(v - V[ref])); r["minus_" + ref + "_ci"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        out[k] = r
    # how often the per-type model picks someone other than the 1-D choice
    return out

res = {}
sm = ["adele", "share_fam", "share_tree", "share_dom", "share_famtree", "logit_fam", "logit_tree", "logit_dom", "logit_all", "logit_all_l30",
      "mirt_q3", "mirt_q3_l10", "main_all", "inter_all", "adele_onehot"]
tasks = pd.read_csv(REPO / "data/sopbench/tasks.csv")
for sc in ["new_procedures", "new_domain"]:
    obs = pd.read_csv(RUN / f"obs_{sc}.csv.gz")
    res[f"sopbench|{sc}"] = route(obs, tasks, sm, "sop")
t2 = pd.read_csv(REPO / "data/tau2/tasks.csv")
for sc in ["new_procedures", "new_domain"]:
    obs = pd.read_csv(REPO / f"results/tau2/obs_{sc}.csv.gz")
    res[f"tau2|{sc}"] = route(obs, t2, ["adele"], "tau2")
json.dump(res, open("routing.json", "w"), indent=1)
for k, r in res.items():
    print(k)
    for m, v in r.items():
        print(f"  {m:22s} {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]", ("diff vs adele %+.3f [%.3f,%.3f]" % (v['minus_adele'], *v['minus_adele_ci'])) if 'minus_adele' in v else "")
