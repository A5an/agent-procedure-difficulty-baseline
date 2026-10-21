"""Frozen analysis of the ThinkingBox-Bench blind test (written before any agent run).

Inputs: outcomes.csv (task_id, agent, trial, success 0/1, valid 0/1; written by the run collector), eval_set.json
({"n_per_domain": N}, fixed after the pilot from cost alone), order.csv, predictions_D.csv, predictions_baseline.csv.

Target: difficulty of a task = 1 - mean success over its valid runs (all agents and trials pooled). A task enters the
evaluation if its rank in order.csv is below N in its domain and at least 6 of its 8 runs are valid.
Primary metric: Spearman rho inside each of the five domains, pooled with weights = number of task pairs (as in
src/evaluate.py). Uncertainty: 2,000 bootstrap draws that resample task clusters inside each domain (retail: the
STnnn template, whose variants share one scenario; other domains: single tasks), paired across predictors.

  python evaluate.py   -> results.json, results.txt"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from tbpaths import HERE, ORDER

SEED = 20261015
N_BOOT = 2000
SCORES = ["score_D", "score_baseline", "score_length"]
PAIRS = [("score_D", "score_baseline"), ("score_D", "score_length"), ("score_baseline", "score_length")]


def cluster(task_id):
    name = task_id.split(":")[1]
    if name.startswith("test_case_ST"):
        return "retail:" + name.split("_")[2]
    return task_id


def pooled_rho(dom, s, y):
    num = den = 0.0
    for d in np.unique(dom):
        m = dom == d
        n = int(m.sum())
        if n < 3:
            continue
        pairs = n * (n - 1) / 2
        rs, ry = rankdata(s[m]), rankdata(y[m])
        if rs.std() > 0 and ry.std() > 0:
            num += np.corrcoef(rs, ry)[0, 1] * pairs
        den += pairs
    return num / den if den else np.nan


def pooled_pacc(dom, s, y):
    num = den = 0.0
    for d in np.unique(dom):
        m = dom == d
        p, t = s[m], y[m]
        i, j = np.triu_indices(len(t), 1)
        keep = t[i] != t[j]
        sgn = np.sign(p[i] - p[j])[keep] * np.sign(t[i] - t[j])[keep]
        num += np.where(sgn == 0, 0.5, sgn > 0).sum()
        den += keep.sum()
    return num / den if den else np.nan


def table():
    out = pd.read_csv(HERE / "outcomes.csv")
    n_dom = json.loads((HERE / "eval_set.json").read_text())["n_per_domain"]
    order = pd.read_csv(ORDER)
    keep = set(order[order["rank"] < n_dom].task_id)
    v = out[(out.valid == 1) & out.task_id.isin(keep)]
    g = v.groupby("task_id").agg(runs=("success", "size"), success=("success", "mean")).reset_index()
    g = g[g.runs >= 6]
    g["difficulty"] = 1 - g.success
    t = (g.merge(order[["task_id", "domain"]], on="task_id")
          .merge(pd.read_csv(HERE / "predictions_D.csv")[["task_id", "p_success", "score_D"]], on="task_id")
          .merge(pd.read_csv(HERE / "predictions_baseline.csv")[["task_id", "score_baseline", "score_length"]],
                 on="task_id"))
    t["cluster"] = t.task_id.map(cluster)
    per_agent = v.groupby(["task_id", "agent"]).success.mean().unstack()
    return t, per_agent, len(keep), out


def main():
    t, per_agent, n_sel, out = table()
    dom, y = t.domain.values, t.difficulty.values
    S = {k: t[k].values for k in SCORES}
    res = {"n_selected": n_sel, "n_evaluated": len(t), "runs_valid": int(out.valid.sum()), "runs_total": len(out),
           "mean_success": float(t.success.mean()),
           "per_agent_success": {a: float(per_agent[a].mean()) for a in per_agent.columns}}
    point = {k: pooled_rho(dom, S[k], y) for k in SCORES}
    res["rho"] = point
    res["pacc"] = {k: pooled_pacc(dom, S[k], y) for k in SCORES}
    res["rho_by_domain"] = {d: {k: float(spearmanr(t[t.domain == d][k], t[t.domain == d].difficulty).correlation)
                                for k in SCORES} for d in sorted(set(dom))}
    # bootstrap: resample clusters inside each domain, the same draw for every predictor
    rng = np.random.default_rng(SEED)
    groups = {d: [np.flatnonzero((t.domain == d).values & (t.cluster == c).values)
                  for c in sorted(t[t.domain == d].cluster.unique())] for d in sorted(set(dom))}
    draws = {k: [] for k in SCORES}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([gs[i] for i in rng.integers(0, len(gs), len(gs))])
                              for gs in groups.values()])
        for k in SCORES:
            draws[k].append(pooled_rho(dom[idx], S[k][idx], y[idx]))
    draws = {k: np.array(v) for k, v in draws.items()}
    res["rho_ci"] = {k: [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]
                     for k in SCORES}
    res["diff"] = {}
    for a, b in PAIRS:
        d = draws[a] - draws[b]
        res["diff"][f"{a} - {b}"] = {"point": point[a] - point[b], "ci": [float(np.nanpercentile(d, 2.5)),
                                                                          float(np.nanpercentile(d, 97.5))]}
    # secondary: rho against each agent's own success, and calibration of p_success against the observed share
    res["rho_vs_each_agent"] = {a: {k: pooled_rho(dom, S[k], 1 - per_agent.reindex(t.task_id)[a].values)
                                    for k in SCORES} for a in per_agent.columns}
    p = t.p_success.clip(0.02, 0.98).values
    res["brier_p_success"] = float(np.mean((p - t.success.values) ** 2))
    res["brier_constant"] = float(np.mean((t.success.mean() - t.success.values) ** 2))
    pd_ = res["diff"]["score_D - score_baseline"]
    res["verdict_point5"] = bool(pd_["point"] > 0 and pd_["ci"][0] > 0)
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    lines = [f"tasks evaluated {len(t)} of {n_sel} selected; valid runs {res['runs_valid']}/{res['runs_total']}; "
             f"mean success {res['mean_success']:.3f}"]
    for k in SCORES:
        lines.append(f"{k:16s} rho {point[k]:.3f} [{res['rho_ci'][k][0]:.3f}, {res['rho_ci'][k][1]:.3f}]  "
                     f"pacc {res['pacc'][k]:.3f}")
    for k, d in res["diff"].items():
        lines.append(f"{k:34s} {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    lines.append(f"point 5 of the improvement rule met: {res['verdict_point5']}")
    (HERE / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
