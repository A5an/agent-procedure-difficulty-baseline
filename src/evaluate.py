"""Final evaluation of the baseline and its references under the protocol in EVALUATION_PROTOCOL.md.

  python src/evaluate.py        -> results/final_evaluation.json and results/final_evaluation.txt

Reads the out-of-fold predictions of run_baseline.py (results/<bench>/obs_<scheme>.csv.gz): every
method was fitted on the same folds, so all comparisons are paired. For each benchmark and split:

  primary     task difficulty ranking inside a domain: Spearman rho and pairwise accuracy, computed
              inside each (seed, fold, domain) cell so that predictions from different fold models are
              never compared with each other, then pooled over cells (weights = number of task pairs)
  secondary   AUC within one agent configuration (per seed, fold, agent; the constant scores 0.5);
              Brier and log-loss skill against the constant (1 - loss / loss of constant);
              pooled AUC gain over the constant, only for comparison with Agent psychometrics
  diagnostic  SOPBench procedure level: pairwise accuracy of mean difficulty between procedures
              of the same domain (new-domain split, where each domain is one fold)

Uncertainty: points are means over seeds; 95% intervals from 1,000 bootstrap draws over procedures
(task groups) on seed 0; paired intervals for the difference to the baseline. The summary is the
mean primary score over the four new-process scenarios (SOPBench and tau2, new procedures and new
domain), with a joint bootstrap.
"""

import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr

from common import BENCHES, DATA, RESULTS

KEY = ["seed", "fold", "agent", "case_id"]
# logit averages with equal weights, fixed before looking at test data
COMBOS = [("emb_ap", "adele"), ("knn_router", "amortized_mirt", "emb_ap")]
BASELINE = "adele"
METHODS = {
    "constant": "agent ability only (reference)",
    "length": "text length (control)",
    "emb_ap": "Agent psychometrics, text embedding",
    "adele": "BASELINE: Agent psychometrics + ADeLe demands",
    "adele16": "baseline without the 2 demands that did not replicate",
    "ap_combined": "their grouped ridge: embedding + ADeLe demands",
    "emb_ap+adele": "equal-weight logit average of embedding and ADeLe models",
    "lltm_adele": "same, fitted as LLTM (joint)",
    "knn_router": "kNN router (past success on similar tasks)",
    "amortized_mirt": "IRT-Router-style multidimensional IRT",
    "knn_router+amortized_mirt+emb_ap": "routers + psychometrics (best on AUC)",
    "human_time": "estimated human time (METR idea)",
    "oracle": "oracle (fitted difficulty)",
}
N_BOOT = 1000
DATASETS = BENCHES
# the scenarios the research question is about: procedures or domains never seen in training
NEW = ["sopbench|new_procedures", "sopbench|new_domain", "tau2|new_procedures", "tau2|new_domain"]


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(d, s):
    m = pd.read_csv(RESULTS / d / f"obs_{s}.csv.gz")
    c = lambda x: np.clip(x, 1e-4, 1 - 1e-4)  # noqa: E731
    for combo in COMBOS:
        m["+".join(combo)] = expit(np.mean([logit(c(m[k])) for k in combo], axis=0))
    return m


def difficulty(m, col):
    """Per (seed, fold, case): predicted difficulty = minus the mean logit of P(success) over agents."""
    u = m.drop_duplicates(KEY)
    return (u.assign(l=-logit(np.clip(u[col], 1e-4, 1 - 1e-4))).groupby(["seed", "fold", "case_id"]).l.mean()
            .reset_index())


def rank_cells(bh, tasks):
    """Spearman and pairwise accuracy inside each (seed, fold, domain) cell, pooled by pairs."""
    x = bh.merge(tasks[["case_id", "domain", "procedure_id", "b"]], on="case_id")
    rho_num = rho_den = acc_num = acc_den = 0.0
    for _, g in x.groupby(["fold", "domain"]):
        n = len(g)
        if n < 3:
            continue
        pairs = n * (n - 1) / 2
        if g.l.nunique() > 1 and g.b.nunique() > 1:
            rho_num += spearmanr(g.l, g.b).correlation * pairs
        rho_den += pairs
        p, t = g.l.values, g.b.values
        i, j = np.triu_indices(n, 1)
        keep = t[i] != t[j]
        sgn = np.sign(p[i] - p[j])[keep] * np.sign(t[i] - t[j])[keep]
        acc_num += np.where(sgn == 0, 0.5, sgn > 0).sum()
        acc_den += keep.sum()
    return rho_num / rho_den if rho_den else np.nan, acc_num / acc_den if acc_den else np.nan


def rank_arrays(cell, l, b):
    """Spearman inside each cell, pooled by number of pairs (array version for the bootstrap)."""
    num = den = 0.0
    order = np.argsort(cell, kind="stable")
    cell, l, b = cell[order], l[order], b[order]
    bounds = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1], True])
    for a, z in zip(bounds[:-1], bounds[1:]):
        n = z - a
        if n < 3:
            continue
        pairs = n * (n - 1) / 2
        rl, rb = rankdata(l[a:z]), rankdata(b[a:z])
        if rl.std() > 0 and rb.std() > 0:
            num += np.corrcoef(rl, rb)[0, 1] * pairs
        den += pairs
    return num / den if den else np.nan


def proc_pairs(bh, tasks):
    x = bh.merge(tasks[["case_id", "domain", "procedure_id", "b"]], on="case_id")
    pm = x.groupby(["fold", "domain", "procedure_id"])[["l", "b"]].mean().reset_index()
    num = den = 0.0
    for _, g in pm.groupby(["fold", "domain"]):
        if len(g) < 2:
            continue
        p, t = g.l.values, g.b.values
        i, j = np.triu_indices(len(g), 1)
        keep = t[i] != t[j]
        sgn = np.sign(p[i] - p[j])[keep] * np.sign(t[i] - t[j])[keep]
        num += np.where(sgn == 0, 0.5, sgn > 0).sum(); den += keep.sum()
    return num / den if den else np.nan


def scores(m, col):
    """Per-fold AUC gain, within-configuration AUC, Brier and log-loss skill (mean over folds)."""
    gains, briers, logs = [], [], []
    for _, f in m.groupby(["seed", "fold"]):
        if f.y.nunique() < 2:
            continue
        gains.append(auc(f.y.values, f[col].values) - auc(f.y.values, f.constant.values))
        p, pc = np.clip(f[col].values, 1e-4, 1 - 1e-4), np.clip(f.constant.values, 1e-4, 1 - 1e-4)
        y = f.y.values
        briers.append(1 - np.mean((p - y) ** 2) / np.mean((pc - y) ** 2))
        ll = lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        logs.append(1 - ll(p) / ll(pc))
    vals, w = [], []
    for _, g in m.groupby(["seed", "fold", "agent"]):
        a = auc(g.y.values, g[col].values)
        if a is not None:
            vals.append(a); w.append(len(g))
    return {"auc_gain": float(np.mean(gains)), "auc_within_config": float(np.average(vals, weights=w)),
            "brier_skill": float(np.mean(briers)), "logloss_skill": float(np.mean(logs))}


def evaluate(d, s):
    tasks = pd.read_csv(DATA / d / "tasks.csv")
    tasks["b"] = tasks.case_id.map(pd.read_csv(DATA / d / "irt/1d_1pl/items.csv", index_col=0).b)
    m = load(d, s)
    cols = [c for c in METHODS if c in m and m[c].notna().all()]
    res = {}
    diffs = {c: difficulty(m, c) for c in cols}
    for c in cols:
        r = {}
        if c not in ("constant", "oracle"):
            per_seed = [rank_cells(g, tasks) for _, g in diffs[c].groupby("seed")]
            r["rho_within_domain"] = float(np.nanmean([x[0] for x in per_seed]))
            r["pacc_within_domain"] = float(np.nanmean([x[1] for x in per_seed]))
            if d == "sopbench" and s == "new_domain":
                r["pacc_procedures"] = proc_pairs(diffs[c], tasks)
        r.update(scores(m, c) if c != "constant" else {"auc_gain": 0.0, "brier_skill": 0.0, "logloss_skill": 0.0,
                                                       "auc_within_config": 0.5})
        res[c] = r
    # procedure bootstrap on seed 0 for the primary metric and the paired difference to the baseline.
    # One generator per benchmark with a fixed seed, so draw k resamples the same procedures in every
    # split of that benchmark; this lets pooled() average draws across splits.
    rng = np.random.default_rng([20260928, DATASETS.index(d)])
    boots = {c: [] for c in cols if c not in ("constant", "oracle")}
    dboots = {c: [] for c in boots if c != BASELINE}
    per_proc = {}
    for c in boots:
        x = diffs[c][diffs[c].seed == 0].merge(tasks[["case_id", "domain", "procedure_id", "b"]], on="case_id")
        x["cell"] = x.fold.astype(str) + "|" + x.domain
        per_proc[c] = {p: (g.cell.values, g.l.values, g.b.values) for p, g in x.groupby("procedure_id")}
    procs = list(per_proc[next(iter(boots))])
    for _ in range(N_BOOT):
        pick = rng.choice(procs, len(procs))
        vals = {}
        for c in boots:
            cell = np.concatenate([per_proc[c][p][0] for p in pick])
            l = np.concatenate([per_proc[c][p][1] for p in pick])
            b = np.concatenate([per_proc[c][p][2] for p in pick])
            vals[c] = rank_arrays(cell, l, b)
            boots[c].append(vals[c])
        for c in dboots:
            dboots[c].append(vals[c] - vals[BASELINE])
    for c in boots:
        res[c]["rho_ci"] = [float(np.nanpercentile(boots[c], 2.5)), float(np.nanpercentile(boots[c], 97.5))]
    for c in dboots:
        res[c]["rho_minus_baseline_ci"] = [float(np.nanpercentile(dboots[c], 2.5)), float(np.nanpercentile(dboots[c], 97.5))]
    return res, {c: np.array(v) for c, v in boots.items()}


def pooled(out, draws):
    """Mean primary score over the four new-process scenarios, with a joint bootstrap interval
    (draws of the two splits of one benchmark share their procedure resample; benchmarks independent)."""
    cols = [c for c in METHODS if all(c in draws[k] for k in NEW)]
    res = {}
    for c in cols:
        per = [out[k][c]["rho_within_domain"] for k in NEW]
        dist = np.nanmean([draws[k][c] for k in NEW], axis=0)
        r = {"rho_mean": float(np.mean(per)), "rho_worst": float(min(per)), "rho_by_scenario": per,
             "positive_everywhere": bool(min(per) > 0),
             "rho_ci": [float(np.nanpercentile(dist, 2.5)), float(np.nanpercentile(dist, 97.5))]}
        if c != BASELINE:
            dd = dist - np.nanmean([draws[k][BASELINE] for k in NEW], axis=0)
            r["minus_baseline_ci"] = [float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
        res[c] = r
    return res


def main():
    out, draws, lines = {}, {}, []
    say = lambda x: (print(x, flush=True), lines.append(x))  # noqa: E731
    for d in DATASETS:
        for s in ["random_cases", "new_procedures", "new_domain"]:
            if (RESULTS / d / f"obs_{s}.csv.gz").exists():
                out[f"{d}|{s}"], draws[f"{d}|{s}"] = evaluate(d, s)
                say(f"{d} {s}")
                for c, r in out[f"{d}|{s}"].items():
                    say(f"  {c:34s} " + " ".join(f"{k}={v:+.3f}" if isinstance(v, float) else f"{k}={[round(x, 3) for x in v]}"
                                                 for k, v in r.items()))
    out["pooled_new_process"] = pooled(out, draws)
    say(f"pooled over {NEW}")
    for c, r in sorted(out["pooled_new_process"].items(), key=lambda x: -x[1]["rho_mean"]):
        say(f"  {c:34s} rho {r['rho_mean']:+.3f} {[round(x, 3) for x in r['rho_ci']]}  worst {r['rho_worst']:+.3f}  by scenario "
            f"{[round(x, 3) for x in r['rho_by_scenario']]}  vs baseline {[round(x, 3) for x in r.get('minus_baseline_ci', [])]}")
    (RESULTS / "final_evaluation.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "final_evaluation.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
