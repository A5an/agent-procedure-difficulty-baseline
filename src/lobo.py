"""Leave-one-benchmark-out across SOPBench, tau2-bench and tau-Knowledge banking (Krsteski & Meyer
protocol, arXiv 2608.05797).

Target: full-data 1PL difficulty standardised inside each benchmark, z(beta). Their ridge
(FeatureBasedPredictor: StandardScaler + RidgeCV over the grid of run_baseline.py) is fitted on two
benchmarks and predicts the third; reported: Spearman rho and pairwise accuracy inside the held-out
benchmark, with a bootstrap over task groups (procedures in SOPBench, resolution paths in telecom,
single tasks elsewhere). No agent ability is involved, so benchmarks with different agents pool.

  python src/lobo.py        -> results/lobo.json
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BENCHES, DATA, RESULTS, use_agent_psychometrics

use_agent_psychometrics()
from experiment_new_tasks.feature_predictor import FeatureBasedPredictor  # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
FEATURES = ["length", "emb_ap", "adele", "human_time"]
N_BOOT = 1000


class ArraySource:
    """Minimal feature source: task id -> row of a matrix."""

    def __init__(self, name, ids, X):
        self.name = name
        self._idx = {t: i for i, t in enumerate(ids)}
        self._X = X

    def get_features(self, task_ids):
        return self._X[[self._idx[t] for t in task_ids]]


def load():
    tasks, feats = [], {k: [] for k in FEATURES}
    for b in BENCHES:
        d = DATA / b
        t = pd.read_csv(d / "tasks.csv")[["case_id", "domain", "procedure_id"]]
        beta = t.case_id.map(pd.read_csv(d / "irt/1d_1pl/items.csv", index_col=0).b)
        t["bench"] = b
        t["z"] = ((beta - beta.mean()) / beta.std()).values
        t["uid"] = b + ":" + t.case_id
        tasks.append(t)
        for name in ("length", "adele", "human_time"):
            c = pd.read_csv(d / "features" / f"{name}.csv").set_index("task_id").reindex(t.case_id)
            feats[name].append(c.values.astype(np.float32))
        z = np.load(d / "features" / "ap_deepseek_r1_qwen_1.5b.npz", allow_pickle=True)
        idx = {str(k): i for i, k in enumerate(z["task_ids"])}
        feats["emb_ap"].append(z["X"][[idx[c] for c in t.case_id]].astype(np.float32))
    tasks = pd.concat(tasks, ignore_index=True)
    return tasks, {k: ArraySource(k, tasks.uid.tolist(), np.vstack(v)) for k, v in feats.items()}


def pairwise_acc(pred, true):
    i, j = np.triu_indices(len(true), 1)
    keep = true[i] != true[j]
    s = np.sign(pred[i] - pred[j])[keep] * np.sign(true[i] - true[j])[keep]
    return float(np.mean(np.where(s == 0, 0.5, s > 0)))


def main():
    tasks, sources = load()
    out = {}
    for h, held in enumerate(BENCHES):
        tr = tasks[tasks.bench != held]
        te = tasks[tasks.bench == held].reset_index(drop=True)
        out[held] = {}
        for n, name in enumerate(FEATURES):
            p = FeatureBasedPredictor(sources[name], alphas=ALPHAS)
            p.fit(tr.uid.tolist(), tr.z.values)
            pr = p.predict(te.uid.tolist())
            pred = np.array([pr[u] for u in te.uid])
            groups = te.procedure_id.unique()
            gi = {g: np.where(te.procedure_id == g)[0] for g in groups}
            rng = np.random.default_rng([20260928, h, n])
            br, bp = [], []
            for _ in range(N_BOOT):
                idx = np.concatenate([gi[g] for g in rng.choice(groups, len(groups), replace=True)])
                br.append(spearmanr(pred[idx], te.z.values[idx]).correlation)
                bp.append(pairwise_acc(pred[idx], te.z.values[idx]))
            out[held][name] = {
                "rho": float(spearmanr(pred, te.z).correlation),
                "rho_ci": [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))],
                "pacc": pairwise_acc(pred, te.z.values),
                "pacc_ci": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
                "n_test": len(te), "alpha": p._best_alpha}
            print(held, name, json.dumps(out[held][name]), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lobo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
