"""Out-of-fold predictions of the baseline and of every reference method, on one benchmark.

  python src/run_baseline.py <bench> [scheme ...]      bench = sopbench | tau2 | tauk_banking

The baseline (Agent psychometrics, Ge et al., COLM 2026, their code unchanged):
  1. fit 1PL IRT on the training tasks of the fold only: P(success) = sigmoid(theta_agent - beta_task)
  2. freeze beta of the training tasks; ridge regression from task features to beta, penalty by
     inner 5-fold CV over their grid (their FeatureBasedPredictor)
  3. for a held-out task, predict beta_hat from its features; P = sigmoid(theta_agent - beta_hat)
Task features of the baseline: the 18 ADeLe demand levels of the task statement (Zhou et al.,
Nature 2026), see adele_annotate.py.

Every method below is fitted on the same folds, so all comparisons in evaluate.py are paired:
  constant        agent ability only, every task equally hard (reference, their ConstantPredictor)
  oracle          full-data IRT difficulty (upper bound, their OraclePredictor)
  length          log(1 + characters) of the statement (control, Krsteski & Meyer)
  emb_ap          their text embedding recipe (DeepSeek-R1-Distill-Qwen-1.5B backbone)
  adele           BASELINE: their ridge on the 18 ADeLe levels
  adele16         the same without AS and MCu (robustness)
  human_time      their ridge on log estimated human minutes (reference, not a candidate)
  lltm_adele      ADeLe levels fitted jointly with abilities as an LLTM (methods.AmortizedIRT)
  ap_combined     their grouped ridge on embedding + ADeLe, one penalty per family
  knn_router      kNN over bge-base-en-v1.5 embeddings (methods.KNNRouter)
  amortized_mirt  IRT-Router-style multidimensional IRT from bge embeddings (methods.AmortizedMIRT)
Logit averages of several methods (e.g. emb_ap + adele) are formed in evaluate.py.

Splits:
  random_cases    KFold over tasks, 5 folds x 5 seeds (tasks of a procedure land in train and test;
                  sanity check only)
  new_procedures  GroupKFold over procedures, 5 folds x 5 seeds
  new_domain      leave one domain out (one fixed partition, so one seed)

Writes results/<bench>/obs_<scheme>.csv.gz (one row per held-out attempt: seed, fold, agent, task,
outcome y, and P(success) of every method) and betas_<scheme>.csv (predicted difficulties).
Fold IRT fits are cached in results/<bench>/irt_splits/.
"""

import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    # Two things make the fold IRT of agent-psychometrics change from run to run: it writes the
    # training responses by iterating over a Python set (order depends on the string hash seed of
    # the process), and its trainer does not seed the random generators. The hash seed is fixed
    # here; the generators are seeded before every fold in run_scheme().
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import random  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyro  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

import methods  # noqa: E402
from common import AP_REPO, DATA, RESULTS, SCHEMES, use_agent_psychometrics  # noqa: E402

use_agent_psychometrics()
os.chdir(AP_REPO)  # their modules resolve some paths relative to the repository root
from experiment_new_tasks.dataset import load_dataset_for_fold  # noqa: E402
from experiment_new_tasks.difficulty_predictors import (  # noqa: E402
    ConstantPredictor, DifficultyPredictorAdapter, OraclePredictor)
from experiment_new_tasks.feature_predictor import FeatureBasedPredictor, GroupedRidgePredictor  # noqa: E402
from experiment_new_tasks.feature_source import (  # noqa: E402
    CSVFeatureSource, EmbeddingFeatureSource, GroupedFeatureSource)

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]  # their default grid
N_SEEDS = int(os.environ.get("SEEDS", 5))


def sources(bench):
    f = DATA / bench / "features"
    out = {n: CSVFeatureSource(f / f"{n}.csv", name=n) for n in ("length", "adele", "adele16", "human_time")}
    out["emb_ap"] = EmbeddingFeatureSource(f / "ap_deepseek_r1_qwen_1.5b.npz", name="emb_ap")
    out["emb_bge"] = EmbeddingFeatureSource(f / "bge_base_en_v1.5.npz", name="emb_bge")
    return out


def predictors(src):
    ridge = lambda s: DifficultyPredictorAdapter(FeatureBasedPredictor(s, alphas=ALPHAS))  # noqa: E731
    grids = GroupedRidgePredictor.SOURCE_ALPHA_GRIDS
    return {
        "constant": ConstantPredictor(),
        "oracle": OraclePredictor(),
        "length": ridge(src["length"]),
        "emb_ap": ridge(src["emb_ap"]),
        "adele": ridge(src["adele"]),
        "adele16": ridge(src["adele16"]),
        "human_time": ridge(src["human_time"]),
        "lltm_adele": methods.AmortizedIRT(src["adele"], name="lltm_adele"),
        "ap_combined": DifficultyPredictorAdapter(GroupedRidgePredictor(
            GroupedFeatureSource([src["emb_ap"], src["adele"]]),
            alpha_grids={"emb_ap": grids["Embedding"], "adele": grids["LLM Judge"]})),
        "knn_router": methods.KNNRouter(src["emb_bge"]),
        "amortized_mirt": methods.AmortizedMIRT(src["emb_bge"]),
    }


def splits(tasks, scheme, seed):
    ids = tasks.case_id.tolist()
    if scheme == "random_cases":
        it = KFold(5, shuffle=True, random_state=seed).split(ids)
    elif scheme == "new_procedures":
        it = GroupKFold(5, shuffle=True, random_state=seed).split(ids, groups=tasks.procedure_id)
    elif scheme == "new_domain":
        doms = sorted(tasks.domain.unique())
        it = [((tasks.domain != d).to_numpy().nonzero()[0], (tasks.domain == d).to_numpy().nonzero()[0])
              for d in doms]
    else:
        raise ValueError(scheme)
    return [([ids[i] for i in tr], [ids[i] for i in te]) for tr, te in it]


def run_scheme(bench, tasks, scheme, src):
    d, res = DATA / bench, RESULTS / bench
    seeds = [0] if scheme == "new_domain" else list(range(N_SEEDS))
    rows, betas = [], []
    for seed in seeds:
        folds = splits(tasks, scheme, seed)
        for k, (tr, te) in enumerate(folds):
            t0 = time.time()
            fold_seed = 1000 * seed + k
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            pyro.set_rng_seed(fold_seed)
            data = load_dataset_for_fold(
                abilities_path=d / "irt/1d_1pl/abilities.csv", items_path=d / "irt/1d_1pl/items.csv",
                responses_path=d / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k,
                k_folds=len(folds), split_seed=seed, irt_cache_dir=res / "irt_splits" / scheme)
            preds = predictors(src)
            for p in preds.values():
                p.fit(data, tr)
            for t in te:
                for a in data.train_abilities.index:
                    if t not in data.responses.get(a, {}):
                        continue
                    probs = {name: p.predict_probability(data, a, t) for name, p in preds.items()}
                    labels, _ = data.expand_for_auc(a, t, 0.0)  # binary cell: 1 row; binomial: one per attempt
                    for y in labels:
                        rows.append({"seed": seed, "fold": k, "agent": a, "case_id": t, "y": int(y), **probs})
            for name, p in preds.items():
                for t, b in getattr(p, "_predicted_difficulties", {}).items():
                    betas.append({"seed": seed, "fold": k, "method": name, "case_id": t, "beta_hat": b})
            print(f"{bench} {scheme} seed {seed} fold {k + 1}/{len(folds)} ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(betas)


def main(bench, schemes):
    tasks = pd.read_csv(DATA / bench / "tasks.csv")
    if tasks.domain.nunique() < 2:
        schemes = [s for s in schemes if s != "new_domain"]
    if tasks.procedure_id.nunique() == len(tasks):
        schemes = [s for s in schemes if s != "new_procedures"]  # identical to random_cases
    src = sources(bench)
    res = RESULTS / bench
    res.mkdir(parents=True, exist_ok=True)
    for scheme in schemes:
        obs, betas = run_scheme(bench, tasks, scheme, src)
        obs.to_csv(res / f"obs_{scheme}.csv.gz", index=False)
        betas.to_csv(res / f"betas_{scheme}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or SCHEMES)
