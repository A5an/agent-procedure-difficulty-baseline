"""Frozen comparators on ThinkingBox-Bench, fitted only on the three development benchmarks.

baseline: the 18 ADeLe demand levels of each task (src/adele_annotate.py, unchanged: Gemini 3.8 Flash, temperature 0,
          the ADeLe rubrics and parser) and the ridge of src/lobo.py (Agent psychometrics FeatureBasedPredictor, same
          alpha grid) fitted on SOPBench + tau2 + tau-Knowledge banking with the target z(beta) inside each benchmark.
          This is the leave-one-benchmark-out baseline of the protocol with ThinkingBox as the held-out benchmark.
length:   the same ridge on log characters of the task text (control).
The annotated text of a task is the request of build_items.py (the user's first message and the simulated user's
instructions), as for tau2, whose statements are the user scenario.

  python baseline.py annotate   ADeLe levels -> adele_responses.jsonl (needs LLM_BACKEND=vertex and GOOGLE_CLOUD_PROJECT)
  python baseline.py predict    -> predictions_baseline.csv"""
import json
import sys

import numpy as np
import pandas as pd

from tbpaths import HERE, ITEMS, REPO

sys.path.insert(0, str(REPO / "src"))
import adele_annotate as AA  # noqa: E402
from common import BENCHES, DATA, use_agent_psychometrics  # noqa: E402
from lobo import ALPHAS, ArraySource  # noqa: E402

use_agent_psychometrics()
from experiment_new_tasks.feature_predictor import FeatureBasedPredictor  # noqa: E402

CACHE = HERE / "adele_responses.jsonl"


def items():
    return [json.loads(l) for l in ITEMS.read_text().splitlines()]


def annotate():
    AA.run([(it["task_id"], it["request"]) for it in items()], CACHE)


def dev():
    rows, feats = [], {"adele": [], "length": []}
    for b in BENCHES:
        d = DATA / b
        t = pd.read_csv(d / "tasks.csv")[["case_id"]]
        beta = t.case_id.map(pd.read_csv(d / "irt/1d_1pl/items.csv", index_col=0).b)
        t["z"] = ((beta - beta.mean()) / beta.std()).values
        t["uid"] = b + ":" + t.case_id
        rows.append(t)
        for name in feats:
            c = pd.read_csv(d / "features" / f"{name}.csv").set_index("task_id").reindex(t.case_id)
            feats[name].append(c.values.astype(np.float32))
    return pd.concat(rows, ignore_index=True), {k: np.vstack(v) for k, v in feats.items()}


def predict():
    tr, Xtr = dev()
    its = items()
    lv = AA.levels(CACHE).reindex([it["task_id"] for it in its])
    unparsed = int(lv.isna().sum().sum())
    lv = lv.fillna(lv.median())  # as in adele_annotate.annotate, counted and reported
    cols = list(pd.read_csv(DATA / BENCHES[0] / "features" / "adele.csv", nrows=0).columns[1:])  # dev column order
    Xte = {"adele": lv[cols].values.astype(np.float32),
           "length": np.array([[it["log_chars"]] for it in its], dtype=np.float32)}
    te_uid = ["tb:" + it["task_id"] for it in its]
    out = pd.DataFrame({"task_id": [it["task_id"] for it in its], "log_chars": Xte["length"][:, 0]})
    for name in ("adele", "length"):
        src = ArraySource(name, tr.uid.tolist() + te_uid, np.vstack([Xtr[name], Xte[name]]))
        p = FeatureBasedPredictor(src, alphas=ALPHAS)
        p.fit(tr.uid.tolist(), tr.z.values)
        pr = p.predict(te_uid)
        out[f"score_{'baseline' if name == 'adele' else 'length'}"] = [pr[u] for u in te_uid]
        print(name, "alpha", p._best_alpha)
    out.to_csv(HERE / "predictions_baseline.csv", index=False)
    print("done", len(out), "unparsed ADeLe cells filled with the column median:", unparsed)


if __name__ == "__main__":
    {"annotate": annotate, "predict": predict}[sys.argv[1]]()
