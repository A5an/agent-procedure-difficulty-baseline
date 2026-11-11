"""Exploratory (PREREG.md): does a commit-point failure monitor trained on synthetic episodes transfer to real
TauIndianBank episodes? Same 13 features and model as ../monitor/eval.py. Output monitor_transfer.out (stdout).
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monitor"))
from build import features, tau_events  # noqa: E402
import dataset, engine  # noqa: E402

GENS, AGENTS = ["haiku", "sonnet", "opus"], ["lite", "flash", "g38"]
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def within_agent_auc(d, s):
    vals, ws = [], []
    for a, g in d.groupby("agent"):
        if 0 < g.fail.mean() < 1:
            vals.append(roc_auc_score(g.fail, s[g.index.values])); ws.append(len(g))
    return float(np.average(vals, weights=ws)) if vals else np.nan


rows = []
for g in GENS:
    items = {it["uid"]: it for it in dataset.cases(g)}
    for a in AGENTS:
        p = HERE / "runs" / f"{g}_{a}.jsonl"
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("reward") is None or r["task_id"] not in items:
                continue
            it = items[r["task_id"]]
            writes = {t["name"] for t in it["spec"]["tools"] if engine.tool_is_write(t)}
            msgs = json.load(open(HERE / "runs" / "msgs" / f"{g}_{a}" / f"{r['task_id']}.json"))
            rows.append({"gen": g, "agent": a, "proc": f"{g}/{it['pid']}", "fail": 1 - int(r["reward"] >= 1),
                         **features(tau_events(msgs), writes)})
S = pd.DataFrame(rows).drop_duplicates()
T = pd.read_csv(HERE.parent / "monitor" / "episodes.csv")
T = T[T.bench == "tib"].reset_index(drop=True)
print(f"synthetic episodes {len(S)}, TauIndianBank real episodes {len(T)} (fail {T.fail.mean():.2f})")
for name, tr in [(g, S[S.gen == g]) for g in GENS] + [("all three", S)]:
    tr = tr.reset_index(drop=True)
    if tr.empty or tr.fail.nunique() < 2:
        continue
    s_in = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr.proc):
        s_in[b] = gbm().fit(tr.loc[a, FEATS], tr.loc[a, "fail"]).predict_proba(tr.loc[b, FEATS])[:, 1]
    m = gbm().fit(tr[FEATS], tr.fail)
    s = m.predict_proba(T[FEATS])[:, 1]
    print(f"train {name:10s} n {len(tr):4d}: inside synthetic (5-fold by procedure) AUC {roc_auc_score(tr.fail, s_in):.3f}, "
          f"within-agent {within_agent_auc(tr, s_in):.3f} | on TauIndianBank AUC {roc_auc_score(T.fail, s):.3f}, "
          f"within-agent {within_agent_auc(T, s):.3f}")
