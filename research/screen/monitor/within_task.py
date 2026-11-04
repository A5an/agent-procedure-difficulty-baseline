"""Exp 2 follow-up: does the monitor add anything beyond how hard the task is? (lit check: pooled AUC mostly measures
task difficulty). Within-task AUC = AUC among episodes of the same task (different agents / trials), averaged over tasks
with both outcomes. Difficulty baseline = the task's failure rate among the other agents (leave-agent-out)."""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "episodes.csv"))
df["bench2"] = df.bench.replace({"tau2_airline": "tau2", "tau2_retail": "tau2"})
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]
def within_task(d, s):
    v, w = [], []
    for t, g in d.groupby("task"):
        y = g.fail.values
        if 0 < y.mean() < 1: v.append(roc_auc_score(y, s[g.index.values])); w.append(len(g))
    return np.average(v, weights=w), len(v)
for b in ("sopbench", "tau2", "tib"):
    d = df[df.bench2 == b].reset_index(drop=True)
    # leave-agent-out task difficulty
    tot = d.groupby("task").fail.agg(["sum", "count"])
    own = d.groupby(["task", "agent"]).fail.agg(["sum", "count"])
    diff = np.array([(tot.loc[t, "sum"] - own.loc[(t, a), "sum"] + 0.5) / (tot.loc[t, "count"] - own.loc[(t, a), "count"] + 1) for t, a in zip(d.task, d.agent)])
    s = np.zeros(len(d)); s2 = np.zeros(len(d))
    X = d[FEATS].copy(); X2 = X.copy(); X2["task_difficulty"] = diff
    for tr, te in GroupKFold(5).split(d, groups=d.task):
        s[te] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X.loc[tr], d.fail[tr]).predict_proba(X.loc[te])[:, 1]
        s2[te] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X2.loc[tr], d.fail[tr]).predict_proba(X2.loc[te])[:, 1]
    wt_m, n_t = within_task(d, s)
    pooled = lambda x: roc_auc_score(d.fail, x)
    print(f"{b:9s} tasks with mixed outcomes {n_t:4d} | within-task AUC monitor {wt_m:.3f} | pooled AUC: difficulty only {pooled(diff):.3f}, monitor {pooled(s):.3f}, difficulty + monitor {pooled(s2):.3f}")
