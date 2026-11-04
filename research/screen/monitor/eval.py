"""Exp 2: does a monitor at the commit point predict episode failure, inside a benchmark and across benchmarks?

Metrics: AUC over all episodes, AUC within agent (mean over agents weighted by episodes; removes "strong agents fail
less"), and selective escalation: send the riskiest 10% / 20% of each agent's episodes to a human, report the share of
that agent's failures caught. Inside a benchmark: 5-fold CV grouped by task. Across: train on the other benchmarks.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "episodes.csv"))
df["bench2"] = df.bench.replace({"tau2_airline": "tau2", "tau2_retail": "tau2"})
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]
rng = np.random.default_rng(0)


def models():
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
            "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)}


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.mean(y) < 1 else np.nan


def within_agent_auc(d, s):
    vals, ws = [], []
    for a, g in d.groupby("agent"):
        v = auc(g.fail.values, s[g.index.values])
        if not np.isnan(v): vals.append(v); ws.append(len(g))
    return float(np.average(vals, weights=ws)) if vals else np.nan


def caught(d, s, frac):
    tot = got = 0
    for a, g in d.groupby("agent"):
        k = int(round(frac * len(g)))
        if k == 0: continue
        top = g.index.values[np.argsort(-s[g.index.values])[:k]]
        got += d.loc[top, "fail"].sum(); tot += g.fail.sum()
    return float(got / tot) if tot else np.nan


def boot_ci(d, s, B=500):
    """bootstrap over tasks for the within-agent AUC"""
    tasks = d.task.unique(); idx = {t: d.index[d.task == t].values for t in tasks}
    vals = []
    for _ in range(B):
        pick = rng.choice(tasks, len(tasks))
        ii = np.concatenate([idx[t] for t in pick])
        dd = d.loc[ii].reset_index(drop=True); ss = s[ii]
        vals.append(within_agent_auc(dd, ss))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def report(name, d, s):
    s = pd.Series(s, index=d.index).values if not isinstance(s, np.ndarray) else s
    out = {"n": len(d), "fail": float(d.fail.mean()), "auc": auc(d.fail.values, s), "auc_within_agent": within_agent_auc(d, s),
           "ci_within": boot_ci(d.reset_index(drop=True), s), "caught@10%": caught(d, s, .1), "caught@20%": caught(d, s, .2)}
    print(f"{name:<52} n {out['n']:>6}  fail {out['fail']:.2f}  AUC {out['auc']:.3f}  within-agent {out['auc_within_agent']:.3f} "
          f"[{out['ci_within'][0]:.3f}, {out['ci_within'][1]:.3f}]  caught@10% {out['caught@10%']:.2f}  @20% {out['caught@20%']:.2f}", flush=True)
    return out


R = {}
# 1. inside each benchmark
for b in ("sopbench", "tau2", "tib"):
    d = df[df.bench2 == b].reset_index(drop=True)
    for mname in ("logreg", "gbm"):
        s = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(d, groups=d.task):
            m = models()[mname].fit(d.loc[tr, FEATS], d.loc[tr, "fail"])
            s[te] = m.predict_proba(d.loc[te, FEATS])[:, 1]
        R[f"within {b} {mname}"] = report(f"within {b}, {mname}, 5-fold by task", d, s)
    for f in ("has_commit", "n_calls", "grounding", "n_tool_errors"):
        sgn = -1 if f == "grounding" else 1
        R[f"within {b} only {f}"] = report(f"within {b}, single feature {f}", d, sgn * d[f].values.astype(float) + 1e-6 * rng.standard_normal(len(d)))

# 2. across benchmarks: train on the others
for b in ("sopbench", "tau2", "tib"):
    tr = df[df.bench2 != b].reset_index(drop=True); te = df[df.bench2 == b].reset_index(drop=True)
    for mname in ("logreg", "gbm"):
        m = models()[mname].fit(tr[FEATS], tr.fail)
        R[f"cross ->{b} {mname}"] = report(f"train on others -> {b}, {mname}", te, m.predict_proba(te[FEATS])[:, 1])
for src, dst in (("sopbench", "tau2"), ("sopbench", "tib"), ("tau2", "sopbench"), ("tau2", "tib")):
    tr = df[df.bench2 == src].reset_index(drop=True); te = df[df.bench2 == dst].reset_index(drop=True)
    m = models()["logreg"].fit(tr[FEATS], tr.fail)
    R[f"cross {src}->{dst}"] = report(f"train {src} -> {dst}, logreg", te, m.predict_proba(te[FEATS])[:, 1])

# coefficients of the pooled logistic model (standardized features)
m = models()["logreg"].fit(df[FEATS], df.fail)
coef = dict(zip(FEATS, m.named_steps["logisticregression"].coef_[0].round(3)))
print("\npooled logreg coefficients (per SD):", coef)
R["coef"] = {k: float(v) for k, v in coef.items()}
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
