"""Exp 5 (PREREG.md): family probes + commit-point monitor router on TauIndianBank, recorded data only.

  python combo.py   -> prints tables, writes results.json
"""
import json, sys, collections
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
T = HERE.parent / "tib"
sys.path.insert(0, str(HERE.parent / "monitor"))
from build import features, tau_events, TIB_WRITES  # noqa: E402

rng = np.random.default_rng(20261110)
sel = json.load(open(T / "selection.json"))
FAMS = sel["test_families"]
AGS = ["g38", "flash"]
API = {"g38": 0.0385, "flash": 0.0060}
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]
RATES = [round(0.1 * i, 1) for i in range(10)]
COSTS = [(20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)]


def fam_of(tid):
    return next(f for f in FAMS if f"_{f}_" in tid)


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is not None:
            out[r["task_id"]] = 1 - int(r["reward"] >= 1)
    return out


# ---------------------------------------------------------------- episodes with features
rows = []
for a in AGS:
    for which, lab, msgdir in (("real", f"real_{a}.jsonl", f"real_{a}"), ("doc", f"probes_{a}.v2.jsonl", f"probes_{a}"),
                               ("seeded", f"seeded_{a}.jsonl", f"seeded_{a}")):
        for tid, fail in load(T / "runs" / lab).items():
            msgs = json.load(open(T / "runs" / "msgs" / msgdir / f"{tid}.json"))
            rows.append({"src": which, "agent": a, "task": tid, "fam": fam_of(tid), "fail": fail,
                         **features(tau_events(msgs), TIB_WRITES)})
E = pd.DataFrame(rows)
print(E.groupby(["src", "agent"]).fail.agg(["size", "mean"]).round(3))
chk = pd.read_csv(HERE.parent / "monitor" / "episodes.csv")
chk = chk[chk.bench == "tib"].set_index(["agent", "task"]).fail
R = E[E.src == "real"].set_index(["agent", "task"])
assert (R.fail == chk.reindex(R.index)).all(), "real labels differ from the monitor table"

REAL = E[E.src == "real"].reset_index(drop=True)
PROBE = E[E.src != "real"].reset_index(drop=True)

# ---------------------------------------------------------------- family estimates
def rate(df, a, f):
    x = df[(df.agent == a) & (df.fam == f)].fail
    return (x.sum() + 0.5) / (len(x) + 1)


real_rate = {a: {f: REAL[(REAL.agent == a) & (REAL.fam == f)].fail.mean() for f in FAMS} for a in AGS}
EST = {}
for src in ("seeded", "doc"):
    for cal in (False, True):
        e = {}
        for a in AGS:
            raw = {f: rate(PROBE[PROBE.src == src], a, f) for f in FAMS}
            e[a] = {}
            for f in FAMS:
                p = raw[f]
                if cal:
                    others = [g for g in FAMS if g != f]
                    p = p + np.mean([real_rate[a][g] for g in others]) - np.mean([raw[g] for g in others])
                e[a][f] = float(np.clip(p, 0.01, 0.99))
        EST[f"{src}{'+cal' if cal else ''}"] = e
EST["no probes"] = {a: {f: float(np.mean([real_rate[a][g] for g in FAMS if g != f])) for f in FAMS} for a in AGS}

# ---------------------------------------------------------------- monitor scores and operating curves
def curve(scores, fails):
    """threshold, TPR, FPR per flag rate, from out-of-fold scores of one agent"""
    out = {}
    for r in RATES:
        thr = np.inf if r == 0 else np.quantile(scores, 1 - r)
        flag = scores > thr if r else np.zeros(len(scores), bool)
        tpr = flag[fails == 1].mean() if (fails == 1).any() else 0.0
        fpr = flag[fails == 0].mean() if (fails == 0).any() else 0.0
        out[r] = (thr, float(tpr), float(fpr))
    return out


MON = {}  # name -> {family: {"score": series indexed like REAL rows of that family, "curve": {agent: curve}}}
# M1: real episodes of the other families
M1 = {}
for f in FAMS:
    tr = REAL[REAL.fam != f]
    te = REAL[REAL.fam == f]
    model = gbm().fit(tr[FEATS], tr.fail)
    s_te = pd.Series(model.predict_proba(te[FEATS])[:, 1], index=te.index)
    oof = pd.Series(np.nan, index=tr.index)
    for g in FAMS:
        if g == f:
            continue
        a_, b_ = tr[tr.fam != g], tr[tr.fam == g]
        oof[b_.index] = gbm().fit(a_[FEATS], a_.fail).predict_proba(b_[FEATS])[:, 1]
    M1[f] = {"score": s_te, "curve": {a: curve(oof[tr.agent == a].values, tr[tr.agent == a].fail.values) for a in AGS}}
MON["M1 real logs of other families"] = M1
# M2: probe episodes only
oof = pd.Series(np.nan, index=PROBE.index)
for g in FAMS:
    a_, b_ = PROBE[PROBE.fam != g], PROBE[PROBE.fam == g]
    oof[b_.index] = gbm().fit(a_[FEATS], a_.fail).predict_proba(b_[FEATS])[:, 1]
full = gbm().fit(PROBE[FEATS], PROBE.fail)
s_all = pd.Series(full.predict_proba(REAL[FEATS])[:, 1], index=REAL.index)
cur2 = {a: curve(oof[PROBE.agent == a].values, PROBE[PROBE.agent == a].fail.values) for a in AGS}
MON["M2 probe episodes only"] = {f: {"score": s_all[REAL.fam == f], "curve": cur2} for f in FAMS}

for name, M in MON.items():
    sc = pd.concat([M[f]["score"] for f in FAMS])
    y = REAL.loc[sc.index, "fail"]
    wa = np.mean([roc_auc_score(y[REAL.loc[sc.index, "agent"] == a], sc[REAL.loc[sc.index, "agent"] == a]) for a in AGS])
    print(f"{name}: AUC on real episodes {roc_auc_score(y, sc):.3f}, within agent {wa:.3f}")


# ---------------------------------------------------------------- routing
def route(est, mon, F, h, q, allow_monitor=True, allow_agent=True):
    """per family: chosen option and realized mean loss per case; also escalation and failure counts"""
    H = h + F * (1 - q)
    res = []
    for f in FAMS:
        opts = [("human", None, None, H)]
        for a in AGS:
            p = est[a][f]
            if allow_agent:
                opts.append(("agent", a, 0.0, API[a] + F * p))
            if allow_monitor and mon is not None:
                for r, (thr, tpr, fpr) in mon[f]["curve"][a].items():
                    if r == 0:
                        continue
                    exp = API[a] + (p * tpr + (1 - p) * fpr) * H + p * (1 - tpr) * F
                    opts.append(("agent+monitor", a, r, exp))
        kind, a, r, _ = min(opts, key=lambda o: o[3])
        if kind == "human":
            res.append({"fam": f, "choice": "human", "loss": H, "escalated": 1.0, "fails": 0.0}); continue
        te = REAL[(REAL.fam == f) & (REAL.agent == a)]
        fail = te.fail.values
        if kind == "agent":
            loss = API[a] + F * fail
            res.append({"fam": f, "choice": a, "loss": loss.mean(), "escalated": 0.0, "fails": fail.mean()}); continue
        thr = mon[f]["curve"][a][r][0]
        flag = mon[f]["score"][te.index].values > thr
        loss = np.where(flag, API[a] + H, API[a] + F * fail)
        res.append({"fam": f, "choice": f"{a}+mon@{r}", "loss": loss.mean(), "escalated": flag.mean(), "fails": (fail * ~flag).mean()})
    return pd.DataFrame(res)


def boot(d, B=4000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


OUT = {}
for F, h, q in COSTS:
    key = f"F={F} h={h} q={q}"
    H = h + F * (1 - q)
    per = {"always human": np.full(len(FAMS), H)}
    for a in AGS:
        per[f"always {a}"] = np.array([API[a] + F * real_rate[a][f] for f in FAMS])
    best = min(per, key=lambda k: per[k].mean())
    per["oracle per family"] = np.min(np.vstack(list(per.values())), 0)
    detail = {}
    for en, est in EST.items():
        r0 = route(est, None, F, h, q, allow_monitor=False)
        per[f"probes only [{en}]"] = r0.loss.values; detail[f"probes only [{en}]"] = r0
        for mn, mon in MON.items():
            r1 = route(est, mon, F, h, q)
            per[f"combined [{en} | {mn[:2]}]"] = r1.loss.values; detail[f"combined [{en} | {mn[:2]}]"] = r1
    o = {"best constant": best, "mean loss": {k: float(v.mean()) for k, v in per.items()}, "vs best constant": {}, "vs probes only": {}}
    for k in per:
        if k.startswith("combined") or k.startswith("probes only"):
            o["vs best constant"][k] = boot(per[k] - per[best])
        if k.startswith("combined"):
            en = k[len("combined ["):].split(" | ")[0]
            o["vs probes only"][k] = boot(per[k] - per[f"probes only [{en}]"])
    o["choices"] = {k: collections.Counter(v.choice).most_common() for k, v in detail.items() if k.startswith("combined")}
    o["escalation and residual failures"] = {k: (float(v.escalated.mean()), float(v.fails.mean())) for k, v in detail.items()}
    OUT[key] = o
    print(f"\n== {key}  (best constant: {best} {per[best].mean():.2f}, oracle per family {per['oracle per family'].mean():.2f})")
    for k in per:
        if k.startswith(("combined", "probes only")):
            m, lo, hi = o["vs best constant"][k]
            extra = ""
            if k in o["vs probes only"]:
                m2, lo2, hi2 = o["vs probes only"][k]; extra = f" | vs probes only {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]"
            esc, res_f = o["escalation and residual failures"][k]
            print(f"  {k:52s} {per[k].mean():5.2f}  vs best {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{extra}  human share {esc:.2f} residual fail {res_f:.2f}")
json.dump(OUT, open(HERE / "results.json", "w"), indent=1, default=str)
