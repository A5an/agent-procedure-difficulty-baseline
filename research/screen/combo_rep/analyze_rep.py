"""Replication analysis (PREREG.md of this folder), frozen before the first replication episode.

The decision rule, monitor, operating curves, costs and bootstrap are those of ../combo/combo.py, unchanged.
Scope "new": the 30 replication families. Scope "all": the 20 families of P0-2 / Exp 3 plus the 30 new ones.

  python analyze_rep.py  -> analyze_rep.out (stdout), results_rep.json
"""
import json, sys, collections
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
T = HERE.parent / "tib"
sys.path.insert(0, str(HERE.parent / "monitor"))
from build import features, tau_events, TIB_WRITES  # noqa: E402

AGS = ["g38", "flash"]
API = {"g38": 0.0385, "flash": 0.0060}
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]
RATES = [round(0.1 * i, 1) for i in range(10)]
COSTS = [(20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)]
rng = np.random.default_rng(20261112)


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def load(path):
    out = {}
    if not path.exists():
        return out
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is not None:
            out[r["task_id"]] = 1 - int(r["reward"] >= 1)
    return out


def episodes(fams, base, files):
    """files: {src: (label file pattern, msg dir pattern)} with {a} for the agent"""
    rows = []
    for a in AGS:
        for src, (lab, md) in files.items():
            for tid, fail in load(base / "runs" / lab.format(a=a)).items():
                fam = next((f for f in fams if f"_{f}_" in tid), None)
                if fam is None:
                    continue
                msgs = json.load(open(base / "runs" / "msgs" / md.format(a=a) / f"{tid}.json"))
                rows.append({"src": src, "agent": a, "task": tid, "fam": fam, "fail": fail, **features(tau_events(msgs), TIB_WRITES)})
    return rows


new_sel = json.load(open(HERE / "selection_rep.json"))
old_sel = json.load(open(T / "selection.json"))
NEW, OLD = new_sel["test_families"], old_sel["test_families"]
E_new = pd.DataFrame(episodes(NEW, HERE, {"real": ("real_{a}.jsonl", "real_{a}"), "seeded": ("seeded_{a}.jsonl", "seeded_{a}"),
                                          "doc": ("doc_{a}.jsonl", "doc_{a}")}))
E_old = pd.DataFrame(episodes(OLD, T, {"real": ("real_{a}.jsonl", "real_{a}"), "seeded": ("seeded_{a}.jsonl", "seeded_{a}"),
                                       "doc": ("probes_{a}.v2.jsonl", "probes_{a}")}))


def curve(scores, fails):
    out = {}
    for r in RATES:
        thr = np.inf if r == 0 else np.quantile(scores, 1 - r)
        flag = scores > thr if r else np.zeros(len(scores), bool)
        out[r] = (thr, float(flag[fails == 1].mean()) if (fails == 1).any() else 0.0,
                  float(flag[fails == 0].mean()) if (fails == 0).any() else 0.0)
    return out


def boot(d, B=4000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze(E, FAMS, label):
    REAL = E[E.src == "real"].reset_index(drop=True)
    PROBE = E[E.src != "real"].reset_index(drop=True)
    FAMS = [f for f in FAMS if ((REAL.fam == f) & (REAL.agent == "g38")).any() and ((REAL.fam == f) & (REAL.agent == "flash")).any()]
    print(f"\n######## {label}: {len(FAMS)} families with real episodes of both agents")
    print(E.groupby(["src", "agent"]).fail.agg(["size", "mean"]).round(3).to_string())
    real_rate = {a: {f: REAL[(REAL.agent == a) & (REAL.fam == f)].fail.mean() for f in FAMS} for a in AGS}

    def rate(df, a, f):
        x = df[(df.agent == a) & (df.fam == f)].fail
        return (x.sum() + 0.5) / (len(x) + 1)

    EST = {}
    for src in ("seeded", "doc"):
        P = PROBE[PROBE.src == src]
        if P.empty or not all(((P.fam == f) & (P.agent == a)).any() for f in FAMS for a in AGS):
            continue
        for cal in (False, True):
            e = {}
            for a in AGS:
                raw = {f: rate(P, a, f) for f in FAMS}
                e[a] = {}
                for f in FAMS:
                    p = raw[f]
                    if cal:
                        o = [g for g in FAMS if g != f]
                        p = p + np.mean([real_rate[a][g] for g in o]) - np.mean([raw[g] for g in o])
                    e[a][f] = float(np.clip(p, 0.01, 0.99))
            EST[f"{src}{'+cal' if cal else ''}"] = e

    MON = {}
    # M1: real episodes of the other families (as combo.py)
    M1 = {}
    for f in FAMS:
        tr = REAL[(REAL.fam != f) & REAL.fam.isin(FAMS)]
        te = REAL[REAL.fam == f]
        s_te = pd.Series(gbm().fit(tr[FEATS], tr.fail).predict_proba(te[FEATS])[:, 1], index=te.index)
        oof = pd.Series(np.nan, index=tr.index)
        for g in FAMS:
            if g == f:
                continue
            a_, b_ = tr[tr.fam != g], tr[tr.fam == g]
            oof[b_.index] = gbm().fit(a_[FEATS], a_.fail).predict_proba(b_[FEATS])[:, 1]
        M1[f] = {"score": s_te, "curve": {a: curve(oof[tr.agent == a].values, tr[tr.agent == a].fail.values) for a in AGS}}
    MON["M1"] = M1
    for name, srcs in (("M2 seeded", ["seeded"]), ("M2 doc+seeded", ["seeded", "doc"])):
        P = PROBE[PROBE.src.isin(srcs) & PROBE.fam.isin(FAMS)].reset_index(drop=True)
        if not set(srcs) <= set(P.src):
            continue
        oof = pd.Series(np.nan, index=P.index)
        for g in FAMS:
            a_, b_ = P[P.fam != g], P[P.fam == g]
            if b_.empty:
                continue
            oof[b_.index] = gbm().fit(a_[FEATS], a_.fail).predict_proba(b_[FEATS])[:, 1]
        s = pd.Series(gbm().fit(P[FEATS], P.fail).predict_proba(REAL[FEATS])[:, 1], index=REAL.index)
        cur = {a: curve(oof[P.agent == a].dropna().values, P[(P.agent == a) & oof.notna()].fail.values) for a in AGS}
        MON[name] = {f: {"score": s[REAL.fam == f], "curve": cur} for f in FAMS}
    res = {"families": FAMS, "monitor_auc": {}, "routing": {}}
    for name, M in MON.items():
        sc = pd.concat([M[f]["score"] for f in FAMS])
        y = REAL.loc[sc.index, "fail"]
        wf = []
        for f in FAMS:
            for a in AGS:
                m = (REAL.loc[sc.index, "fam"] == f) & (REAL.loc[sc.index, "agent"] == a)
                if 0 < y[m].mean() < 1:
                    wf.append(roc_auc_score(y[m], sc[m]))
        res["monitor_auc"][name] = (float(roc_auc_score(y, sc)), float(np.mean(wf)) if wf else None)
        print(f"monitor {name:14s}: AUC {res['monitor_auc'][name][0]:.3f}, within family x agent {res['monitor_auc'][name][1]:.3f}")

    def route(est, mon, F, h, q, allow_monitor=True):
        H = h + F * (1 - q)
        out = []
        for f in FAMS:
            opts = [("human", None, None, H)]
            for a in AGS:
                p = est[a][f]
                opts.append(("agent", a, None, API[a] + F * p))
                if allow_monitor and mon is not None:
                    for r, (thr, tpr, fpr) in mon[f]["curve"][a].items():
                        if r:
                            opts.append(("mon", a, r, API[a] + (p * tpr + (1 - p) * fpr) * H + p * (1 - tpr) * F))
            kind, a, r, _ = min(opts, key=lambda o: o[3])
            if kind == "human":
                out.append((H, 1.0)); continue
            te = REAL[(REAL.fam == f) & (REAL.agent == a)]
            fail = te.fail.values
            if kind == "agent":
                out.append(((API[a] + F * fail).mean(), 0.0)); continue
            flag = mon[f]["score"][te.index].values > mon[f]["curve"][a][r][0]
            out.append((np.where(flag, API[a] + H, API[a] + F * fail).mean(), flag.mean()))
        return np.array([x[0] for x in out]), float(np.mean([x[1] for x in out]))

    for F, h, q in COSTS:
        H = h + F * (1 - q)
        key = f"F={F} h={h} q={q}"
        per = {"always human": np.full(len(FAMS), H)}
        for a in AGS:
            per[f"always {a}"] = np.array([API[a] + F * real_rate[a][f] for f in FAMS])
        best = min(per, key=lambda k: per[k].mean())
        oracle = np.min(np.vstack(list(per.values())), 0)
        o = {"best": best, "best_mean": float(per[best].mean()), "oracle": float(oracle.mean()), "rows": {}}
        print(f"== {key}: best constant {best} {per[best].mean():.2f}, oracle per family {oracle.mean():.2f}")
        for en, est in EST.items():
            l0, h0 = route(est, None, F, h, q, False)
            m0 = boot(l0 - per[best])
            o["rows"][f"probes only [{en}]"] = {"mean": float(l0.mean()), "vs best": m0, "human share": h0}
            print(f"   probes only [{en:10s}]                 {l0.mean():5.2f}  vs best {m0[0]:+.2f} [{m0[1]:+.2f}, {m0[2]:+.2f}]")
            for mn, mon in MON.items():
                l1, h1 = route(est, mon, F, h, q)
                m1, m2 = boot(l1 - per[best]), boot(l1 - l0)
                o["rows"][f"combined [{en} | {mn}]"] = {"mean": float(l1.mean()), "vs best": m1, "vs probes only": m2, "human share": h1}
                print(f"   combined [{en:10s} | {mn:13s}] {l1.mean():5.2f}  vs best {m1[0]:+.2f} [{m1[1]:+.2f}, {m1[2]:+.2f}]"
                      f" | vs probes only {m2[0]:+.2f} [{m2[1]:+.2f}, {m2[2]:+.2f}]  human share {h1:.2f}")
        res["routing"][key] = o
    return res


OUT = {}
if not E_new.empty:
    OUT["new"] = analyze(E_new, NEW, "NEW: 30 replication families")
OUT["all"] = analyze(pd.concat([E_old, E_new], ignore_index=True), OLD + NEW, "ALL: 20 earlier + 30 new families")
json.dump(OUT, open(HERE / "results_rep.json", "w"), indent=1, default=float)
