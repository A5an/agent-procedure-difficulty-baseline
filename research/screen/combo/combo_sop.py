"""Exp 5, SOPBench replication (PREREG.md addendum): g38 probes + probe-trained monitor router, recorded data only.

  python combo_sop.py  -> combo_sop.out (stdout), results_sop.json
"""
import csv, json, sys, collections
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
G = HERE.parent / "ideas5" / "smart_gemini"
sys.path.insert(0, str(HERE.parent / "monitor"))
from build import features  # noqa: E402

rng = np.random.default_rng(20261111)
API = 0.021
FEATS = ["has_commit", "n_calls", "n_distinct_tools", "n_tool_errors", "n_empty_arg_calls", "n_repeat_calls", "n_user_msgs",
         "n_agent_texts", "log_chars", "user_confirmed", "agent_asked", "n_commit_args", "grounding"]
RATES = [round(0.1 * i, 1) for i in range(10)]
COSTS = [(20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)]


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def events(inter):
    ev = []
    for m in inter:
        if m.get("sender") == "user":
            ev.append({"kind": "user", "text": str(m.get("content") or "")})
        elif "tool_name" in m:
            ev.append({"kind": "tool", "text": str(m.get("content")), "error": False})
        else:
            if m.get("content"):
                ev.append({"kind": "say", "text": str(m["content"])})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc); a = fn.get("arguments") or "{}"
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                except Exception:
                    a = {}
                ev.append({"kind": "call", "name": fn.get("name"), "args": a if isinstance(a, dict) else {}})
    return ev


lab_real = {r["case_id"]: 1 - int(r["success"]) for r in csv.DictReader(open(G / "real_results.csv"))}
lab_syn = {r["synthetic_case_id"]: 1 - int(r["success"]) for r in csv.DictReader(open(G / "synth_results.csv"))}
rows = []
for src, path, lab in (("real", "real.jsonl", lab_real), ("probe", "synth.jsonl", lab_syn)):
    for l in open(G / "episodes" / path):
        r = json.loads(l)
        cid = r["case_id"]
        if cid not in lab or not r.get("interaction"):
            continue
        dom, goal = cid.split("/")[:2]
        rows.append({"src": src, "case": cid, "proc": f"{dom}/{goal}", "dom": dom, "fail": lab[cid],
                     **features(events(r["interaction"]), {goal})})
E = pd.DataFrame(rows)
REAL = E[E.src == "real"].reset_index(drop=True)
PROBE = E[E.src == "probe"].reset_index(drop=True)
PROCS = sorted(set(REAL.proc) & set(PROBE.proc))
REAL = REAL[REAL.proc.isin(PROCS)].reset_index(drop=True)
PROBE = PROBE[PROBE.proc.isin(PROCS)].reset_index(drop=True)
print(f"{len(PROCS)} procedures; real {len(REAL)} (fail {REAL.fail.mean():.3f}), probes {len(PROBE)} (fail {PROBE.fail.mean():.3f})")

real_rate = REAL.groupby("proc").fail.mean()
raw = PROBE.groupby("proc").fail.agg(lambda x: (x.sum() + 0.5) / (len(x) + 1))
dom_of = REAL.groupby("proc").dom.first()
EST = {"probes": raw.clip(0.01, 0.99)}
cal = {}
for p in PROCS:
    others = [q for q in PROCS if dom_of[q] != dom_of[p]]
    cal[p] = raw[p] + real_rate[others].mean() - raw[others].mean()
EST["probes+cal"] = pd.Series(cal).clip(0.01, 0.99)
EST["no probes"] = pd.Series({p: real_rate[[q for q in PROCS if dom_of[q] != dom_of[p]]].mean() for p in PROCS})


def curve(scores, fails):
    out = {}
    for r in RATES:
        thr = np.inf if r == 0 else np.quantile(scores, 1 - r)
        flag = scores > thr if r else np.zeros(len(scores), bool)
        out[r] = (thr, float(flag[fails == 1].mean()) if (fails == 1).any() else 0.0,
                  float(flag[fails == 0].mean()) if (fails == 0).any() else 0.0)
    return out


def oof_by(df, key):
    s = pd.Series(np.nan, index=df.index)
    for g in df[key].unique():
        a_, b_ = df[df[key] != g], df[df[key] == g]
        s[b_.index] = gbm().fit(a_[FEATS], a_.fail).predict_proba(b_[FEATS])[:, 1]
    return s


MON = {}
# M2: all probe episodes; curve from out-of-fold by procedure
o = oof_by(PROBE, "proc")
full = gbm().fit(PROBE[FEATS], PROBE.fail)
s2 = pd.Series(full.predict_proba(REAL[FEATS])[:, 1], index=REAL.index)
c2 = curve(o.values, PROBE.fail.values)
MON["M2 probe episodes"] = {p: {"score": s2[REAL.proc == p], "curve": c2} for p in PROCS}
# M2 without the target procedure's probes (one model per procedure; curve from the all-probe OOF scores)
s2b = pd.Series(np.nan, index=REAL.index)
for p in PROCS:
    tr = PROBE[PROBE.proc != p]
    te = REAL[REAL.proc == p]
    s2b[te.index] = gbm().fit(tr[FEATS], tr.fail).predict_proba(te[FEATS])[:, 1]
MON["M2 other procedures' probes"] = {p: {"score": s2b[REAL.proc == p], "curve": c2} for p in PROCS}
# M1: real episodes of the other domains
s1 = pd.Series(np.nan, index=REAL.index)
M1 = {}
for d in REAL.dom.unique():
    tr = REAL[REAL.dom != d]
    te = REAL[REAL.dom == d]
    s1[te.index] = gbm().fit(tr[FEATS], tr.fail).predict_proba(te[FEATS])[:, 1]
    c1 = curve(oof_by(tr, "dom").values, tr.fail.values)
    for p in te.proc.unique():
        M1[p] = {"score": s1[REAL.proc == p], "curve": c1}
MON["M1 real logs of other domains"] = M1


def within_proc_auc(s):
    v, w = [], []
    for p in PROCS:
        m = REAL.proc == p
        y = REAL[m].fail
        if 0 < y.mean() < 1:
            v.append(roc_auc_score(y, s[REAL[m].index])); w.append(m.sum())
    return float(np.average(v, weights=w)), len(v)


for name, s in (("M2 probe episodes", s2), ("M2 other procedures' probes", s2b), ("M1 real logs of other domains", s1)):
    wp, k = within_proc_auc(s)
    print(f"{name:32s} AUC on real {roc_auc_score(REAL.fail, s):.3f}, within procedure {wp:.3f} ({k} procedures with both outcomes)")


def route(est, mon, F, h, q, allow_monitor=True):
    H = h + F * (1 - q)
    res = []
    for p in PROCS:
        pp = est[p]
        opts = [("human", None, H), ("agent", None, API + F * pp)]
        if allow_monitor and mon is not None:
            for r, (thr, tpr, fpr) in mon[p]["curve"].items():
                if r:
                    opts.append(("agent+monitor", r, API + (pp * tpr + (1 - pp) * fpr) * H + pp * (1 - tpr) * F))
        kind, r, _ = min(opts, key=lambda o: o[2])
        fail = REAL[REAL.proc == p].fail.values
        if kind == "human":
            res.append((p, kind, H, 1.0, 0.0)); continue
        if kind == "agent":
            res.append((p, kind, (API + F * fail).mean(), 0.0, fail.mean())); continue
        flag = mon[p]["score"].values > mon[p]["curve"][r][0]
        res.append((p, f"monitor@{r}", np.where(flag, API + H, API + F * fail).mean(), flag.mean(), (fail * ~flag).mean()))
    return pd.DataFrame(res, columns=["proc", "choice", "loss", "human_share", "residual_fail"])


def boot(d, B=4000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


OUT = {}
for F, h, q in COSTS:
    H = h + F * (1 - q)
    key = f"F={F} h={h} q={q}"
    per = {"always human": np.full(len(PROCS), H), "always g38": np.array([API + F * real_rate[p] for p in PROCS])}
    best = min(per, key=lambda k: per[k].mean())
    per["oracle per procedure"] = np.minimum(per["always human"], per["always g38"])
    rows_ = {}
    print(f"\n== {key}  best constant {best} {per[best].mean():.2f}, oracle per procedure {per['oracle per procedure'].mean():.2f}")
    for en, est in EST.items():
        r0 = route(est, None, F, h, q, allow_monitor=False)
        per[f"probes only [{en}]"] = r0.loss.values; rows_[f"probes only [{en}]"] = r0
        for mn, mon in MON.items():
            r1 = route(est, mon, F, h, q)
            per[f"combined [{en} | {mn}]"] = r1.loss.values; rows_[f"combined [{en} | {mn}]"] = r1
    o = {}
    for k, v in per.items():
        if k.startswith(("probes", "combined")):
            m, lo, hi = boot(v - per[best])
            line = f"  {k:58s} {v.mean():5.2f}  vs best {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            o[k] = {"mean": float(v.mean()), "vs best": (m, lo, hi)}
            if k.startswith("combined"):
                en = k[len("combined ["):].split(" | ")[0]
                m2, lo2, hi2 = boot(v - per[f"probes only [{en}]"])
                o[k]["vs probes only"] = (m2, lo2, hi2)
                line += f" | vs probes only {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]"
            r = rows_[k]
            line += f"  human share {r.human_share.mean():.2f} residual fail {r.residual_fail.mean():.3f}"
            print(line)
    OUT[key] = {"best": best, "best_mean": float(per[best].mean()), "oracle": float(per["oracle per procedure"].mean()), "rows": o}
json.dump(OUT, open(HERE / "results_sop.json", "w"), indent=1)
