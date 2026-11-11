"""Exp 5, third testbed (exploratory, written after combo_sop.out): the synthetic procedures of ../synth, round 1.
Within every procedure the odd cases play the probes and the even cases the real deployment. Agents lite, flash, g38.
Monitor M2 trained on the probe-half episodes (all procedures), applied to the real half; router per procedure over
{human, agent a, agent a + monitor} with the probe-half failure rate (calibrated by the level shift on the other
procedures). Same decision rule and costs as combo.py; API per episode from the synth ledger.

  python combo_synth.py -> combo_synth.out (stdout)
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
S = HERE.parent / "synth"
sys.path.insert(0, str(S)); sys.path.insert(0, str(HERE.parent / "monitor"))
import dataset, engine  # noqa: E402
from build import features, tau_events  # noqa: E402
import combo as C  # noqa: E402  (reuses gbm, curve, boot, FEATS, RATES)

AGS = ["lite", "flash", "g38"]
API = {"lite": 0.0009, "flash": 0.0046, "g38": 0.0101}   # agent USD per episode, round 1 of ../synth (ledger / episodes)
rows = []
for g in ("haiku", "sonnet", "opus"):
    items = {it["uid"]: it for it in dataset.cases(g)}
    for a in AGS:
        for l in open(S / "runs" / f"{g}_{a}.jsonl"):
            r = json.loads(l)
            if r.get("reward") is None or r["task_id"] not in items:
                continue
            it = items[r["task_id"]]
            writes = {t["name"] for t in it["spec"]["tools"] if engine.tool_is_write(t)}
            msgs = json.load(open(S / "runs" / "msgs" / f"{g}_{a}" / f"{r['task_id']}.json"))
            k = int(it["case"]["case_id"][1:])
            rows.append({"agent": a, "proc": f"{g}/{it['pid']}", "half": "probe" if k % 2 else "real", "uid": r["task_id"],
                         "fail": 1 - int(r["reward"] >= 1), **features(tau_events(msgs), writes)})
E = pd.DataFrame(rows).drop_duplicates("uid" if False else ["uid", "agent"])
PROBE = E[E.half == "probe"].reset_index(drop=True)
REAL = E[E.half == "real"].reset_index(drop=True)
PROCS = sorted(set(REAL.proc) & set(PROBE.proc))
print(f"{len(PROCS)} procedures; probe half {len(PROBE)} episodes, real half {len(REAL)}")
print(REAL.groupby("agent").fail.mean().round(3).to_dict(), "real-half failure")

real_rate = {a: REAL[REAL.agent == a].groupby("proc").fail.mean() for a in AGS}
raw = {a: PROBE[PROBE.agent == a].groupby("proc").fail.agg(lambda x: (x.sum() + 0.5) / (len(x) + 1)) for a in AGS}
est = {a: {} for a in AGS}
for a in AGS:
    for p in PROCS:
        o = [x for x in PROCS if x != p]
        est[a][p] = float(np.clip(raw[a][p] + real_rate[a][o].mean() - raw[a][o].mean(), 0.01, 0.99))

oof = pd.Series(np.nan, index=PROBE.index)
for p in PROCS:
    a_, b_ = PROBE[PROBE.proc != p], PROBE[PROBE.proc == p]
    oof[b_.index] = C.gbm().fit(a_[C.FEATS], a_.fail).predict_proba(b_[C.FEATS])[:, 1]
s = pd.Series(C.gbm().fit(PROBE[C.FEATS], PROBE.fail).predict_proba(REAL[C.FEATS])[:, 1], index=REAL.index)
cur = {a: C.curve(oof[PROBE.agent == a].values, PROBE[PROBE.agent == a].fail.values) for a in AGS}
for a in AGS:
    m = REAL.agent == a
    print(f"monitor on real half, {a}: AUC {roc_auc_score(REAL[m].fail, s[m]):.3f}")


def route(F, h, q, use_monitor=True):
    H = h + F * (1 - q)
    out = []
    for p in PROCS:
        opts = [("human", None, None, H)]
        for a in AGS:
            pp = est[a][p]
            opts.append(("agent", a, None, API[a] + F * pp))
            if use_monitor:
                for r, (thr, tpr, fpr) in cur[a].items():
                    if r:
                        opts.append(("mon", a, r, API[a] + (pp * tpr + (1 - pp) * fpr) * H + pp * (1 - tpr) * F))
        kind, a, r, _ = min(opts, key=lambda o: o[3])
        if kind == "human":
            out.append(H); continue
        te = REAL[(REAL.proc == p) & (REAL.agent == a)]
        fail = te.fail.values
        if kind == "agent":
            out.append((API[a] + F * fail).mean()); continue
        flag = s[te.index].values > cur[a][r][0]
        out.append(np.where(flag, API[a] + H, API[a] + F * fail).mean())
    return np.array(out)


for F, h, q in C.COSTS:
    H = h + F * (1 - q)
    per = {"always human": np.full(len(PROCS), H)}
    for a in AGS:
        per[f"always {a}"] = np.array([API[a] + F * real_rate[a][p] for p in PROCS])
    best = min(per, key=lambda k: per[k].mean())
    oracle = np.min(np.vstack(list(per.values())), 0)
    r0, r1 = route(F, h, q, False), route(F, h, q, True)
    m0 = C.boot(r0 - per[best]); m1 = C.boot(r1 - per[best]); m2 = C.boot(r1 - r0)
    print(f"F={F} h={h} q={q}: best constant {best} {per[best].mean():.3f}, oracle per procedure {oracle.mean():.3f} | "
          f"probes only {r0.mean():.3f} ({m0[0]:+.3f} [{m0[1]:+.3f}, {m0[2]:+.3f}]) | probes + monitor {r1.mean():.3f} "
          f"({m1[0]:+.3f} [{m1[1]:+.3f}, {m1[2]:+.3f}]); monitor adds {m2[0]:+.3f} [{m2[1]:+.3f}, {m2[2]:+.3f}]")
