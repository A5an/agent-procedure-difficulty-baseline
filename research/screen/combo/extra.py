"""Exp 5, exploratory after seeing combo.out: why is the probe-trained monitor (M2) so much better than M1, and does it
need the seeded probes? Within-family AUC; M2 trained on documentation probes only / seeded only; routers with them.

  python extra.py  -> extra.out (stdout)
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import combo as C  # runs combo.py's data loading and main tables (printed once more)

REAL, PROBE, FAMS, AGS = C.REAL, C.PROBE, C.FAMS, C.AGS


def within_family_auc(score):
    vals, ws = [], []
    for f in FAMS:
        for a in AGS:
            m = (REAL.fam == f) & (REAL.agent == a)
            y = REAL[m].fail
            if 0 < y.mean() < 1:
                vals.append(roc_auc_score(y, score[REAL[m].index])); ws.append(len(y))
    return float(np.average(vals, weights=ws)), len(vals)


def m2(src_filter):
    P = PROBE[PROBE.src.isin(src_filter)].reset_index(drop=True)
    oof = pd.Series(np.nan, index=P.index)
    for g in FAMS:
        a_, b_ = P[P.fam != g], P[P.fam == g]
        oof[b_.index] = C.gbm().fit(a_[C.FEATS], a_.fail).predict_proba(b_[C.FEATS])[:, 1]
    full = C.gbm().fit(P[C.FEATS], P.fail)
    s = pd.Series(full.predict_proba(REAL[C.FEATS])[:, 1], index=REAL.index)
    cur = {a: C.curve(oof[P.agent == a].values, P[P.agent == a].fail.values) for a in AGS}
    return s, {f: {"score": s[REAL.fam == f], "curve": cur} for f in FAMS}


# M2 trained on probe episodes of the OTHER families only (no probe of the target family): does it still work?
def m2_lofo(src_filter):
    P = PROBE[PROBE.src.isin(src_filter)].reset_index(drop=True)
    s = pd.Series(np.nan, index=REAL.index)
    mon = {}
    for f in FAMS:
        tr = P[P.fam != f].reset_index(drop=True)
        model = C.gbm().fit(tr[C.FEATS], tr.fail)
        te = REAL[REAL.fam == f]
        s[te.index] = model.predict_proba(te[C.FEATS])[:, 1]
        oof = pd.Series(np.nan, index=tr.index)
        for g in FAMS:
            if g == f:
                continue
            a_, b_ = tr[tr.fam != g], tr[tr.fam == g]
            oof[b_.index] = C.gbm().fit(a_[C.FEATS], a_.fail).predict_proba(b_[C.FEATS])[:, 1]
        mon[f] = {"score": s[te.index], "curve": {a: C.curve(oof[tr.agent == a].values, tr[tr.agent == a].fail.values) for a in AGS}}
    return s, mon


print("\n== monitor quality on the 240 real episodes")
variants = {}
for name, srcs in (("probes doc+seeded", ["doc", "seeded"]), ("probes doc only", ["doc"]), ("probes seeded only", ["seeded"])):
    variants[f"M2 {name}"] = m2(srcs)
    variants[f"M2 {name}, other families only"] = m2_lofo(srcs)
variants["M1 real logs of other families"] = (pd.concat([C.MON["M1 real logs of other families"][f]["score"] for f in FAMS]), C.MON["M1 real logs of other families"])
for name, (s, _) in variants.items():
    s = s[REAL.index]
    wf, k = within_family_auc(s)
    print(f"  {name:48s} AUC {roc_auc_score(REAL.fail, s):.3f}  within family x agent {wf:.3f} ({k} cells)")

print("\n== routers at the primary costs (F 20, h 5, q 0.99); best constant = always human 5.20")
F, h, q = 20, 5, 0.99
for en in ("seeded+cal", "doc+cal", "doc", "no probes"):
    for name, (_, mon) in variants.items():
        r = C.route(C.EST[en], mon, F, h, q)
        d = r.loss.values - (h + F * (1 - q))
        m, lo, hi = C.boot(d)
        p0 = C.route(C.EST[en], None, F, h, q, allow_monitor=False).loss.values
        m2_, lo2, hi2 = C.boot(r.loss.values - p0)
        print(f"  [{en:10s} | {name:48s}] {r.loss.mean():.2f}  vs human {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  vs probes only {m2_:+.2f} [{lo2:+.2f}, {hi2:+.2f}]  human share {r.escalated.mean():.2f}")
